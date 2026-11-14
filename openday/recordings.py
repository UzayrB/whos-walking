"""The thesis recordings, played to the app without the boards: Room B walks by participant code (P1-P6), stitched
into one continuous RX2 stream, the replay playlists, and the stand-in serial port that feeds the stream in.

This file, in order:
  1. The thesis recordings, by participant code: stem (with the laptop-only model manifest in state/model, the
     only places a code becomes a name), load_rx2, load_gt, passes, gt_speed, segment.
  2. Stitching one stream from many pieces: Stitcher, IdleSource (empty-room frames for the gaps), walk_piece.
  3. Playlists (scripts/*.yaml): Plan and build_plan.
  4. The stand-in serial port: Replay plays the stream to the thesis recorder and types the operator's keys.

The replay feeds a stitched stream through the chain: walk segments (first anchor - pre_s to last anchor +
post_s) joined by real empty-room frames, each join cross-faded, each piece rebased
so the device clock runs on with native spacing, the host stamps keep their recorded serial batching, `pkt_id`
counts up by one and `local_timestamp` stays monotonic (mod 2^32, as the receiver's counter does).
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import yaml

import openday

# --- 1. The thesis recordings, by participant code ---


def stem(code: str, day: int, block: str) -> str:
    """A recording's file name without its ending. stem and the model manifest (state/model, laptop only) are the
    only places a participant code becomes the first name the thesis filed the recording under
    (openday.recording_name, from the laptop-only configs/people.local.yaml)."""
    return f"participant_{openday.recording_name(code)}_idpath_d{int(day)}_{block}"


@lru_cache(maxsize=64)
def load_rx2(code: str, day: int, block: str) -> np.ndarray:
    """RX2 frames of one Room B identification recording (RX2 stood 0.5 m beside the TX: the demo geometry)."""
    openday.use_code()
    from pipeline.io.Capture.capture_logger import read_capture_file
    p = os.path.join(openday.ROOM_B_ID, f"session_0{int(day)}", f"{stem(code, day, block)}_RX2.bin")
    return read_capture_file(p)[1]


@lru_cache(maxsize=64)
def load_gt(code: str, day: int, block: str) -> dict:
    p = os.path.join(openday.ROOM_B_ID, f"session_0{int(day)}", f"{stem(code, day, block)}.gt.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def passes(code: str, day: int, block: str, direction: Optional[str] = None) -> list[dict]:
    """Unflagged passes with at least two anchors, optionally one direction (S = towards the boards)."""
    out = []
    for p in load_gt(code, day, block)["passes"]:
        if p.get("flagged") or len(p.get("anchor_times", [])) < 2:
            continue
        if direction and p["direction"] != direction:
            continue
        out.append(p)
    return out


def gt_speed(p: dict, cfg: dict) -> float:
    """Ground-truth pace: the lane length (5 m) over the first-to-last anchor time."""
    a = p["anchor_times"]
    return float(cfg["eval"]["lane_m"]) / (a[-1] - a[0])


def segment(fr: np.ndarray, t0: float, t1: float) -> np.ndarray:
    tm = fr["t_mono"]
    return fr[(tm >= t0) & (tm <= t1)]


# --- 2. Stitching one stream from many pieces ---


class Stitcher:
    """Joins frame pieces into one stream. `add` returns the shift from the piece's recorded t_mono to the
    stream's, so recorded times (anchors) can be moved onto the stream clock.

    Two recordings never match where they meet: each subcarrier's static level jumps, and the high-pass turns
    that step into a burst the chain reads as motion. So the first `xfade_s` of every piece after the first is
    cross-faded in amplitude (raised cosine) from the previous piece's last frames, keeping the new piece's
    phase (the chain uses amplitude only). This reduces the burst; it does not remove it.
    """

    def __init__(self, cfg: dict) -> None:
        self.mod = int(cfg["clock"]["modulus_us"])
        self.dt_us = int(round(1e6 / float(cfg["radio"]["fs_hz"])))
        self.n_fade = int(round(float(cfg["replay"]["xfade_s"]) * float(cfg["radio"]["fs_hz"])))
        self.lt_next, self.tm_base = 1_000_000, 1000.0     # the stream starts at device 1 s, host 1000 s
        self.parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(self, fr: np.ndarray) -> float:
        if not len(fr):
            return 0.0
        if self.parts and self.n_fade:
            fr = _crossfade(self.parts[-1][0], fr, self.n_fade)
        lt = fr["local_timestamp"].astype(np.int64)
        tm = fr["t_mono"].astype(np.float64)
        lag = tm - lt * 1e-6
        lt_new = lt - lt[0] + self.lt_next
        tm_new = lt_new * 1e-6 + (lag - lag.min()) + self.tm_base
        dt = int(np.median(np.diff(lt))) if len(lt) > 1 else self.dt_us
        self.lt_next = int(lt_new[-1]) + dt
        self.parts.append((fr, lt_new, tm_new))
        return float(tm_new[0] - tm[0])

    @property
    def t_end(self) -> float:
        """Stream t_mono just after the last frame so far."""
        return self.lt_next * 1e-6 + self.tm_base

    def frames(self) -> np.ndarray:
        out = np.concatenate([p[0] for p in self.parts]).copy()
        tm = np.maximum.accumulate(np.concatenate([p[2] for p in self.parts]))
        lt = np.concatenate([p[1] for p in self.parts])
        out["t_host"] = out["t_host"] - out["t_mono"] + tm
        out["t_mono"] = tm
        out["local_timestamp"] = (lt % self.mod).astype(np.uint64)
        out["pkt_id"] = np.arange(1, 1 + len(out), dtype=np.int64)
        return out


def _crossfade(prev: np.ndarray, fr: np.ndarray, n: int) -> np.ndarray:
    """fr with its first n frames' amplitudes blended from prev's last n (raised cosine), fr's own phase."""
    n = min(n, len(prev), len(fr))
    if n < 2:
        return fr
    fr = fr.copy()
    a = prev["data"][-n:].astype(np.float32).reshape(n, 128, 2)
    b = fr["data"][:n].astype(np.float32).reshape(n, 128, 2)
    amp_a, amp_b = np.sqrt((a * a).sum(-1)), np.sqrt((b * b).sum(-1))
    w = (0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n))[:, None]
    amp = (1 - w) * amp_a + w * amp_b
    unit = np.where(amp_b[..., None] > 0, b / np.maximum(amp_b, 1e-9)[..., None], np.array([0.0, 1.0]))
    fr["data"][:n] = np.clip(np.round(unit * amp[..., None]), -128, 127).astype(np.int8).reshape(n, 256)
    return fr


class IdleSource:
    """Empty-room frames for idle gaps, cycling through one empty_idspot recording."""

    def __init__(self, code: str, day: int) -> None:
        self.fr = load_rx2(code, day, "empty_idspot")
        self.i = 0

    def take(self, st: Stitcher, seconds: float) -> None:
        need = int(round(seconds * len(self.fr) / max(self.fr["t_mono"][-1] - self.fr["t_mono"][0], 1e-9)))
        while need > 0:
            j = min(len(self.fr), self.i + need)
            st.add(self.fr[self.i:j])
            need -= j - self.i
            self.i = 0 if j >= len(self.fr) else j


def walk_piece(st: Stitcher, p: dict, fr: np.ndarray, cfg: dict) -> float:
    """Add one pass's walk segment; returns its first anchor (the click) on the stream clock."""
    r = cfg["replay"]
    a = p["anchor_times"]
    shift = st.add(segment(fr, a[0] - r["pre_s"], a[-1] + r["post_s"]))
    return a[0] + shift


# --- 3. Playlists (scripts/*.yaml) ---


@dataclass
class Plan:
    """A stitched replay: the frames, the operator's keys on the stream clock, and the truth for each walk."""

    frames: np.ndarray
    keys: list[tuple[float, str]]
    walks: list[dict] = field(default_factory=list)
    players: int = 0                                       # new_visitor steps (the replay checks none was refused)


def build_plan(script_path: str, cfg: dict, speed: float = 1.0) -> Plan:
    """Parse a playlist (scripts/*.yaml) and stitch its stream for a replay at `speed`. Passes are never reused
    within one run. A walk after an enrol_uzayr step carries expect: enrol_uzayr (the replay checks the app took
    it as his learning walk)."""
    with open(script_path, encoding="utf-8") as f:
        sc = yaml.safe_load(f)
    r = cfg["replay"]
    rng = random.Random(sc.get("seed", 0))
    gap = float(sc.get("gap_s", r["gap_s"]))
    b = sc["baseline"]
    idle = IdleSource(b["name"], b["day"])
    st = Stitcher(cfg)
    keys: list[tuple[float, str]] = []
    walks: list[dict] = []
    used: set = set()
    idle.take(st, float(r["lead_s"]))
    uz_left = players = 0
    for step in sc["steps"]:
        act = step.get("action")
        if act == "baseline":
            t0 = st.t_end + 0.5
            keys.append((t0, "ctrl+b"))
            idle.take(st, float(cfg["baseline"]["live_s"]) + 2.0)
            continue
        if act == "enrol_uzayr":
            # Redo my walks is refused while the answer is still coming (ui.reading_s of real time, so reading_s x
            # --speed of stream time): press it once the answer is up, at any --speed
            wait = max(3.0, float(cfg["ui"]["reading_s"]) * float(speed) + 1.5)
            keys.append((st.t_end + wait, "ctrl+u"))
            idle.take(st, wait + 0.7)
            uz_left = int(cfg["pools"].get("n_enrol_uzayr", cfg["pools"]["n_enrol"]))   # his learning walks
            continue
        if act == "new_visitor":
            players += 1
            t = st.t_end + 0.3
            for k in ["n", *[c if c != " " else "space" for c in step["name"]], "enter"]:
                keys.append((t, k))
                t += float(r["key_interval_s"])
            idle.take(st, t - st.t_end + 1.0)
            continue
        name, day, block = step["walk"], int(step["day"]), step["block"]
        cands = [p for p in passes(name, day, block, "S") if (name, day, block, p["pass_id"]) not in used]
        if not cands:
            raise ValueError(f"no unused {name} d{day} {block} passes left for step {step}")
        p = rng.choice(cands)                              # (a random unused S pass)
        used.add((name, day, block, p["pass_id"]))
        click = walk_piece(st, p, load_rx2(name, day, block), cfg)
        keys.append((click, "space"))
        role = "uzayr" if (uz_left > 0 or name == sc["uzayr_standin"]) else "visitor"
        expect = "enrol_uzayr" if uz_left > 0 else None
        uz_left = max(0, uz_left - 1)
        walks.append({"t_click": click, "role": role, "gt_speed": gt_speed(p, cfg), "expect": expect})
        idle.take(st, gap)
    idle.take(st, r["tail_s"])
    return Plan(st.frames(), sorted(keys), walks, players)


# --- 4. The stand-in serial port ---


class Replay:
    """The stitched stream as a stand-in `serial` module (Code/development/live/Checks/stream_source.py), and the
    operator's keys injected on the same clock. Registered in sys.modules["serial"] only here, never in live mode."""

    NODE, PORT = "RX", "replay://RX"

    def __init__(self, plan: Plan, cfg: dict, speed: float, put_key: Callable[[str, float], None]) -> None:
        openday.use_code()
        from development.live.Checks.stream_source import ReplayPort, serial_module
        self.plan, self.speed, self.put_key = plan, float(speed), put_key
        self.t_rec0 = float(plan.frames["t_mono"][0])
        self.port = ReplayPort(plan.frames, self.t_rec0, 0.0, self.speed)
        sys.modules["serial"] = serial_module({self.PORT: self.port})
        self.delay = float(cfg["replay"]["start_delay_s"])
        self.t_wall0 = 0.0

    @property
    def nodes(self) -> dict:
        return {self.NODE: self.PORT}

    @property
    def duration(self) -> float:
        return self.delay + (float(self.plan.frames["t_mono"][-1]) - self.t_rec0) / self.speed + 1.0

    def start(self) -> None:
        """Start the stream's clock `start_delay_s` from now, and the key injector."""
        self.t_wall0 = time.monotonic() + self.delay
        self.port.t_wall0 = self.t_wall0

        def inject() -> None:
            for t, key in self.plan.keys:
                due = self.t_wall0 + (t - self.t_rec0) / self.speed
                while time.monotonic() < due:
                    time.sleep(min(0.002, max(0.0, due - time.monotonic())))
                self.put_key(key, time.monotonic())
        threading.Thread(target=inject, name="replay-keys", daemon=True).start()
