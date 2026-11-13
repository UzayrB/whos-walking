"""The app: what each click and key does, whose turn it is, the light, and the view the screen draws; and, at the
bottom, running it live on the boards or on a replay.

This file, in order:
  1. RunLog (the run log) and small text helpers.
  2. App: packets in (handle, tick), keys (on_key_now), the round (turns, Uzayr's kept walks, Redo my walks, the
     room measurement, walks and Undo walk), timers and the light (health), the view (view, _op_hint, _buttons,
     _card, _dots) and the replay's summary.
  3. Running it: the receiver's port, then run_live (the boards), run_replay (a playlist, no boards) and keytest.

`handle` only queues the packet (no work, never raises). `tick` runs every loop iteration (about 150 times a
second with one receiver): it drains keys, runs the chain once a batch has built up, advances timers and draws at
most `fps_max` times a second, slowing down by itself if the stream starts to lag.

Privacy: the visitor's name exists only in `self.name`. Logs, events and saved files say "visitor".

The screen: crowd text on the card and operator hints beside the buttons, notices that replace only the
instruction line, one main button that never moves (Start walk / They've stopped / Next walk), no public score,
and the room measured with Measure room (after a countdown).

The room level of the day: the first measurement of the calendar day that nobody moved in is the one in use, and
it is kept in state/baseline_live.json, so Use last measurement offers it all day, also after a restart. A later
Measure room replaces it only when clearly quieter (under baseline.quieter_ratio times it); anything else, or one
someone moved in, keeps it (the operator's line says so, and Uzayr's kept walks stay). live --new-room forgets it
(the boards were moved). A test recording (--record) always measures, always uses its own measurement, as the bench
tools expect, and never writes the file.

The round (pools.enrol_order visitor_first): New player, the name, the visitor's pools.n_enrol learning walks, then
Uzayr's, then the game (alternate: Uzayr first, then taking turns). Uzayr's learning walks are kept in memory across
players and reused while fresh (pools.uzayr_reuse_min: all younger than that when New player is clicked, taken after
the current room measurement, in this app run), so most rounds need only the visitor's walks; Redo my walks (Ctrl+U)
takes his again at any time between walks (Undo walk, or Ctrl+U again, keeps the old ones until his first new walk
counts). New player and End game wipe the visitor only; Quit wipes everything.

Uzayr walks pools.n_enrol_uzayr learning walks (5; his are kept for the next players), the visitor pools.n_enrol (2).

The second look (classifier.second_look): a game walk that is a close call (its margin below the model's tau) names
nobody yet. After Thinking ... the card says "Walk once more!" ("Too close to call: the same walker again.", state
AGAIN), the same walker walks once more, and the answer is decided on both walks, their readings averaged
(models.average_walks; the tiles show those averaged readings, "Both walks"), and says "(from both walks)", How sure
from the model's two-walk curve (Model.how_sure_both: cal2_m / cal2_p, fitted separately after training and added to
the model file, laptop only; the one-walk curve when the file has none, logged once at start). At most one extra walk
per game walk; a missed second walk leaves the first walk's own answer. The close call waits for its second walk
(look2) only on "walk once more" (and its own Thinking step): a second look starts from nowhere else. Every way off
that card comes back to it (Cancel walk or a lost signal during the second walk, Measure room whether it works or
not, a lost signal or a receiver restart, New player then Cancel, the End game and Quit boxes' Keep going: _resting
rests in AGAIN while the close call waits), or drops the close call with the player (Undo walk: "Last walk removed.",
back to the game, nobody named; End game, a new player, Quit). Undo walk on the answer from both walks throws both
away. A recording links the second walk to the first ("second_of" in its walk event), and an It was ... click labels
both.

Labels: a learning walk is labelled by its turn (Undo walk redoes a wrong turn) and joins that person's pool; those
walks are the pools for the whole game. A game walk never joins a pool and nothing is tallied. On the day there are
no It was ... buttons (U / V do nothing). Only in a test recording (--record) It was Uzayr / It was {V} show on an
answer: optional, one click is final, and it only writes the label event (for the bench tools to score later), so a
recording behaves exactly like the day. The answer always names the nearer person ("How sure" is lower on a close
call); the margin and the would-be "not sure" stay in the log and the events.
"""

from __future__ import annotations

import collections
import copy
import json
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

import openday
from openday import chain, models, recordings, screen, walks

NEEDS_BASELINE, BUSY, IDLE, NAME_ENTRY = "NEEDS_BASELINE", "BUSY", "IDLE", "NAME_ENTRY"
ENROL_VISITOR, ENROL_UZAYR, READY, WALKING = "ENROL_VISITOR", "ENROL_UZAYR", "READY", "WALKING"
DECIDING, RESULT = "DECIDING", "RESULT"
AGAIN = "AGAIN"                                            # a close call: the same walker walks once more (second look)
# (short labels, so "like {V}" has room beside them in the narrow tiles: on the MacBook "like Lerato" keeps about
# 0.9 of its size, an 8-letter name about two-thirds, 9 letters 0.6; the steps value says "/min")
TILE = {"speed": ("Speed", "{:.2f} m/s"), "steps": ("Steps", "{:.0f} /min"), "swing": ("Swing", "{:.1f}x"),
        "bounce": ("Bounce", "{:.0f}%")}


class RunLog:
    """logs/openday_<date>.log: one line per event. Never a visitor's name."""

    def __init__(self, path: Optional[str]) -> None:
        self.f = open(path, "a", encoding="utf-8") if path else None

    def __call__(self, msg: str) -> None:
        if self.f:
            self.f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            self.f.flush()

    def close(self) -> None:
        if self.f:
            self.f.close()


def _words(items: list) -> str:
    """["a", "b", "c"] -> "a, b and c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


BIG_NAME = 13                                              # the card's big text: a name up to 13 letters whole
                                                           # (longer: the first word, see App._vname)


def _times(k: int) -> str:
    """1 -> "once", 2 -> "twice", 3 -> "3 times"."""
    return {1: "once", 2: "twice"}.get(k, f"{k} times")


def _today(t: float) -> bool:
    """time.time() t is on today's calendar date (the laptop's clock): the room level is kept for the day."""
    return time.strftime("%Y%m%d", time.localtime(t)) == time.strftime("%Y%m%d")


def _age_text(minutes: float) -> str:
    """How old the last room measurement is, for its button: "7 min old", "3 h old" (whole hours)."""
    m = int(round(minutes))
    return f"{m} min old" if m < 60 else f"{m // 60} h old"


class App:
    """One run of the demo: the chain, the round's state and the screen."""

    def __init__(self, cfg: dict, *, state_dir: str, log: RunLog, headless: bool = False, fullscreen: bool = True,
                 record_dir: Optional[str] = None, baseline_cache: bool = True, keep_log: bool = False,
                 new_room: bool = False) -> None:
        self.cfg, self.log, self.keep_log = cfg, log, keep_log
        self.names = list(cfg["features"]["names"])
        self.use = list(cfg["classifier"].get("use", self.names))
        self.hop_s = chain.hop_s(cfg)
        self.chain = chain.Chain(cfg)
        self.lt_mod = int(cfg["clock"]["modulus_us"])
        self.keys: "queue.Queue[tuple[str, float]]" = queue.Queue()
        self.stop = threading.Event()
        self.pending: list = []
        self.n_pkts, self.last_pkt_wall, self.newest_tm, self.last_id, self.lost = 0, None, None, None, 0
        os.makedirs(state_dir, exist_ok=True)
        self.baseline_path = os.path.join(state_dir, "baseline_live.json") if baseline_cache else None
        self.visitor = models.Pool(cfg["pools"]["n_visitor"])             # memory only, never written
        self.tally = models.Tally(os.path.join(state_dir, "tally.json"))
        n_v = int(cfg["pools"]["n_enrol"])                 # learning walks: the visitor's, and Uzayr's own (the same
        self.n_enrol = {"visitor": n_v, "uzayr": int(cfg["pools"].get("n_enrol_uzayr", n_v))}   # when not set)
        self.uz_fresh = models.Pool(max(int(cfg["pools"]["n_uzayr"]), self.n_enrol["uzayr"]))   # his, memory only
        self.second_look = bool(cfg["classifier"].get("second_look", False))   # a close call: walk once more
        self.look2: Optional[dict] = None                  # the close call waiting for its second walk (state AGAIN)
        self.order = str(cfg["pools"].get("enrol_order", "visitor_first")).lower()
        if self.order not in ("visitor_first", "alternate"):
            raise ValueError(f"pools.enrol_order must be visitor_first or alternate, not {self.order!r}")
        self.reuse_min = float(cfg["pools"].get("uzayr_reuse_min", 0) or 0)   # 0 = he walks every round
        self.room_at = 0.0                                 # time.time() of the last room measurement (captured, reused)
        self.uz_reused = False                             # this round reuses Uzayr's walks from an earlier round
        self.uz_redo = False                               # Redo my walks was clicked in this player's game
        self.uz_held: Optional[dict] = None                # Redo my walks: his old walks (memory) till a new one
        self.model = models.load_model(cfg, self.log)       # m1 / m2 (thesis data only); None = m0
        if self.second_look and self.model is not None and self.model.cal2_m is None:
            self.log("second look: the model has no two-walk How sure curve (cal2_m / cal2_p, fitted separately "
                     "after training): an answer from both walks reads the one-walk curve")
        self.name: Optional[str] = None                                   # the visitor's name: memory only
        self.vidx: Optional[int] = None
        self.visitor_ready = False
        self.state, self.prev = NEEDS_BASELINE, IDLE
        self.uz_enrolling = False                          # Ctrl+U pressed and Uzayr's walks still wanted
        self.done = {"uzayr": 0, "visitor": 0}             # enrolment walks done, kept apart per person
        self.typed = ""
        self.quit_until = 0.0
        self.end_until = 0.0                               # the End game box is open until then
        self.debug = False
        self.walk: Optional[dict] = None
        self.last_end_dev: Optional[float] = None          # device time the last walk ended (walk.after_end_lock_s)
        self.last: Optional[dict] = None
        self.walk_log: list[dict] = []
        self.n_walks = 0                                   # walk ids never restart, so bench events stay unambiguous
        self.timer = 0.0
        self.capture: Optional[dict] = None
        self.ring: collections.deque = collections.deque(maxlen=int(round(cfg["ui"]["waterfall_s"] / self.hop_s)))
        self.recent: collections.deque = collections.deque(maxlen=int(cfg["walk"]["preclick_cols"]))
        self.rate_hist: collections.deque = collections.deque()
        self.stats = {"draw_s": [], "draw_at": [], "lag": [], "tick_s": []}
        self.fps = float(cfg["ui"]["fps_max"])
        self.last_pump = self.last_fps_adj = self.next_draw = 0.0
        self.record_dir = record_dir
        self.events = open(os.path.join(record_dir, "events.jsonl"), "a", encoding="utf-8") if record_dir else None
        self.closed = False
        # the 3D picture above the waterfall: how much the signal wobbles on each subcarrier (display only)
        self.ridge = chain.Ridge(cfg)
        self.ridge_cols = 0                                # waterfall columns since the ridges last restarted
        self._ridge_err_at = -1e9                          # (a ridge fault is logged at most every 10 s)
        self.notice: Optional[tuple] = None                # (text, colour, until): replaces the instruction line
        self.hint_ev: Optional[tuple] = None               # (text, colour, until): an event hint for the operator
        self.misses = 0                                    # walks in a row the app did not see
        self.last_end_wall: Optional[float] = None         # host time the last walk ended (the light's grey)
        self.busy_since: Optional[float] = None            # busy >= health.busy_amber since then
        self.quiet_e0: Optional[float] = None              # the quietest room measurement this run (a recording's
                                                           # 1.3x check)
        self.screen = screen.Screen(cfg, self.on_key, self.on_close, headless=headless, fullscreen=fullscreen,
                                record=record_dir is not None)
        nb = int(np.sum(chain.speed_of(self.chain.freqs, cfg) <= float(cfg["ui"]["speed_max_mps"])))
        self.n_bins = max(nb, 2)
        df = self.chain.freqs[1] - self.chain.freqs[0]
        self.screen.set_speed_axis(float(chain.speed_of(self.chain.freqs[self.n_bins - 1] + df / 2, cfg)), self.n_bins)
        if new_room and self.baseline_path and os.path.exists(self.baseline_path):
            os.remove(self.baseline_path)                  # --new-room: the boards were moved, today's level is wrong
        self.cached = None if record_dir else self._load_cached_baseline()   # a bench run always takes Ctrl+B
        self.log(f"start variant {cfg['dsp']['variant']} nfft {cfg['dsp']['nfft']} record {bool(record_dir)}"
                 + (" new room: today's room level forgotten" if new_room else ""))

    # -- run_capture consumer -------------------------------------------------------------------
    def handle(self, rec) -> None:
        """Queue one packet. Nothing else: the chain runs in tick."""
        # the firmware prints the 32-bit counter with %d, so after 35.8 min it reads negative: back to unsigned
        self.pending.append((int(rec.local_timestamp) % self.lt_mod, rec.t_mono, rec.data))
        self.n_pkts += 1
        self.last_pkt_wall = time.monotonic()
        self.newest_tm = rec.t_mono
        if self.last_id is not None and rec.pkt_id > self.last_id:
            self.lost += max(0, rec.pkt_id - self.last_id - 1)
        self.last_id = rec.pkt_id

    def tick(self, now: float) -> None:
        if self.closed:
            return
        t0 = time.perf_counter()
        try:
            while True:
                try:
                    key, t = self.keys.get_nowait()
                except queue.Empty:
                    break
                self.on_key_now(key, t)
            d = self.cfg["dsp"]
            if self.pending and (len(self.pending) >= int(d["tick_batch"])
                                 or now - self.pending[0][1] > float(d["tick_max_wait_s"])):
                self._process()
            self._timers(now)
        except Exception as exc:  # noqa: BLE001 - the show must go on; log it and keep the stream flowing
            import traceback
            traceback.print_exc()
            self.log(f"error {type(exc).__name__}: {exc}")
            self._hint("Something went wrong; carrying on.", "red")
        self.stats["tick_s"].append(time.perf_counter() - t0)
        if len(self.stats["tick_s"]) > 20000:
            del self.stats["tick_s"][:10000]
        try:
            self._maybe_draw(now)
        except Exception as exc:  # noqa: BLE001 - a drawing fault must not stop the stream
            if now - getattr(self, "_draw_err_at", -1e9) > 10.0:
                self._draw_err_at = now
                import traceback
                traceback.print_exc()
                self.log(f"draw error {type(exc).__name__}: {exc}")

    def close(self) -> None:
        """Quit: wipe the visitor and keep the tally. Uzayr's learning walks (memory only) go too: they never outlive
        the app run."""
        if self.closed:
            return
        self.closed = True
        if self.walk is not None and "rec" in self.walk:   # its answer is made and only the screen still waited:
            try:                                           # logged and recorded as any walk, before the files close
                self._walk_shown()
            except Exception as exc:                       # (a quit must always close the files and the screen)
                self.log(f"quit: the last walk could not be recorded ({exc})")
        self.wipe_visitor()                                # (it retires the last walk itself)
        self._drop_uzayr()
        self.uz_held = None
        self.log("quit")
        if self.events:
            self.events.close()
        self.screen.close()

    # -- packets -> columns -----------------------------------------------------------------------
    def _process(self) -> None:
        lt = np.array([p[0] for p in self.pending], np.int64)
        tm = np.array([p[1] for p in self.pending], np.float64)
        data = np.stack([p[2] for p in self.pending])
        self.pending = []
        r0, s0 = self.chain.reboots, self.chain.silences
        cols = self.chain.push(lt, tm, data)
        if (self.chain.reboots, self.chain.silences) != (r0, s0):   # rebooted or silent: start clean
            self.recent.clear()
            self._abort("Receiver restarted" if self.chain.reboots != r0 else "Signal lost")
            if self.chain.reboots != r0:                   # the device clock starts a new epoch
                self.last_end_dev = None
            self.ridge.reset()
            self.ridge_cols = 0                            # (the waterfall's older columns aren't theirs)
        try:                                               # this batch's samples before the high-pass (display
            self.ridge.push(self.chain.front.raw_out[1])   # only: a fault here never stops the walk's own work)
        except Exception as exc:  # noqa: BLE001
            self.ridge.reset()
            self.ridge_cols = 0
            self._ridge_fault(exc)
        if not len(cols):
            return
        if self.capture is not None:
            self.capture["cols"].append(cols)
        base = self.chain.baseline
        for i in range(len(cols)):
            if base is not None:
                db = 10 * np.log10(np.maximum(cols.power[i, :self.n_bins], 1e-12) / np.maximum(base.floor[:self.n_bins], 1e-12))
                act = cols.energy[i] > float(self.cfg["dsp"]["active_k"]) * base.e0 and not cols.gap[i]
                v = float(chain.speed_of(cols.f50[i], self.cfg)) if act and np.isfinite(cols.f50[i]) else np.nan
                self.ring.append((np.clip(db, 0, None).astype(np.float32), v, float(cols.energy[i])))
                self.ridge_cols += 1
                self.recent.append((cols.t[i], cols.energy[i], cols.f50[i], cols.f90[i], cols.gap[i]))
            else:
                self.ring.append((np.zeros(self.n_bins, np.float32), np.nan, 0.0))
                self.ridge_cols += 1
        if self.walk is not None and base is not None:
            self._feed_walk(cols)

    def _abort(self, why: str) -> None:
        """A walk or a baseline capture cannot finish (receiver restarted or silent): drop it, say so. A walk whose
        answer was already made (only the screen was still waiting, walk.screen_hold_s) is shown instead."""
        if self.walk is not None and self.walk["w"].ended:
            self._walk_shown()
        if self.walk is not None:
            self.log(f"walk {self.walk['id']} dropped: {why.lower()}")
            self._event("discard", walk_id=self.walk["id"], t_mono=time.monotonic(), why=why)
            self.state = self.walk["from"] if self.walk["from"] not in (RESULT, DECIDING) else self._resting()
            self.walk = None
            self.last_end_wall = time.monotonic()
            self._notice("Lost the signal. Walk again when the light is green.", "red", 4.0)
        if self.capture is not None:
            self.log(f"baseline dropped: {why.lower()}")
            self.capture = None
            self.state = NEEDS_BASELINE if self.chain.baseline is None else self._resting()
            level = self._day_level() if self.chain.baseline is not None else None   # (today's level stays in use)
            self._hint(f"{why}: kept the earlier room level {level.e0:.2f}." if level is not None
                       else f"{why}: click Measure room again.", "amber", 8.0)

    def _recent_columns(self) -> chain.Columns:
        if not self.recent:
            z = np.zeros(0)
            return chain.Columns(z, None, np.zeros(0, bool), None, z, z, z)
        t, e, f5, f9, g = (np.array(x) for x in zip(*self.recent))
        return chain.Columns(t, None, g.astype(bool), None, e, f5, f9)

    # -- keys -----------------------------------------------------------------------------------
    def on_key(self, key: str, t: float) -> None:
        """From the window (called inside the draw's event loop) or the replay injector: queue it."""
        self.keys.put((key, t))

    def on_close(self) -> None:
        self.stop.set()

    def _is(self, key: str, action: str) -> bool:
        names = self.cfg["keys"][action]
        return key in names or (len(key) == 1 and key.lower() in names)

    def _event(self, kind: str, **kw) -> None:
        if self.events:
            self.events.write(json.dumps({"kind": kind, **kw}, default=float) + "\n")
            self.events.flush()

    def on_key_now(self, key: str, t: float) -> None:
        """One key. Next = start_walk's keys (the main button sends space); clicker back =
        keys.clicker_back; Backspace (the Undo walk button) = discard. The quit box and the End game box swallow
        every key but Y (which quits / ends the game); any other key only closes them."""
        key = screen.norm_key(key)
        if self.state == NAME_ENTRY:                       # the name is typed here: never logged, never saved
            self._event("key", key="<name entry>", t_mono=t, state=self.state)
            return self._name_key(key)
        quiet = self.state in (NEEDS_BASELINE, BUSY, WALKING)  # letters do nothing here: never record them
        quitting = bool(self.quit_until) and time.monotonic() < self.quit_until
        acts = any(self._is(key, a) for a in self.cfg["keys"]) and (not quiet or (quitting and self._is(key, "quit_yes")))
        self._event("key", key=key if acts or len(key) != 1 else "<other>", t_mono=t, state=self.state)
        if quitting:
            if self._is(key, "quit_yes"):
                self.log("quit confirmed")
                self.stop.set()
                return
            self.quit_until = 0.0                          # any other key only closes the box: it starts nothing
            return
        if self._ending(time.monotonic()):                 # the End game box: Y ends the game, any other key closes it
            self.end_until = 0.0
            if self._is(key, "quit_yes") and self.state != WALKING:
                self.wipe_visitor()
                self.state = self._resting()
                self.notice = None
                self._hint("Player wiped.")
            return
        nxt, back = self._is(key, "start_walk"), self._is(key, "clicker_back")
        st = self.state
        if self._is(key, "quit"):
            self.quit_until = time.monotonic() + float(self.cfg["ui"]["quit_confirm_s"])
        elif self._is(key, "fullscreen"):                  # Ctrl+F (no button: the app opens full screen)
            self.screen.toggle_fullscreen()
        elif self._is(key, "debug"):
            self.debug = not self.debug
        elif self._is(key, "baseline"):
            self.start_baseline(t)
        elif self._is(key, "enrol_uzayr"):                 # Redo my walks (only with pools.uzayr_reuse_min)
            self._redo_uzayr()
        elif nxt:
            if st == NEEDS_BASELINE:
                self.start_baseline(t)
            elif st == WALKING:
                self.request_stop(t)                       # ignored under walk.stop_min_s after the click
            elif st != BUSY:
                self.start_walk(t)
        elif back:
            if st == WALKING or (st in (ENROL_VISITOR, ENROL_UZAYR) and self.last is not None) \
                    or self._can_unredo():                 # (right after Redo my walks: keeps his old walks)
                self.discard()
            elif st not in (NEEDS_BASELINE, BUSY, DECIDING):
                self._hint("To throw away the last walk: click Undo walk.")
        elif self._is(key, "discard"):                     # (not while the answer is coming: Undo walk is dim)
            if st not in (NEEDS_BASELINE, BUSY, DECIDING):
                self.discard()
        elif self._is(key, "reuse_baseline"):
            if st == NEEDS_BASELINE and self._reusable() is not None:
                self._set_baseline(self.cached, "reused")
        elif st in (NEEDS_BASELINE, BUSY):
            if self._is(key, "new_visitor"):
                self._hint("Wait: measuring the room." if st == BUSY else "Measure the room first.")
        elif self._is(key, "new_visitor"):
            if st == WALKING:
                self._hint("Finish or cancel the walk first.")
            else:
                self.prev, self.state, self.typed = self._resting(), NAME_ENTRY, ""   # (Cancel: back; on a close
                                                                                       # call "walk once more")
        elif self._is(key, "wipe_visitor"):                # End game: asks first, like Quit
            if self.name is None:
                self._hint("No player to end.")
            elif st != WALKING:
                self.end_until = time.monotonic() + float(self.cfg["ui"]["quit_confirm_s"])
        elif self._is(key, "confirm_uzayr") or self._is(key, "confirm_visitor"):
            if self.events is not None:                    # a test recording only: on the day U / V do nothing
                self.confirm("u" if self._is(key, "confirm_uzayr") else "v")

    def _ending(self, now: float) -> bool:
        """The End game box is open."""
        return bool(self.end_until) and now < self.end_until

    def _lock_left(self, t: float) -> float:
        """How long (s) a start is still refused after the last walk ended (walk.after_end_lock_s, on the device
        clock; host time t). 0 when a start is free, and after a receiver reboot (a new clock epoch)."""
        if self.last_end_dev is None:
            return 0.0
        t_dev = self.chain.to_device(t)
        lock = float(self.cfg["walk"].get("after_end_lock_s", 0.0))
        since = None if t_dev is None else t_dev - self.last_end_dev
        return lock - since if since is not None and 0.0 <= since < lock else 0.0

    def _reusable(self) -> Optional[chain.Baseline]:
        """The cached room measurement if it is from today (checked now, not only at start-up)."""
        b = self.cached
        if b is None or not _today(b.created):
            return None
        return None if b.disturbed else b                  # (someone moved during it: measure again instead)

    def _day_level(self) -> Optional[chain.Baseline]:
        """Today's room level: a measurement nobody moved in, today. It is the one in use (only a clearly quieter
        measurement replaces it), or, until the first measurement after a restart, the one Use last measurement
        offers. None while there is none, and in a test recording (it uses every measurement)."""
        if self.record_dir:
            return None
        b = self.chain.baseline if self.chain.baseline is not None else self._reusable()
        return None if b is None or b.disturbed else b

    def _vname(self, n: int) -> str:
        """The visitor's name in at most n characters ("the visitor" when there is none): the whole name, else its
        first word ("Thandiwe Mokoena" -> "Thandiwe"), else the part before a hyphen ("Nomvula-Precious" ->
        "Nomvula"), else that cut to n - 1 letters and a "." (never "Nomvula-." or "Thandiwe ."). One rule for the
        hint, the dots, the tiles, the It was button (all n = 9, so one short form) and the card's big text."""
        if self.name is None:
            return "the visitor"
        if len(self.name) <= n:
            return self.name
        first = self.name.split()[0]
        if len(first) <= n:
            return first
        part = first.split("-")[0].rstrip(" '")
        if 1 < len(part) <= n:
            return part
        return (first[:n - 1].rstrip(" -'") or first[:n - 1]) + "."

    def _notice(self, text: str, colour: str, secs: float) -> None:
        """A notice takes the card's instruction line for secs (the big text and dots stay; on an answer the line is
        added and the name is drawn smaller while it shows)."""
        self.notice = (text, screen.STATUS.get(colour, colour), time.monotonic() + secs)

    def _hint(self, text: str, level: str = "info", secs: float = 4.0, light: Optional[str] = None) -> None:
        """An event hint for the operator (beside the buttons, never on the card) for secs. With light, the light at
        the top says it too (amber, or red), so it can be read from the back mark."""
        self.hint_ev = (text, level, time.monotonic() + secs, light)

    def _name_key(self, key: str) -> None:
        if self._is(key, "clicker_back"):
            return                                         # the clicker's back key never edits the name
        if key != "space" and self._is(key, "start_walk"):
            if not self.typed.strip():
                return                                     # Next with nothing typed does nothing
            key = "enter"
        if key == "enter":
            name, self.typed = self.typed.strip() or "Guest", ""
            self.new_visitor(name)
        elif key == "escape":
            self.state, self.typed = self.prev, ""
        elif key == "backspace":
            self.typed = self.typed[:-1]
        elif len(self.typed) < int(self.cfg["ui"]["name_max_len"]):
            ch = " " if key == "space" else key
            if len(ch) == 1 and (ch.isalpha() or ch in " -'"):
                self.typed += ch

    # -- actions --------------------------------------------------------------------------------
    def _turn(self) -> Optional[str]:
        """Whose learning walk is next, None when both are done. Redo my walks: Uzayr's. pools.enrol_order
        visitor_first: the visitor's n_enrol walks, then Uzayr's (none when his are reused); alternate: Uzayr first,
        then taking turns."""
        nu, nv = self.n_enrol["uzayr"], self.n_enrol["visitor"]
        u, v = self.done["uzayr"], self.done["visitor"]
        if self.uz_enrolling and u < nu:
            return "uzayr"
        if self.order == "visitor_first":
            return "visitor" if v < nv else ("uzayr" if u < nu else None)
        if u < nu and u <= v:
            return "uzayr"
        if v < nv:
            return "visitor"
        return "uzayr" if u < nu else None

    # -- Uzayr's learning walks, kept across players while fresh ------------------------------------
    def _keep_uzayr(self) -> bool:
        """With pools.uzayr_reuse_min > 0: Uzayr's learning walks outlive a player (memory only)."""
        return self.reuse_min > 0

    def _drop_uzayr(self) -> None:
        self.uz_fresh.clear()
        self.done["uzayr"] = 0

    def _uz_age(self) -> Optional[float]:
        """Minutes since Uzayr's whole set of learning walks was taken (its oldest walk); None when he has no whole
        set, or it was taken before the current room measurement (Measure room / Use last measurement)."""
        n = self.n_enrol["uzayr"]
        ws = self.uz_fresh.walks
        if self.done["uzayr"] < n or len(ws) < n:
            return None
        t0 = min(w["t"] for w in ws)
        return None if t0 < self.room_at else max(0.0, (time.time() - t0) / 60.0)

    def _uz_reusable(self) -> bool:
        age = self._uz_age()
        return self._keep_uzayr() and age is not None and age < self.reuse_min

    def _uz_note(self) -> str:
        """The operator's line between walks (IDLE and an answer): how old Uzayr's walks are (whole minutes), or
        that he walks in the next round. Only on this line: the light stays green (amber there means "wait", and
        the crowd reads it too). In the third person, as the crowd can read this line as well; short, so the line
        keeps two lines at full size in a recording on the MacBook too."""
        if not self._keep_uzayr():
            return ""
        if self._uz_reusable():
            return f" Uzayr's walks: {max(1, int(self._uz_age())):d} min old."
        return " Uzayr walks next round."

    def _uz_details(self) -> str:
        """Details: the age of Uzayr's walks."""
        if not self._keep_uzayr():
            return ""
        age = self._uz_age()
        if age is None:
            n = self.n_enrol["uzayr"]
            return "  your walks: " + ("none" if self.done["uzayr"] < n else "room measured since")
        return f"  your walks {age:.1f} min (reused under {self.reuse_min:g})"

    def _silent(self, now: float) -> bool:
        """No packets for health.silent_s: the light's red "No signal"."""
        return self.last_pkt_wall is None or now - self.last_pkt_wall > float(self.cfg["health"]["silent_s"])

    def _redo_uzayr(self) -> None:
        """Redo my walks (Ctrl+U, top right): Uzayr's learning walks again now, without a new player: between
        walks only (IDLE, the game, an answer), and not with "No signal". His old walks are set aside in memory
        (uz_held) until his first new walk counts: Undo walk, or Ctrl+U again, before that keeps them
        (_keep_old_walks). After his n_enrol walks the game goes on (or IDLE comes back), and the next players
        reuse the new ones."""
        if not self._keep_uzayr():
            return                                         # he walks in every round: nothing to redo
        st = self.state
        if self._can_unredo():                             # Ctrl+U again before his first new walk: keep the old
            return self._keep_old_walks()
        if st not in (IDLE, READY, RESULT):
            if st in (WALKING, DECIDING, ENROL_VISITOR, ENROL_UZAYR, AGAIN):
                self._hint("Not now: finish this step first.")
            return
        if self._silent(time.monotonic()):                 # his walks could not be taken now
            self._hint("No signal: check RX2's USB and the TX charger.", "red")
            return
        self._retire(self.last)
        self.last = None                                   # Undo walk now means his turns only
        self.uz_held = {"walks": [dict(w) for w in self.uz_fresh.walks], "done": self.done["uzayr"],
                        "reused": self.uz_reused, "redo": self.uz_redo}
        self._drop_uzayr()
        self.uz_enrolling, self.uz_redo = True, self.name is not None
        self.visitor_ready = False
        self.state = self._resting()
        self.notice = None
        self.log("uzayr redo")
        self._event("uzayr_redo", index=self.vidx, t_mono=time.monotonic())

    def _can_unredo(self) -> bool:
        """Redo my walks was clicked and none of his new walks has counted yet: his old walks can come back."""
        return self.uz_held is not None and self.state == ENROL_UZAYR and self.last is None

    def _keep_old_walks(self) -> None:
        """Undo walk (or Ctrl+U again) right after Redo my walks: his old walks back (from memory), the game or
        IDLE as before. The recording writes uzayr_kept with their walk ids (bench_rounds)."""
        held, self.uz_held = self.uz_held, None
        self.uz_fresh.walks = held["walks"]
        self.done["uzayr"] = held["done"]
        self.uz_enrolling = False
        self.uz_reused, self.uz_redo = held["reused"] and self.name is not None, held["redo"] and self.name is not None
        self.visitor_ready = self._both_learnt()
        self.state = self._resting()
        self.log("uzayr redo undone: old walks kept")
        self._event("uzayr_kept", index=self.vidx, t_mono=time.monotonic(),
                    walk_ids=[w.get("walk_id") for w in self.uz_fresh.walks])
        self.notice = None
        if self.visitor_ready:                             # the crowd: the game goes on as before
            self._notice("Back to the game.", "green", 3.0)
        self._hint("Kept the old walks." if held["walks"] else "Redo my walks cancelled.", "green")

    def _decide(self, f: dict) -> Optional[models.Decision]:
        if self.model is None:
            X = {"uzayr": self.uz_fresh.matrix(self.use), "visitor": self.visitor.matrix(self.use)}
            return models.decide_m0(np.array([f[n] for n in self.use]), X, self.use, self.cfg)
        return models.decide(self.model, f, {"uzayr": [w["f"] for w in self.uz_fresh.walks],
                                             "visitor": [w["f"] for w in self.visitor.walks]})

    def _both_learnt(self) -> bool:
        """A player, and both people have walked all their learning walks: the game can start."""
        return self.name is not None and self.done["visitor"] >= self.n_enrol["visitor"] \
            and self.done["uzayr"] >= self.n_enrol["uzayr"]

    def _resting(self) -> str:
        """The state to rest in when nothing is happening."""
        if self.chain.baseline is None:
            return NEEDS_BASELINE
        if self.uz_enrolling and self.done["uzayr"] < self.n_enrol["uzayr"]:
            return ENROL_UZAYR
        if self.name is not None and not self.visitor_ready:
            return ENROL_VISITOR
        if self._close_call_waits():                       # a close call still waiting for its second walk: every
            return AGAIN                                   # way back (Measure room, a lost signal, a box) comes here
        return READY if (self.visitor_ready and len(self.uz_fresh)) else IDLE

    def _close_call_waits(self) -> bool:
        """A close call is waiting for its second walk (the second look): it is the last walk and the game is on."""
        return self.look2 is not None and self.look2 is self.last and self.visitor_ready and len(self.uz_fresh) > 0

    def _drop_close_call(self, why: str) -> None:
        """The close call can no longer get its second walk: it goes (nobody was named for it), logged."""
        if self.look2 is not None:
            self.log(f"walk {self.look2['id']} close call dropped: {why}")
            self.look2 = None

    def start_baseline(self, t: float) -> None:
        """Measure the room, after a countdown of baseline.delay_s (time to step back from the boards); the event
        and the capture window record the real start, so the bench evaluation finds the same columns."""
        if self.state in (WALKING, NAME_ENTRY):
            return
        if self.state == BUSY:                             # Ctrl+B again restarts the capture
            self.capture = None
            self.state = self.prev
        secs = float(self.cfg["baseline"]["bench_s" if self.record_dir else "live_s"])
        delay = float(self.cfg["baseline"].get("delay_s", 0.0))
        t_dev = self.chain.to_device(t)
        if t_dev is None:
            self._hint("No signal yet: check RX2's USB and the TX charger.", "red")
            return
        self.prev = self.state if self.state != NEEDS_BASELINE else IDLE
        self.capture = {"t0": t_dev + delay, "t1": t_dev + delay + secs, "cols": [], "t_mono": t, "secs": secs,
                        "delay": delay}
        self.state = BUSY
        self.hint_ev = None
        self._event("baseline_start", t_mono=t + delay, t_dev=t_dev + delay, secs=secs)
        self.log(f"baseline start {secs:.0f} s" + (f" after {delay:.0f} s" if delay else ""))

    def _finish_baseline(self) -> None:
        cap, self.capture = self.capture, None
        cs = [c for c in cap["cols"] if len(c)]
        t = np.concatenate([c.t for c in cs]) if cs else np.zeros(0)
        m = (t >= cap["t0"]) & (t <= cap["t1"])
        cols = chain.Columns(t[m], np.concatenate([c.power for c in cs])[m] if cs else np.zeros((0, 1)),
                           np.concatenate([c.gap for c in cs])[m] if cs else np.zeros(0, bool))
        try:
            if len(cols) < int(self.cfg["baseline"]["min_cols"]):
                raise ValueError("too few packets")
            b = chain.room_baseline(cols, self.cfg, self.chain.freqs, self.chain.spec.variant, self.chain.spec.nfft,
                                    created=time.time())
        except (ValueError, IndexError) as exc:
            self.log(f"baseline failed: {exc}")
            self.state = NEEDS_BASELINE if self.chain.baseline is None else self._resting()
            self._hint("Measuring failed: check RX2, then measure again.", "red", 8.0)
            return
        self._event("baseline_end", t_mono=time.monotonic(), n_cols=len(cols), e0=b.e0, disturbed=b.disturbed,
                    d=b.d)
        ratio = float(self.cfg["baseline"].get("recheck_ratio", 1.3))
        level = self._day_level()
        quieter = float(self.cfg["baseline"].get("quieter_ratio", 0.9))
        if level is not None and (b.disturbed or b.e0 >= quieter * level.e0):   # not clearly quieter: keep today's
            return self._keep_level(b, level, ratio)
        ref = self.quiet_e0
        self._set_baseline(b, "captured")
        if level is not None:                              # (it replaces today's level)
            self._hint(f"Quieter than earlier: room level {b.e0:.2f}.", "green", 8.0)
        elif b.disturbed:                                  # (its own hint already asks to measure again)
            self.log(f"baseline disturbed: d {b.d:.1f}, the quiet part's e0 {b.e0:.3g}")
        elif ref is not None and b.e0 > ratio * ref:        # busier than this run's quietest (a test recording: on
            self.log(f"baseline busy: e0 {b.e0:.3g} is {b.e0 / ref:.1f}x the quietest {ref:.3g}")   # the day, kept)
            self._hint("Room was busy: click Measure room again.", "amber", 15.0, light="Measure the room again")

    def _keep_level(self, b: chain.Baseline, level: chain.Baseline, ratio: float) -> None:
        """Measure room found the room not clearly quieter than today's level (or someone moved): today's level
        stays in use, and with it Uzayr's kept walks. The operator's line says why, in green: nothing to do. After a
        restart (no level in use yet) today's level is taken from the file, as Use last measurement would."""
        why = ("Someone moved" if b.disturbed else "Busier than earlier" if b.e0 > ratio * level.e0
               else "About the same as earlier")
        self.log(f"baseline kept: today's e0 {level.e0:.3g}; this one's e0 {b.e0:.3g}"
                 + (f" (someone moved: d {b.d:.1f})" if b.disturbed else f" ({b.e0 / level.e0:.2f}x)"))
        if self.chain.baseline is None:
            self._set_baseline(level, "reused")
        else:
            self._back_from_measuring()
        self._hint(f"{why}: kept the earlier room level {level.e0:.2f}.", "green", 8.0)

    def _back_from_measuring(self) -> None:
        """Measure room is done (or Use last measurement): back to where it was clicked."""
        self.state = self.prev if self.prev not in (NEEDS_BASELINE, BUSY, NAME_ENTRY) else IDLE
        self.state = self._resting() if self.state in (IDLE, READY) else self.state

    def _set_baseline(self, b: chain.Baseline, how: str) -> None:
        self.chain.baseline = b
        self.room_at = time.time()                         # Uzayr's walks from before this are no longer reused
        self.uz_held = None                                # (old walks set aside by Redo my walks: stale now)
        if self._keep_uzayr() and len(self.uz_fresh) and (
                (self.uz_reused and not self.visitor_ready) or self.done["uzayr"] < self.n_enrol["uzayr"]):
            # still learning on reused walks, or half a set (his turns, or Redo my walks): he walks his whole set
            # again after this measurement, so it can be reused; a game already under way keeps its pools. The
            # bench tools rebuild the pools from uzayr_dropped.
            if self.last is not None and self.last.get("pool") == "uzayr":
                self._retire(self.last)
                self.last = None
            self._drop_uzayr()
            self.uz_reused = False
            self._event("uzayr_dropped", index=self.vidx, t_mono=time.monotonic(), why="room")
        if self.baseline_path and how == "captured" and not self.record_dir:   # (a recording's is not the day's)
            with open(self.baseline_path, "w", encoding="utf-8") as f:
                json.dump(b.to_json(), f)
        self.log(f"baseline {how} cols {b.n_cols}")
        self._back_from_measuring()
        low = self.quiet_e0                                # the run's quietest measurement before this one
        if not b.disturbed:                                # (one someone moved in never sets it)
            self.quiet_e0 = b.e0 if low is None else min(low, b.e0)
        if how != "captured":
            self._hint("Using the last measurement.", "green")
        elif b.disturbed:                                  # its quiet part is in use (chain.room_baseline)
            self._hint("Someone moved: using the quiet part. Measure room again when everyone is still.", "amber",
                       20.0, light="Measure the room again")
        else:                                              # the number, and the run's quietest when that was lower
            quietest = f" (quietest today {low:.2f})" if low is not None and round(low, 2) < round(b.e0, 2) else ""
            self._hint(f"Room level {b.e0:.2f}{quietest}.", "green", 8.0)

    def _load_cached_baseline(self) -> Optional[chain.Baseline]:
        if not self.baseline_path or not os.path.exists(self.baseline_path):
            return None
        try:
            with open(self.baseline_path, encoding="utf-8") as f:
                b = chain.Baseline.from_json(json.load(f))
            same = b.variant == self.chain.spec.variant and b.nfft == self.chain.spec.nfft
            ok = _today(b.created) and same and len(b.floor) == len(self.chain.freqs)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
            return None                                    # (a damaged file: as if there were none)
        return b if ok else None

    def new_visitor(self, name: str) -> None:
        self.wipe_visitor()
        self._retire(self.last)
        self.last = None                                   # nothing from before this visitor can be relabelled
        self.uz_enrolling = False
        self.uz_held = None                                # (a Redo my walks left unfinished: its old walks go)
        self.name, self.vidx = name, self.tally.new_visitor()
        self.visitor_ready, self.done["visitor"] = False, 0
        self.uz_reused = self.uz_redo = False
        age = self._uz_age()
        if self._uz_reusable():                            # his walks are fresh: only the visitor walks
            self.uz_reused = True
        elif self._keep_uzayr():                           # too old, from before a room measurement, or not whole
            self._drop_uzayr()
        self.state = ENROL_VISITOR
        self.notice = None
        self.log(f"new visitor #{self.vidx}" + (f", Uzayr's walks reused ({age:.1f} min old)" if self.uz_reused
                                                else ""))
        self._event("new_visitor", index=self.vidx, t_mono=time.monotonic())
        if self.uz_reused:                                 # which earlier walks form his pool (bench tools)
            self._event("uzayr_reused", index=self.vidx, t_mono=time.monotonic(), age_min=round(age, 3),
                        walk_ids=[w.get("walk_id") for w in self.uz_fresh.walks])

    def wipe_visitor(self) -> None:
        self.visitor.clear()
        self.name, self.vidx, self.visitor_ready, self.typed = None, None, False, ""
        self.misses = 0                                    # "Missed twice" belonged to the last player
        self._drop_close_call("the player was wiped")      # (a close call waiting for its second walk too)
        self.end_until = 0.0
        self.done["visitor"] = 0
        self.uz_reused = self.uz_redo = False
        if not self._keep_uzayr():                         # Uzayr's walks with this visitor go with them (else they
            self._drop_uzayr()                             # stay in memory for the next player, while fresh)
        if self.last is not None:
            self._retire(self.last)
            self.last = None
        if not self.keep_log:                              # the visitor's walks go with them
            self.walk_log.clear()

    def start_walk(self, t: float) -> None:
        if self.state in (WALKING, BUSY, NEEDS_BASELINE) or self.chain.baseline is None:
            return
        if self._silent(time.monotonic()):
            self._hint("No signal: check RX2's USB and the TX charger.", "red")
            return
        t_dev = self.chain.to_device(t)
        if t_dev is None:
            return
        left = self._lock_left(t)                          # a late They've stopped click must not start a stray walk
        if left > 0:
            lock = float(self.cfg["walk"].get("after_end_lock_s", 0.0))
            self.log(f"start ignored: {lock - left:.2f} s after the last walk ended")
            self._hint("Ignored: that walk had already ended by itself.", "amber")
            return
        if self.state == ENROL_UZAYR:
            purpose = "enrol_uzayr"
        elif self.state == ENROL_VISITOR:
            purpose = "enrol_uzayr" if self._turn() == "uzayr" else "enrol_visitor"
        elif self.visitor_ready and len(self.uz_fresh):
            purpose = "test"
        else:
            purpose = "free"
        thr = float(self.cfg["dsp"]["active_k"]) * self.chain.baseline.e0
        w = walks.WalkWindow(self.cfg, t_dev, thr, self.hop_s)    # the model's window: the walk and its answer
        hold = float(self.cfg["walk"].get("screen_hold_s", 0.0) or 0.0)
        sw = None                                          # the screen's window: the same columns, a longer hold
        if hold > w.arrive_hold:                           # (walk.screen_hold_s; else the screen ends with the answer)
            sw = walks.WalkWindow(self.cfg, t_dev, thr, self.hop_s, hold_s=hold)
        pre = self._recent_columns()
        w.add(pre)
        if sw is not None:
            sw.add(pre)
        self.n_walks += 1
        # the second look at a waiting close call: only from "walk once more" (AGAIN) or its own Thinking step (a
        # replay has no after-walk lock); from anywhere else a game walk is a new one and the close call goes
        waiting = self._close_call_waits() and self.state in (AGAIN, DECIDING)
        first = self.look2 if purpose == "test" and waiting else None
        if self.look2 is not None and first is None:
            self._drop_close_call(f"walk {self.n_walks} started from {self.state}")
        self.walk = {"w": w, "screen": sw, "purpose": purpose, "t_click": t, "t_dev": t_dev,
                     "from": AGAIN if first is not None else self.state, "id": self.n_walks}
        if first is not None:                              # (Cancel walk, or a lost signal, goes back to AGAIN)
            self.walk["second_of"] = first["id"]
            back = AGAIN
        else:
            back = self._resting() if self.state in (RESULT, DECIDING) else self.state
        self.prev, self.state = back, WALKING
        self.log(f"walk {self.walk['id']} start {purpose}"
                 + (f" (second look at walk {first['id']})" if first is not None else ""))
        self.notice = None                                 # starting a walk clears any notice
        if self.hint_ev is not None and self.hint_ev[1] != "red":
            self.hint_ev = None

    def request_stop(self, t: float) -> bool:
        """The operator says the walker has stopped (host time t): the walk ends "operator" on the first column at or
        after that moment. Under walk.stop_min_s after the click it is a double press and is ignored (False). When
        the answer is already made and only the screen is still waiting (walk.screen_hold_s), it shows it now."""
        if self.state != WALKING or self.walk is None:
            return False
        t_dev = self.chain.to_device(t)
        wk = self.walk
        if t_dev is not None and wk["w"].ended:            # (the walk itself is already whole: nothing to stop)
            self.log(f"walk {wk['id']} stop pressed at {t_dev - wk['t_dev']:.1f} s, after its end")
            self._walk_shown(max(t_dev, wk["w"].t_end_col))
            return True
        if t_dev is None or not wk["w"].request_stop(t_dev):
            return False
        if wk.get("screen") is not None:                   # (the screen's window stops there too)
            wk["screen"].request_stop(t_dev)
        self.log(f"walk {wk['id']} stop pressed at {t_dev - wk['t_dev']:.1f} s")
        self._event("stop", walk_id=wk["id"], t_mono=t, t_dev=t_dev)
        if wk["w"].ended:
            self._walk_ended()
        return True

    def _feed_walk(self, cols: chain.Columns) -> None:
        """New columns into the running walk: first the model's window (its end makes the answer: _walk_ended), then
        the screen's (its end shows it: _walk_shown), which waits at most walk.screen_max_extra_s past the model's."""
        wk = self.walk
        w, sw = wk["w"], wk.get("screen")
        if not w.ended and w.add(cols):
            self._walk_ended()
        if sw is None or self.walk is not wk:              # (no screen window, or the walk is already shown)
            return
        if w.ended:
            # the screen waits at most walk.screen_max_extra_s past the model's end: its window's own "limit" rule
            # then ends it on its last column inside that
            extra = float(self.cfg["walk"].get("screen_max_extra_s", 1.5))
            sw.max_s = min(sw.max_s, w.t_end_col - sw.t0 + max(extra - self.hop_s, 0.0))
        if sw.add(cols):
            self._walk_shown()

    def _walk_ended(self) -> None:
        """The model's window has ended: the walk's numbers and the answer, made now as always, and kept on the walk.
        The screen shows them now (_walk_shown), unless the walk "arrived" by itself and the screen's own window (a
        longer hold, walk.screen_hold_s) is still waiting. Every other end (stopped, operator, nothing, limit, no
        signal) shows at once, and so does an arrival after They've stopped was pressed."""
        wk, self.walk = self.walk, None                    # (if the numbers or the answer fail, the walk is dropped at
        w: walks.WalkWindow = wk["w"]                      # once, as before the screen wait: tick logs the error)
        arr = w.arrays()
        f = walks.walk_features(arr, self.hop_s, self.cfg) if w.seen else None
        if f is not None:                                  # the single-link gait set, from the click to the arrival
            t_arr = float(w.t_arrive)
            f.update(walks.walk_gait(*self.chain.raw.window(wk["t_dev"], t_arr), wk["t_dev"], t_arr, self.cfg))
        rel = lambda x: float(x - wk["t_dev"]) if x is not None else None  # noqa: E731
        rec = {"id": wk["id"], "purpose": wk["purpose"], "t_click": wk["t_click"], "t_dev": wk["t_dev"],
               "t_end_wall": None, "seen": f is not None, "reason": w.reason, "f": f,
               "label": None, "decision": None, "pool": None, "entry": None,
               "active_s": w.active_s, "t_end_dev": w.t_end_col, "t_end": rel(w.t_end_col),
               "t_arrive": rel(w.t_arrive), "screen_t_end": None}
        first = self.look2 if self.look2 is not None and wk.get("second_of") == self.look2["id"] else None
        if first is not None:
            rec["second_of"] = first["id"]                 # (also when it was missed: the first walk then decides)
        if f is not None and wk["purpose"] == "test":      # (the pools can't change while a walk is on screen)
            if first is not None:                          # the second look: decided on both walks, averaged
                rec["alone"] = self._decide(f)             # (this walk on its own: the log and the events only)
                rec["f_both"] = models.average_walks([first["f"], f])   # (what it was decided on: the tiles)
                rec["decision"], rec["both"], rec["first"] = self._decide(rec["f_both"]), True, first
            else:
                rec["decision"] = self._decide(f)
                d = rec["decision"]
                if self.second_look and d is not None and d.not_sure:
                    rec["again"] = True                    # too close to call: the same walker walks once more
        wk["rec"] = rec
        wk["end_dev"] = w.t_end_col if w.t_end_col is not None else (w.t[-1] if w.t else wk["t_dev"])  # model's end
        wk["ended_at"] = time.monotonic()                  # (host time: _timers shows it if the columns stop coming)
        self.walk = wk                                     # (back on: shown below, or when the screen's wait ends)
        if wk.get("screen") is None or w.reason != "arrived":   # shown now, ending where the model's window ended
            self._walk_shown(wk["end_dev"])
        elif w.t_stop is not None:                         # They've stopped was pressed before the end: shown now,
            self._walk_shown(max(w.t_stop, wk["end_dev"]))  # the after-end lock counting from the press

    def _walk_shown(self, t_end_dev: Optional[float] = None) -> None:
        """The walk ends on screen, at device time t_end_dev (by default the column that ended the screen's window,
        else the newest column it saw): the answer _walk_ended made, its notice, the run log and the events."""
        wk, self.walk = self.walk, None
        if t_end_dev is None:
            win = wk.get("screen") or wk["w"]
            t_end_dev = win.t_end_col if win.t_end_col is not None else (win.t[-1] if win.t else wk["t_dev"])
        rec = wk["rec"]
        f = rec["f"]
        self.last_end_dev = t_end_dev                      # (the after-end lock counts from the screen's end)
        self.last_end_wall = rec["t_end_wall"] = time.monotonic()
        # on screen at the model's own end: its t_end (None after "no signal"), so the wait off logs as before it
        rec["screen_t_end"] = rec["t_end"] if t_end_dev == wk["end_dev"] else float(t_end_dev - wk["t_dev"])
        self.walk_log.append(rec)
        self.state = wk["from"] if wk["from"] not in (RESULT, DECIDING) else self.prev
        if f is None:                                      # (Undo walk can still remove the walk before it)
            self.misses += 1
            same = " Same turn again: start from the back mark." \
                if self.state in (ENROL_VISITOR, ENROL_UZAYR) else " Walk again from the back mark."
            alone = rec.get("second_of") is not None and self.look2 is not None \
                and self.look2["id"] == rec["second_of"]
            if alone:                                      # the second look was missed: the first walk decides alone
                self.look2 = None
                self.state, self.timer = DECIDING, time.monotonic() + float(self.cfg["ui"]["reading_s"])
                same = " The answer is from the first walk."
            self._notice("Missed that one." + same, "amber", 4.0)
            self.log(f"walk {rec['id']} not seen, ended {rec['reason']} ({rec['active_s']:.1f} s active)"
                     + (f": walk {rec['second_of']} decides alone" if alone else ""))
            self._event("walk", **self._walk_event(rec))
            return
        self._retire(self.last)
        self.last = rec
        p = rec["purpose"]
        rec["final"] = False                               # the label can no longer change (one click)
        self.misses = 0
        if p in ("enrol_uzayr", "enrol_visitor"):
            who = "u" if p == "enrol_uzayr" else "v"
            rec["final"] = True                            # labelled by its turn; Undo walk redoes it
            rec["label"], rec["pool"] = who, "uzayr" if who == "u" else "visitor"
            pool = self.uz_fresh if who == "u" else self.visitor
            rec["entry"] = pool.add(f)
            pool.walks[-1]["walk_id"] = rec["id"]          # (memory only: the reuse event names walks by id)
            self._notice("Got it!", "green", 1.5)          # _enrol_count says what comes next
            self._enrol_count(rec, +1)
            self.state = self._resting()
        elif p == "test":                                  # (its answer was made when the model's window ended)
            self.state, self.timer = DECIDING, time.monotonic() + float(self.cfg["ui"]["reading_s"])
            self.look2 = rec if rec.get("again") else None  # a close call: AGAIN after Thinking (_timers)
        d = rec["decision"]
        at = f" at {rec['t_end']:.1f} s" if rec["t_end"] is not None else ""
        arr_at = f", arrived {rec['t_arrive']:.1f} s" if rec["t_arrive"] is not None else ""
        shown = f", on screen {rec['screen_t_end']:.1f} s" if rec["screen_t_end"] != rec["t_end"] else ""
        ended = f" ended {rec['reason']}{at} ({rec['active_s']:.1f} s active{arr_at}{shown})"
        self.log(f"walk {rec['id']} {p} seen{ended}" + (self._feat_text(f) if self.keep_log else "")
                 + (f" -> {d.label} margin {d.margin:.3f}{' not sure' if d.not_sure else ''} speed-only "
                    f"{d.speed_only.label if d.speed_only else '-'}" if d else "")
                 + (": too close to call, walk once more" if rec.get("again") else "")
                 + (f" (from both walks {rec['second_of']} and {rec['id']}; this walk alone {rec['alone'].label} "
                    f"margin {rec['alone'].margin:.3f})" if rec.get("both") and rec.get("alone") else ""))
        self._event("walk", **self._walk_event(rec))
        if rec["final"]:                                   # the same label event an It was ... click writes
            self._event("label", walk_id=rec["id"], label=rec["label"], t_mono=time.monotonic(), counted=None,
                        source="turn")

    def _retire(self, rec: Optional[dict]) -> None:
        """A walk can no longer be relabelled: only now is it logged, and only if it is Uzayr's (a visitor's walk
        numbers are never written). Its numbers only in a recording: on the day his walks stay in memory
        only."""
        if rec is not None and rec.get("seen") and rec.get("label") == "u" and not self.keep_log:
            self.log(f"walk {rec['id']} uzayr" + (self._feat_text(rec["f"]) if self.events else ""))

    @staticmethod
    def _feat_text(f: dict) -> str:
        return f" speed {f['speed']:.2f} steps {f['steps']:.0f} swing {f['swing']:.2f} bounce {f['bounce']:.3f}"

    def _enrol_count(self, rec: dict, step: int) -> None:
        """Count (step +1) or uncount (-1) a learning walk towards its own person's progress, exactly once: the walk
        counts for whose turn it was. Uzayr's first new walk after Redo my walks lets go of his old ones (uz_held);
        his whole set ends a Redo; Undo walk on his walk with no player, or in a Redo, brings his turn back (the
        card: "Uzayr walks once more, then ..."). The notices say what comes next (pools.enrol_order visitor_first,
        and Redo): "Got it! Once more." (or "Got it! 4 more walks.") between one person's walks, "Got it! Now
        Uzayr walks 5 times." (pools.n_enrol_uzayr) at the hand-over; when the game starts "Got you both!" after
        both walked, "Got it!" when his walks were reused (or redone in the game). The alternating order keeps a
        plain "Got it!" (the next turn is the other person's)."""
        if not rec["purpose"].startswith("enrol") or bool(rec.get("enrol_counted")) == (step > 0):
            return
        rec["enrol_counted"] = step > 0
        who = "uzayr" if rec["purpose"] == "enrol_uzayr" else "visitor"
        n = self.n_enrol[who]
        if who == "visitor" and self.name is None:
            return
        self.done[who] = max(0, self.done[who] + step)
        if who == "uzayr" and step > 0:
            self.uz_held = None                            # his old walks are gone for good now
        if step > 0 and (self.order == "visitor_first" or self.uz_enrolling):
            left = n - self.done[who]
            if left > 0:
                self._notice("Got it! " + ("Once more." if left == 1 else f"{left} more walks."), "green", 2.0)
            elif who == "visitor" and self._turn() == "uzayr":
                self._notice(f"Got it! Now Uzayr walks {_times(self.n_enrol['uzayr'])}.", "green", 3.0)
        if who == "uzayr":
            if self.done["uzayr"] >= n and self.uz_enrolling:
                self.uz_enrolling = False                  # Redo my walks: done
                if self.name is None and step > 0:
                    self._notice("Got it! Ready for the next player.", "green", 3.0)
            elif step < 0 and (self.name is None or self.uz_redo):
                self.uz_enrolling = True                   # Undo walk with no player, or in a Redo: his turn again
        if self.name is None:
            return
        was = self.visitor_ready
        self.visitor_ready = self._both_learnt()
        if self.visitor_ready and not was and step > 0:
            if who == "uzayr" and self.uz_redo:
                text = "Got it! Back to the game."
            elif self.uz_reused:
                text = "Got it! Now the guessing game."
            else:
                text = "Got you both! Now the guessing game."
            self._notice(text, "green", 3.0)

    def _walk_event(self, rec: dict) -> dict:
        """The walk event: reason, t_end and t_arrive are the model's window (the bench tools re-run it), screen_t_end
        is when the walk ended on screen."""
        d = rec["decision"]
        out = {k: rec.get(k) for k in ("id", "purpose", "t_click", "t_dev", "seen", "reason", "f", "label", "active_s",
                                       "t_end_dev", "t_end", "t_arrive", "screen_t_end")}
        if d is not None:
            out["decision"] = {"label": d.label, "margin": d.margin, "not_sure": d.not_sure,
                               "speed_only": d.speed_only.label if d.speed_only else None}
        if rec.get("again"):                               # the second look (only with classifier.second_look):
            out["second_look"] = True                      # this close call asked for one more walk ...
        if rec.get("second_of") is not None:               # ... and this walk was it: its decision is from both
            out["second_of"] = rec["second_of"]            # walks (bench.second_looks rebuilds it)
            if d is not None:
                out["decision"]["both"] = True
            a = rec.get("alone")
            if a is not None:
                out["alone"] = {"label": a.label, "margin": a.margin, "not_sure": a.not_sure}
        return out

    def discard(self) -> None:
        if self.state == WALKING and self.walk is not None:
            self.log(f"walk {self.walk['id']} discarded while walking")
            self._event("discard", walk_id=self.walk["id"], t_mono=time.monotonic())
            back = self.walk["from"]                       # (the second look: back to "walk once more")
            self.state, self.walk = back if back not in (RESULT, DECIDING) else self._resting(), None
            self.last_end_wall = time.monotonic()
            again = " Same turn again." if self.state in (ENROL_VISITOR, ENROL_UZAYR) else \
                (" Same walker again." if self.state == AGAIN else "")
            self._notice("Walk cancelled." + again, "amber", 2.5)
            return
        rec = self.last
        if rec is None:
            if self._can_unredo():                         # right after Redo my walks: keep his old walks
                self._keep_old_walks()
            return
        self._unapply(rec)
        self._enrol_count(rec, -1)
        self.last = None
        self._drop_close_call("Undo walk")                 # (on "walk once more": the close call goes)
        self.state = self._resting()
        self._event("discard", walk_id=rec["id"], t_mono=time.monotonic())
        self.log(f"walk {rec['id']} discarded")
        first = rec.get("first") if rec.get("both") else None
        if first is not None:                              # an answer from both walks: both go
            self._event("discard", walk_id=first["id"], t_mono=time.monotonic())
            self.log(f"walk {first['id']} discarded (the first of the two)")
        again = " Same turn again." if self.state in (ENROL_VISITOR, ENROL_UZAYR) else ""
        self._notice(("Both walks removed." if first is not None else "Last walk removed.") + again, "amber", 2.5)

    def _unapply(self, rec: dict) -> None:
        if rec.get("entry") is not None:
            (self.uz_fresh if rec["pool"] == "uzayr" else self.visitor).remove(rec["entry"])
        rec["entry"], rec["pool"] = None, None

    def _answer(self, st: str) -> Optional[dict]:
        """A test recording (--record) only: the walk whose It was ... buttons show in state st (a game answer, or a
        speed-only walk with no player), else None. Never on the day, never for a learning walk."""
        rec = self.last
        if self.events is None or rec is None or not rec["seen"]:
            return None
        tail = self.walk_log[-1] if self.walk_log else None     # a missed walk came after it: not this answer's
        if tail is not rec and not (tail is not None and not tail["seen"] and tail.get("second_of") == rec["id"]):
            return None                                    # (unless it was its missed second look)
        if (st == RESULT and rec["decision"] is not None) or (st == IDLE and self.name is None
                                                              and rec["purpose"] == "free"):
            return rec
        return None

    def confirm(self, who: str) -> None:
        """A test recording only: It was Uzayr / It was {V} (U / V) on an answer. Optional, and one click is final:
        it writes the label event (counted right or wrong: the answer always names someone) and thanks;
        the walk joins no pool and nothing is tallied, as on the day. A second click only says so (Undo walk removes
        the walk). A learning walk is labelled by its turn."""
        rec = self._answer(self.state)
        if rec is None:
            last, learning = self.last, (ENROL_VISITOR, ENROL_UZAYR, READY, IDLE)
            if last is not None and last["purpose"].startswith("enrol") and self.state in learning:
                self._hint("Learning walks go by turn. Undo walk redoes one.")
            return
        if who == "v" and self.name is None:               # a speed-only answer: It was Uzayr only
            return
        if rec.get("final"):
            self._hint("Already noted. Undo walk removes the walk.")
            return
        truth = "uzayr" if who == "u" else "visitor"
        rec["final"], rec["label"] = True, who
        d = rec["decision"]
        counted = None
        if d is not None:
            counted = "right" if d.label == truth else "wrong"
        self._event("label", walk_id=rec["id"], label=who, t_mono=time.monotonic(), counted=counted)
        self.log(f"walk {rec['id']} labelled {truth} {counted or ''}")
        first = rec.get("first") if rec.get("both") else None
        if first is not None:                              # an answer from both walks: the same walker walked both
            first["final"], first["label"] = True, who
            self._event("label", walk_id=first["id"], label=who, t_mono=time.monotonic(), counted=None,
                        both_with=rec["id"])
            self.log(f"walk {first['id']} labelled {truth} (with walk {rec['id']})")
        against = d is not None and d.label != truth
        self._notice(f"Thanks: that was {'Uzayr' if who == 'u' else self.name}.",
                     screen.INK2 if against else "green", 2.5)  # a label against the answer: thanks, but not in green

    # -- timers and health ----------------------------------------------------------------------
    def _timers(self, now: float) -> None:
        if self.state == DECIDING and now >= self.timer:   # a close call waiting: "walk once more", else the answer
            self.state = AGAIN if self._close_call_waits() else RESULT
        if self.capture is not None:
            newest = max((c.t[-1] for c in self.capture["cols"] if len(c)), default=-np.inf)
            if newest >= self.capture["t1"]:
                self._finish_baseline()
            elif now - self.capture["t_mono"] > self.capture["secs"] + self.capture.get("delay", 0.0) \
                    + float(self.cfg["walk"]["guard_s"]):
                self._abort("Receiver silent")
        if self.walk is not None and now - self.walk["t_click"] > float(self.cfg["walk"]["max_s"]) \
                + float(self.cfg["walk"]["guard_s"]):          # columns stopped coming: end on what was seen
            w = self.walk["w"]
            if not w.ended:
                w.ended, w.reason = True, "no signal"
                self._walk_ended()
            if self.walk is not None:                      # (only the screen was still waiting: show it now)
                self._walk_shown()
        elif self.walk is not None and "ended_at" in self.walk and now - self.walk["ended_at"] \
                > float(self.cfg["walk"].get("screen_max_extra_s", 1.5)) + float(self.cfg["walk"]["guard_s"]):
            self.log(f"walk {self.walk['id']} shown: no columns came after its answer")
            self._walk_shown()                             # only the screen waits, and the columns stopped coming
        self._repair()
        b = self.busy()
        if b is not None and b >= float(self.cfg["health"]["busy_amber"]):
            self.busy_since = now if self.busy_since is None else self.busy_since
        else:
            self.busy_since = None
        self.rate_hist.append((now, self.n_pkts))
        while self.rate_hist and now - self.rate_hist[0][0] > float(self.cfg["health"]["rate_window_s"]):
            self.rate_hist.popleft()

    def _repair(self) -> None:
        """A state that has lost what it shows goes back to resting (never stuck on a dead screen). "Walk once more"
        without its waiting close call drops it and says so."""
        st = self.state
        lost = st == AGAIN and not self._close_call_waits()
        if (st == WALKING and self.walk is None) or (st == BUSY and self.capture is None) \
                or (st in (DECIDING, RESULT) and (self.last is None or self.last.get("decision") is None)) or lost:
            if lost:
                self._drop_close_call("its walk is gone")
            self.state = self._resting()
            self.log(f"state repaired: {st} -> {self.state}")
            if lost:
                self._notice("Walk again from the back mark.", "amber", 4.0)

    def rate(self) -> float:
        if len(self.rate_hist) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self.rate_hist[0], self.rate_hist[-1]
        return (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0

    def health(self, now: float) -> tuple[str, str, bool]:
        """The light: can the next walk start? (It never blocks Start walk.) Red is for the signal
        only; grey while watching or typing, and while the walker goes back after a walk. Amber also carries the
        warnings the operator must see from the back mark: "Measure the room again" (a busy measurement, for as
        long as its hint) and "Missed twice". (The age of Uzayr's walks is only on the operator's line: amber means
        "wait", and his walks never stop a walk.)"""
        h = self.cfg["health"]
        silent = self._silent(now)
        if silent or self.rate() < float(h["rate_red"]):
            return "red", "No signal" if silent else "Weak signal", silent
        if self.state == BUSY:
            return "amber", "Measuring the room", False
        if self.chain.baseline is None:
            return "amber", "Room not measured", False
        if self.state in (WALKING, DECIDING):
            return "grey", "Watching", False
        if self.state == NAME_ENTRY:
            return "grey", "Typing a name", False
        ev = self.hint_ev                                  # a warning the operator must see from the back mark
        if ev is not None and len(ev) > 3 and ev[3] and now < ev[2]:
            return ("red" if ev[1] == "red" else "amber"), ev[3], False
        if self.misses >= 2:
            return "amber", "Missed twice", False
        if self._lock_left(now) > 0:                       # the main button is grey for these few seconds too
            return "grey", "Walk back to the start", False
        b = self.busy()
        if b is not None and b >= float(h["busy_amber"]):
            if self.last_end_wall is not None and now - self.last_end_wall < float(h.get("after_walk_s", 8.0)):
                return "grey", "Walk back to the start", False
            if self.busy_since is not None and now - self.busy_since >= float(h.get("busy_hold_s", 3.0)):
                return "amber", "Wait for green", False
        return "green", "Ready", False

    def busy(self) -> Optional[float]:
        if self.chain.baseline is None or not self.ring:
            return None
        n = max(1, int(round(float(self.cfg["health"]["busy_window_s"]) / self.hop_s)))
        e = [x[2] for x in list(self.ring)[-n:]]
        return float(np.median(e)) / self.chain.baseline.e0

    # -- screen ---------------------------------------------------------------------------------
    def _maybe_draw(self, now: float) -> None:
        if now < self.next_draw:                           # frames on a fixed cadence, not "period after the last"
            if now - self.last_pump > 0.03 and self.next_draw - now > 0.015:   # keys arrive between frames
                self.last_pump = now
                t0 = time.perf_counter()
                self.screen.pump()
                ps = self.stats.setdefault("pump_s", [])
                ps.append(time.perf_counter() - t0)
                if len(ps) > 40000:
                    del ps[:20000]
            return
        period = 1.0 / self.fps
        self.next_draw = self.next_draw + period if now - self.next_draw < period else now + period
        self.last_pump = now
        self.screen.render(self.view(now))
        if self.screen.ridge_error is not None:            # the ridges failed to draw (the rest did): say so
            exc, self.screen.ridge_error = self.screen.ridge_error, None
            self._ridge_fault(exc)
        dt = self.screen.draw()
        fresh = self.last_pkt_wall is not None and now - self.last_pkt_wall < float(self.cfg["ui"]["lag_fresh_s"])
        lag = (time.monotonic() - self.newest_tm) if (self.newest_tm is not None and fresh) else 0.0
        s = self.stats
        s["draw_s"].append(dt)
        s["draw_at"].append(now)
        s["lag"].append(lag)
        if len(s["draw_at"]) > 20000:                      # a whole day: keep the newest, all lists aligned
            for k in ("draw_s", "draw_at", "lag"):
                del s[k][:10000]
        if now - self.last_fps_adj > 1.0:                  # drawing must never starve the dispatcher
            u = self.cfg["ui"]
            if lag > float(u["lag_limit_s"]):
                self.fps = max(float(u["fps_min"]), self.fps / 2)
                self.last_fps_adj = now
            elif lag < float(u["lag_ok_s"]) and self.fps < float(u["fps_max"]):
                self.fps = min(float(u["fps_max"]), self.fps + 1)
                self.last_fps_adj = now

    def _tile_bars(self, name: str, this: Optional[float], first: str = "Last walk") -> list:
        """The last walk (first: its row's name; "Both walks" for an answer from both walks), Uzayr's average and
        the visitor's, as (short label, value text, bar fraction)."""
        lo, hi = self.cfg["ui"]["tile_ranges"][name]
        frac = lambda v: None if v is None or not np.isfinite(v) else (min(max(v, lo), hi) - lo) / (hi - lo)  # noqa: E731
        fmt = TILE[name][1].split(" ")[0]
        show = lambda v: "-" if v is None or not np.isfinite(v) else fmt.format(v * 100 if name == "bounce" else v)  # noqa: E731
        bars = [(first, show(this), frac(this))]
        if self.name is None:                              # nobody to compare with yet
            return bars
        u = self.uz_fresh.means(self.names)[name] if len(self.uz_fresh) else None
        bars.append(("Uzayr", show(u), frac(u)))
        v = self.visitor.means(self.names)[name] if len(self.visitor) else None
        bars.append((self._vname(9), show(v), frac(v)))
        return bars

    def _ridge_view(self, silent: bool) -> tuple:
        """The ridges (oldest first) and how much the room moved for each: the waterfall's energy over the room's (0
        before the room is measured), from the columns made since the ridges last restarted only (older ones belong to
        the signal before a restart: those ridges stay calm). A ridge's own column and the screen.RIDGE_TRAIL - 1 before
        it, as a geometric mean (each at least screen.RIDGE_E_MIN; before this restart's first column, that column again
        but never above the room's own level), so a lone busy column barely counts, the first after a restart too, and a
        walk's run of them does. Silent: none, rather than frozen ones."""
        z = self.ridge.heights()
        act = np.zeros(len(z))
        if silent:
            return z[:0], act[:0]
        k = min(len(z), self.ridge_cols, len(self.ring))
        e0 = self.chain.baseline.e0 if self.chain.baseline is not None else 0.0
        if k and e0 > 0:
            ring = list(self.ring)
            w = screen.RIDGE_TRAIL
            take = min(self.ridge_cols, len(ring), k + w - 1)   # (at least k: this restart's columns only)
            e = np.log(np.maximum(np.array([c[2] for c in ring[len(ring) - take:]], float) / e0, screen.RIDGE_E_MIN))
            e = np.concatenate((np.full(k + w - 1 - take, min(e[0], 0.0)), e))   # (never above the room)
            act[len(z) - k:] = np.exp(np.convolve(e, np.ones(w) / w, mode="valid"))
        return z, act

    def _ridge_fault(self, exc: Exception) -> None:
        """Log a fault in the ridges (display only) at most every 10 s; the show carries on without them."""
        if time.monotonic() - self._ridge_err_at > 10.0:
            self._ridge_err_at = time.monotonic()
            self.log(f"ridge error {type(exc).__name__}: {exc}")

    def view(self, now: float) -> dict:
        light, light_text, silent = self.health(now)
        st = self.state
        vm: dict = {"recording": self.events is not None, "light": light, "light_text": light_text,
                    "silent": "Receiver silent:\ncheck USB and the TX power" if silent else ""}
        vm["who"] = f"Uzayr vs {self.name}" if self.name is not None else ""
        vm["hint"] = self._op_hint(st, now)
        # waterfall
        img = np.zeros((self.n_bins, self.ring.maxlen), np.float32)
        lx = ly = np.zeros(0)
        if self.ring:
            cols = list(self.ring)
            img[:, -len(cols):] = np.stack([c[0] for c in cols], axis=1)
            ly = self._smooth_line(np.array([c[1] for c in cols]))
            lx = (np.arange(len(cols)) - len(cols) + 1) * self.hop_s
        vm["wf"], vm["line"] = img, (lx, ly)
        try:                                               # (a fault here never stops the rest of the frame)
            vm["ridge"] = self._ridge_view(silent)
        except Exception as exc:  # noqa: BLE001
            vm["ridge"] = (np.zeros((0, 2)), np.zeros(0))
            self._ridge_fault(exc)
        # tiles
        rec = self.last
        f = rec["f"] if rec is not None and st in (DECIDING, RESULT, READY, IDLE, ENROL_UZAYR, ENROL_VISITOR, AGAIN) \
            and rec["seen"] else None
        both = f is not None and rec.get("f_both") is not None   # an answer from both walks: the readings it was
        if both:                                                  # decided on (both walks averaged), "Both walks"
            f = rec["f_both"]
        tiles = {}
        for n in self.names:
            label, fmt = TILE[n]
            if st == WALKING:
                val = "..."
                if n == "speed":
                    k = max(1, int(round(float(self.cfg["ui"]["speed_smooth_s"]) / self.hop_s)))
                    v = [c[1] for c in list(self.ring)[-k:] if np.isfinite(c[1])]
                    val = fmt.format(float(np.median(v))) if v else "..."
                tiles[n] = {"label": label, "value": val, "bars": []}
                continue
            this = f[n] if f else None
            val = "-" if this is None or not np.isfinite(this) else fmt.format(this * 100 if n == "bounce" else this)
            t = {"label": label, "value": val, "bars": self._tile_bars(n, this, "Both walks" if both else "Last walk")}
            if rec is not None and f:                      # a learning walk's bar in the colour of whose turn it was
                t["this_color"] = ({"u": screen.UZAYR, "v": screen.VISITOR}.get(rec.get("label"), screen.THIS)
                                   if rec["purpose"].startswith("enrol")
                                   else screen.THIS)             # (a game walk: as on the day)
            d = rec["decision"] if rec else None
            if d is not None and st == RESULT and n in d.contrib:
                c = d.contrib[n]
                nm = self._vname(9) if self.name is not None else "visitor"
                t["pull"] = "" if abs(c) < 0.05 else ("like Uzayr" if c > 0 else f"like {nm}")
                t["pull_color"] = screen.UZAYR if c > 0 else screen.VISITOR
            tiles[n] = t
        vm["tiles"] = tiles
        vm["card"] = self._card(st, now)
        vm["overlay"] = self._overlay(st, now)
        vm["buttons"] = self._buttons(st, now)
        if self.debug:                                     # the operator's numbers first, short enough for the screen
            s = self.stats
            ds = s["draw_s"][-30:]
            fps = (len(s["draw_at"][-30:]) - 1) / (s["draw_at"][-1] - s["draw_at"][-30]) if len(s["draw_at"]) > 30 else 0
            b = self.chain.baseline
            vm["debug"] = (f"link {self.rate():.0f} pkt/s  lost {self.lost}"
                + (f"  room e0 {b.e0:.3g}" if b is not None else "  room not measured")
                + self._uz_details()
                + f"  fps {fps:.0f}/{self.fps:.0f}  draw {1e3 * np.median(ds) if ds else 0:.0f} ms"
                + f"  lag {s['lag'][-1] if s['lag'] else 0:.2f} s  restarts {self.chain.restarts}")
        return vm

    # -- the hint, the card, the buttons ------------------------------------------------------------
    def _op_hint(self, st: str, now: float) -> tuple:
        """The operator's line beside the buttons: (text, colour). An open box's own hint, then an event hint (4 s),
        then the warnings, then the state's own hint. Mouse wording; never on the
        card; short (about 45 letters), so it fits two lines at 20 pt, on the MacBook's own screen too, and can be
        read from the back mark."""
        col = {"info": screen.INK2, "green": screen.STATUS["green"], "amber": screen.STATUS["amber"],
               "red": screen.STATUS["red"]}
        if st != NAME_ENTRY and self.quit_until and now < self.quit_until:
            return "Click Yes, quit or Keep going (it closes by itself).", screen.INK2
        if st != NAME_ENTRY and self._ending(now):
            return "Click Yes, end it or Keep going (it closes by itself).", screen.INK2
        if self.hint_ev is not None and now < self.hint_ev[2]:
            return self.hint_ev[0], col.get(self.hint_ev[1], screen.INK2)
        rest = st not in (WALKING, DECIDING, NAME_ENTRY, BUSY, NEEDS_BASELINE)
        if self.misses >= 2 and rest:
            return "Missed twice: is anyone near the boards?", screen.STATUS["amber"]
        own = self._turn() == "uzayr"                      # Uzayr's own learning turn: he is the walker
        if st == ENROL_UZAYR and self._can_unredo():       # right after Redo my walks: how to take it back
            return ("Click Start walk as you set off. "
                    + ("Undo walk keeps the old ones." if self.uz_held["walks"] else "Undo walk cancels this."),
                    screen.INK2)
        text = {                                           # the card already says what the walker does
            NEEDS_BASELINE: "Clear the boards, then click Measure room.",
            BUSY: "Someone moved? Click Measure room again.",
            IDLE: "Click New player to start." + self._uz_note(),
            NAME_ENTRY: "The visitor types their first name, then OK.",
            ENROL_VISITOR: ("Click Start walk, then walk to the X yourself." if own
                            else f"Click Start walk as {self._vname(9)} sets off."),
            ENROL_UZAYR: "Click Start walk, then walk to the X yourself.",
            WALKING: "Click They've stopped when they stand still (optional).",
            DECIDING: "",
            RESULT: "Click Next walk as they set off." + self._uz_note(),
            READY: "One of them walks: click Start walk as they set off.",
            AGAIN: "The same walker again: click Start walk as they set off.",
        }.get(st, "")
        return text, screen.INK2

    def _overlay(self, st: str, now: float) -> Optional[tuple]:
        """The open box (the name, Quit, End game) as (title, line), else None."""
        if st == NAME_ENTRY:
            sub = "Type your first name, then OK or Enter. Names are never saved."
            if self.name is not None:
                sub += f" This ends {self.name}'s game."
            if self.events is not None:
                sub += " Radio data is being recorded."
            return f"Name: {self.typed}_", sub
        if self.quit_until and now < self.quit_until:
            return "Quit?", (f"{self.name}'s name and walks will be forgotten." if self.name is not None
                             else "The demo closes.")
        if self._ending(now) and self.name is not None:
            return "End the game?", f"{self.name}'s name and walks will be forgotten."
        return None

    def _buttons(self, st: str, now: float) -> dict:
        """The buttons, as name -> {look, label, key}. Every button has a fixed
        place (screen.py); one that exists in a state but can't act is drawn dim, so nothing jumps around. The main
        button (start_walk, sends Next = space) means Measure room / Start walk / They've stopped / Next walk by state;
        the row left of it: Undo walk (Backspace, so it never throws away an answer by accident; Cancel walk during a
        walk), New player and, only in a test recording on an answer and left of Undo walk, It was {V} / It was Uzayr
        (dim once used, the one clicked ticked); the top-right group: Redo my walks (only with pools.uzayr_reuse_min:
        Uzayr's learning walks again, between walks), End game, Measure room, Details, Quit. An open box (name, quit,
        end game) dims everything behind it."""
        quitting = bool(self.quit_until) and now < self.quit_until
        ending = self._ending(now) and self.name is not None
        naming = st == NAME_ENTRY
        if naming:                                         # (behind the name box: the buttons as they were)
            st = self.prev
        keys = self.cfg["keys"]
        setup = st in (NEEDS_BASELINE, BUSY)
        busy = st in (WALKING, DECIDING)                   # a walk under way or its answer coming
        b = {"debug": {"look": "on"}, "quit": {"look": "on"},
             "wipe_visitor": {"look": "on" if self.name is not None and not (setup or busy) else "off",
                              "label": "End game", "key": keys["wipe_visitor"][0]}}
        # Measure room (in set-up it is the main button's job too)
        b["baseline"] = {"look": "off" if busy else "on", "label": "Measure room", "key": keys["baseline"][0]}
        if self._keep_uzayr():                             # Uzayr's learning walks again: between walks, with signal
            free = st in (IDLE, READY, RESULT) and not self._silent(now)
            b["enrol_uzayr"] = {"look": "on" if free else "off", "label": "Redo my walks",
                                "key": keys["enrol_uzayr"][0]}
        main = {"key": "space", "label": "Start walk", "look": "next"}
        undo = (self.last is not None and not (setup or st == DECIDING)) or self._can_unredo()
        b["discard"] = {"look": "on" if undo else "off",   # (dim while the answer is coming; right after Redo my
                        "label": "Undo walk", "key": "backspace"}   # walks it keeps his old walks)
        b["new_visitor"] = {"look": "off" if setup or busy else "on", "label": "New player"}
        rec = self.last
        if st == NEEDS_BASELINE:
            main.update(label="Measure room")
            if self._reusable() is not None:               # in the slots Undo walk and New player use later
                age = (time.time() - self.cached.created) / 60.0
                b["reuse_baseline"] = {"look": "on", "label": f"Use last measurement ({_age_text(age)})",
                                       "key": "enter"}
                del b["discard"], b["new_visitor"]
        elif st == BUSY:
            main.update(label="Measuring \u2026", look="off")
        elif st == WALKING:
            early = self.walk is not None and now - self.walk["t_click"] < float(self.cfg["walk"].get("stop_min_s", 1.5))
            main.update(label="They've stopped", look="off" if early else "next")
            b["discard"] = {"look": "on", "label": "Cancel walk", "key": "backspace"}
        elif st == DECIDING:
            main.update(label="Next walk", look="off")
        else:
            if st == RESULT:
                main.update(label="Next walk")
            elif st == IDLE:
                main.update(look="on")
                b["new_visitor"]["look"] = "next"
            if self._answer(st) is not None:               # --record only; optional; one click is final (then dim,
                final = bool(rec.get("final"))             # the one clicked ticked)
                look = "dim" if final else "on"
                tick = lambda who: " \u2713" if final and rec.get("label") == who else ""  # noqa: E731
                b["confirm_uzayr"] = {"look": look, "label": "It was Uzayr" + tick("u"),
                                      "key": keys["confirm_uzayr"][0]}
                if self.name is not None:
                    b["confirm_visitor"] = {"look": look, "label": f"It was {self._vname(9)}" + tick("v"),
                                            "key": keys["confirm_visitor"][0]}
            if self._lock_left(now) > 0:                   # just after a walk ended: a late click starts nothing
                main["look"] = "off"
        b["start_walk"] = main
        if not (naming or quitting or ending):
            return b
        b = {k: dict(v, look="off") for k, v in b.items()}
        if naming:
            b["ov_a"] = {"look": "next", "label": "OK", "key": "enter"}
            b["ov_b"] = {"look": "on", "label": "Cancel", "key": "escape"}
        else:                                              # the quit box, or the End game box
            b["ov_a"] = {"look": "next", "label": "Yes, quit" if quitting else "Yes, end it",
                         "key": keys["quit_yes"][0]}
            b["ov_b"] = {"look": "on", "label": "Keep going", "key": "escape"}
        return b

    def _card(self, st: str, now: float) -> dict:
        """The card is for the crowd: no keys, no clicks, no score. A notice
        takes the instruction line (an answer has none: there it adds the line, and the screen draws the name at the
        smaller size while it shows); the big text's words, the dots and the bar stay. No notice while setting up
        or measuring the room."""
        nm = self.name or "the visitor"
        if st == NEEDS_BASELINE:
            return {"title": "Measure the room first", "sub": "Everyone step back from the boards."}
        if st == BUSY and self.capture is not None:
            cols = [c for c in self.capture["cols"] if len(c)]
            newest = cols[-1].t[-1] if cols else self.capture["t0"] - self.capture.get("delay", 0.0)
            if newest < self.capture["t0"]:
                return {"title": f"Step back from the boards \u2026 {np.ceil(self.capture['t0'] - newest):.0f} s",
                        "sub": "Then everyone keep still for a moment."}
            return {"title": f"Hold still: measuring the room \u2026 {np.ceil(max(0.0, self.capture['t1'] - newest)):.0f} s",
                    "sub": "Everyone keep still, nobody near the boards."}
        if st == WALKING:
            p = self.walk["purpose"] if self.walk is not None else "free"
            if p.startswith("enrol"):
                out = {"big": "Walk to the X", "big_color": screen.UZAYR if p == "enrol_uzayr" else screen.VISITOR,
                       "sub": "Then stand still on the X until it says \u201cGot it!\u201d", "dots": self._dots()}
            else:
                out = {"big": "Walk to the X", "sub": "Then stand still on the X until the answer appears."}
        elif st in (ENROL_VISITOR, ENROL_UZAYR):           # (ENROL_UZAYR: Redo my walks)
            turn = "uzayr" if st == ENROL_UZAYR else (self._turn() or "visitor")
            who = "Uzayr" if turn == "uzayr" else self._vname(BIG_NAME)
            sub = "Stand on the back mark. Walk to the X, then stand still."
            nu = self.n_enrol["uzayr"]
            if turn == "uzayr" and nu != self.n_enrol["visitor"]:   # (his count differs: which walk of his this is)
                sub = f"Walk {min(self.done['uzayr'] + 1, nu)} of {nu}: to the X, then stand still."
            if st == ENROL_UZAYR:                          # Redo my walks: say what happens after his walks
                k = self.done["uzayr"]
                left = max(1, nu - k)                      # ("Uzayr walks 4 more times", "twice more", "once more")
                words = f"Uzayr walks {_times(left)}" if not k else (
                    f"Uzayr walks {left} more times" if left > 2 else f"Uzayr walks {_times(left)} more")
                sub = words + (", then the game carries on." if self.name is not None
                               else ", then on to the next player.")
            out = {"big": f"{who}'s turn", "big_color": screen.UZAYR if turn == "uzayr" else screen.VISITOR,
                   "sub": sub, "dots": self._dots()}
        elif st == DECIDING:                               # (after a second look: both walks, averaged)
            walk = "both walks" if self.last is not None and self.last.get("both") else "this walk"
            out = {"big": "Thinking \u2026", "sub": f"Comparing {walk} with Uzayr's and {nm}'s"}
        elif st == AGAIN:                                  # a close call: nobody named yet, the action is the headline
            out = {"big": "Walk once more!", "sub": "Too close to call: the same walker again."}
        elif st == RESULT and self.last is not None and self.last["decision"] is not None:
            d = self.last["decision"]                      # (the nearer person, however close)
            both = bool(self.last.get("both"))             # (the second look: decided on both walks)
            who = "Uzayr" if d.label == "uzayr" else self._vname(BIG_NAME)
            why = d.why or models.why(d.contrib, d.label)
            colour = screen.UZAYR if d.label == "uzayr" else screen.VISITOR
            p = None
            if self.model is not None:                     # from both walks: the model's two-walk curve
                p = self.model.how_sure_both(d.margin) if both else self.model.how_sure(d.margin)
            conf = p if p is not None else d.margin
            # (no line under the name: the screen draws it bigger; Next walk is the operator's)
            out = {"big": who, "big_color": colour, "conf": conf,
                   "conf_text": f"How sure: {100 * conf:.0f}%", "conf_color": colour,
                   "why": ("Told apart by: " + _words(why[:2])) if why else ""}
            if both:                                       # (a line under the name: drawn at the smaller size)
                out["sub"] = "(from both walks)"
        elif st == READY:
            out = {"big": "Who's walking?",
                   "sub": f"Uzayr or {nm}: one of you walks to the X. The Wi-Fi guesses who."}
        elif self.last is not None and self.last["seen"] and self.last["purpose"] == "free" and self.last["f"]:
            out = {"big": f"{self.last['f']['speed']:.2f} m/s", "sub": "Your walking speed, measured by Wi-Fi"}
        else:
            out = {"big": "Who's walking?",
                   "sub": "Can Wi-Fi tell you and Uzayr apart by the way you walk? Ask to play!"}
        if self.notice is not None and now < self.notice[2] and st != BUSY:
            out["sub"], out["sub_color"] = self.notice[0], self.notice[1]
        return out

    def _dots(self) -> dict:
        """The learning walks as two rows of dots, in the order they are walked (pools.enrol_order visitor_first:
        the visitor's row on top; alternate: Uzayr's): done / now / next, no letters. Uzayr's row is all done when
        his walks are reused; with no player (Redo my walks) only his row. Each row has that person's learning walks
        (pools.n_enrol_uzayr for Uzayr), at most 5 dots."""
        turn = self._turn()
        walking = self.walk is not None and self.walk["purpose"].startswith("enrol")
        if walking:                                        # the turn being walked is the "now" dot
            turn = "uzayr" if self.walk["purpose"] == "enrol_uzayr" else "visitor"
        rows = []
        people = [("uzayr", "Uzayr", screen.UZAYR), ("visitor", self._vname(9), screen.VISITOR)]
        if self.order == "visitor_first":
            people.reverse()
        for who, name, colour in people:
            if who == "visitor" and self.name is None:
                continue
            done = self.done[who]
            rows.append((name, colour, ["done" if j < done else ("now" if who == turn and j == done else "next")
                                        for j in range(min(self.n_enrol[who], screen.DOTS_MAX))]))
        return {"rows": rows}

    def _smooth_line(self, y: np.ndarray) -> np.ndarray:
        """Display only: a running median over `line_smooth` columns of the body-speed line (gaps stay gaps)."""
        k = int(self.cfg["ui"]["line_smooth"])
        if k <= 1 or len(y) < k:
            return y
        pad = np.concatenate((np.full(k // 2, np.nan), y, np.full(k - 1 - k // 2, np.nan)))
        win = np.lib.stride_tricks.sliding_window_view(pad, k)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out = np.nanmedian(win, axis=1)
        return np.where(np.isfinite(y), out, np.nan)

    # -- results --------------------------------------------------------------------------------
    def summary(self) -> dict:
        s = self.stats
        at = np.array(s["draw_at"])
        fps = []
        if len(at) > 2:
            for t in np.arange(at[0] + 1.0, at[-1], 1.0):
                fps.append(int(((at > t - 1.0) & (at <= t)).sum()))
        lag = np.array(s["lag"]) if s["lag"] else np.zeros(1)
        return {"frames": len(at), "fps_median": float(np.median(fps)) if fps else 0.0,
                "fps_p5": float(np.percentile(fps, 5)) if fps else 0.0,
                "draw_ms_median": 1e3 * float(np.median(s["draw_s"])) if s["draw_s"] else 0.0,
                "draw_ms_p95": 1e3 * float(np.percentile(s["draw_s"], 95)) if s["draw_s"] else 0.0,
                "ui_busy_share": float((sum(s["draw_s"]) + sum(s.get("pump_s", []))) / max(at[-1] - at[0], 1e-9))
                if len(at) > 1 else 0.0,
                "lag_p95_s": float(np.percentile(lag, 95)), "lag_max_s": float(lag.max()),
                "lag_max_at_s": float(at[int(np.argmax(lag))] - at[0]) if len(at) == len(lag) and len(at) else None,
                "lag_over_0.3_s": int((lag > 0.3).sum()),
                "tick_ms_median": 1e3 * float(np.median(s["tick_s"])) if s["tick_s"] else 0.0,
                "packets": self.n_pkts, "lost": self.lost, "chain_restarts": self.chain.restarts,
                "tally": dict(self.tally.doc, visitors=len(self.tally.doc["visitors"]))}


# ------------------------------------------------------------------------------------------------ running it

# -- the receiver's port (live only) ---------------------------------------------------------------------

PORT_GLOB = "/dev/cu.usbserial-*"


def identify_receivers() -> list[str]:
    """Ports the thesis board identifier (development/io/RunLive/board_identify.py) reports as RECEIVER (parsed as
    development/live/RunLive/support/rig_run.py identify() does).

    Note: identify_board pulses each board's reset line once to read its boot banner.
    """
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-m", "development.io.RunLive.board_identify"], cwd=openday.CODE, env=env,
                         capture_output=True, text=True, timeout=600).stdout
    found = []
    for ln in out.splitlines():
        p = ln.split()
        if p and p[0].startswith("/dev/") and "RECEIVER" in ln:
            found.append(p[0])
    return found


def unsigned_timestamps() -> None:
    """The receiver firmware prints its 32-bit microsecond counter with %d, so after 35.8 min of uptime it reads
    negative, and the thesis recorder drops every such frame (it will not write a negative number into its
    unsigned field) until the counter wraps at 71.6 min. Within this process only, wrap the recorder's parser so
    the counter comes back unsigned, as the receiver meant it. Nothing in Code/ is changed."""
    openday.use_code()
    import pipeline.io.Capture.capture_logger as CL
    if getattr(CL.parse_line, "openday_unsigned", False):
        return
    orig, mod = CL.parse_line, 1 << 32

    def parse_line(*a, **k):
        rec = orig(*a, **k)
        if rec is not None and rec.local_timestamp < 0:
            rec.local_timestamp %= mod
        return rec

    parse_line.openday_unsigned = True
    CL.parse_line = parse_line


def csi_lines(port: str, secs: float = 1.5) -> int:
    """CSI lines a port prints in `secs` (-1: it would not open). Opened exactly as the thesis SerialReader opens a
    port without --reset, so nothing is reset: this only listens."""
    openday.use_code()
    import serial
    from pipeline.io.Capture.capture_logger import DEFAULT_BAUD
    try:
        s = serial.Serial(port, DEFAULT_BAUD, timeout=0.05)
    except Exception:  # noqa: BLE001 - a port that will not open is simply not the receiver
        return -1
    buf, t0 = bytearray(), time.monotonic()
    try:
        while time.monotonic() - t0 < secs:
            buf.extend(s.read(4096))
    finally:
        s.close()
    return buf.count(b"CSI_DATA")


def find_port(port: Optional[str], identify: bool) -> str:
    """--port wins; else the only /dev/cu.usbserial-* port; with several, the one port streaming CSI (each is
    listened to for 1.5 s, no reset), or --identify."""
    if port:
        return port
    ports = sorted(glob.glob(PORT_GLOB))
    if len(ports) == 1:
        return ports[0]
    if not ports:
        raise SystemExit("No receiver found (no /dev/cu.usbserial-* port). Plug the receiver into the Mac by USB.")
    if not identify:
        print(f"Several boards on USB ({', '.join(ports)}): listening for the one sending CSI ...", flush=True)
        live = [p for p in ports if csi_lines(p) >= 30]          # a receiver sends about 225 lines in 1.5 s
        if len(live) == 1:
            print(f"Receiver: {live[0]}", flush=True)
            return live[0]
        why = "none of them is sending CSI (is the TX powered?)" if not live else f"{len(live)} are sending CSI"
        raise SystemExit(f"Several boards on USB and {why}. Unplug all but the receiver, or pass --port PORT.")
    print("Identifying boards (each one is reset once to read its banner) ...", flush=True)
    rx = identify_receivers()
    if len(rx) != 1:
        raise SystemExit(f"Expected exactly one receiver, found {rx or 'none'}. Pass --port PORT.")
    return rx[0]


# -- live, replay and the key test ------------------------------------------------------------------------

def _run(cfg: dict, app: App, nodes: dict, reset: bool, duration: Optional[float], out_dir: Optional[str],
         on_start=None) -> tuple[int, dict]:
    openday.use_code()
    from pipeline.io.Capture.capture_logger import run_capture
    unsigned_timestamps()                                  # the recorder keeps frames past 35.8 min of uptime
    states: dict = {}
    if on_start:
        on_start()
    try:
        rc = run_capture(nodes, duration=duration, out_dir=out_dir, reset=reset, extra_consumers=[app],
                         stop_when=app.stop.is_set, quiet_monitor=True, states_out=states,
                         timestamped_subdir=False, writer_prefix="bench_" if out_dir else "",
                         run_config={"openday": True, "variant": cfg["dsp"]["variant"]})
    finally:
        app.close()
    return rc, states


def _say_if_no_model(cfg: dict, app: App) -> None:
    """A trained model is expected (classifier.kind is not m0) but did not load: say so on the terminal too, not only
    in the run log (the screen does not show it)."""
    if cfg["classifier"].get("kind", "m0") != "m0" and app.model is None:
        print(f"No trained model at {models.model_file(cfg)}: using m0 (How sure shows the raw margin)",
              file=sys.stderr, flush=True)


def run_live(cfg: dict, port: Optional[str], identify: bool, reset: bool, record: Optional[str],
             fullscreen: bool = True, new_room: bool = False) -> int:
    openday.use_code()                                     # first: a copy without ../Code stops here, before any
    openday.ensure_dirs()                                  # port check or window
    p = find_port(port, identify)
    rec_dir = None
    if record:
        rec_dir = os.path.join(os.path.abspath(record), time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(rec_dir, exist_ok=True)
        print(f"RECORDING raw CSI and events into {rec_dir}", flush=True)
    log = RunLog(os.path.join(openday.LOGS, f"openday_{time.strftime('%Y%m%d')}.log"))
    app = App(cfg, state_dir=openday.STATE, log=log, fullscreen=fullscreen, record_dir=rec_dir,
              new_room=new_room)                       # (--new-room: today's room level forgotten)
    _say_if_no_model(cfg, app)
    print(f"Receiver {p}. Click the window once so it has the keyboard.", flush=True)
    rc, states = _run(cfg, app, {"RX": p}, reset, None, rec_dir)
    st = states.get("RX")
    if st is not None and not st.opened:
        print(f"Could not open {p}: {st.error}. Is another program using it?", flush=True)
    log.close()
    return 0 if st is not None and st.opened else 1


def replay_rows(truths: list[dict], seen: list[dict]) -> list[dict]:
    """A replay's walks, one row each: the app's walk records (walk_log) beside the script's truth (who walked),
    each answer scored against that walker (the app tallies nothing, and a replay clicks no It was ...). A close
    call answered from both walks (classifier.second_look) never showed its own answer (only "walk once more"): its
    row says "shown": False (a missed second look shows the first walk's answer: that one is shown)."""
    rnd = lambda x: round(x, 2) if x is not None else None  # noqa: E731
    unshown = {w["second_of"] for w in seen if w.get("both")}
    res = []
    for truth, w in zip(truths, seen):
        d = w.get("decision")
        res.append({"role": truth["role"], "purpose": w["purpose"], "seen": w["seen"],
                    "decision": d.label if d else None, "margin": round(d.margin, 3) if d else None,
                    "not_sure": d.not_sure if d else None, "speed": round(w["f"]["speed"], 2) if w["f"] else None,
                    "gt_speed": round(truth["gt_speed"], 2),
                    "scored": None if d is None else ("right" if d.label == truth["role"] else "wrong"),
                    "reason": w["reason"], "active_s": round(w["active_s"], 2), "t_end": rnd(w.get("t_end")),
                    "t_arrive": rnd(w.get("t_arrive")), "screen_t_end": rnd(w.get("screen_t_end"))})
        if w.get("second_of") is not None:                 # (the second look: walked by the script's next walker)
            res[-1].update(second_of=w["second_of"], both=bool(w.get("both")))
        if w.get("id") in unshown:
            res[-1]["shown"] = False                       # (its answer never shown: walked once more)
    return res


def replay_tally(res: list[dict]) -> dict:
    """The answers the screen showed, right and wrong (replay_rows); with second looks also how many there were, and
    the close calls answered from both walks apart: how their own answer (never shown) would have scored."""
    dec = [r for r in res if r["decision"] and r.get("shown", True)]
    out = {"decisions": len(dec), "decisions_right": sum(r["scored"] == "right" for r in dec),
           "decisions_wrong": sum(r["scored"] == "wrong" for r in dec),
           "decisions_close_calls": sum(bool(r["not_sure"]) for r in dec)}   # margin below tau (named anyway)
    if any("second_of" in r for r in res):                 # classifier.second_look: close calls walked again (the
        first = [r for r in res if r.get("shown") is False]   # script's next walk is the second look, whoever walks it)
        out["second_looks"] = sum("second_of" in r for r in res)
        out["close_calls_walked_again"] = {"walks": len(first),
                                           "right_alone": sum(r["scored"] == "right" for r in first),
                                           "wrong_alone": sum(r["scored"] == "wrong" for r in first)}
    return out


def run_replay(cfg: dict, script: str, speed: float = 1.0, headless: bool = False, fullscreen: bool = False) -> int:
    cfg = copy.deepcopy(cfg)                                   # the replay's own settings, never the caller's
    cfg["baseline"]["delay_s"] = 0.0                           # the script's Ctrl+B measures at once
    cfg["walk"]["after_end_lock_s"] = 0.0                      # scripted clicks are never double clicks
    openday.ensure_dirs()
    t0 = time.time()
    plan = recordings.build_plan(script, cfg, speed=speed)
    print(f"Replay {os.path.basename(script)}: {len(plan.frames)} packets, {len(plan.walks)} walks, "
          f"{(plan.frames['t_mono'][-1] - plan.frames['t_mono'][0]) / speed:.0f} s (built in {time.time() - t0:.1f} s)",
          flush=True)
    sdir = os.path.join(openday.STATE, "replay")                 # never the real pool or tally
    shutil.rmtree(sdir, ignore_errors=True)
    log = RunLog(os.path.join(openday.LOGS, f"replay_{time.strftime('%Y%m%d')}.log"))
    app = App(cfg, state_dir=sdir, log=log, headless=headless, fullscreen=fullscreen, baseline_cache=False,
              keep_log=True)
    _say_if_no_model(cfg, app)
    rp = recordings.Replay(plan, cfg, speed, app.on_key)
    rc, states = _run(cfg, app, rp.nodes, False, rp.duration, None, on_start=rp.start)
    summ = app.summary()
    st = states.get("RX")
    summ["logger_pkt_s"] = (st.counters.ok / max(rp.duration - rp.delay, 1e-9)) if st is not None else 0.0
    summ["speed"] = speed
    summ["logger_pkt_per_stream_s"] = summ["logger_pkt_s"] / speed
    summ["queue_dropped"] = int(getattr(st, "queue_dropped", 0)) if st is not None else 0
    seen = [w for w in app.walk_log]
    res = replay_rows(plan.walks, seen)
    summ["walks"] = res
    summ["n_walks_planned"], summ["n_walks_seen_by_app"] = len(plan.walks), len(seen)
    summ["script"] = os.path.basename(script)
    summ.update(replay_tally(res))
    summ["end_reasons"] = dict(collections.Counter(r["reason"] for r in res))
    # a scripted step the app refused sends the replay off script: say so, and return 1
    off = [f"walk {i + 1} should be Uzayr's learning walk (after Redo my walks) but the app took it as "
           f"{w['purpose']}" for i, (truth, w) in enumerate(zip(plan.walks, seen))
           if truth.get("expect") and w["purpose"] != truth["expect"]]
    if len(seen) != len(plan.walks):
        off.append(f"the script has {len(plan.walks)} walks, the app started {len(seen)}")
    if plan.players and len(app.tally.doc["visitors"]) != plan.players:
        off.append(f"the script has {plan.players} New player steps, the app took "
                   f"{len(app.tally.doc['visitors'])}")
    summ["off_script"] = off
    out = os.path.join(openday.LOGS, f"replay_{os.path.splitext(os.path.basename(script))[0]}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "walks"}, indent=1, default=str), flush=True)
    for r in res:
        print(f"  {r['role']:<8} {r['purpose']:<14} seen {str(r['seen']):<5} -> {str(r['decision']):<8} "
              f"margin {r['margin']}  speed {r['speed']} (truth {r['gt_speed']})  ended {r['reason']} at "
              f"{r['t_end']} s, arrived {r['t_arrive']} s, on screen {r['screen_t_end']} s  {r['scored'] or ''}"
              + ("  (from both walks)" if r.get("both") else "")
              + ("  (not shown: walked once more)" if r.get("shown") is False else ""), flush=True)
    log.close()
    for msg in off:
        print(f"OFF SCRIPT: {msg}", flush=True)
    return 1 if off else 0


def keytest(cfg: dict) -> int:
    """Open the window and print every key name received (map the presenter clicker)."""
    import matplotlib.pyplot as plt
    for k in list(plt.rcParams):
        if k.startswith("keymap."):
            plt.rcParams[k] = []
    plt.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(10, 5), facecolor=screen.BG)
    txt = fig.text(0.5, 0.55, "Press keys (clicker buttons too).\nCtrl+Q or close the window to stop.",
                   ha="center", va="center", color=screen.INK, fontsize=22)
    seen: list[str] = []
    done = threading.Event()

    def on_key(e) -> None:
        k = screen.norm_key(e.key)
        seen.append(k)
        print(f"key: {k!r}   (matplotlib: {e.key!r})", flush=True)
        txt.set_text(f"{k}\n\nlast keys: {'  '.join(seen[-8:])}")
        if k == "ctrl+q":
            done.set()
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: done.set())
    plt.ion()
    plt.show(block=False)
    while not done.is_set() and plt.fignum_exists(fig.number):
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.03)
    plt.close(fig)
    print("Keys seen:", ", ".join(dict.fromkeys(seen)), flush=True)
    return 0
