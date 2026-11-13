"""Where the walker is on the lane, for the screen only: from the back mark (START) to the X.

Nothing here reaches a walk, its end, its answer, the room level or the tally. The app feeds it the same columns
as the walk (app.py, _feed_walk) and draws what it says (screen.py: the lane above the 3D picture). `lane.show:
false` in configs/openday.yaml switches it off, and the screen is exactly as it was without it.

How far: the distance walked since the click, divided by the lane's length (`lane.length_m`). The distance is the
Speed tile's own readout (the middle value of the valid Doppler speeds over the last ui.speed_smooth_s; a column is
valid when its energy is over dsp.active_k x the room level and it is no gap, as for the tile) added up column by
column, leaving out the odd column over MAX_MPS (a far walker's noise, at home). So the marker moves only while the
Wi-Fi sees someone moving, never backwards, and stands at START until the walker is first seen.

Near the X: from NEAR of the lane on, the marker slows down towards TOP, so it never reaches the X by itself. It goes
onto the X only when the app's own walk end fires (arrived, stopped, They've stopped, or the 9 s limit) and the walk
was seen (arrive). A missed walk, Cancel walk or a lost signal never puts it on the X.

How close (checked in scratch): on the thesis walks (Room B, a 5 m line with a press at every metre) the distance it
added up by the last press was 4.95 m of 5 (middle value), and halfway the dot was about 0.3 to 0.4 m from the walker,
counting the screen's own delay of about 0.2 s; on the home walks (no presses: a steady walk assumed) about 0.6 m.
At home the dot often waits 1 to 2 s on START (a far walker is faint). "On the X" comes about a second after the
walker stops, sometimes 2 to 3 s, and in about 1 walk in 4 or 5 a little early (that walk ended early).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from openday.chain import Columns, speed_of

NEAR, TOP = 0.75, 0.95       # the marker follows the distance up to NEAR of the lane, then slows towards TOP
MAX_MPS = 2.0                # a column's speed over this is left out (walking stays under it)
WORDS = ("Walking \u2026", "Nearly there", "On the X")   # beside the lane: where the walker is


def enabled(cfg: dict) -> bool:
    """lane.show in the config (off when the config has no lane section)."""
    return bool((cfg.get("lane") or {}).get("show", False))


class Lane:
    """One walk's place on the lane. `add` the walk's columns (any before the click only fill the speed
    readout), `arrive` when the app's walk end fires on a seen walk; `frac` is where the marker is (0 = START,
    1 = on the X)."""

    def __init__(self, cfg: dict, t_click: float, e0: float, hop_s: float) -> None:
        self.cfg, self.t0, self.hop = cfg, float(t_click), float(hop_s)
        self.length = max(float(cfg["lane"]["length_m"]), 0.1)
        self.thr = float(cfg["dsp"]["active_k"]) * float(e0)
        n = max(1, int(round(float(cfg["ui"]["speed_smooth_s"]) / self.hop)))
        self.recent: deque = deque(maxlen=n)               # the last n columns' speeds (NaN: not valid)
        self.dist = 0.0                                    # metres walked since the click (the estimate)
        self.on_x = False

    def add(self, cols: Columns) -> None:
        for t, e, f, g in zip(cols.t, cols.energy, cols.f50, cols.gap):
            v = float(speed_of(f, self.cfg)) if (e > self.thr and not g and np.isfinite(f)) else np.nan
            self.recent.append(v if v <= MAX_MPS else np.nan)
            if t < self.t0 or self.on_x:
                continue
            ok = [u for u in self.recent if np.isfinite(u)]
            if ok:
                self.dist += float(np.median(ok)) * self.hop

    def arrive(self) -> None:
        """The app's walk end fired and the walk was seen: the walker is on the X."""
        self.on_x = True

    @property
    def walked(self) -> float:
        """The distance walked over the lane's length (may pass 1)."""
        return self.dist / self.length

    @property
    def frac(self) -> float:
        """Where the marker is: the distance walked over the lane's length up to NEAR, then slowing towards TOP
        (never past it), and 1 once the walker has arrived."""
        if self.on_x:
            return 1.0
        x = self.walked
        if x <= NEAR:
            return x
        s = TOP - NEAR
        return NEAR + s * (1.0 - float(np.exp(-(x - NEAR) / s)))

    def words(self) -> str:
        """The few words beside the lane (WORDS): where the walker is, never how fast."""
        if self.on_x:
            return WORDS[2]
        return WORDS[1] if self.frac >= NEAR else WORDS[0]


def colour_for(purpose: Optional[str], uzayr: str, visitor: str, this: str) -> str:
    """The marker's colour: whose learning turn it is (Uzayr's, the visitor's), else a game or free walk's."""
    return {"enrol_uzayr": uzayr, "enrol_visitor": visitor}.get(purpose or "", this)
