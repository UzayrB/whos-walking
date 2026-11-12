"""One walk: when it ends, its four numbers (speed, steps, swing, bounce) and the thesis's gait features.

This file, in order:
  1. WalkWindow: collects one walk's columns from the click and decides when the walk ended.
  2. The four numbers the screen shows: step_rate and walk_features.
  3. The thesis's gait features (walk_gait), copied unchanged from the thesis code (tests/test_walks.py checks them).

Walk windowing: a walk starts on the click and ends when the walker has arrived (the energy stopped climbing),
gone quiet, been stopped by the operator, was never seen (give-up), or at a hard limit.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from typing import Optional

import numpy as np
from scipy import signal

from openday.chain import Columns, speed_of

# --- 1. One walk's window ---


class WalkWindow:
    """Collects the columns of one walk.

    The threshold is applied to a short causal running median of the energy (`active_smooth` columns, pre-click
    columns included), so single noise spikes neither count as walking nor break a quiet run. `seen` is False when
    fewer than `min_active_s` were active. Each column after the click may end the walk; the first rule that holds
    wins, in this order:

    - "stopped": seen, and the smoothed energy has stayed below the active threshold for `quiet_s`;
    - "arrived": seen, the running maximum of the smoothed energy since the click is at least `arrive_min_x` x e0
      and at least `arrive_jump_x` x the median smoothed energy of the walk's active columns so far, and no new
      maximum has appeared for `arrive_hold_s` (the energy jumps as the walker reaches the X close to the boards
      and stops climbing once they stand still; the jump test keeps a bump in the middle of a walk from ending it);
    - "operator": the column is at or after the operator's stop request (`request_stop`);
    - "nothing": `give_up_s` after the click with less than `give_up_active_s` active (never seen);
    - "limit": `max_s` after the click.

    `t_arrive` is the time of the running maximum when it passes both arrival tests, else the last active column
    (None with no active column). The gait window closes there. e0 (the room's median energy) is
    threshold / dsp.active_k unless given. A rule whose key is 0 is off; `arrive=False` turns "arrived" off for
    offline windows that already close at a known stop line. `hold_s` replaces `arrive_hold_s` (the app's second,
    screen-only window waits walk.screen_hold_s).
    """

    def __init__(self, cfg: dict, t_click: float, threshold: float, hop_s: float, e0: Optional[float] = None,
                 arrive: bool = True, hold_s: Optional[float] = None) -> None:
        w = cfg["walk"]
        self.t0, self.thr, self.hop_s = float(t_click), float(threshold), float(hop_s)
        self.min_active, self.quiet, self.max_s = float(w["min_active_s"]), float(w["quiet_s"]), float(w["max_s"])
        self.e0 = float(e0) if e0 is not None else self.thr / float(cfg["dsp"]["active_k"])
        self.arrive_x = float(w.get("arrive_min_x", 80.0)) if arrive else 0.0
        self.arrive_hold = float(w.get("arrive_hold_s", 0.5)) if hold_s is None else float(hold_s)
        self.arrive_jump = float(w.get("arrive_jump_x", 4.0))
        self.give_up_s = float(w.get("give_up_s", 5.0))
        self.give_up_active = float(w.get("give_up_active_s", 0.3))
        self.stop_min = float(w.get("stop_min_s", 1.5))
        self.recent: deque = deque(maxlen=max(1, int(w["active_smooth"])))
        self.t: list[float] = []
        self.energy: list[float] = []
        self.f50: list[float] = []
        self.f90: list[float] = []
        self.active: list[bool] = []
        self.sm: list[float] = []                          # the smoothed energy of every column
        self.n_active = 0
        self.quiet_since: Optional[float] = None
        self.peak = -np.inf                                # running maximum of the smoothed energy (active columns)
        self.t_peak: Optional[float] = None
        self.t_last_active: Optional[float] = None
        self.sm_active: list[float] = []                   # smoothed energy of the active columns, sorted
        self.t_stop: Optional[float] = None                # the operator's stop request (device clock)
        self.ended = False
        self.reason = ""
        self.t_end_col: Optional[float] = None

    @property
    def active_s(self) -> float:
        return self.n_active * self.hop_s

    @property
    def seen(self) -> bool:
        return self.active_s >= self.min_active - 1e-9

    @property
    def peaked(self) -> bool:
        """The smoothed energy's peak passes the arrival tests: at least arrive_min_x x e0, and at least
        arrive_jump_x x the median smoothed energy of the active columns so far."""
        if self.arrive_x <= 0 or self.t_peak is None or self.peak < self.arrive_x * self.e0:
            return False
        if self.arrive_jump <= 0:
            return True
        a, n = self.sm_active, len(self.sm_active)
        med = a[n // 2] if n % 2 else 0.5 * (a[n // 2 - 1] + a[n // 2])
        return self.peak >= self.arrive_jump * med

    @property
    def t_arrive(self) -> Optional[float]:
        """When the walker arrived: the energy's peak if it reached the arrival level, else the last active column."""
        return self.t_peak if self.peaked else self.t_last_active

    def request_stop(self, t_dev: float) -> bool:
        """The operator says the walker has stopped (device time t_dev). Ignored (False) when under `stop_min_s`
        after the click (a double press) or once the walk has ended. Otherwise the walk ends "operator" on the first
        column at or after t_dev, at once if that column is already in (the columns after it are dropped, so the
        walk is the one the offline evaluation sees with the stop applied first); returns True when accepted."""
        if self.ended or t_dev - self.t0 < self.stop_min - 1e-9:
            return False
        self.t_stop = float(t_dev) if self.t_stop is None else min(self.t_stop, float(t_dev))
        if self.t and self.t[-1] >= self.t_stop:
            self._cut(bisect_left(self.t, self.t_stop) + 1)
            self.ended, self.reason, self.t_end_col = True, "operator", self.t[-1]
        return True

    def _cut(self, n: int) -> None:
        """Keep only the first n columns and recount what they gave (none of them ended the walk)."""
        for a in (self.t, self.energy, self.f50, self.f90, self.active, self.sm):
            del a[n:]
        self.n_active, self.peak, self.t_peak, self.t_last_active, self.sm_active = 0, -np.inf, None, None, []
        for t, act, sm in zip(self.t, self.active, self.sm):
            if act:
                self.n_active += 1
                self.t_last_active = t
                insort(self.sm_active, sm)
                if sm > self.peak:
                    self.peak, self.t_peak = sm, t

    def add(self, cols: Columns) -> bool:
        """Feed new columns (any before the click are skipped). Returns True once the walk has ended."""
        for t, e, f5, f9, g in zip(cols.t, cols.energy, cols.f50, cols.f90, cols.gap):
            if self.ended:
                break
            self.recent.append(float(e) if not g else 0.0)
            if t < self.t0:
                continue
            sm = float(np.median(self.recent))
            act = bool(sm > self.thr and not g and np.isfinite(f5))
            self.t.append(float(t))
            self.energy.append(float(e))
            self.f50.append(float(f5) if np.isfinite(f5) else np.nan)
            self.f90.append(float(f9) if np.isfinite(f9) else np.nan)
            self.active.append(act)
            self.sm.append(sm)
            if act:
                self.n_active += 1
                self.quiet_since = None
                self.t_last_active = float(t)
                insort(self.sm_active, sm)
                if sm > self.peak:
                    self.peak, self.t_peak = sm, float(t)
            elif self.quiet_since is None:
                self.quiet_since = float(t)
            if self.seen and self.quiet_since is not None and t - self.quiet_since >= self.quiet - 1e-9:
                self.reason = "stopped"
            elif self.seen and self.peaked and t - self.t_peak >= self.arrive_hold - 1e-9:
                self.reason = "arrived"
            elif self.t_stop is not None and t >= self.t_stop:
                self.reason = "operator"
            elif self.give_up_s > 0 and t - self.t0 >= self.give_up_s - 1e-9 \
                    and self.active_s < self.give_up_active - 1e-9:
                self.reason = "nothing"
            elif t - self.t0 >= self.max_s:
                self.reason = "limit"
            if self.reason:
                self.ended, self.t_end_col = True, float(t)
        return self.ended

    def arrays(self) -> dict[str, np.ndarray]:
        return {"t": np.asarray(self.t), "energy": np.asarray(self.energy), "f50": np.asarray(self.f50),
                "f90": np.asarray(self.f90), "active": np.asarray(self.active, bool)}


# --- 2. The four numbers: speed, steps, swing, bounce ---

# Per-walk features. Motion (Doppler) only: no static amplitude, subcarrier shape, RSSI or walk duration.


def step_rate(energy: np.ndarray, active: np.ndarray, hop_s: float, cfg: dict) -> float:
    """Steps/min from the walk's energy envelope: the autocorrelation peak for lags in `step_lag_s`, parabolic.

    The envelope is the log energy from the first to the last active column (echo power spans decades as the
    walker nears), detrended by a polynomial of `step_detrend_order`, then first-differenced (`step_diff`).
    """
    f = cfg["features"]
    idx = np.flatnonzero(active)
    if len(idx) < 8:
        return float("nan")
    env = np.log(np.maximum(energy[idx[0]:idx[-1] + 1], 1e-12))
    x = np.arange(len(env))
    env = env - np.polyval(np.polyfit(x, env, int(f["step_detrend_order"])), x)
    if f["step_diff"]:
        env = np.diff(env)
    lo, hi = (s / hop_s for s in f["step_lag_s"])
    lags = np.arange(max(1, int(np.floor(lo))), int(np.ceil(hi)) + 2)
    lags = lags[lags < len(env) - 1]
    inside = (lags >= lo) & (lags <= hi)
    if not inside.any():
        return float("nan")
    den = float((env * env).sum()) or 1.0
    r = np.array([(env[:-L] * env[L:]).sum() / den for L in lags])
    j = int(np.flatnonzero(inside)[np.argmax(r[inside])])
    lag = float(lags[j])
    if 0 < j < len(r) - 1:
        a, b, c = r[j - 1], r[j], r[j + 1]
        d = a - 2 * b + c
        if d < 0:
            lag += 0.5 * (a - c) / d
    return 60.0 / (min(max(lag, lo), hi) * hop_s)


def walk_features(w: dict[str, np.ndarray], hop_s: float, cfg: dict) -> Optional[dict[str, float]]:
    """Speed (m/s), steps (/min), swing (f90/f50) and bounce (detrended f50 SD / mean) over the active columns."""
    act = w["active"] & np.isfinite(w["f50"]) & (w["f50"] > 0)
    if act.sum() < 3:
        return None
    f50, f90, t = w["f50"][act], w["f90"][act], w["t"][act]
    trend = np.polyval(np.polyfit(t - t[0], f50, 1), t - t[0]) if len(f50) > 2 else np.full_like(f50, f50.mean())
    return {
        "speed": float(speed_of(np.median(f50), cfg)),
        "steps": step_rate(w["energy"], w["active"], hop_s, cfg),
        "swing": float(np.median(f90 / f50)),
        "bounce": float(np.std(f50 - trend) / max(np.mean(f50), 1e-9)),
    }


# --- 3. The thesis's gait features (copied unchanged from the thesis code; tests/test_walks.py checks them) ---

# The thesis's single-link gait features, RX2 only: Doppler contours, modulation, cadence and the Wi-Fi speed.
#
# Copied, not imported: importing Code/pipeline/identification/engine.py reads the thesis's feature-name bank from
# Code/runs/ when the module loads. Every function marked "copy of" is that function verbatim (Code as of 6 Nov
# 2026), apart from the module prefixes; tests/test_walks.py checks that the copies give the originals' numbers.
# `single_link` follows engine.live_features for one link. Left out on purpose: static amplitude levels,
# subcarrier shape, the RSSI and the walk's duration. (GAIT_NAMES and GAIT_PLAIN are OpenDay's own lists.)

FS = 150.0                                     # copy of identification/features.py FS
STEP_LO, STEP_HI = 1.3, 2.8                    # copy of identification/features.py: plausible step rate, Hz
BANDS = {"b03_2": (0.3, 2.0), "b08_35": (0.8, 3.5), "dop": (4.0, 70.0), "full": (0.3, 70.0)}
CADENCE_VOTE_MIN_ACF = 0.30                    # copy of identification/engine.py


def bandpass(x: np.ndarray, band: str, fs: float = FS, axis: int = 0) -> np.ndarray:
    """Copy of identification/features.py bandpass."""
    lo, hi = BANDS[band]
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float), axis=axis)


def reduce_link(a: np.ndarray, band: str, reduction: str, n_pc: int = 6,
                bmat: np.ndarray = None) -> np.ndarray:
    """Copy of identification/features.py reduce_link: (n, n_sub) amplitude of ONE link -> (n_channels, n)."""
    a = np.asarray(a, float)
    if reduction == "mean":
        return bandpass(a.mean(axis=1), band)[None, :]
    b = bandpass(a, band, axis=0) if bmat is None else bmat
    if reduction == "rms":
        e = np.sqrt(np.mean(b * b, axis=1))
        return bandpass(e, "b03_2" if band == "b03_2" else "b08_35")[None, :]
    if reduction == "pca":
        b = b - b.mean(axis=0, keepdims=True)
        w, V = np.linalg.eigh(b.T @ b)
        V = V[:, np.argsort(w)[::-1][:n_pc]]
        comp = (b @ V).T
        for i in range(comp.shape[0]):
            if np.mean(comp[i] ** 3) < 0:
                comp[i] = -comp[i]
        return comp
    raise ValueError(reduction)


def _acf(x: np.ndarray) -> np.ndarray:
    """Copy of identification/features.py _acf."""
    x = np.asarray(x, float) - np.mean(x)
    if not np.any(x):
        return np.zeros_like(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[:n]
    return r / max(r[0], 1e-12)


def cadence_estimates(x: np.ndarray, fs: float = FS) -> dict:
    """Copy of identification/features.py cadence_estimates."""
    x = np.asarray(x, float)
    out = {"welch": None, "acf_cycle": None, "peaks": None, "acf_peak": None}
    if len(x) < int(1.5 * fs):
        return out
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), 512), nfft=4096)
    m = (f >= 0.5) & (f <= STEP_HI)
    if m.any():
        out["welch"] = float(f[m][np.argmax(p[m])])
    r = _acf(x)
    lo, hi = int(0.3 * fs), min(int(2.2 * fs), len(r) - 1)
    if hi > lo + 3:
        pk, pr = signal.find_peaks(r[lo:hi], height=0.0)
        if len(pk):
            k = pk[int(np.argmax(pr["peak_heights"]))] + lo
            out["acf_cycle"] = float(fs / k)
            out["acf_peak"] = float(r[k])
    pk, _ = signal.find_peaks(x, distance=max(int(fs / STEP_HI), 2),
                              prominence=0.3 * np.std(x) if np.std(x) > 0 else None)
    if len(pk) >= 3:
        out["peaks"] = float(fs / np.median(np.diff(pk)))
    return out


def _candidates(v):
    """Copy of identification/features.py _candidates."""
    if v is None or not np.isfinite(v) or v <= 0:
        return []
    return [c for c in (0.5 * v, v, 2.0 * v) if STEP_LO <= c <= STEP_HI]


def resolve_cadence(estimate_sets: list) -> dict:
    """Copy of identification/features.py resolve_cadence: harmonic resolution by agreement."""
    cands = []
    for es in estimate_sets:
        for k in ("welch", "acf_cycle", "peaks"):
            cands.append((k, _candidates(es.get(k))))
    pool = sorted({c for _k, cs in cands for c in cs})
    if not pool:
        return {"cadence_hz": None, "agree_frac": 0.0, "n_estimates": 0}
    best = None
    for c0 in pool:
        near = [min(cs, key=lambda c: abs(c - c0)) for _k, cs in cands if cs]
        near = np.asarray(near)
        support = int(np.sum(np.abs(near - c0) <= 0.1))
        score = (-support, float(np.median(np.abs(near - c0))))
        if best is None or score < best[0]:
            best = (score, c0, near)
    _s, c0, near = best
    agree = np.abs(near - c0) <= 0.1
    return {"cadence_hz": float(np.median(near[agree])) if agree.any() else float(c0),
            "agree_frac": float(agree.mean()), "n_estimates": int(len(near))}


def strong_cadence(estimate_sets: list) -> dict:
    """Copy of identification/engine.py strong_cadence."""
    strong = [e for e in estimate_sets if e.get("acf_peak") is not None and e["acf_peak"] >= CADENCE_VOTE_MIN_ACF]
    if not strong:
        return {"cadence_hz": None, "agree_frac": 0.0, "n_estimates": 0}
    return resolve_cadence(strong)


def contour_interp(f: np.ndarray, mag: np.ndarray, pct: float) -> np.ndarray:
    """Copy of identification/engine.py contour_interp."""
    f = np.asarray(f, float)
    df = float(f[1] - f[0])
    assert np.allclose(np.diff(f), df), "contour_interp needs uniform bins"
    m = np.asarray(mag, float) ** 2
    tot = m.sum(axis=1)
    c = np.cumsum(m, axis=1)
    out = np.full(m.shape[0], np.nan)
    ok = tot > 0
    if ok.any():
        thr = (pct / 100.0) * tot[ok]
        cc, mm = c[ok], m[ok]
        j = (cc >= thr[:, None]).argmax(axis=1)
        r = np.arange(len(j))
        prev = np.where(j > 0, cc[r, np.maximum(j - 1, 0)], 0.0)
        frac = (thr - prev) / np.maximum(mm[r, j], 1e-300)
        out[ok] = f[j] + (frac - 0.5) * df
    return out


def doppler_features_h(x: np.ndarray, fs: float = FS, win_s: float = 0.2, cadence_hz=None) -> dict:
    """Copy of identification/engine.py doppler_features_h."""
    from scipy import signal
    x = np.atleast_2d(np.asarray(x, float))
    n = int(round(win_s * fs))
    if x.shape[1] < 4 * n:
        return {}
    f, t, Z = signal.stft(x, fs=fs, nperseg=n, noverlap=int(0.75 * n), axis=-1, nfft=max(n, 64), boundary=None, padded=False)
    mag = np.abs(Z).sum(axis=0).T
    keep = (f >= 4.0) & (f <= 0.95 * fs / 2)
    f, mag = f[keep], mag[:, keep]
    P = mag ** 2
    tot = P.sum(axis=1)
    ok = tot > 0
    if ok.sum() < 8:
        return {}
    cont = {q: contour_interp(f, mag[ok], float(q)) for q in (25, 50, 75, 95)}
    med50 = max(float(np.median(cont[50])), 1e-6)
    out = {"leg_torso": float(np.median(cont[95]) / med50), "q25_torso": float(np.median(cont[25]) / med50),
           "q75_torso": float(np.median(cont[75]) / med50)}
    fr = 1.0 / max(np.median(np.diff(t)), 1e-9)
    for q in (50, 95):
        cq = cont[q] / med50
        out[f"c{q}_cv"] = float(np.std(cq) / max(np.mean(cq), 1e-9))
        r = _acf(cq)
        lo, hi = int(0.3 * fr), min(int(2.2 * fr), len(r) - 1)
        out[f"c{q}_periodicity"] = float(r[lo:hi].max()) if hi > lo else np.nan
        if cadence_hz:
            L = int(round(2.0 * fr / cadence_hz))
            segs = [cq[s:s + L] for s in range(0, len(cq) - L + 1, L)] if L >= 4 else []
            if len(segs) >= 2:
                Pc = np.vstack([np.interp(np.linspace(0, 1, 16, endpoint=False), np.linspace(0, 1, len(s), endpoint=False), s)
                                for s in segs]).mean(axis=0)
                Pc = np.roll(Pc, -int(np.argmax(Pc)))
                Pc = (Pc - Pc.mean()) / max(Pc.std(), 1e-9)
                for i, v in enumerate(Pc):
                    out[f"c{q}_cyc{i:02d}"] = float(v)
    fr_u = f / med50
    Pk = P[ok]
    prof = np.asarray([Pk[:, (fr_u >= a) & (fr_u < a + 0.5)].mean() if ((fr_u >= a) & (fr_u < a + 0.5)).any() else 0.0
                       for a in np.arange(0, 4.0, 0.5)])
    prof = prof / max(prof.sum(), 1e-20)
    for i, v in enumerate(prof):
        out[f"uprof_{i}"] = float(v)
    return out


def torso_hz_h(a_link: np.ndarray, bmat: np.ndarray = None) -> float:
    """Copy of identification/engine.py torso_hz_h: the median 50 % contour (4 to 70 Hz) of PCs 2 to 4."""
    from scipy import signal
    P = reduce_link(np.asarray(a_link, float), "dop", "pca", n_pc=4, bmat=bmat)[1:]
    f, _t, Z = signal.stft(P, fs=FS, nperseg=32, noverlap=24, nfft=64, axis=-1, boundary=None, padded=False)
    mag = np.abs(Z).sum(axis=0).T
    keep = (f >= 4.0) & (f <= 70.0)
    c = contour_interp(f[keep], mag[:, keep], 50.0)
    c = c[np.isfinite(c)]
    return float(np.median(c)) if len(c) else float("nan")


# ---------------------------------------------------------------------------------------- the per-walk set

DOP = (["leg_torso", "q25_torso", "q75_torso", "c50_cv", "c95_cv", "c50_periodicity", "c95_periodicity"]
       + [f"c50_cyc{i:02d}" for i in range(16)] + [f"c95_cyc{i:02d}" for i in range(16)]
       + [f"uprof_{i}" for i in range(8)])
GAIT_NAMES = ["cadence_hz"] + DOP + ["moddepth_dop", "sub_dispersion", "torso_hz"]

# visitor-friendly words for the "what separated this walk" line: (words, the features they cover)
GAIT_PLAIN = [("speed", ("speed", "torso_hz")),
              ("step rate", ("steps", "cadence_hz")),
              ("leg swing", ("swing", "leg_torso", "q75_torso", "q25_torso")),
              ("bounce", ("bounce", "c50_cv", "c95_cv")),
              ("stride rhythm", ("c50_periodicity", "c95_periodicity") + tuple(f"c50_cyc{i:02d}" for i in range(16))
               + tuple(f"c95_cyc{i:02d}" for i in range(16))),
              ("body motion", tuple(f"uprof_{i}" for i in range(8))),
              ("movement strength", ("moddepth_dop", "sub_dispersion"))]


def single_link(AN: np.ndarray) -> dict:
    """engine.live_features for one link: AN (n, n_sub) amplitude under per-packet normalisation, 150 Hz.

    Cadence from the link's rms and PC1 channels in 0.8 to 3.5 Hz (the thesis's live channels; the mean channel
    is constant under normalisation); Doppler contours of PCs 2 to 6 in 4 to 70 Hz with 0.2 s frames; modulation
    depth and subcarrier dispersion; the torso's Doppler frequency. Missing values are NaN.
    """
    out = {n: float("nan") for n in GAIT_NAMES}
    AN = np.asarray(AN, float)
    if AN.ndim != 2 or AN.shape[0] < int(1.5 * FS) or AN.shape[1] < 8:
        return out
    bg = bandpass(AN, "b08_35", axis=0)
    rms = bandpass(np.sqrt(np.mean(bg ** 2, axis=1)), "b08_35")[None, :]
    pca = reduce_link(AN, "b08_35", "pca", n_pc=6, bmat=bg)
    bd = bandpass(AN, "dop", axis=0)
    dop = reduce_link(AN, "dop", "pca", n_pc=6, bmat=bd)
    cad = strong_cadence([cadence_estimates(rms[0]), cadence_estimates(pca[0])])
    c: Optional[float] = cad["cadence_hz"]
    out["cadence_hz"] = float("nan") if c is None else float(c)
    for k, v in doppler_features_h(dop[1:], win_s=0.2, cadence_hz=c).items():
        out[k] = v
    lvl = max(float(AN.mean()), 1e-6)
    d = bandpass(np.sqrt(np.mean(bd ** 2, axis=1)), "b08_35")
    out["moddepth_dop"] = float(np.std(d) / lvl)
    out["sub_dispersion"] = float(np.median(np.std(bg, axis=0)) / 1e-9)
    out["torso_hz"] = torso_hz_h(AN, bmat=bd)
    return {k: (float(v) if v is not None and np.isfinite(v) else float("nan")) for k, v in out.items()}


def walk_gait(raw_t: np.ndarray, raw_a: np.ndarray, t_click: float, t_end: float, cfg: dict) -> dict:
    """The gait set over one walk: the raw samples from the click to t_end (live and bench: the walk's
    arrival, WalkWindow.t_arrive; eval on the thesis recordings: the last active column before the stop line), at most gait.max_window_s."""
    g = cfg["gait"]
    t1 = min(float(t_end), float(t_click) + float(g["max_window_s"]))
    m = (raw_t >= t_click) & (raw_t <= t1)
    if (t1 - t_click) < float(g["min_window_s"]) or m.sum() < int(float(g["min_window_s"]) * FS):
        return {n: float("nan") for n in GAIT_NAMES}
    return single_link(raw_a[m])
