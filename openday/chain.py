"""From the receiver's Wi-Fi packets to the speed picture (one column every hop, 53 ms) and the 3D bars.

This file, in order:
  1. The signal chain, run incrementally: packets in, spectrogram columns and contours out.
         packets -> DeviceClock -> FrontEnd (resample 150 Hz, Hampel, high-pass) -> Spectrogram (A or B) -> contours
     `Chain` runs them all; `Baseline` is the empty room they are measured against.
  2. hop_s: the time from one column to the next.
  3. The 3D picture (display only): `Ridge`, one row of bars per column, from the front end's samples.

Timing comes from the receiver's own microsecond counter (`local_timestamp`), not from `t_mono`: the serial
reader stamps one `t_mono` per read, so about seven packets share each host stamp (Code/pipeline/io/Capture/clean.py,
`fit_timebase`). `t_mono` is used only through the clock offset, to place the operator's click on the device clock.

Every stage keeps its own state and emits outputs in sample order, so the result does not depend on how the
packets are batched: the bench tools feed large batches, the live app small ones, and both get the same columns
(tests/test_chain.py checks this).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

# --- 1. The signal chain ---

_EPS = 1e-12


def live_bins(cfg: dict) -> np.ndarray:
    """The 103 live subcarrier bins: KEEP (id19_bank) minus NULL_SC (id19_featbank)."""
    sc = cfg["subcarriers"]
    drop = set(sc["drop"]) | set(sc["null_sc"])
    bins = np.array([i for i in range(128) if i not in drop])
    assert len(bins) == sc["n_live"] == 103, len(bins)
    return bins


def amplitudes(data: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Raw int8 CSI (n, 256) -> live-bin amplitudes (n, 103), each packet divided by its own mean.

    The amplitude is Code/pipeline/tracking/run.py `amplitude`. Per-packet normalisation removes AGC level jumps and
    makes the separate gain-compensation factor irrelevant.
    """
    d = np.asarray(data, np.int8).astype(np.float32).reshape(-1, 128, 2)
    amp = np.sqrt((d * d).sum(-1))[:, bins]
    return amp / np.maximum(amp.mean(axis=1, keepdims=True), _EPS)


class DeviceClock:
    """Receiver counter -> device seconds (unwrapped), plus the host-minus-device offset.

    The offset is the lower envelope of t_mono - t_dev (a packet drained at once has the smallest lag), kept as
    per-block minima so slow drift between the two clocks is followed.
    """

    def __init__(self, cfg: dict) -> None:
        c = cfg["clock"]
        self.mod, self.margin = int(c["modulus_us"]), int(c["wrap_margin_us"])
        self.block_s, self.n_blocks = float(c["offset_block_s"]), int(c["offset_blocks"])
        self.reset()

    def reset(self) -> None:
        self.last: Optional[int] = None
        self.wraps = 0
        self.blocks: deque = deque()

    def push(self, lt: np.ndarray, tm: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Returns (device seconds, host stamps, restarted) for the packets of the current device epoch."""
        lt = np.asarray(lt, np.int64)
        tm = np.asarray(tm, np.float64)
        restarted = False
        if len(lt) == 0:
            return np.zeros(0), tm, False
        prev = np.concatenate(([self.last if self.last is not None else lt[0]], lt[:-1]))
        back = lt < prev
        wrap = back & (prev > self.mod - self.margin)
        restart = np.flatnonzero(back & ~wrap)
        if restart.size:                                   # the board rebooted: start again from the new epoch
            k = int(restart[-1])
            lt, tm, wrap = lt[k:], tm[k:], np.zeros(len(lt) - k, bool)
            self.reset()
            restarted = True
        n_wrap = self.wraps + np.cumsum(wrap)
        t_dev = (lt + n_wrap * self.mod) * 1e-6
        self.wraps = int(n_wrap[-1])
        self.last = int(lt[-1])
        off = tm - t_dev
        blk = np.floor(tm / self.block_s).astype(np.int64)
        for b in np.unique(blk):
            m = float(off[blk == b].min())
            if self.blocks and self.blocks[-1][0] == b:
                self.blocks[-1] = (b, min(self.blocks[-1][1], m))
            else:
                self.blocks.append((b, m))
        while len(self.blocks) > self.n_blocks:
            self.blocks.popleft()
        return t_dev, tm, restarted

    @property
    def offset(self) -> Optional[float]:
        """Host monotonic time minus device time, or None before any packet."""
        return min(m for _b, m in self.blocks) if self.blocks else None


class FrontEnd:
    """Uniform 150 Hz resampling (linear, on device time), causal Hampel, causal Butterworth high-pass."""

    def __init__(self, cfg: dict) -> None:
        d = cfg["dsp"]
        self.fs = float(cfg["radio"]["fs_hz"])
        self.gap_s, self.restart_s = float(d["gap_s"]), float(d["restart_gap_s"])
        self.half = int(d["hampel_window"]) // 2
        self.nsig, self.mad_k = float(d["hampel_sigmas"]), float(d["hampel_mad_scale"])
        self.sos = signal.butter(int(d["filter_order"]), d["filter_low_hz"], btype="highpass", fs=self.fs,
                                 output="sos")
        self.reset()

    def reset(self) -> None:
        self.t_last: Optional[float] = None
        self.a_last: Optional[np.ndarray] = None
        self.k_next = 0
        self.hist: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.zi: Optional[np.ndarray] = None
        self.raw_out: tuple = (np.zeros(0, np.int64), None)   # the last batch before the high-pass (RawRing)

    def push(self, t: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """New packets -> (grid indices, filtered samples (n, 103), gap flags, restarted)."""
        t = np.asarray(t, np.float64)
        restarted = False
        self.raw_out = (np.zeros(0, np.int64), None)
        if self.t_last is not None and len(t):
            big = np.flatnonzero(np.diff(np.concatenate(([self.t_last], t))) > self.restart_s)
            if big.size:                                   # a long silence: start again after it
                k = int(big[-1])
                t, a = t[k:], a[k:]
                self.reset()
                restarted = True
        if self.t_last is None:
            if not len(t):
                return np.zeros(0, np.int64), np.zeros((0, a.shape[1])), np.zeros(0, bool), restarted
            self.t_last, self.a_last = float(t[0]), a[0]
            self.k_next = int(np.ceil(t[0] * self.fs))
            t, a = t[1:], a[1:]
        keep = t > np.maximum.accumulate(np.concatenate(([self.t_last], t)))[:-1]
        t, a = t[keep], a[keep]
        n_sc = self.a_last.shape[0]
        if not len(t):
            return np.zeros(0, np.int64), np.zeros((0, n_sc)), np.zeros(0, bool), restarted
        T = np.concatenate(([self.t_last], t))
        A = np.concatenate((self.a_last[None, :], a))
        k_end = int(np.floor(T[-1] * self.fs))
        ks = np.arange(self.k_next, k_end + 1, dtype=np.int64)
        self.k_next, self.t_last, self.a_last = k_end + 1, float(T[-1]), A[-1]
        g = ks / self.fs
        i = np.clip(np.searchsorted(T, g, side="right") - 1, 0, len(T) - 2)
        span = T[i + 1] - T[i]
        w = ((g - T[i]) / span)[:, None]
        x = A[i] * (1.0 - w) + A[i + 1] * w
        gap = span > self.gap_s
        ks, x, gap = self._hampel(ks, x, gap)
        self.raw_out = (ks, x)
        return ks, self._filter(x), gap, restarted

    def _hampel(self, ks: np.ndarray, x: np.ndarray, gap: np.ndarray):
        """Centred Hampel window, emitted `half` samples late so it stays causal."""
        if self.hist is not None:
            ks = np.concatenate((self.hist[0], ks))
            x = np.concatenate((self.hist[1], x))
            gap = np.concatenate((self.hist[2], gap))
        h = self.half
        n = len(ks) - 2 * h
        if n <= 0:
            self.hist = (ks, x, gap)
            return ks[:0], x[:0], gap[:0]
        win = sliding_window_view(x, 2 * h + 1, axis=0)                     # (n, n_sc, window)
        med = np.median(win, axis=-1)
        mad = self.mad_k * np.median(np.abs(win - med[..., None]), axis=-1)
        c = x[h:h + n]
        y = np.where(np.abs(c - med) > self.nsig * mad, med, c)
        self.hist = (ks[-2 * h:], x[-2 * h:], gap[-2 * h:])
        return ks[h:h + n], y, gap[h:h + n]

    def _filter(self, x: np.ndarray) -> np.ndarray:
        if not len(x):
            return x
        if self.zi is None:                                # start in steady state on the first sample
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


@dataclass
class Columns:
    """A batch of spectrogram columns. `t` is the device time of each column's centre."""

    t: np.ndarray
    power: np.ndarray
    gap: np.ndarray
    above: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    f50: Optional[np.ndarray] = None
    f90: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.t)


class Spectrogram:
    """Short-time power columns, every hop, from the filtered samples.

    Variant A (thesis recipe): PCA across subcarriers over the last `pca_window_s`, refitted every `pca_refit_s`;
    drop PC1, keep PCs 2-4, sum their STFT power. Variant B: sum the STFT power of every live subcarrier.
    """

    def __init__(self, cfg: dict, variant: Optional[str] = None, nfft: Optional[int] = None) -> None:
        d = cfg["dsp"]
        self.fs = float(cfg["radio"]["fs_hz"])
        self.variant = variant or d["variant"]
        self.nperseg, self.hop = int(d["nperseg"]), int(d["nperseg"]) - int(d["noverlap"])
        self.nfft = int(nfft or d["nfft"])
        self.win = signal.get_window(d["window"], self.nperseg)[:, None]
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / self.fs)
        self.pca_n = int(round(d["pca_window_s"] * self.fs))
        self.refit_n = int(round(d["pca_refit_s"] * self.fs))
        self.keep = np.asarray(d["pca_keep"], int) - 1
        self.reset()

    def reset(self) -> None:
        self.buf: Optional[np.ndarray] = None
        self.kbuf = np.zeros(0, np.int64)
        self.gbuf = np.zeros(0, bool)
        self.n = 0                                         # samples seen since the last reset
        self.next_col = self.nperseg if self.variant == "B" else self.pca_n
        self.next_fit = self.pca_n
        self.W: Optional[np.ndarray] = None

    def push(self, ks: np.ndarray, y: np.ndarray, gap: np.ndarray) -> Columns:
        nf = len(self.freqs)
        if len(ks):
            self.buf = y if self.buf is None else np.concatenate((self.buf, y))
            self.kbuf = np.concatenate((self.kbuf, ks))
            self.gbuf = np.concatenate((self.gbuf, gap))
            self.n += len(ks)
        base = self.n - len(self.kbuf)                     # absolute index of buf[0]
        cols = []
        while self.next_col <= self.n:
            if self.variant == "A":
                while self.next_fit <= self.next_col:      # refits fall in sample order, before their columns
                    seg = self.buf[self.next_fit - self.pca_n - base:self.next_fit - base]
                    seg = seg - seg.mean(axis=0)
                    _w, v = np.linalg.eigh(seg.T @ seg)
                    self.W = v[:, ::-1][:, self.keep]
                    self.next_fit += self.refit_n
                cols.append((self.next_col, self.W))
            else:
                cols.append((self.next_col, None))
            self.next_col += self.hop
        if not cols:
            self._trim()
            return Columns(np.zeros(0), np.zeros((0, nf)), np.zeros(0, bool))
        ends = np.array([c for c, _ in cols]) - base
        idx = ends[:, None] + np.arange(-self.nperseg, 0)[None, :]      # (n_cols, nperseg)
        t = (self.kbuf[idx[:, 0]] + self.kbuf[idx[:, -1]]) / 2.0 / self.fs
        g = self.gbuf[idx].any(axis=1)
        if self.variant == "A":
            seg = np.stack([self.buf[ix] @ W for ix, (_c, W) in zip(idx, cols)])   # (n_cols, nperseg, 3)
        else:
            seg = self.buf[idx]                                                # (n_cols, nperseg, 103)
        spec = np.fft.rfft(seg * self.win[None], n=self.nfft, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).sum(axis=2)
        self._trim()
        return Columns(t, power, g)

    def _trim(self) -> None:
        keep = max(self.pca_n, self.nperseg) + self.hop
        if self.buf is not None and len(self.kbuf) > keep:
            self.buf, self.kbuf, self.gbuf = self.buf[-keep:], self.kbuf[-keep:], self.gbuf[-keep:]


@dataclass
class Baseline:
    """Empty-room reference for one extractor: per-frequency median power and the median above-floor energy.
    `disturbed`: someone moved while the room was measured, so both come from its quiet part; `d`: how loud its
    loudest second was against that part (room_baseline)."""

    floor: np.ndarray
    e0: float
    n_cols: int
    variant: str
    nfft: int
    created: float = 0.0
    disturbed: bool = False
    d: float = 0.0

    def to_json(self) -> dict:
        return {"floor": [float(v) for v in self.floor], "e0": self.e0, "n_cols": self.n_cols,
                "variant": self.variant, "nfft": self.nfft, "created": self.created, "disturbed": self.disturbed,
                "d": self.d}

    @classmethod
    def from_json(cls, d: dict) -> "Baseline":
        return cls(np.asarray(d["floor"], float), float(d["e0"]), int(d["n_cols"]), d["variant"], int(d["nfft"]),
                   float(d.get("created", 0.0)), bool(d.get("disturbed", False)), float(d.get("d", 0.0)))


def band_mask(cfg: dict, freqs: np.ndarray) -> np.ndarray:
    lo, hi = cfg["dsp"]["band_hz"]
    return (freqs >= lo) & (freqs <= hi)


def contours(cols: Columns, base: Baseline, cfg: dict, freqs: np.ndarray) -> Columns:
    """Fill `above`, `energy`, `f50`, `f90` on a batch of columns, in place."""
    above = np.clip(cols.power - float(cfg["dsp"]["floor_scale"]) * base.floor[None, :], 0.0, None)
    band = band_mask(cfg, freqs)
    p, fb = above[:, band], freqs[band]
    energy = p.sum(axis=1)
    cols.above, cols.energy = above, energy
    cols.f50 = quantile_freq(p, fb, 0.5, energy)
    cols.f90 = quantile_freq(p, fb, 0.9, energy)
    return cols


def quantile_freq(p: np.ndarray, fb: np.ndarray, q: float, total: np.ndarray) -> np.ndarray:
    """Frequency below which a share q of each row's power lies; NaN for an empty row.

    Each bin's power is spread evenly over its own width (centre +/- half a bin), so a tone exactly on a bin
    centre reads as that centre (interpolating between centres would read half a bin low).
    """
    if not len(p):
        return np.zeros(0)
    df = float(fb[1] - fb[0]) if len(fb) > 1 else 1.0
    cum = np.cumsum(p, axis=1) / np.maximum(total, _EPS)[:, None]
    k = np.argmax(cum >= q, axis=1)
    rows = np.arange(len(p))
    c_hi = cum[rows, k]
    c_lo = np.where(k > 0, cum[rows, np.maximum(k - 1, 0)], 0.0)
    frac = np.clip((q - c_lo) / np.maximum(c_hi - c_lo, _EPS), 0.0, 1.0)
    out = fb[k] - df / 2 + frac * df
    return np.where(total > _EPS, out, np.nan)


def make_baseline(cols: Columns, cfg: dict, freqs: np.ndarray, variant: str, nfft: int, created: float = 0.0
                  ) -> Baseline:
    """Floor = per-frequency median of the baseline columns; e0 = their median above-floor band energy."""
    ok = cols.power[~cols.gap] if (~cols.gap).sum() >= 10 else cols.power
    floor = np.median(ok, axis=0)
    b = Baseline(floor, 0.0, int(len(ok)), variant, int(nfft), created)
    tmp = contours(Columns(np.zeros(len(ok)), ok, np.zeros(len(ok), bool)), b, cfg, freqs)
    b.e0 = float(max(np.median(tmp.energy), _EPS))
    return b


def room_baseline(cols: Columns, cfg: dict, freqs: np.ndarray, variant: str, nfft: int, created: float = 0.0
                  ) -> Baseline:
    """A Measure room capture's Baseline: make_baseline's, unless someone moved while the room was measured; then
    the floor and e0 come from the capture's quiet part, and it is marked disturbed.

    The quiet part: the quarter of the columns whose energy above make_baseline's floor, as a centred 1-s running
    median, is lowest. The floor is measured again on those columns and e0 is their median energy above it. d = the
    loudest 1-s running median of every column's energy above that floor, over that e0: 3 to 12 on 30 s of a still
    room (the thesis's 12 empty-room recordings and 10 home measurements), 27 and 36 on the two home measurements
    someone moved in. From baseline.quiet_flag_d up, the quiet part is used."""
    b = make_baseline(cols, cfg, freqs, variant, nfft, created)
    ok = cols.power[~cols.gap] if (~cols.gap).sum() >= 10 else cols.power
    band, k = band_mask(cfg, freqs), float(cfg["dsp"]["floor_scale"])

    def energy(floor: np.ndarray) -> np.ndarray:           # each column's above-floor band energy, as contours
        return np.clip(ok - k * floor[None, :], 0.0, None)[:, band].sum(axis=1)

    def run_median(x: np.ndarray) -> np.ndarray:           # centred, over 1 s (fewer columns at the two ends)
        h = min(max(1, int(round(1.0 / hop_s(cfg)))), len(x)) // 2
        pad = np.concatenate((np.full(h, np.nan), x, np.full(h, np.nan)))
        return np.nanmedian(sliding_window_view(pad, 2 * h + 1), axis=1)

    s = run_median(energy(b.floor))
    quiet = s <= np.quantile(s, 0.25)
    floor = np.median(ok[quiet], axis=0)
    e = energy(floor)
    e0 = float(max(np.median(e[quiet]), _EPS))
    b.d = float(run_median(e).max() / e0)
    if b.d >= float(cfg["baseline"].get("quiet_flag_d", 15.0)):
        b.floor, b.e0, b.disturbed = floor, e0, True
    return b


def speed_of(f_hz, cfg: dict):
    """Doppler frequency -> walking speed: v = f * lambda / (2 F)."""
    r = cfg["radio"]
    return np.asarray(f_hz) * float(r["wavelength_m"]) / (2.0 * float(r["bistatic_factor"]))


class RawRing:
    """The front end's samples before the high-pass (Hampel'd amplitude, each packet divided by its mean, on the
    150 Hz grid) for the last `keep_s`. The gait features (walks.py) need the 0.8 to 3.5 Hz band the high-pass
    removes, and they run on the finished walk, so the same samples feed them live and offline."""

    def __init__(self, fs: float, keep_s: float) -> None:
        self.fs, self.n = float(fs), int(round(keep_s * fs))
        self.reset()

    def reset(self) -> None:
        self.chunks: deque = deque()
        self.count = 0

    def push(self, ks: np.ndarray, a: Optional[np.ndarray]) -> None:
        if a is None or not len(ks):
            return
        if self.chunks and ks[0] <= self.chunks[-1][0][-1]:   # the grid went backwards: a restart
            self.reset()
        self.chunks.append((ks, a))
        self.count += len(ks)
        while self.chunks and self.count - len(self.chunks[0][0]) >= self.n:
            self.count -= len(self.chunks.popleft()[0])

    def window(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """(device times, samples (n, n_sub)) with t0 <= t <= t1."""
        if not self.chunks:
            return np.zeros(0), np.zeros((0, 0))
        k = np.concatenate([c[0] for c in self.chunks])
        t = k / self.fs
        m = (t >= t0) & (t <= t1)
        return t[m], np.concatenate([c[1] for c in self.chunks])[m]


class Chain:
    """Clock + front end + one spectrogram extractor + (once a baseline exists) contours."""

    def __init__(self, cfg: dict, variant: Optional[str] = None, nfft: Optional[int] = None,
                 baseline: Optional[Baseline] = None) -> None:
        self.cfg = cfg
        self.bins = live_bins(cfg)
        self.clock = DeviceClock(cfg)
        self.front = FrontEnd(cfg)
        self.spec = Spectrogram(cfg, variant, nfft)
        self.raw = RawRing(float(cfg["radio"]["fs_hz"]), float(cfg["gait"]["keep_s"]))
        self.baseline = baseline
        self.reboots = 0                                   # the receiver's counter restarted (board rebooted)
        self.silences = 0                                  # a gap longer than restart_gap_s (signal lost)

    @property
    def freqs(self) -> np.ndarray:
        return self.spec.freqs

    def push(self, lt: np.ndarray, tm: np.ndarray, data: np.ndarray) -> Columns:
        """Packets (local_timestamp, t_mono, raw data) -> new columns."""
        t_dev, tm, restarted = self.clock.push(lt, tm)
        if restarted:                                      # the receiver rebooted: everything re-warms
            self.front.reset()
            self.spec.reset()
            self.raw.reset()
            self.reboots += 1
        a = amplitudes(data, self.bins)
        a = a[len(a) - len(t_dev):]
        ks, y, gap, silence = self.front.push(t_dev, a)
        if silence:
            self.spec.reset()
            self.raw.reset()
            self.silences += 1
        self.raw.push(*self.front.raw_out)
        cols = self.spec.push(ks, y, gap)
        if self.baseline is not None:
            contours(cols, self.baseline, self.cfg, self.freqs)
        return cols

    @property
    def restarts(self) -> int:
        return self.reboots + self.silences

    def to_device(self, t_mono: float) -> Optional[float]:
        """A host monotonic time (a click) on the device clock."""
        off = self.clock.offset
        return None if off is None else float(t_mono) - off


# --- 2. The time between columns ---


def hop_s(cfg: dict) -> float:
    """Seconds from one spectrogram column (and one ridge) to the next: nperseg - noverlap samples at fs_hz."""
    return (int(cfg["dsp"]["nperseg"]) - int(cfg["dsp"]["noverlap"])) / float(cfg["radio"]["fs_hz"])


# --- 3. The 3D picture (display only) ---

# The top picture: how much the Wi-Fi signal wobbles on each subcarrier, in 3D, one ridge per moment (newest in
# front; screen.py draws each ridge as a row of bars).
#
# Display only: nothing here reaches a walk, a feature or a decision. One ridge per waterfall hop (8 samples, 53 ms),
# made from the front end's samples before its high-pass (FrontEnd.raw_out), put in frequency order and lightly
# smoothed across neighbouring subcarriers.
#
# Why a wobble: a walk barely moves the signal's slow level (on the owner's home run a hop's average moved only 2-3x
# as much while someone walked as in a still room, a few %): it makes the signal flicker some 5-40 times a second (the
# waterfall's band), and a hop's average smooths that flicker away. So each ridge is the flicker itself: the samples
# high-passed like the front end's (FrontEnd), kept to the movement the waterfall uses (its PCs 2-4 across the
# subcarriers, refitted every pca_refit_s over pca_window_s: Spectrogram's variant A recipe), and each
# subcarrier's RMS over the waterfall's window (nperseg, ending at the hop) as a share of its level (the hop's raw
# average). +27 and +28 (the HT-LTF's alone, at the band's edge) wobble about twice as much as the rest even in a
# still room: each is capped at +26's wobble before the smoothing. A ridge is that wobble's shape across the
# subcarriers, its mean 1; how tall the ridges stand comes from the waterfall's own energy over the room's (app.py,
# screen.py), so in a still room they lie flat and a walk stands them up (on the owner's home run that energy was about
# 1x the room's still and 10x, often 50x or more, during a walk).
# No ridge until the movement is first fitted (pca_window_s after a start, like the waterfall's first column).
#
# Frequency order: the receiver's 128 bins are two estimates of the same channel (the legacy LTF in 0-63, the HT-LTF
# in 64-127), each in FFT order (subcarrier 0, +1..+31, then -32..-1). The 103 live bins cover subcarriers -26..-1
# and +1..+28, 54 in all, 49 of them twice. A ridge runs from -26 to +28 (left to right), the two estimates of a
# subcarrier averaged (each as a share of its own level, so the two halves' different levels drop out), so a
# neighbour on the ridge is a neighbouring frequency and the axis can carry true subcarrier numbers.


def subcarriers(bins) -> tuple:
    """Live bins -> (the subcarrier numbers they measure, sorted; each bin's place in the sorted numbers)."""
    bins = np.asarray(bins, int)
    k = bins % 64
    num = np.where(k < 32, k, k - 64)
    sc = np.unique(num)
    return sc, np.searchsorted(sc, num)


class Ridge:
    """The 3D picture's ridges (display only): one per waterfall column, each subcarrier's wobble in frequency
    order, shape mean 1; see the comment above."""

    def __init__(self, cfg: dict) -> None:
        u, d = cfg["ui"], cfg["dsp"]
        fs = float(cfg["radio"]["fs_hz"])
        self.hop = int(d["nperseg"]) - int(d["noverlap"])    # samples per ridge (= one waterfall column)
        self.n = max(2, int(round(float(u.get("ridge_s", 2.0)) * fs / self.hop)))
        self.smooth = int(u.get("ridge_smooth", 3)) | 1     # odd: a running mean centred on its subcarrier
        self.sc, self._at = subcarriers(live_bins(cfg))
        self.points = len(self.sc)                         # one point per subcarrier (54)
        self._count = np.bincount(self._at, minlength=self.points).astype(float)
        self.sos = FrontEnd(cfg).sos                       # the front end's high-pass and the waterfall's movement
        self.win = int(d["nperseg"])                       # (its PCs), over its window
        self.pca_n = int(round(float(d["pca_window_s"]) * fs))
        self.refit_n = max(1, int(round(float(d["pca_refit_s"]) * fs)))
        self.keep = np.asarray(d["pca_keep"], int) - 1
        self.ease = min(1.0, max(0.0, float(u.get("ridge_ease", 0.5))))   # each ridge's share of the new shape
        self.reset()

    def reset(self) -> None:
        """Start empty (a restart, a silence): the old ridges belong to a signal that is gone."""
        self.rows: deque = deque(maxlen=self.n)
        self.zi: Optional[np.ndarray] = None               # the high-pass's state
        self.hp: Optional[np.ndarray] = None               # the latest high-passed samples (and the raw ones),
        self.raw: Optional[np.ndarray] = None              # enough for a fit and a window
        self.seen = 0                                      # samples since the start
        self.next_row = self.pca_n                         # (a ridge ends a hop: the first once the movement fits)
        self.next_fit = self.pca_n
        self.P: Optional[np.ndarray] = None                # the projection onto the movement (bins x bins)

    def push(self, x: Optional[np.ndarray]) -> None:
        """New samples (n, live bins) from the front end, in time order: high-pass them, refit the movement on
        schedule and add a ridge every hop."""
        if x is None or not len(x):
            return
        x = np.asarray(x, np.float64)
        if self.hp is not None and self.hp.shape[1] != x.shape[1]:
            self.reset()
        if self.zi is None:                                # start in steady state on the first sample (FrontEnd)
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        self.hp = y if self.hp is None else np.concatenate((self.hp, y))
        self.raw = x.copy() if self.raw is None else np.concatenate((self.raw, x))
        self.seen += len(x)
        base = self.seen - len(self.hp)                    # the sample number of hp[0]
        while self.next_row <= self.seen:
            while self.next_fit <= self.next_row:          # refits fall in sample order, before their ridges
                seg = self.hp[self.next_fit - self.pca_n - base:self.next_fit - base]
                seg = seg - seg.mean(axis=0)
                _w, v = np.linalg.eigh(seg.T @ seg)
                w = v[:, ::-1][:, self.keep]
                self.P = w @ w.T
                self.next_fit += self.refit_n
            e = self.next_row - base
            self.add_wobble(self.hp[e - self.win:e] @ self.P, self.raw[e - self.hop:e].mean(axis=0))
            self.next_row += self.hop
        keep = max(self.pca_n, self.win) + self.hop       # (a fit may still be due for the last hop; never a view of
                                                           # the chain's own array: copied above)
        self.hp, self.raw = self.hp[-keep:], self.raw[-keep:]

    def add_wobble(self, y: np.ndarray, level: np.ndarray) -> None:
        """One ridge from a window of the movement's samples (n, live bins) and each bin's level: each subcarrier's
        RMS as a share of its level, the two estimates' powers averaged, in frequency order, +27 and +28 capped at
        +26, smoothed; its shape (mean 1), eased in from the last ridge's (ridge_ease)."""
        p = (np.asarray(y, np.float64) ** 2).mean(axis=0) / np.maximum(np.asarray(level, np.float64), 1e-9) ** 2
        q = np.sqrt(np.bincount(self._at, p, minlength=self.points) / np.maximum(self._count, 1.0))
        if len(q) > 3:                                     # +27, +28: no more than +26 (before the smoothing, else
            q[-2:] = np.minimum(q[-2:], q[-3])             # +26 takes up their noise)
        self.add_shape(q)

    def add_shape(self, q: np.ndarray) -> None:
        """One ridge from each subcarrier's wobble in frequency order (also how screenshots.py draws made-up ones):
        smoothed, as a shape (mean 1; none at all: flat), eased in from the last ridge's."""
        q = self._smooth(np.asarray(q, np.float64))
        mean = float(q.mean()) if len(q) else 0.0
        z = q / mean if mean > 1e-12 and np.isfinite(mean) else np.ones_like(q)
        if self.rows and len(self.rows[-1]) == len(z):
            z = self.rows[-1] + self.ease * (z - self.rows[-1])
        self.rows.append(z)

    def _smooth(self, q: np.ndarray) -> np.ndarray:
        k = self.smooth
        if k > 1 and len(q) > k:
            q = np.convolve(np.pad(q, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        return q

    def heights(self) -> np.ndarray:
        """The ridges, oldest first: (rows, points), each one's shape, mean 1 (screen.py stands them up by the room's
        movement)."""
        return np.array(self.rows) if self.rows else np.zeros((0, self.points))
