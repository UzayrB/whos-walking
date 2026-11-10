"""Offline evaluation on the Room B identification walks (RX2 only), and retraining the model.

This file, in order:
  1. The thesis passes through the app's own chain: MultiChain, baselines_for, run_pass, walk_of, collect.
  2. Analysis: speed against ground truth, false walks in an empty room, the readouts' spread, the two-person
     game, the "not sure" threshold tau, and timing.
  3. Outputs: write_fitted (the config lines marked "fitted by eval"), the plots, and run (`eval`).
  4. Fitting a model on the thesis people: the feature sets, fit_m0, fit_m1, fit_m2 and calibrate ("How sure").
  5. Leave two people out (`eval-pair`): run_pair compares the models, chooses one and, with --write, ships it.

Scoring the home-test (bench) recordings is in bench.py.

Every pass is streamed through the same causal chain the live app runs (chain.py, walks.py), as one
continuous stretch of its own recording: from warmup_s + pre_s before the first anchor (the click) to post_s
after the last. Nothing is joined: a join between two recordings is itself read as motion. The noise floor comes
from that participant's empty_idspot recording. The walk window closes at the last anchor (end_after_last_s), the
stand-in for the demo's stop line, because the Room B walkers turned round there instead of standing still.
Nothing here uses a non-causal shortcut.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import random
import re
import time
from typing import Callable, Optional

import numpy as np
from scipy import optimize, stats

import openday
from openday import chain, models as MD, recordings, walks

# --- 1. The thesis passes through the app's own chain ---

BLOCKS = ("natural", "slow", "fast")
VARIANTS = [("A", 128), ("A", 64), ("B", 128), ("B", 64)]


class MultiChain:
    """One clock and front end feeding several extractors, so variants are compared on identical samples."""

    def __init__(self, cfg: dict, keys: list[tuple[str, int]], baselines: Optional[dict] = None) -> None:
        self.cfg = cfg
        self.bins = chain.live_bins(cfg)
        self.clock, self.front = chain.DeviceClock(cfg), chain.FrontEnd(cfg)
        self.specs = {k: chain.Spectrogram(cfg, k[0], k[1]) for k in keys}
        self.raw = chain.RawRing(float(cfg["radio"]["fs_hz"]), float(cfg["gait"]["keep_s"]))
        self.baselines = baselines or {}
        self.cols: dict = {k: [] for k in keys}

    def push(self, lt: np.ndarray, tm: np.ndarray, data: np.ndarray) -> None:
        t_dev, tm, restarted = self.clock.push(lt, tm)
        if restarted:
            self.front.reset()
            self.raw.reset()
            for s in self.specs.values():
                s.reset()
        a = chain.amplitudes(data, self.bins)[len(data) - len(t_dev):]
        ks, y, gap, silence = self.front.push(t_dev, a)
        if silence:
            self.raw.reset()
        self.raw.push(*self.front.raw_out)
        for k, s in self.specs.items():
            if silence:
                s.reset()
            c = s.push(ks, y, gap)
            if k in self.baselines:
                chain.contours(c, self.baselines[k], self.cfg, s.freqs)
            self.cols[k].append(c)

    def feed(self, fr: np.ndarray, batch: int) -> None:
        for i in range(0, len(fr), batch):
            f = fr[i:i + batch]
            self.push(f["local_timestamp"], f["t_mono"], f["data"])

    def columns(self, k) -> chain.Columns:
        cs = [c for c in self.cols[k] if len(c)]
        cat = lambda a: np.concatenate([getattr(c, a) for c in cs]) if cs and getattr(cs[0], a) is not None else None  # noqa: E731
        return chain.Columns(cat("t"), cat("power"), cat("gap"), cat("above"), cat("energy"), cat("f50"), cat("f90"))


def baselines_for(cfg: dict, name: str, day: int, keys: list, batch: int) -> dict:
    """Noise floor per extractor from that participant's empty_idspot recording, through the same chain."""
    mc = MultiChain(cfg, keys)
    mc.feed(recordings.load_rx2(name, day, "empty_idspot"), batch)
    return {k: chain.make_baseline(mc.columns(k), cfg, mc.specs[k].freqs, k[0], k[1]) for k in keys}


def run_pass(cfg: dict, name: str, day: int, block: str, p: dict, keys: list, bases: dict, batch: int,
             gait_key=None) -> dict:
    """One pass, one continuous stretch of its own recording, through the chain; per extractor, its columns.
    With gait_key, also the gait set (walks.walk_gait) from the click to that extractor's last active column."""
    e, rp = cfg["eval"], cfg["replay"]
    a = p["anchor_times"]
    st = recordings.Stitcher(cfg, xfade=False)
    click = a[0] + st.add(recordings.segment(recordings.load_rx2(name, day, block), a[0] - rp["pre_s"] - e["warmup_s"],
                                     a[-1] + rp["post_s"]))
    mc = MultiChain(cfg, keys, bases)
    fr = st.frames()
    i_click = int(np.searchsorted(fr["t_mono"], click, side="right"))
    mc.feed(fr[:i_click], batch)
    t_click = click - mc.clock.offset                      # the click on the device clock, as live maps it
    mc.feed(fr[i_click:], batch)
    out = {"name": name, "day": day, "block": block, "pass_id": p["pass_id"], "dir": p["direction"],
           "gt": recordings.gt_speed(p, cfg), "dur": a[-1] - a[0]}
    end = t_click + out["dur"] + float(e["end_after_last_s"])
    for k in keys:
        c = mc.columns(k)
        m = (c.t >= t_click - 1.0) & (c.t <= end)          # after the click (plus the few just before, which the
                                                           # energy smoothing sees) up to the stop line
        out[k] = {"t": c.t[m] - t_click, "energy": c.energy[m], "f50": c.f50[m], "f90": c.f90[m], "gap": c.gap[m],
                  "e0": bases[k].e0}
    if gait_key is not None:
        w = walk_of(out[gait_key], cfg)
        out["gait"] = walks.walk_gait(*mc.raw.window(t_click, end), t_click, t_click + w["t_last_active"], cfg)
    return out


def walk_of(rec: dict, cfg: dict, k_active: Optional[float] = None) -> dict:
    """Walk windowing and features on a pass's stored columns (click at t = 0), for a given active threshold.
    A thesis pass's columns already close at its last anchor (the stop line), so the live "arrived" end, which
    estimates that stop, is off: the offline windows (and the model's training rows) stay as they were."""
    k = float(cfg["dsp"]["active_k"] if k_active is None else k_active)
    w = walks.WalkWindow(cfg, 0.0, k * rec["e0"], chain.hop_s(cfg), e0=rec["e0"], arrive=False)
    w.add(chain.Columns(rec["t"], None, rec["gap"], None, rec["energy"], rec["f50"], rec["f90"]))
    arr = w.arrays()
    f = walks.walk_features(arr, chain.hop_s(cfg), cfg) if w.seen else None
    act_t = arr["t"][arr["active"]] if len(arr["t"]) else np.zeros(0)
    return {"f": f, "active_s": w.active_s, "seen": w.seen, "reason": w.reason or "open",
            "t_end": w.t_end_col if w.t_end_col is not None else float("nan"),
            "t_last_active": float(act_t.max() - w.t0) if len(act_t) else float("nan")}


def collect(cfg: dict, keys: list, batch: int = 75, log=print) -> list[dict]:
    """Every pass of every person, both days, every block, both directions, through the chain."""
    rows = []
    for name in openday.PEOPLE:
        for day in (1, 2):
            bases = baselines_for(cfg, name, day, keys, batch)
            for block in BLOCKS:
                for p in recordings.passes(name, day, block):
                    rows.append(run_pass(cfg, name, day, block, p, keys, bases, batch))
            log(f"  {name} d{day}: {len(rows)} passes so far")
    return rows


# --- 2. Analysis ---


def speed_stats(g: np.ndarray, v: np.ndarray) -> dict:
    if len(g) < 3:
        return {"n": int(len(g))}
    return {"n": int(len(g)), "spearman": float(stats.spearmanr(g, v)[0]), "pearson": float(stats.pearsonr(g, v)[0]),
            "medae": float(np.median(np.abs(v - g))), "bias": float(np.median(v - g))}


def speed_table(rows: list[dict], key, cfg: dict) -> dict:
    """Doppler speed against ground truth for each direction, day and block, and for all of them together."""
    out = {}
    w = [(r, walk_of(r[key], cfg)) for r in rows]
    for d in ("S", "N"):
        for day in (None, 1, 2):
            for block in (None,) + BLOCKS:
                sub = [(r["gt"], x["f"]["speed"]) for r, x in w if r["dir"] == d and x["f"]
                       and (day is None or r["day"] == day) and (block is None or r["block"] == block)]
                seen = [x["seen"] for r, x in w if r["dir"] == d and (day is None or r["day"] == day)
                        and (block is None or r["block"] == block)]
                g, v = (np.array(z) for z in zip(*sub)) if sub else (np.zeros(0), np.zeros(0))
                st = speed_stats(g, v)
                st["seen"] = float(np.mean(seen)) if seen else float("nan")
                out[(d, day, block)] = st
    return out


def empty_false_walks(cfg: dict, key, k_values, batch: int = 75) -> dict:
    """Clicks on an empty room: floor from the first half of each empty_idspot recording, clicks every 1 s in the
    second half. A click with no walker that still yields a walk is a false walk."""
    out = {k: [0, 0] for k in k_values}
    for name in openday.PEOPLE:
        for day in (1, 2):
            fr = recordings.load_rx2(name, day, "empty_idspot")
            mc = MultiChain(cfg, [key])
            mc.feed(fr[:len(fr) // 2], batch)
            base = chain.make_baseline(mc.columns(key), cfg, mc.specs[key].freqs, key[0], key[1])
            mc2 = MultiChain(cfg, [key], {key: base})
            mc2.feed(fr, batch)
            c = mc2.columns(key)
            mid = c.t[0] + (c.t[-1] - c.t[0]) / 2
            for t0 in np.arange(mid, c.t[-1] - cfg["walk"]["max_s"] - 0.5, 1.0):
                rec = {"t": c.t - t0, "energy": c.energy, "f50": c.f50, "f90": c.f90, "gap": c.gap, "e0": base.e0}
                for k in k_values:
                    out[k][0] += walk_of(rec, cfg, k)["seen"]
                    out[k][1] += 1
    return out


def feature_rows(rows: list[dict], key, cfg: dict, names) -> list[dict]:
    out = []
    for r in rows:
        w = walk_of(r[key], cfg)
        if w["f"] is not None:
            out.append({**{k: r[k] for k in ("name", "day", "block", "dir", "pass_id", "gt")},
                        "x": np.array([w["f"][n] for n in names], float)})
    return out


def spread_table(fr: list[dict], names) -> dict:
    """Within-person SD (pooled) against between-person SD (of person means), per feature."""
    X = {p: np.array([r["x"] for r in fr if r["name"] == p]) for p in openday.PEOPLE}
    X = {p: x for p, x in X.items() if len(x) > 1}
    ss = sum(np.nansum((x - np.nanmean(x, 0)) ** 2, 0) for x in X.values())
    within = np.sqrt(ss / sum(len(x) - 1 for x in X.values()))
    means = np.array([np.nanmean(x, 0) for x in X.values()])
    between = np.std(means, axis=0, ddof=1)
    return {n: {"within": float(within[i]), "between": float(between[i]), "ratio": float(between[i] / within[i]),
                "means": {p: float(means[j, i]) for j, p in enumerate(X)}} for i, n in enumerate(names)}


def two_person(fr_enrol: list[dict], test_of: Callable, cfg: dict, names, use, n_enrol: int, draws: int, seed: int,
               tau: float = 0.0) -> dict:
    """All 15 pairs; per draw, n_enrol random enrolment walks each, test on test_of(a, b, used). Mean and SD over
    draws of the share right (not sure counts as a miss), plus the speed-only result and every margin."""
    rng = random.Random(seed)
    idx = [list(names).index(n) for n in use]
    acc, acc_s, ns, margins, correct = [], [], [], [], []
    for _ in range(draws):
        right = right_s = n_ns = tot = 0
        for a, b in itertools.combinations(openday.PEOPLE, 2):
            ra = [r for r in fr_enrol if r["name"] == a]
            rb = [r for r in fr_enrol if r["name"] == b]
            if len(ra) < n_enrol or len(rb) < n_enrol:
                continue
            ea, eb = rng.sample(ra, n_enrol), rng.sample(rb, n_enrol)
            pools = {"uzayr": np.array([r["x"][idx] for r in ea]), "visitor": np.array([r["x"][idx] for r in eb])}
            used = {id(r) for r in ea + eb}
            for r in test_of(a, b, used):
                d = MD.decide_m0(r["x"][idx], pools, use, cfg, tau=tau)
                truth = "uzayr" if r["name"] == a else "visitor"
                ok = d.label == truth
                right += ok and not d.not_sure
                n_ns += d.not_sure
                if d.speed_only is not None:
                    right_s += d.speed_only.label == truth
                margins.append(d.margin)
                correct.append(ok)
                tot += 1
        if tot:
            acc.append(right / tot)
            acc_s.append(right_s / tot)
            ns.append(n_ns / tot)
    return {"mean": float(np.mean(acc)), "sd": float(np.std(acc)), "speed_only": float(np.mean(acc_s)),
            "speed_only_sd": float(np.std(acc_s)), "not_sure": float(np.mean(ns)), "n_test": len(margins) // draws,
            "margins": np.array(margins), "correct": np.array(correct, bool)}


def tau_curve(margins: np.ndarray, correct: np.ndarray, max_ns: float) -> tuple[float, list]:
    """Not-sure rate and accuracy on answered walks against tau; the chosen tau maximises the latter with at most
    max_ns not sure."""
    curve, best = [], (0.0, -1.0)
    for tau in np.round(np.arange(0.0, 0.401, 0.005), 3):
        ans = margins >= tau
        nsr = 1.0 - ans.mean()
        acc = float(correct[ans].mean()) if ans.any() else float("nan")
        curve.append((float(tau), float(nsr), acc, float((correct & ans).mean())))
        if nsr <= max_ns + 1e-9 and acc > best[1] + 1e-9:
            best = (float(tau), acc)
    return best[0], curve


def timing(cfg: dict, key) -> dict:
    """Live-like batches of 8 packets through one chain: time per column; and the end-of-walk decision cost."""
    name, day, block = "P4", 1, "natural"
    base = baselines_for(cfg, name, day, [key], 75)[key]
    p = recordings.passes(name, day, block, "S")[0]
    a = p["anchor_times"]
    st = recordings.Stitcher(cfg, xfade=False)
    click = a[0] + st.add(recordings.segment(recordings.load_rx2(name, day, block), a[0] - cfg["replay"]["pre_s"]
                                     - cfg["eval"]["warmup_s"], a[-1] + cfg["replay"]["post_s"]))
    fr = st.frames()
    ch = chain.Chain(cfg, key[0], key[1], base)
    per, n_cols, cols = [], 0, []
    for i in range(0, len(fr), int(cfg["dsp"]["tick_batch"])):
        f = fr[i:i + int(cfg["dsp"]["tick_batch"])]
        t0 = time.perf_counter()
        c = ch.push(f["local_timestamp"], f["t_mono"], f["data"])
        per.append(time.perf_counter() - t0)
        n_cols += len(c)
        cols.append(c)
    w = walks.WalkWindow(cfg, click - ch.clock.offset, float(cfg["dsp"]["active_k"]) * base.e0, chain.hop_s(cfg))
    for c in cols:
        if len(c):
            w.add(c)
    names = cfg["features"]["names"]
    pools = {"uzayr": np.random.default_rng(0).normal(1, 0.1, (5, 4)), "visitor": np.random.default_rng(1).normal(1.2, 0.1, (3, 4))}
    t0 = time.perf_counter()
    for _ in range(20):
        f = walks.walk_features(w.arrays(), chain.hop_s(cfg), cfg)
        MD.decide_m0(np.array([f[n] for n in names]), pools, names, cfg)
    dec = (time.perf_counter() - t0) / 20
    d = cfg["dsp"]
    chain_delay = (int(d["nperseg"]) / 2 + int(d["hampel_window"]) // 2) / float(cfg["radio"]["fs_hz"])
    batch_wait = int(d["tick_batch"]) / float(cfg["radio"]["fs_hz"])
    return {"ms_per_column": 1e3 * sum(per) / max(n_cols, 1), "ms_per_push_p95": 1e3 * float(np.percentile(per, 95)),
            "ms_per_push_max": 1e3 * max(per), "decision_ms": 1e3 * dec, "chain_delay_s": chain_delay,
            "batch_wait_s": batch_wait, "latency_s": chain_delay + batch_wait + dec}


# --- 3. Outputs and the eval report ---


def write_fitted(path: str, updates: dict) -> None:
    """Rewrite the values of the given (section, key[, subkey]) lines in the YAML, keeping comments and layout."""
    lines = open(path, encoding="utf-8").read().split("\n")
    stack: list[str] = []
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)([A-Za-z_0-9]+):(\s*)([^#]*?)(\s*)(#.*)?$", ln)
        if not m:
            continue
        depth = len(m.group(1)) // 2
        stack = stack[:depth] + [m.group(2)]
        key = tuple(stack)
        if key in updates and m.group(4).strip():
            head = f"{m.group(1)}{m.group(2)}: {updates[key]}"
            com = m.group(6) or ""
            col = len(ln) - len(com) if com else 0
            lines[i] = (head.ljust(col - 1) + " " + com if com else head).rstrip()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"natural": "#2a78d6", "slow": "#eb6834", "fast": "#1baf7a"}


def plot_speed(rows: list[dict], key, cfg: dict, path: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True, facecolor=SURF)
    for ax, d, title in zip(axes, ("S", "N"), ("Towards the boards (S, the demo)", "Away from the boards (N)")):
        ax.set_facecolor(SURF)
        allg, allv = [], []
        for block, col in SERIES.items():
            pts = [(r["gt"], walk_of(r[key], cfg)["f"]) for r in rows if r["dir"] == d and r["block"] == block]
            pts = [(g, f["speed"]) for g, f in pts if f]
            if not pts:
                continue
            g, v = np.array(pts).T
            allg += list(g)
            allv += list(v)
            ax.scatter(g, v, s=26, color=col, edgecolor=SURF, linewidth=1.0, label=block, zorder=3)
        lim = (0.5, 2.2)
        ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=2, label="equal")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        rho = stats.spearmanr(allg, allv)[0] if len(allg) > 2 else float("nan")
        ax.set_title(f"{title}\nSpearman {rho:.3f}, n = {len(allg)}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Ground-truth pace (m/s, 5 m over the anchor times)", fontsize=9, color=INK2)
        ax.grid(color=GRID, lw=0.6)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel(f"Wi-Fi speed (m/s): median f50 x lambda / (2F), F = {cfg['radio']['bistatic_factor']:.3f}",
                       fontsize=9, color=INK2)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def plot_tau(curve: list, tau: float, max_ns: float, path: str) -> None:
    plt = _plt()
    t, nsr, acc, _ov = (np.array(c) for c in zip(*curve))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.plot(t, acc, color="#2a78d6", lw=2, label="accuracy on answered walks")
    ax.plot(t, nsr, color="#eb6834", lw=2, label="share 'Not sure'")
    ax.axhline(max_ns, color=INK2, lw=0.8, ls=":")
    ax.axvline(tau, color=INK, lw=1, ls="--")
    ax.text(tau, 0.02, f"  tau = {tau:.3f}", color=INK, fontsize=9)
    ax.set_xlabel("margin threshold tau", fontsize=9, color=INK2)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_title("Day 1 natural S, two people, three enrolment walks each", fontsize=10, color=INK, loc="left")
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def _f(x, p=3) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"


def _acc(r: dict) -> str:
    return f"{100 * r['mean']:.1f} ± {100 * r['sd']:.1f} %"


def run(cfg: dict, write: bool = True, log=print) -> str:
    """The offline evaluation (`eval`). Writes reports/eval_<date>.md, two PNGs, and the fitted YAML values."""
    openday.ensure_dirs()
    t_cpu, t_wall = time.process_time(), time.time()
    e = cfg["eval"]
    names = list(cfg["features"]["names"])
    log("1/5 streaming every Room B pass (RX2) through the causal chain, four extractors ...")
    rows = collect(cfg, VARIANTS, log=log)

    # 1. speedometer: variants at F = 1 (the bistatic factor is fitted just below), pick by S-pass Spearman
    cfg_f1 = json.loads(json.dumps(cfg))
    cfg_f1["radio"]["bistatic_factor"] = 1.0
    tables = {k: speed_table(rows, k, cfg_f1) for k in VARIANTS}
    best = max(VARIANTS, key=lambda k: tables[k][("S", None, None)]["spearman"])
    cfg["dsp"]["variant"], cfg["dsp"]["nfft"] = best[0], best[1]
    # the bias says how far the effective bistatic factor is from 1: fit it on day 1 S, check it on day 2 S
    ratio = lambda day: [walk_of(r[best], cfg_f1)["f"]["speed"] / r["gt"] for r in rows  # noqa: E731
                         if r["dir"] == "S" and r["day"] == day and walk_of(r[best], cfg_f1)["f"]]
    F_d1 = float(np.median(ratio(1)))
    F_all = round(float(np.median(ratio(1) + ratio(2))), 3)
    cfg_cv = json.loads(json.dumps(cfg_f1))
    cfg_cv["radio"]["bistatic_factor"] = F_d1
    cv = speed_table([r for r in rows if r["day"] == 2], best, cfg_cv)[("S", 2, None)]
    cfg["radio"]["bistatic_factor"] = F_all
    k_values = [3.0, 4.0, 6.0, 8.0, 12.0]
    log("   empty-room false-walk check ...")
    fw = empty_false_walks(cfg, best, k_values)
    k_table = {}
    for k in k_values:
        c2 = json.loads(json.dumps(cfg_f1))
        c2["dsp"]["active_k"] = k
        k_table[k] = (speed_table([r for r in rows if r["dir"] == "S"], best, c2)[("S", None, None)], fw[k])

    # 2. features (chosen extractor, calibrated F): spreads on day 1 natural S; prior_sd = within-person SDs
    log("2/5 features ...")
    fr = feature_rows(rows, best, cfg, names)
    sel = lambda day, block, d="S": [r for r in fr if r["day"] == day and r["block"] == block and r["dir"] == d]  # noqa: E731
    d1, d2 = sel(1, "natural"), sel(2, "natural")
    spreads = spread_table(d1, names)
    prior = {n: float(f"{spreads[n]['within']:.3g}") for n in names}
    cfg["classifier"]["prior_sd"] = prior
    nan_steps = sum(not np.isfinite(r["x"][names.index("steps")]) for r in fr if r["dir"] == "S")
    within_d1 = lambda a, b, used: [r for r in d1 if r["name"] in (a, b) and id(r) not in used]  # noqa: E731
    within_d2 = lambda a, b, used: [r for r in d2 if r["name"] in (a, b) and id(r) not in used]  # noqa: E731
    cross = lambda blk: (lambda a, b, used: [r for r in sel(2, blk) if r["name"] in (a, b)])  # noqa: E731
    D, S0, n3 = int(e["draws"]), int(e["seed"]), int(e["enrol_n"])

    # 4 first (it needs only forced decisions): tau from day-1 margins, then every accuracy with that tau
    log("3/5 two-person accuracy, 15 pairs x 50 draws per condition ...")
    base_d1 = two_person(d1, within_d1, cfg, names, names, n3, D, S0, tau=0.0)
    tau, curve = tau_curve(base_d1["margins"], base_d1["correct"], float(e["not_sure_max"]))
    cfg["classifier"]["tau"] = tau
    res = {}
    for label, enrol, test in (("Within day 1 (natural)", d1, within_d1), ("Within day 2 (natural)", d2, within_d2),
                               ("Cross-day: enrol day 1, test day 2 natural", d1, cross("natural")),
                               ("Cross-day: enrol day 1, test day 2 slow", d1, cross("slow")),
                               ("Cross-day: enrol day 1, test day 2 fast", d1, cross("fast"))):
        res[label] = (two_person(enrol, test, cfg, names, names, n3, D, S0, 0.0),
                      two_person(enrol, test, cfg, names, names, n3, D, S0, tau))
    sizes = {n: two_person(d1, within_d1, cfg, names, names, n, D, S0, 0.0) for n in e["enrol_sizes"]}
    loo = {n: two_person(d1, within_d1, cfg, names, [m for m in names if m != n], n3, D, S0, 0.0) for n in names}
    log("4/5 timing ...")
    tm = timing(cfg, best)
    cpu = time.process_time() - t_cpu

    # outputs
    day = time.strftime("%Y%m%d")
    png_s = os.path.join(openday.REPORTS, f"eval_{day}_speed.png")
    png_t = os.path.join(openday.REPORTS, f"eval_{day}_tau.png")
    plot_speed(rows, best, cfg, png_s)
    plot_tau(curve, tau, float(e["not_sure_max"]), png_t)
    if write:
        write_fitted(openday.CONFIG, {("dsp", "variant"): best[0], ("dsp", "nfft"): best[1],
                                    ("radio", "bistatic_factor"): F_all, ("classifier", "tau"): tau,
                                    **{("classifier", "prior_sd", n): v for n, v in prior.items()}})
    S = tables[best][("S", None, None)]
    four, spd = res["Within day 1 (natural)"][0]["mean"], res["Within day 1 (natural)"][0]["speed_only"]
    gain = 100 * (four - spd)
    wx0 = res["Cross-day: enrol day 1, test day 2 natural"][0]
    gain_x = 100 * (wx0["mean"] - wx0["speed_only"])
    help_ = {n: -100 * (loo[n]["mean"] - four) for n in names if n != "speed"}      # points lost by leaving it out
    role = lambda v: "helps" if v >= 1.0 else ("hurts" if v <= -1.0 else "adds nothing")  # noqa: E731
    verdict = (f"Within a day the extra features add {gain:+.1f} points over speed alone ("
               + "; ".join(f"{n} {role(v)}: leaving it out changes accuracy by {-v:+.1f}" for n, v in help_.items())
               + ")"
               + f". Across days they change accuracy by {gain_x:+.1f} points"
               + (": speed holds up across days and the gait features do not, so the within-day figure is the "
                  "upper end for a stale pool." if gain_x < 0 else ".")
               + (" Speed does most of the work." if gain < 5 else ""))
    ends = {}
    for r_ in rows:
        if r_["dir"] == "S":
            why = walk_of(r_[best], cfg)["reason"]
            ends[why] = ends.get(why, 0) + 1
    L = []
    L.append(f"# Offline evaluation, {time.strftime('%d %B %Y')}")
    L.append("")
    L.append(f"Room B identification walks, RX2 only (0.5 m beside the TX, the demo geometry), {len(rows)} unflagged "
             f"passes, every one streamed through the live app's causal chain. CPU {cpu:.0f} s.")
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append(f"- **Speedometer:** S-pass Spearman **{S['spearman']:.3f}** (n = {S['n']}), extractor "
             f"{best[0]} with nfft {best[1]}. {'Above' if S['spearman'] >= 0.5 else 'BELOW'} the 0.5 bar; "
             f"the thesis measured 0.744 on RX2 with a different extractor.")
    w1, wx = res["Within day 1 (natural)"][0], res["Cross-day: enrol day 1, test day 2 natural"][0]
    L.append(f"- **Two people, three walks each:** within day 1 **{_acc(w1)}**, within day 2 "
             f"{_acc(res['Within day 2 (natural)'][0])}, cross-day (day 2 natural) {_acc(wx)}. Chance is 50 %.")
    L.append(f"- **Speed alone** gets {100 * spd:.1f} % within day 1. {verdict}")
    L.append(f"- **Not sure threshold:** tau = **{tau:.3f}** (at most {100 * float(e['not_sure_max']):.0f} % "
             f"not sure within day 1).")
    L.append("")
    L.append("prior_sd and tau are fitted on the day-1 natural S walks by design, and the within-day-1 "
             "accuracy is measured on the same walks; prior_sd only floors the spread, tau only sets the not sure "
             "share.")
    L.append("")
    L.append("**Caveat.** Within-day results are optimistic: each person's passes were one contiguous recording "
             "block (the thesis's recording-block confound), so a feature can tell blocks apart rather than people. "
             "Cross-day results are pessimistic: everything about the room and the person's day has changed. The "
             "live demo, with both people enrolled within the same few minutes, sits between the two.")
    L.append("")
    L.append("## 1. Speedometer")
    L.append("")
    L.append(f"Doppler speed = median over active columns of f50 x lambda / (2F). Ground truth = 5 m over (last "
             f"anchor - first anchor). Variants compared at F = 1, active threshold k = {cfg['dsp']['active_k']} "
             f"x the baseline's median energy (smoothed over {cfg['walk']['active_smooth']} columns).")
    L.append("")
    L.append("| extractor | S Spearman | S Pearson | S median abs err (m/s) | S bias (m/s) | N Spearman | S walks seen |")
    L.append("|---|---|---|---|---|---|---|")
    for k in VARIANTS:
        a, b = tables[k][("S", None, None)], tables[k][("N", None, None)]
        mark = " **(chosen)**" if k == best else ""
        L.append(f"| {k[0]} nfft {k[1]}{mark} | {_f(a.get('spearman'))} | {_f(a.get('pearson'))} | "
                 f"{_f(a.get('medae'))} | {_f(a.get('bias'))} | {_f(b.get('spearman'))} | {100 * a['seen']:.0f} % |")
    L.append("")
    L.append("A = PCA across subcarriers (2 s window, refit every 0.5 s), PC1 dropped, PCs 2-4 kept. "
             "B = power summed over all 103 live subcarriers.")
    L.append("")
    L.append("Chosen extractor by subset (F = 1):")
    L.append("")
    L.append("| direction | day | block | n | Spearman | Pearson | median abs err | bias |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in ("S", "N"):
        for dd in (1, 2):
            for blk in (None,) + BLOCKS:
                a = tables[best][(d, dd, blk)]
                L.append(f"| {d} | {dd} | {blk or 'all'} | {a['n']} | {_f(a.get('spearman'))} | "
                         f"{_f(a.get('pearson'))} | {_f(a.get('medae'))} | {_f(a.get('bias'))} |")
    L.append("")
    L.append(f"**Calibration.** At F = 1 the readout runs slow (S bias {S['bias']:+.3f} m/s). Fitting F on day 1 "
             f"S passes gives F = {F_d1:.3f}; with it, day 2 S passes have median absolute error "
             f"{_f(cv.get('medae'))} m/s and bias {_f(cv.get('bias'))} m/s (held out). F = {F_all:.3f} (all S passes) "
             f"is written to the config, so the on-screen speed reads true in this geometry. Spearman does not depend "
             f"on F.")
    L.append("")
    L.append(f"![speed](eval_{day}_speed.png)")
    L.append("")
    lim = f"at the {cfg['walk']['max_s']:g} s limit"
    L.append("**How the offline walks end** (S passes): " + ", ".join(
        f"{n} {('at the stop line (last anchor)' if k == 'open' else 'on quiet' if k == 'stopped' else lim if k == 'limit' else k)}"
        for k, n in sorted(ends.items(), key=lambda kv: -kv[1])) + ". Each pass is one continuous stretch of its own "
        "recording; the window closes at the last anchor because the Room B walkers turned round there rather than "
        "standing still, so the live 'stopped on quiet' rule cannot be exercised offline.")
    L.append("")
    L.append("**Active threshold.** Clicks on an empty room (floor from the first half of each empty_idspot "
             "recording, a click every second in the second half) show whether noise alone makes a walk:")
    L.append("")
    L.append("| k | S Spearman | S walks seen | empty-room false walks |")
    L.append("|---|---|---|---|")
    for k, (a, (n_fw, n_cl)) in k_table.items():
        L.append(f"| {k:g}{' (config)' if k == float(cfg['dsp']['active_k']) else ''} | {_f(a.get('spearman'))} | "
                 f"{100 * a['seen']:.0f} % | {n_fw} of {n_cl} |")
    L.append("")
    L.append("## 2. Feature quality (day 1 natural, S passes)")
    L.append("")
    L.append("| feature | within-person SD | between-person SD | between / within | per-person means |")
    L.append("|---|---|---|---|---|")
    for n in names:
        s_ = spreads[n]
        means = ", ".join(f"{p} {v:.3g}" for p, v in s_["means"].items())
        L.append(f"| {n} | {s_['within']:.4g} | {s_['between']:.4g} | {s_['ratio']:.2f} | {means} |")
    L.append("")
    L.append(f"prior_sd is set to the within-person SDs: {', '.join(f'{n} {v:g}' for n, v in prior.items())}. "
             f"Step rate has no ground truth; it is the weakest feature (ratio {spreads['steps']['ratio']:.2f}) and "
             f"{nan_steps} S walks gave no step-rate peak. Speed's within-person SD is in calibrated m/s.")
    L.append("")
    L.append("## 3. Two-person accuracy")
    L.append("")
    L.append(f"All 15 pairs of the six participants, {D} random draws; per draw, random enrolment walks "
             f"(S natural, day 1 unless stated) and every remaining S walk of both as tests. Mean ± SD over draws. "
             f"'Forced' always names the nearer person; 'with tau' counts not sure as a miss.")
    L.append("")
    L.append("| condition | tests per draw | four features (forced) | speed only (forced) | with tau | not sure |")
    L.append("|---|---|---|---|---|---|")
    for label, (a, b) in res.items():
        L.append(f"| {label} | {a['n_test']} | {_acc(a)} | {100 * a['speed_only']:.1f} ± "
                 f"{100 * a['speed_only_sd']:.1f} % | {_acc(b)} | {100 * b['not_sure']:.0f} % |")
    L.append("")
    L.append("Enrolment size (within day 1, forced):")
    L.append("")
    L.append("| walks each | tests per draw | four features | speed only |")
    L.append("|---|---|---|---|")
    for n, a in sizes.items():
        L.append(f"| {n} | {a['n_test']} | {_acc(a)} | {100 * a['speed_only']:.1f} % |")
    L.append("")
    L.append("Each feature left out once (within day 1, three walks each, forced):")
    L.append("")
    L.append("| left out | accuracy | change |")
    L.append("|---|---|---|")
    for n, a in loo.items():
        L.append(f"| {n} | {_acc(a)} | {100 * (a['mean'] - four):+.1f} points |")
    L.append("")
    L.append(f"**Do the extra features earn their place?** Four features {100 * four:.1f} %, speed alone "
             f"{100 * spd:.1f} % within day 1. {verdict} The decision uses classifier.use "
             f"({', '.join(cfg['classifier'].get('use', names))}); the tiles always show all four.")
    L.append("")
    L.append("## 4. Not sure threshold")
    L.append("")
    L.append(f"Margin = (d_far - d_near) / (d_far + d_near). tau is the value that maximises accuracy on answered "
             f"walks with at most {100 * float(e['not_sure_max']):.0f} % not sure, within day 1: **tau = {tau:.3f}**.")
    L.append("")
    L.append("| tau | not sure | accuracy on answered | right overall |")
    L.append("|---|---|---|---|")
    for t, nsr, acc, ov in curve[::4]:
        L.append(f"| {t:.3f} | {100 * nsr:.0f} % | {100 * acc:.1f} % | {100 * ov:.1f} % |")
    L.append("")
    L.append(f"![tau](eval_{day}_tau.png)")
    L.append("")
    L.append("## 5. Timing")
    L.append("")
    L.append(f"- Chain, live-like batches of {cfg['dsp']['tick_batch']} packets: {tm['ms_per_column']:.2f} ms per "
             f"column (one column per 53 ms), {tm['ms_per_push_p95']:.2f} ms per batch at p95, "
             f"{tm['ms_per_push_max']:.2f} ms at most.")
    L.append(f"- End-of-walk decision latency: the deciding column is out {1e3 * tm['chain_delay_s']:.0f} ms after "
             f"its last sample (half the STFT window plus the Hampel delay), plus up to "
             f"{1e3 * tm['batch_wait_s']:.0f} ms waiting for a batch, plus {tm['decision_ms']:.1f} ms for the "
             f"features and the decision: **{1e3 * tm['latency_s']:.0f} ms** (target under 500 ms). The walk "
             f"itself ends about half a second after the walker stops (arrived, or {cfg['walk']['quiet_s']} s of "
             f"quiet), and the 'Thinking \u2026' view then shows for {cfg['ui']['reading_s']} s by design.")
    L.append(f"- Whole evaluation: {cpu:.0f} s CPU, {time.time() - t_wall:.0f} s wall.")
    L.append("")
    out = os.path.join(openday.REPORTS, f"eval_{day}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log(f"5/5 report written: {out}  ({cpu:.0f} s CPU)")
    return out


# --- 4. Fitting a model on the thesis people ---

# "core": no stride-cycle shape, no profile
CORE = MD.READOUTS + ["cadence_hz", "leg_torso", "q25_torso", "q75_torso", "c50_cv", "c95_cv", "c50_periodicity",
                      "c95_periodicity", "moddepth_dop", "sub_dispersion", "torso_hz"]
FEATURE_SETS = {"all": MD.ALL, "core": CORE}


def fit_standard(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu, np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)


def fit_m1(X: np.ndarray, person: np.ndarray, names, shrink: float = 0.3, keep: int = 0,
           ridge: float = 1e-3) -> MD.Model:
    mu, sd = fit_standard(X)
    m = MD.Model("m1", list(names), mu, sd)
    Z = m.z(X)
    ids = np.unique(person)
    means = np.array([Z[person == p].mean(axis=0) for p in ids])
    R = np.concatenate([Z[person == p] - means[i] for i, p in enumerate(ids)])
    W = R.T @ R / max(len(R) - len(ids), 1)
    W = (1 - shrink) * W + shrink * np.diag(np.diag(W)) + ridge * np.eye(len(W))
    e, V = np.linalg.eigh(W)
    T = (V / np.sqrt(np.maximum(e, 1e-9))) @ V.T                     # W^(-1/2)
    D = means - means.mean(axis=0)
    B = D.T @ D / max(len(ids) - 1, 1)
    lam, Q = np.linalg.eigh(T @ B @ T)
    order = np.argsort(lam)[::-1]
    lam, Q = np.maximum(lam[order], 0.0), Q[:, order]
    if keep:
        lam, Q = lam[:keep], Q[:, :keep]
    m.P, m.lam = Q.T @ T, lam
    return m


def fit_m2(X: np.ndarray, person: np.ndarray, recording: np.ndarray, names, l2: float = 1e-2) -> MD.Model:
    mu, sd = fit_standard(X)
    m = MD.Model("m2", list(names), mu, sd)
    Z = m.z(X)
    i, j = np.triu_indices(len(Z), 1)
    same = person[i] == person[j]
    use = ~same | (recording[i] != recording[j])                     # same person only across recordings
    i, j, same = i[use], j[use], same[use]
    F = MD._pair_feats(Z[i], Z[j])
    y = same.astype(float)
    wt = np.where(same, 0.5 / max(same.sum(), 1), 0.5 / max((~same).sum(), 1))   # balanced classes

    def loss(theta):
        w, b = theta[:-1], theta[-1]
        s = F @ w + b
        p = 1.0 / (1.0 + np.exp(-s))
        ll = np.logaddexp(0.0, s) - y * s
        g = p - y
        return float((wt * ll).sum() + 0.5 * l2 * (w @ w)), np.append(F.T @ (wt * g) + l2 * w, (wt * g).sum())

    res = optimize.minimize(loss, np.zeros(F.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    m.w, m.b = res.x[:-1], float(res.x[-1])
    return m


def fit_m0(X: np.ndarray, person: np.ndarray) -> MD.Model:
    """The nearest-centre rule (m0): prior_sd = the within-person SD of each readout on the training people."""
    names = MD.READOUTS
    mu, sd = fit_standard(X)
    m = MD.Model("m0", list(names), mu, sd)
    ids = np.unique(person)
    R = np.concatenate([X[person == p] - np.nanmean(X[person == p], axis=0) for p in ids])
    m.prior_sd = {n: float(np.nanstd(R[:, k])) for k, n in enumerate(names)}
    return m


def calibrate(margins: np.ndarray, correct: np.ndarray, bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Margin -> share right: equal-count bins, then pooled-adjacent-violators so it never falls as the margin
    grows. Returns (bin centre margins, share right)."""
    o = np.argsort(margins)
    m, c = np.asarray(margins, float)[o], np.asarray(correct, float)[o]
    parts = np.array_split(np.arange(len(m)), bins)
    x = [float(np.median(m[p])) for p in parts if len(p)]
    y = [float(c[p].mean()) for p in parts if len(p)]
    w = [len(p) for p in parts if len(p)]
    blocks = [[yi, wi, [i]] for i, (yi, wi) in enumerate(zip(y, w))]
    k = 0
    while k < len(blocks) - 1:                         # pool adjacent violators
        if blocks[k][0] > blocks[k + 1][0]:
            a, b = blocks[k], blocks.pop(k + 1)
            a[0] = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1])
            a[1] += b[1]
            a[2] += b[2]
            k = max(k - 1, 0)
        else:
            k += 1
    fit = np.empty(len(y))
    for val, _w, idx in blocks:
        fit[idx] = val
    return np.array(x), fit


# --- 5. Leave two people out (eval-pair) ---

# The only evaluation that counts for telling two NEW people apart.
#
# For each pair (A, B) of the six thesis participants, every learned part (whitening, pairwise model, prior_sd,
# tau) is fitted on the other four people only; A plays Uzayr and B the visitor (and the other way round), each
# enrolled with random S walks, and their remaining S walks are the tests. Four conditions:
#
#   c1 within a recording       enrol and test on day-1 natural (optimistic: each person's walks are one block)
#   c2 across recordings        enrol on day-2 natural, test on day-2 slow and fast (block broken, pace changes)
#   c3 stale Uzayr              A enrolled on day-2 natural, tested on day-1 natural; B enrolled and tested on
#                               day-1 natural (the stale-pool penalty; C3's fresh walks come from day-1 natural)
#   c4 across days              enrol on day-1 natural, test on day-2 natural
#
# Every walk went through the app's own causal chain (run_pass above, with the gait set), so offline = live.

CONDITIONS = {"c1": "within a recording (day-1 natural)", "c2": "across recordings, same day (day-2 natural -> slow/fast)",
              "c3": "stale Uzayr (his enrolment from day 2, tests on day 1)", "c4": "across days (day 1 -> day 2)"}
SELECTION_RULE = (
    "Fixed before running. The candidate with the highest mean on condition 1 (K = 3, no calibration) wins. It "
    "replaces M0 only if it beats M0 by at least 3 points on condition 1 AND does not lose to M0 on condition 2. "
    "If two candidates are within 1 point, the simpler one is taken (M0 < M1 < M2). A calibration is adopted on "
    "top of the winner only if it raises the mean of conditions 2 and 3 by at least 1 point without costing more "
    "than 1 point on condition 1; for K, the smallest K within 1 point of the best. tau is re-fitted from the "
    "leave-two-out decisions so that at most about 15 % of condition-1 walks are not sure.")

# The code files the shipped model's manifest dates (tests/test_models.py checks that each one exists).
CODE_FILES = ("chain.py", "walks.py", "models.py", "evaluate.py")


# -- the six people's walks ------------------------------------------------------------------------

def collect_pair_rows(cfg: dict, log=print) -> list[dict]:
    """Every unflagged RX2 pass of the six people through the causal chain: readouts and gait set."""
    key = (cfg["dsp"]["variant"], int(cfg["dsp"]["nfft"]))
    rows = []
    for name in openday.PEOPLE:
        for day in (1, 2):
            bases = baselines_for(cfg, name, day, [key], 75)
            for block in BLOCKS:
                for p in recordings.passes(name, day, block):
                    r = run_pass(cfg, name, day, block, p, [key], bases, 75, gait_key=key)
                    w = walk_of(r[key], cfg)
                    if w["f"] is None:
                        continue
                    rows.append({"name": name, "day": day, "block": block, "pass_id": p["pass_id"], "dir": r["dir"],
                                 "gt": r["gt"], "f": {**w["f"], **r["gait"]}})
        log(f"  {name}: {len(rows)} walks so far")
    return rows


def load_rows(cfg: dict, log=print, refresh: bool = False) -> list[dict]:
    """The six people's walks through the chain, from the laptop-only cache when it holds exactly P1-P6 (an older
    cache filed by first name is rebuilt, which needs ../Code)."""
    path = os.path.join(openday.STATE, "cache", "pair_rows.json")      # thesis walks only, never a visitor's
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        if {r["name"] for r in rows} == set(openday.PEOPLE):
            return rows
    rows = collect_pair_rows(cfg, log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, default=float)
    return rows


class Data:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.X = np.array([MD.vec(r["f"], MD.ALL) for r in rows])
        self.X0 = np.array([MD.vec(r["f"], MD.READOUTS) for r in rows])
        self.name = np.array([r["name"] for r in rows])
        self.rec = np.array([f"{r['name']}|{r['day']}|{r['block']}" for r in rows])
        self.S = np.array([r["dir"] == "S" for r in rows])
        self.idx: dict = {}
        for i, r in enumerate(rows):
            if r["dir"] == "S":
                self.idx.setdefault((r["name"], r["day"], r["block"]), []).append(i)

    def S_of(self, name: str, day: int, blocks) -> list[int]:
        return [i for b in blocks for i in self.idx.get((name, day, b), [])]


def train_indices(D: Data, test_people, use_N: bool = False) -> np.ndarray:
    keep = ~np.isin(D.name, list(test_people)) & (D.S | use_N)
    idx = np.flatnonzero(keep)
    assert not set(D.name[idx]) & set(test_people), "a test person is in the training set"
    return idx


def fit_fold(D: Data, test_people, use_N: bool = False) -> dict:
    tr = train_indices(D, test_people, use_N)
    nat1 = tr[[D.rows[i]["day"] == 1 and D.rows[i]["block"] == "natural" and D.S[i] for i in tr]]
    out = {"m0": fit_m0(D.X0[nat1], D.name[nat1])}
    for fs, names in FEATURE_SETS.items():
        X = D.X[:, [MD.ALL.index(n) for n in names]][tr]
        out[f"m1_{fs}"] = fit_m1(X, D.name[tr], names)
        out[f"m1lda_{fs}"] = fit_m1(X, D.name[tr], names, keep=3)
        out[f"m2_{fs}"] = fit_m2(X, D.name[tr], D.rec[tr], names)
    out["speed"] = MD.Model("m0", ["speed"], np.zeros(1), np.ones(1), 0.08, {"speed": out["m0"].prior_sd["speed"]})
    out["_train_people"] = sorted(set(D.name[tr]))
    return out


# -- draws -----------------------------------------------------------------------------------------

def draw(D: Data, cond: str, A: str, B: str, rng: random.Random, K: int = 3, fresh: bool = False):
    """(enrol A, enrol B, tests [(index, true side)]) for one random draw of a condition."""
    def split(ix, n):
        ix = list(ix)
        rng.shuffle(ix)
        return ix[:n], ix[n:]
    if cond == "c1":
        ea, ta = split(D.S_of(A, 1, ["natural"]), K)
        eb, tb = split(D.S_of(B, 1, ["natural"]), 3)
    elif cond == "c2":
        ea, _ = split(D.S_of(A, 2, ["natural"]), K)
        eb, _ = split(D.S_of(B, 2, ["natural"]), 3)
        ta, tb = D.S_of(A, 2, ["slow", "fast"]), D.S_of(B, 2, ["slow", "fast"])
    elif cond == "c3":
        if fresh:                                  # C3: Uzayr's walks interleaved with the visitor's, same minutes
            ea, ta = split(D.S_of(A, 1, ["natural"]), 3)
        else:
            ea, _ = split(D.S_of(A, 2, ["natural", "slow", "fast"] if K > 5 else ["natural"]), K)
            ta = D.S_of(A, 1, ["natural"])
        eb, tb = split(D.S_of(B, 1, ["natural"]), 3)
    elif cond == "c4":
        ea, _ = split(D.S_of(A, 1, ["natural"]), K)
        eb, _ = split(D.S_of(B, 1, ["natural"]), 3)
        ta, tb = D.S_of(A, 2, ["natural"]), D.S_of(B, 2, ["natural"])
    else:
        raise ValueError(cond)
    return ea, eb, [(i, "uzayr") for i in ta] + [(i, "visitor") for i in tb]


def decide_all(D: Data, m: MD.Model, ea, eb, tests) -> tuple[np.ndarray, np.ndarray]:
    """(correct, margin) per test walk."""
    X = D.X0 if m.kind == "m0" and m.names == MD.READOUTS else (D.X[:, [MD.ALL.index(n) for n in m.names]]
                                                                if m.names != MD.ALL else D.X)
    enrol = {"uzayr": X[ea], "visitor": X[eb]}
    ok, mg = [], []
    for i, truth in tests:
        dist, margin = MD.scores(m, X[i], enrol)
        ok.append(min(dist, key=dist.get) == truth)
        mg.append(margin)
    return np.array(ok, bool), np.array(mg, float)


def in_sample_tau(D: Data, m: MD.Model, people, rng: random.Random, max_ns: float = 0.15, draws: int = 8) -> float:
    """tau from the training people only (their condition-1 margins, in sample): about max_ns not sure."""
    mg = []
    for A, B in itertools.permutations(people, 2):
        for _ in range(draws):
            ea, eb, tests = draw(D, "c1", A, B, rng)
            mg.append(decide_all(D, m, ea, eb, tests)[1])
    return float(np.quantile(np.concatenate(mg), max_ns))


# -- the run ---------------------------------------------------------------------------------------

def run_configs(D: Data, configs: list[dict], conds, draws: int, seed: int = 0, use_N: bool = False,
                log=print) -> dict:
    """configs: {"label", "model" (fold key), "plda", "session_centre", "K", "fresh"}. Per config and condition:
    per unordered pair the mean forced accuracy, with-tau accuracy and not-sure share; and every (margin,
    correct) for tau."""
    res = {c["label"]: {k: {"pair": {}, "margins": [], "correct": []} for k in conds} for c in configs}
    for A, B in itertools.combinations(openday.PEOPLE, 2):
        folds = fit_fold(D, (A, B), use_N)
        assert A not in folds["_train_people"] and B not in folds["_train_people"]
        others = [p for p in openday.PEOPLE if p not in (A, B)]
        taus = {}
        for c in configs:
            m = copy.copy(folds[c["model"]])
            m.plda, m.session_centre = bool(c.get("plda")), bool(c.get("session_centre"))
            key = (c["model"], m.plda, m.session_centre)
            if key not in taus:
                taus[key] = in_sample_tau(D, m, others, random.Random(seed + 7))
            m.tau = taus[key]
            for cond in conds:
                acc, acc_t, ns = [], [], []
                for u, v in ((A, B), (B, A)):
                    rng = random.Random(f"{seed}|{u}|{v}|{cond}")       # every config sees the same draws
                    for _ in range(draws):
                        ea, eb, tests = draw(D, cond, u, v, rng, K=int(c.get("K", 3)), fresh=bool(c.get("fresh")))
                        ok, mg = decide_all(D, m, ea, eb, tests)
                        sure = mg >= m.tau
                        acc.append(ok.mean())
                        acc_t.append((ok & sure).mean())
                        ns.append((~sure).mean())
                        r = res[c["label"]][cond]
                        r["margins"].append(mg)
                        r["correct"].append(ok)
                res[c["label"]][cond]["pair"][f"{A}-{B}"] = (float(np.mean(acc)), float(np.mean(acc_t)), float(np.mean(ns)))
        log(f"  pair {A}-{B} done")
    for c in res.values():
        for r in c.values():
            r["margins"] = np.concatenate(r["margins"]) if r["margins"] else np.zeros(0)
            r["correct"] = np.concatenate(r["correct"]) if r["correct"] else np.zeros(0, bool)
    return res


def summary(r: dict) -> dict:
    a = np.array([v[0] for v in r["pair"].values()])
    at = np.array([v[1] for v in r["pair"].values()])
    ns = np.array([v[2] for v in r["pair"].values()])
    n = len(a)
    half = 2.145 * a.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")       # t(14), 97.5 %
    return {"mean": 100 * a.mean(), "sd": 100 * a.std(ddof=1), "lo": 100 * (a.mean() - half),
            "hi": 100 * (a.mean() + half), "min": 100 * a.min(), "max": 100 * a.max(),
            "tau_acc": 100 * at.mean(), "not_sure": 100 * ns.mean()}


# -- ship + report ---------------------------------------------------------------------------------

CANDIDATES = ["m0"] + [f"{m}_{fs}" for fs in ("all", "core") for m in ("m1", "m1lda", "m2")]
COMPLEXITY = {"m0": 0, "m1": 1, "m1lda": 1, "m2": 2}
CALIBRATIONS = [  # (label, config on top of the winner, conditions it can be measured on)
    ("none, K = 3", {}, ("c1", "c2", "c3", "c4")),
    ("C1 session centring", {"session_centre": True}, ("c1", "c2", "c3", "c4")),
    ("C2 PLDA shrink, K = 3", {"plda": True}, ("c1", "c2", "c3", "c4")),
    ("C2 PLDA shrink, K = 5", {"plda": True, "K": 5}, ("c1", "c2", "c3", "c4")),
    ("C2 PLDA shrink, K = 10", {"plda": True, "K": 10}, ("c3", "c4")),
    ("K = 5 Uzayr walks, no shrink", {"K": 5}, ("c1", "c2", "c3", "c4")),
    ("K = 10 Uzayr walks, no shrink", {"K": 10}, ("c3", "c4")),
    ("C1 + C2, K = 3", {"session_centre": True, "plda": True}, ("c1", "c2", "c3", "c4")),
    ("C3 interleaved fresh enrolment", {"fresh": True}, ("c3",)),
    ("C3 + C2 PLDA shrink", {"fresh": True, "plda": True}, ("c3",)),
]


def _row(label: str, s: dict, conds) -> str:
    return f"| {label} | " + " | ".join(
        f"{s[c]['mean']:.1f} ± {s[c]['sd']:.1f} [{s[c]['lo']:.1f}, {s[c]['hi']:.1f}]" if c in s else "n/a"
        for c in conds) + " |"


def fit_shipped(D: Data, kind_key: str, tau: float) -> MD.Model:
    """The chosen model fitted on all six people's S walks (thesis data only)."""
    tr = np.flatnonzero(D.S)
    base, fs = kind_key.split("_") if "_" in kind_key else (kind_key, None)
    if base == "m0":
        nat1 = tr[[D.rows[i]["day"] == 1 and D.rows[i]["block"] == "natural" for i in tr]]
        m = fit_m0(D.X0[nat1], D.name[nat1])
    else:
        names = FEATURE_SETS[fs]
        X = D.X[:, [MD.ALL.index(n) for n in names]][tr]
        m = {"m1": lambda: fit_m1(X, D.name[tr], names), "m1lda": lambda: fit_m1(X, D.name[tr], names, keep=3),
             "m2": lambda: fit_m2(X, D.name[tr], D.rec[tr], names)}[base]()
    m.tau = float(tau)
    return m


def run_pair(cfg: dict, draws: int = 50, write: bool = True, log=print) -> str:
    """eval-pair: every candidate model left two people out, the choice by SELECTION_RULE, its calibration and
    tau; returns the report. With write, also saves the report, the shipped model, its manifest and the config's
    classifier lines."""
    t_start = time.process_time()
    day = time.strftime("%Y%m%d")
    log("Collecting the thesis walks through the causal chain (cached in state/cache) ...")
    D = Data(load_rows(cfg, log))
    conds = ("c1", "c2", "c3", "c4")
    log("Models, leave two people out ...")
    res = run_configs(D, [{"label": k, "model": k} for k in ["speed"] + CANDIDATES], conds, draws, log=lambda *a: None)
    S = {k: {c: summary(res[k][c]) for c in conds} for k in res}
    m0 = S["m0"]
    best = max(CANDIDATES, key=lambda k: S[k]["c1"]["mean"])
    near = [k for k in CANDIDATES if S[best]["c1"]["mean"] - S[k]["c1"]["mean"] <= 1.0]
    best = min(near, key=lambda k: (COMPLEXITY[k.split("_")[0]], -S[k]["c1"]["mean"]))
    wins = best != "m0" and S[best]["c1"]["mean"] - m0["c1"]["mean"] >= 3.0 and S[best]["c2"]["mean"] >= m0["c2"]["mean"]
    chosen = best if wins else "m0"
    log(f"Chosen model: {chosen}. Calibrations ...")
    cal = {}
    for label, extra, cs in CALIBRATIONS:
        r = run_configs(D, [{"label": label, "model": chosen, **extra}], list(cs), draws, log=lambda *a: None)
        cal[label] = {c: summary(r[label][c]) for c in cs}
    ref = cal["none, K = 3"]
    adopted, gain = "none", 0.0
    for label, extra, cs in CALIBRATIONS[1:]:
        s = cal[label]
        c23 = [s.get(c, ref[c])["mean"] for c in ("c2", "c3")]
        g = float(np.mean(c23)) - float(np.mean([ref["c2"]["mean"], ref["c3"]["mean"]]))
        cost = ref["c1"]["mean"] - s.get("c1", ref["c1"])["mean"]
        if g >= 1.0 and cost <= 1.0 and g > gain:
            adopted, gain = label, g
    calib = {"none, K = 3": "none"}.get(adopted, adopted)
    extra = dict(next(e for lab, e, _ in CALIBRATIONS if lab == adopted)) if adopted != "none" else {}
    log(f"Calibration: {calib}. tau and S vs S+N ...")
    final = {"label": "final", "model": chosen, **{k: v for k, v in extra.items() if k != "fresh"}}
    r = run_configs(D, [final], ["c1", "c2", "c4"], draws, log=lambda *a: None)["final"]
    r3 = run_configs(D, [{**final, "fresh": bool(extra.get("fresh"))}], ["c3"], draws, log=lambda *a: None)["final"]
    r = {**r, **r3}
    tau = float(np.quantile(r["c1"]["margins"], 0.15))
    tau_rows = []
    for c in conds:
        mg, ok = r[c]["margins"], r[c]["correct"]
        sure = mg >= tau
        tau_rows.append((c, 100 * ok.mean(), 100 * (ok & sure).mean(), 100 * (~sure).mean(),
                         100 * ok[sure].mean() if sure.any() else float("nan")))
    rn = run_configs(D, [final], ["c1", "c2"], draws, use_N=True, log=lambda *a: None)["final"]
    sn = {c: summary(rn[c]) for c in ("c1", "c2")}
    shipped = fit_shipped(D, chosen, tau)
    shipped.cal_m, shipped.cal_p = calibrate(r["c1"]["margins"], r["c1"]["correct"])
    shipped.plda = bool(extra.get("plda"))
    shipped.session_centre = bool(extra.get("session_centre"))
    cpu = time.process_time() - t_start
    head = {c: summary(r[c]) for c in conds}
    lead1 = head["c1"]["mean"] - S["speed"]["c1"]["mean"]
    lead2 = head["c2"]["mean"] - S["speed"]["c2"]["mean"]
    nice = {"m0": "M0 nearest centroid on the four readouts", "m1": "M1 learned whitening", "m1lda": "M1 learned whitening, 3 directions",
            "m2": "M2 pairwise same/different"}
    kind_txt = nice[chosen.split("_")[0]] + (f" on the '{chosen.split('_')[1]}' features ({len(shipped.names)})" if "_" in chosen else "")
    headline = (f"**Chosen: {kind_txt}, calibration {calib} (K = {int(extra.get('K', 3))}). Condition 1: {head['c1']['mean']:.1f} %, "
                f"condition 2: {head['c2']['mean']:.1f} % (speed alone {S['speed']['c1']['mean']:.1f} % and {S['speed']['c2']['mean']:.1f} %: "
                f"a lead of {lead1:+.1f} and {lead2:+.1f} points). "
                + (f"It beat M0 ({m0['c1']['mean']:.1f} % and {m0['c2']['mean']:.1f} %).**" if wins else "Nothing beat M0 by the rule's margin, so M0 ships.**"))
    L = [f"# Two new people: leave-two-out evaluation, {time.strftime('%d %B %Y')}", "", headline, "",
         f"**Selection rule.** {SELECTION_RULE}", "",
         f"Data: Room B identification walks, RX2 only (0.5 m beside the TX), the six participants {', '.join(openday.PEOPLE)}; "
         f"{len(D.rows)} walks the chain saw ({int(D.S.sum())} towards the boards, S). Tests are S walks only. Every walk went "
         f"through the app's own causal chain (chain.py, walks.py), as live. For each of the 15 pairs, "
         f"every learned part (standardisation, whitening, pairwise model, prior_sd, tau) was fitted on the other four "
         f"people only; each person played Uzayr once and the visitor once, {draws} random draws each. Numbers are mean ± SD "
         f"over the 15 pairs [95 % interval of the mean over pairs; pairs share people, so it is approximate]. Chance is 50 %. "
         f"CPU {cpu:.0f} s.", "",
         "Conditions: " + "; ".join(f"**{c}** {t}" for c, t in CONDITIONS.items()) + ".", "",
         "## 1. Models (K = 3, no calibration)", "",
         "| model | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds),
         _row("chance", {c: {"mean": 50, "sd": 0, "lo": 50, "hi": 50} for c in conds}, conds),
         _row("speed only", S["speed"], conds)]
    for k in CANDIDATES:
        L.append(_row(("**" + k + "** (chosen)") if k == chosen else k, S[k], conds))
    L += ["", "m1 = learned whitening (within-person covariance shrunk 30 % towards its diagonal), all directions; "
          "m1lda = the 3 directions with the largest between/within ratio (four training people give at most 3); m2 = "
          "pairwise L2 logistic regression on |za - zb| and za * zb, same-person pairs only across recordings. "
          "'all' = the four readouts and the full single-link set (55 features); 'core' = the readouts and the 11 "
          "single-link features without the stride-cycle shape (c50/c95_cyc) and the Doppler profile (uprof) (15).", "",
          "Spread over pairs, chosen model, condition 1: lowest pair "
          f"{S[chosen]['c1']['min']:.1f} %, highest {S[chosen]['c1']['max']:.1f} %.", "",
          "**Honesty notes.** (1) The 'core' set was added after a 10-draw dry run showed the stride-cycle features are "
          "missing in about 4 walks in 10 (the step rate is found in 61 % of S walks with one link), so the grid of "
          f"{len(CANDIDATES)} candidates was fixed after one look at the data; choosing the best of them on condition 1 "
          "is mildly optimistic. (2) Condition 1 still has each person's walks in one recording block, so a feature can "
          "tell recordings apart; condition 2 breaks that. With interleaved enrolment (C3) both people are recorded in "
          "the same minutes, which removes the recording difference the block confound feeds on: expect the live number "
          "between conditions 2 and 1. (3) M3 (an embedding network) was not built: six people is very few, and M1 "
          "already passed the bar.", "",
          f"## 2. Day-of calibration on top of {chosen}", "",
          "| calibration | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for label, _e, _cs in CALIBRATIONS:
        L.append(_row(("**" + label + "** (adopted)") if label == adopted else label, cal[label], conds))
    L += ["", "C3 is measured on condition 3 only: in conditions 1, 2 and 4 Uzayr's walks already come from the "
          "same recording as his tests or the visitor's. K = 10 needs more walks than one recording has in "
          "conditions 1 and 2. C1 cannot change M1's nearest-centroid choice on its own (a shared shift moves both "
          "centroids); it matters only with C2's shrinkage.", "",
          f"## 3. Not sure: tau = {tau:.4f}", "",
          "Re-fitted from the leave-two-out decisions of the shipped configuration so that 15 % of condition-1 walks "
          "are not sure (the app still names the nearer person; a margin below tau is logged as a close call).", "",
          "| condition | forced | with tau | not sure | right when sure |", "|---|---|---|---|---|"]
    L += [f"| {c} | {a:.1f} % | {b:.1f} % | {n:.1f} % | {s:.1f} % |" for c, a, b, n, s in tau_rows]
    L += ["", "**How sure (on screen).** The share of condition-1 leave-two-out decisions with that margin that were "
          "right, in equal-count bins, made to rise with the margin (pooled adjacent violators):", "",
          "| margin | " + " | ".join(f"{x:.3f}" for x in shipped.cal_m) + " |",
          "|---|" + "---|" * len(shipped.cal_m),
          "| right | " + " | ".join(f"{100 * p:.0f} %" for p in shipped.cal_p) + " |"]
    L += ["", "## 4. Training on S only or S and N", "",
          f"S only: condition 1 {head['c1']['mean']:.1f} %, condition 2 {head['c2']['mean']:.1f} %. S and N: "
          f"{sn['c1']['mean']:.1f} % and {sn['c2']['mean']:.1f} %. "
          + ("N passes do not help on S tests, so the shipped model is trained on S only." if sn["c1"]["mean"] <= head["c1"]["mean"]
             else "N passes help on S tests (noted; the shipped model still uses S only, as fixed above)."), "",
          "## 5. Shipped", "",
          f"`state/model/model.json`: {chosen}, fitted on all six people's S walks ({int(D.S.sum())}), tau {tau:.4f}, "
          f"calibration {calib}. `state/model/manifest.json` lists the training files, the features and these numbers. "
          "No visitor data is in either."]
    text = "\n".join(L) + "\n"
    if write:
        os.makedirs(openday.REPORTS, exist_ok=True)
        rp = os.path.join(openday.REPORTS, f"eval_pair_{day}.md")
        with open(rp, "w", encoding="utf-8") as f:
            f.write(text)
        mdir = os.path.join(openday.STATE, "model")
        os.makedirs(mdir, exist_ok=True)
        shipped.meta = {"chosen": chosen, "calibration": calib, "K": int(extra.get("K", 3))}
        shipped.save(os.path.join(mdir, "model.json"))
        files = sorted({os.path.join("Code/data/room_B/identification", f"session_0{r['day']}",
                                     f"{recordings.stem(r['name'], r['day'], r['block'])}{suf}")
                        for r in D.rows if r["dir"] == "S" for suf in ("_RX2.bin", ".gt.json")})
        code = {f: time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(openday.OPENDAY, "openday", f))))
                for f in CODE_FILES}
        people = [openday.recording_name(p) for p in openday.PEOPLE]     # as the recordings are filed (laptop only)
        manifest = {"made": time.strftime("%Y-%m-%d %H:%M"), "kind": chosen, "calibration": calib,
                    "K": int(extra.get("K", 3)), "tau": tau, "features": shipped.names,
                    "trained_on": {"people": people, "direction": "S", "receiver": "RX2", "walks": int(D.S.sum()),
                                   "files": files},
                    "code_dates": code, "results": {"models": S, "calibrations": cal, "final": head,
                                                    "tau_table": tau_rows, "report": os.path.basename(rp)},
                    "privacy": "thesis participants only; no visitor walk or name is ever written here"}
        with open(os.path.join(mdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, default=float)
        write_fitted(openday.CONFIG, {("classifier", "kind"): chosen.split("_")[0] if chosen != "m0" else "m0",
                                      ("classifier", "model_tau"): f"{tau:.4f}"})
        log(f"Report: {rp}\nModel: {mdir}")
    return text
