"""Home tests (bench sessions): the protocol, scoring a recorded session, and checking walk detection on it.

A bench session is recorded with `live --record DIR` (or run_solo_test.command): the raw CSI (bench_RX.bin) and the
app's events (events.jsonl) of each run. Everything here re-runs those recordings offline through the live chain.

This file, in order:
  1. The two-person bench session protocol (`bench-checklist`).
  2. A recorded run through the live chain again: bench_runs, bench_columns, bench_rounds (the pools each game walk
     was decided on), bench_detect and bench_walks.
  3. Scoring a session (`eval --bench`): run_bench writes reports/bench_<DIR>.md.
  4. Walk detection on recorded runs (`bench-check`): prints only, writes nothing.
"""

from __future__ import annotations

import collections
import copy
import glob
import json
import os
from typing import Callable, Optional

import numpy as np

import openday
from openday import chain, evaluate as E, models as MD, walks

# --- 1. The two-person bench session protocol (bench-checklist) ---

BENCH_CHECKLIST = """\
TWO-PERSON BENCH SESSION (Uzayr plus a friend). The only way to measure how well the app tells two people
apart in a real room. Type the commands in the VS Code terminal, in the OpenDay folder.

 1. Set up as in the README: TX and RX2 on the workbench beside the laptop, a taped X about 30 cm in front of
    the boards, a back mark at least 3 m behind the X. Work the app with the wireless mouse from the back mark:
    the big button at the bottom right never moves, so leave the pointer on it.
 2. Start recording (red RECORDING box at the top; the Mac stays awake):
      ./run_openday.command --record bench/{day}
    Stand on the back mark and click Measure room: 5 s to step back, then everyone keeps still for 30 s.
    Wait for "Who's walking?".
 3. Uzayr alone: 10 walks. For each: click Start walk on the back mark, walk to the X and stand still until
    the speed shows, click It was Uzayr, walk back. (Optional in the app, but only labelled walks are scored.)
 4. Quit (Quit, then Yes, quit). A break of at least 45 minutes.
 5. Start recording again with the same command (each run gets its own folder) and Measure room again.
    Click New player; the friend types their name, then OK. Walk as the screen says: the friend twice, then
    Uzayr twice. These need no other click: each walk is labelled by its turn (the wrong person walked? Undo
    walk, and the turn is walked again). When the game starts, keep alternating until EACH has walked 5 times,
    clicking It was Uzayr or It was <friend> after each game walk.
 6. Test: 20 walks. Before each, flip a coin (heads Uzayr, tails friend); that person walks. After the answer,
    click It was Uzayr or It was <friend> for who really walked. One click is final (both buttons go dim);
    a wrong click: Undo walk removes that walk.
 7. Optional: 4 walks each at a matched pace (both walk to a phone metronome at the same beat). Click It was ...
    for each.
 8. Quit, then score it:
      PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python3 -m openday eval --bench bench/{day}
    The report scores the shipped model (with 5 + 5 and with 3 + 3 learning walks; the app uses 2 + 2 on the
    day) beside the nearest-centre rule (m0) and speed alone.

Someone else walked through? Cancel walk during the walk, Undo walk after it.
Pass bar: the alternating pool (step 5) gets at least 18 of the 20 test walks right.
"""


# --- 2. A recorded run through the live chain again ---


def bench_runs(d: str) -> list[str]:
    """Each `live --record` run's folder under d, oldest first (folder names are timestamps)."""
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(d, "*", "bench_RX.bin")))


def bench_columns(cfg: dict, run_dir: str) -> dict:
    """One pass of the live chain over a recorded run (bench_RX.bin): every column's device time, power and gap,
    its epoch (receiver reboots so far) and silence count, the host-to-device clock per batch, and the events.
    A receiver reboot starts a new epoch of the device clock."""
    openday.use_code()
    from pipeline.io.Capture.capture_logger import read_capture_file
    fr = read_capture_file(os.path.join(run_dir, "bench_RX.bin"))[1]
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        ev = [json.loads(ln) for ln in f if ln.strip()]
    bc = {"fr": fr, "ev": ev, "batch": int(cfg["dsp"]["tick_batch"]) * 8, "allc": None, "ch": chain.Chain(cfg),
          "run": os.path.basename(os.path.normpath(run_dir))}
    if not len(fr):
        return bc
    ch = bc["ch"]
    cols, offs, epochs = [], [], []
    for i in range(0, len(fr), bc["batch"]):
        b = fr[i:i + bc["batch"]]
        c = ch.push(b["local_timestamp"], b["t_mono"], b["data"])
        offs.append((float(b["t_mono"][-1]), ch.clock.offset, ch.reboots))
        if len(c):
            cols.append(c)
            epochs.append(np.full(len(c), ch.reboots))
    if not cols:
        return bc
    t = np.concatenate([c.t for c in cols])
    bc.update(allc=chain.Columns(t, np.concatenate([c.power for c in cols]), np.concatenate([c.gap for c in cols])),
              ep=np.concatenate(epochs), offs=offs,
              off_t=np.array([o[0] for o in offs]))
    return bc


def bench_rounds(ev: list) -> tuple[list[dict], dict]:
    """The pools rebuilt from a recording's events, as the app held them. A learning walk joins its turn's pool
    from its own walk event (purpose enrol_*, seen, label u / v: in older recordings too); a label event moves a
    learning walk that was relabelled (older versions of the app), and Undo walk's discard removes one again. Each New
    player (new_visitor, by its anonymous index) starts the visitor's pool afresh, and Uzayr's too unless a
    uzayr_reused event follows: it names the walk ids of his earlier learning walks the round reuses. Redo my walks
    (uzayr_redo) starts a new set of his walks at any time; uzayr_kept (Undo walk right after it) brings the old set
    back; uzayr_dropped (Measure room while learning) empties his set. Returns (rounds, pools): per player {index,
    t_mono, reused (walk ids), redo (bool), games (game walk ids)}, and per game walk (purpose test) {"u": [walk
    ids], "v": [walk ids]}, the pools it was decided on. Recordings from before reuse (Uzayr walked with every
    player) rebuild the same way, with nothing reused."""
    rounds, pools = [], {}
    uz: list = []                                          # Uzayr's learning walks in use (kept across players)
    vis: list = []                                         # this player's
    cur: Optional[dict] = None
    redo_was = False                                       # the round's redo flag before the last uzayr_redo
    for e in ev:
        k = e.get("kind")
        if k == "new_visitor":
            cur = {"index": e.get("index"), "t_mono": e.get("t_mono"), "reused": [], "redo": False, "games": []}
            rounds.append(cur)
            uz, vis = [], []
        elif k == "uzayr_reused":
            uz = [int(w) for w in e.get("walk_ids") or []]
            if cur is not None:
                cur["reused"] = list(uz)
        elif k == "uzayr_redo":
            uz = []
            if cur is not None and e.get("index") is not None:
                redo_was, cur["redo"] = cur["redo"], True
        elif k == "uzayr_kept":                            # Redo my walks taken back: his old set again
            uz = [int(w) for w in e.get("walk_ids") or []]
            if cur is not None and e.get("index") is not None:
                cur["redo"] = redo_was
        elif k == "uzayr_dropped":                         # Measure room while learning: his set is taken again
            uz = []
            if cur is not None and not cur["games"]:
                cur["reused"] = []
        elif k == "walk" and str(e.get("purpose", "")).startswith("enrol") and e.get("seen") \
                and e.get("label") in ("u", "v"):           # a learning walk joins its turn's pool (its walk event
            (uz if e["label"] == "u" else vis).append(e["id"])   # carries the label: also in older recordings)
        elif k == "label" and e.get("label") in ("u", "v") and (e.get("source") == "turn"
                                                                 or e.get("walk_id") in uz + vis):
            w = e["walk_id"]                               # (a relabelled learning walk, older app) moves
            uz, vis = [x for x in uz if x != w], [x for x in vis if x != w]
            (uz if e["label"] == "u" else vis).append(w)
        elif k == "discard":
            uz = [w for w in uz if w != e.get("walk_id")]
            vis = [w for w in vis if w != e.get("walk_id")]
        elif k == "walk" and e.get("purpose") == "test":
            pools[e["id"]] = {"u": list(uz), "v": list(vis)}
            if cur is not None:
                cur["games"].append(e["id"])
    return rounds, pools


def _to_dev(bc: dict, tm: float) -> tuple[float, int]:
    """A host time on the device clock of the epoch it fell in."""
    o = bc["offs"][min(int(np.searchsorted(bc["off_t"], tm)), len(bc["offs"]) - 1)]
    return float(tm - o[1]), int(o[2])


PROBE_AFTER_WALK_S = 8.0     # bench-check's test clicks on the quiet room start this long after a walk ends


def bench_detect(cfg: dict, run_dir: str, bc: Optional[dict] = None,
                 probes: bool = False) -> tuple[list[dict], dict]:
    """Every recorded walk (click) of one run re-run offline through WalkWindow, with the room level the app had:
    the run's own Measure room (Ctrl+B) captures, the latest ready before the click in the same epoch (as live).

    One row per walk event, labelled or not, with the live and the offline end; a game walk's row carries the
    walk ids of the pools it was decided on ("pools", from bench_rounds; the rounds in info["rounds"]). With
    `probes`, synthetic false clicks every 1 s in each quiet span (from PROBE_AFTER_WALK_S after a walk's end to
    3 s before the next click; after the last walk to 3 s before the quit key or the end), each run through the
    same WalkWindow with the room level at that moment; a probe's columns stop at the next click. Returns
    (rows, info)."""
    bc = bc if bc is not None else bench_columns(cfg, run_dir)
    ev, allc = bc["ev"], bc["allc"]
    info = {"run": bc["run"], "e0": [], "probes": 0, "false": 0}
    info["rounds"], at = bench_rounds(ev)                  # (who walked when: reused and redone walks of Uzayr's)
    if allc is None:
        return [], info
    ch, ep, hop = bc["ch"], bc["ep"], chain.hop_s(cfg)
    k_act, max_s = float(cfg["dsp"]["active_k"]), float(cfg["walk"]["max_s"])
    keys = cfg["keys"]
    is_key = lambda key, a: key in keys.get(a, []) or (len(key) == 1 and key.lower() in keys.get(a, []))  # noqa: E731
    caps = []                                              # (ready host time, epoch, ready device time, Baseline)
    for e in ev:
        if e["kind"] == "baseline_start":
            t0, k0 = _to_dev(bc, e["t_mono"])
            m = (allc.t >= t0) & (allc.t <= t0 + float(e["secs"])) & (ep == k0)
            if m.sum() >= int(cfg["baseline"]["min_cols"]):
                b = chain.make_baseline(chain.Columns(allc.t[m], allc.power[m], allc.gap[m]), cfg, ch.freqs,
                                      ch.spec.variant, ch.spec.nfft)
                caps.append((e["t_mono"] + float(e["secs"]), k0, t0 + float(e["secs"]), b))
    labels, discarded, stops = {}, set(), {}
    for e in ev:
        if e["kind"] == "label":
            labels[e["walk_id"]] = e["label"]
        elif e["kind"] == "discard":
            discarded.add(e["walk_id"])
        elif e["kind"] == "stop":                          # the operator's "they've stopped" press
            stops.setdefault(e["walk_id"], []).append(e["t_mono"])

    def run_walk(base: chain.Baseline, t_dev: float, k: int, stop_ts=(), t_cut: float = np.inf) -> walks.WalkWindow:
        m = (ep == k) & (allc.t >= t_dev - 2.0) & (allc.t <= min(t_dev + max_s + 1.0, t_cut))
        c = chain.contours(chain.Columns(allc.t[m], allc.power[m], allc.gap[m]), base, cfg, ch.freqs)
        w = walks.WalkWindow(cfg, t_dev, k_act * base.e0, hop, e0=base.e0)
        for ts in stop_ts:
            w.request_stop(ts)
        w.add(c)
        return w

    # the timeline: (epoch, device time, order, kind, payload)
    acts = []
    walk_ids = {e["id"] for e in ev if e["kind"] == "walk"}
    last_next = None
    for e in ev:
        if e["kind"] == "key":
            key, st = str(e.get("key", "")), e.get("state")
            if is_key(key, "start_walk") and st not in ("WALKING", "NAME_ENTRY", "NEEDS_BASELINE", "BUSY"):
                last_next = e["t_mono"]
        elif e["kind"] == "walk":
            k_t = _to_dev(bc, e["t_click"])
            acts.append((k_t[1], k_t[0], 2, "walk", e))
        elif e["kind"] == "discard" and e["walk_id"] not in walk_ids and last_next is not None:
            c0, k0 = _to_dev(bc, last_next)                # a walk cancelled while walking: no walk event
            acts.append((k0, c0, 2, "cancel", _to_dev(bc, e["t_mono"])[0]))
            last_next = None
    acts.sort(key=lambda a: a[:3])
    rows, spans = [], []
    for k, t, _o, kind, x in acts:
        if kind == "cancel":
            spans.append((k, t, x))
            continue
        e = x
        usable = [b for t_ready, kb, _td, b in caps if t_ready <= e["t_click"] and kb == k]
        base = usable[-1] if usable else None
        w = run_walk(base, t, k, [_to_dev(bc, ts)[0] for ts in stops.get(e["id"], [])]) if base is not None else None
        t_end = w.t_end_col if w is not None and w.t_end_col is not None else t + max_s
        spans.append((k, t, t_end))
        f = walks.walk_features(w.arrays(), hop, cfg) if w is not None and w.seen else None
        rows.append({"id": e["id"], "t": e["t_click"], "label": labels.get(e["id"]),
                     "purpose": e.get("purpose"), "pools": at.get(e["id"]),
                     "discarded": e["id"] in discarded, "live_seen": bool(e.get("seen")),
                     "live_reason": e.get("reason"), "live_t_end": e.get("t_end"), "live_f": e.get("f"),
                     "e0": base.e0 if base is not None else None,
                     "seen": f is not None, "reason": (w.reason or "open") if w is not None else None,
                     "active_s": w.active_s if w is not None else 0.0,
                     "t_end": (w.t_end_col - t) if w is not None and w.t_end_col is not None else float("nan"),
                     "t_arrive": float(w.t_arrive - t) if w is not None and w.t_arrive is not None else float("nan"),
                     "f": f, "_win": (t, float(w.t_arrive), k) if f is not None else None})
    info["e0"] = [b.e0 for *_x, b in caps]
    if probes:
        quits = [_to_dev(bc, e["t_mono"]) for e in ev if e["kind"] == "key" and is_key(str(e.get("key", "")), "quit")]
        spans.sort()
        for j, (k, t_click, t_end) in enumerate(spans):
            if j + 1 < len(spans) and spans[j + 1][0] == k:
                cut = spans[j + 1][1]
            else:
                cut = float(allc.t[ep == k][-1])
                cut = min([cut] + [tq for tq, kq in quits if kq == k and tq > t_end])
            for tp in np.arange(t_end + PROBE_AFTER_WALK_S, cut - 3.0 + 1e-9, 1.0):
                usable = [b for _r, kb, td, b in caps if kb == k and td <= tp]
                base = usable[-1] if usable else None
                if base is None:
                    continue
                w = run_walk(base, float(tp), k, (), cut)
                info["probes"] += 1
                if w.seen:
                    info["false"] += 1
    return rows, info


def bench_walks(cfg: dict, run_dir: str) -> tuple[list[dict], dict]:
    """Every labelled, undiscarded walk of one recorded run, its features recomputed from the raw CSI through the
    live chain (the run's own room level, see bench_detect; click from events.jsonl), and the skipped counts.
    A walk uses only its own epoch's columns and a level from that epoch. A game walk carries "pools": the walk
    ids each person's pool held when it was decided (Uzayr's reused or redone walks included; bench_rounds)."""
    skipped = {"discarded": 0, "no label": 0, "no baseline": 0, "not seen": 0}
    bc = bench_columns(cfg, run_dir)
    if bc["allc"] is None:
        return [], skipped
    rows, _info = bench_detect(cfg, run_dir, bc=bc)
    out = []
    for r in rows:
        if not r["live_seen"] or r["discarded"]:
            skipped["discarded" if r["discarded"] else "not seen"] += 1
            continue
        if r["label"] not in ("u", "v") or r["e0"] is None:  # a label (an It was ... click, or a learning walk's
                                                             # own turn) is the ground truth
            skipped["no label" if r["label"] not in ("u", "v") else "no baseline"] += 1
            continue
        if not r["seen"]:
            skipped["not seen"] += 1
            continue
        f = r["f"]
        out.append({"id": r["id"], "t": r["t"], "label": r["label"], "f": f, "live_f": r["live_f"],
                    "x": np.array([f[n] for n in cfg["features"]["names"]], float), "run": bc["run"],
                    "reason": r["reason"], "live_reason": r["live_reason"], "active_s": r["active_s"],
                    "t_end": r["t_end"], "t_arrive": r["t_arrive"], "_win": r["_win"],
                    "purpose": r["purpose"], "pools": r["pools"]})
    _bench_gait(cfg, bc["fr"], out, bc["batch"])
    return out, skipped


def _bench_gait(cfg: dict, fr: np.ndarray, rows: list, batch: int) -> None:
    """The gait set of each walk, from a second pass of the same chain (the raw ring only holds seconds, so the
    windows are cut as the stream passes them)."""
    todo = sorted(rows, key=lambda w: w["_win"][1])
    ch, i = chain.Chain(cfg), 0
    for j in range(0, len(fr), batch):
        b = fr[j:j + batch]
        ch.push(b["local_timestamp"], b["t_mono"], b["data"])
        rt, ra = ch.raw.window(-np.inf, np.inf)
        while i < len(todo) and len(rt) and rt[-1] >= todo[i]["_win"][1]:
            t0, t1, k = todo[i]["_win"]
            if ch.reboots == k:
                todo[i]["f"].update(walks.walk_gait(rt, ra, t0, t1, cfg))
            i += 1
    for w in rows:
        w.pop("_win", None)


# --- 3. Scoring a session (eval --bench) ---


def _score(cfg: dict, pools: dict, tests: list, use, tau: Optional[float] = None, model=None) -> dict:
    """With `model` (models.Model) the decisions are that model's, on the full feature dicts; else m0's."""
    names = list(cfg["features"]["names"])
    idx = [names.index(n) for n in use]
    P = {k: np.array([w["x"][idx] for w in v]) for k, v in pools.items()}
    right = forced = speed = 0
    margins, correct = [], []
    for w in tests:
        if model is not None:
            d = MD.decide(model, w["f"], {k: [x["f"] for x in v] for k, v in pools.items()})
        else:
            d = MD.decide_m0(w["x"][idx], P, use, cfg, tau=tau)
        truth = "uzayr" if w["label"] == "u" else "visitor"
        right += d.label == truth and not d.not_sure
        forced += d.label == truth
        speed += d.speed_only is not None and d.speed_only.label == truth
        margins.append(d.margin)
        correct.append(d.label == truth)
    return {"right": right, "forced": forced, "speed_only": speed, "n": len(tests),
            "margins": np.array(margins), "correct": np.array(correct, bool)}


def run_bench(cfg: dict, d: str, log=print) -> str:
    """Score a two-person bench session (bench-checklist): stale and alternating pools, enrolment sizes, speed
    overlap, a re-fitted tau. One line at the top says whether the alternating pool reached the pass bar. The walks
    are recomputed with the runs' own room measurements (bench_detect)."""
    openday.ensure_dirs()
    runs = bench_runs(d)
    if not runs:
        raise SystemExit(f"No recorded runs (*/bench_RX.bin) under {d}")
    names = list(cfg["features"]["names"])
    e = cfg["eval"]
    n_en, n_te = int(e["bench_enrol"]), int(e["bench_tests"])
    openday.use_code()
    from pipeline.io.Capture.capture_logger import read_capture_file
    per_run, spans, has_visitor = [], [], []
    for r in runs:
        ws, sk = bench_walks(cfg, r)
        per_run.append((r, ws))
        log(f"  {os.path.basename(r)}: {len(ws)} labelled walks ({sum(w['label'] == 'u' for w in ws)} U, "
            f"{sum(w['label'] == 'v' for w in ws)} V); skipped "
            + (", ".join(f"{v} {k}" for k, v in sk.items() if v) or "none"))
        _h, fr = read_capture_file(os.path.join(r, "bench_RX.bin"))
        spans.append((float(fr["t_host"][0]), float(fr["t_host"][-1])) if len(fr) else (np.nan, np.nan))
        with open(os.path.join(r, "events.jsonl"), encoding="utf-8") as f:
            evs = [json.loads(x) for x in f if x.strip()]
        has_visitor.append(any(x["kind"] == "new_visitor" or (x["kind"] == "label" and x.get("label") == "v")
                               for x in evs))
    # the break, from what the runs contain: the first run with a visitor (N or a V label) starts the session
    # after the break; the runs before it hold the stored pool. The gap itself is reported from the capture
    # files' own host clock (first and last frame), and checked against bench_break_min.
    cut = has_visitor.index(True) if True in has_visitor else len(runs)
    gap_min = (spans[cut][0] - spans[cut - 1][1]) / 60.0 if 0 < cut < len(runs) else float("nan")
    pre = sorted([w for _, ws in per_run[:cut] for w in ws], key=lambda w: w["t"])
    post = sorted([w for _, ws in per_run[cut:] for w in ws], key=lambda w: (w["run"], w["t"]))
    eu = [w for w in post if w["label"] == "u"][:n_en]
    ev_ = [w for w in post if w["label"] == "v"][:n_en]
    end = max((post.index(w) for w in eu + ev_), default=-1)
    tests = post[end + 1:end + 1 + n_te]
    matched = post[end + 1 + n_te:]
    stale = [w for w in pre if w["label"] == "u"][-n_en:]
    bar = int(cfg["eval"]["bench_pass_bar"])
    L = []
    if not (eu and ev_ and tests):
        raise SystemExit("Not enough labelled walks after the break for the alternating pool and the test walks")
    b = _score(cfg, {"uzayr": eu, "visitor": ev_}, tests, names)
    verdict = "PASS" if b["right"] >= bar and b["n"] >= n_te else "FAIL"
    L.append(f"**{verdict}: alternating pool (b) got {b['right']} of {b['n']} test walks right "
             f"(bar {bar} of {n_te}; not sure counts as a miss).**")
    L.append("")
    L.append(f"# Bench session {os.path.basename(os.path.normpath(d))}")
    L.append("")
    L.append(f"Walk ends: as shipped (arrived, stopped, operator, nothing, {cfg['walk']['max_s']:g} s limit).")
    L.append("")
    L.append(f"Runs: {', '.join(os.path.basename(r) for r, _ in per_run)}. "
             + (f"Stored pool from {', '.join(os.path.basename(r) for r in runs[:cut])}; break {gap_min:.0f} min"
                + (f" (SHORTER than the {float(e['bench_break_min']):.0f} min the protocol asks for)"
                   if gap_min < float(e["bench_break_min"]) else "") + ". " if cut else
                "No run before the visitor's, so no stored pool. ")
             + f"Enrolment (after the break): first {n_en} U and first {n_en} V walks; tests: the {len(tests)} walks "
             f"after them; matched-pace walks after that: {len(matched)}. Only labelled walks count (learning walks by "
             "their turn, game walks by an It was ... click).")
    L.append("")
    L.append("| pool | right (with tau) | forced | speed only | of |")
    L.append("|---|---|---|---|---|")
    if stale:
        a = _score(cfg, {"uzayr": stale, "visitor": ev_}, tests, names)
        L.append(f"| (a) stale: Uzayr's last 5 before the break + friend's 5 | {a['right']} | {a['forced']} | "
                 f"{a['speed_only']} | {a['n']} |")
    else:
        L.append("| (a) stale | - | - | - | no U walks before the break |")
    L.append(f"| (b) alternating 5 + 5 | {b['right']} | {b['forced']} | {b['speed_only']} | {b['n']} |")
    model = MD.load_model(cfg, log=lambda _msg: None)        # (no log line: the report names the model)
    if model is not None:
        bm = _score(cfg, {"uzayr": eu, "visitor": ev_}, tests, names, model=model)
        b3 = _score(cfg, {"uzayr": eu[:3], "visitor": ev_[:3]}, tests, names, model=model)
        b2 = _score(cfg, {"uzayr": eu[:2], "visitor": ev_[:2]}, tests, names, model=model)
        L.append(f"| (b) alternating 5 + 5, shipped model {model.kind} | {bm['right']} | {bm['forced']} | "
                 f"{bm['speed_only']} | {bm['n']} |")
        L.append(f"| (b) 3 + 3 (the older C3 round), shipped model {model.kind} | {b3['right']} | "
                 f"{b3['forced']} | {b3['speed_only']} | {b3['n']} |")
        L.append(f"| (b) 2 + 2 (the app's round), shipped model {model.kind} | {b2['right']} | "
                 f"{b2['forced']} | {b2['speed_only']} | {b2['n']} |")
        if stale:
            am = _score(cfg, {"uzayr": stale, "visitor": ev_}, tests, names, model=model)
            L.append(f"| (a) stale, shipped model {model.kind} | {am['right']} | {am['forced']} | {am['speed_only']} | "
                     f"{am['n']} |")
    L.append("")
    L.append("(c) Enrolment size (first n of each from the alternating pool):")
    L.append("")
    L.append("| walks each | right (with tau) | forced | speed only |")
    L.append("|---|---|---|---|")
    for n in range(1, n_en + 1):
        if len(eu) >= n and len(ev_) >= n:
            c = _score(cfg, {"uzayr": eu[:n], "visitor": ev_[:n]}, tests, names)
            L.append(f"| {n} | {c['right']} / {c['n']} | {c['forced']} | {c['speed_only']} |")
    L.append("")
    L.append("Each feature left out once (alternating pool, forced):")
    L.append("")
    for n in names:
        c = _score(cfg, {"uzayr": eu, "visitor": ev_}, tests, [m for m in names if m != n])
        L.append(f"- without {n}: {c['forced']} / {c['n']}")
    L.append("")
    allw = [w for _, ws in per_run for w in ws]
    sp = {k: np.array([w["f"]["speed"] for w in allw if w["label"] == k]) for k in ("u", "v")}
    L.append("(d) Do the two walkers' speeds overlap?")
    L.append("")
    for k, who in (("u", "Uzayr"), ("v", "friend")):
        if len(sp[k]):
            L.append(f"- {who}: {sp[k].mean():.2f} ± {sp[k].std():.2f} m/s over {len(sp[k])} walks "
                     f"(range {sp[k].min():.2f} to {sp[k].max():.2f})")
    if len(sp["u"]) > 1 and len(sp["v"]) > 1:
        pooled = np.sqrt((sp["u"].var(ddof=1) + sp["v"].var(ddof=1)) / 2)
        L.append(f"- separation: {abs(sp['u'].mean() - sp['v'].mean()) / max(pooled, 1e-9):.2f} pooled SDs; "
                 f"speed alone gets {b['speed_only']} of {b['n']} with the alternating pool, four features "
                 f"{b['forced']} (forced). If those match, the demo is reading pace.")
    if matched:
        m = _score(cfg, {"uzayr": eu, "visitor": ev_}, matched, names)
        L.append(f"- matched pace walks: four features {m['forced']} of {m['n']}, speed alone {m['speed_only']}.")
    L.append("")
    tau, curve = E.tau_curve(b["margins"], b["correct"], float(cfg["eval"]["not_sure_max"]))
    L.append(f"(e) Re-fitted tau for this room (alternating pool's test walks, at most "
             f"{100 * float(cfg['eval']['not_sure_max']):.0f} % not sure): **{tau:.3f}** (config now "
             f"{cfg['classifier']['tau']}). To use it, set classifier.tau in configs/openday.yaml.")
    L.append("")
    diffs = [abs(w["f"][n] - w["live_f"][n]) for w in allw if w.get("live_f") for n in names
             if np.isfinite(w["f"][n]) and np.isfinite(w["live_f"].get(n, np.nan))]
    if diffs:
        L.append(f"Offline equals live: largest difference between the recomputed and the live features "
                 f"{max(diffs):.3g} over {len(diffs)} values.")
    out = os.path.join(openday.REPORTS, f"bench_{os.path.basename(os.path.normpath(d))}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log(L[0])
    log(f"report: {out}")
    return out


# --- 4. Walk detection on recorded runs (bench-check) ---


def _fmt_end(reason: Optional[str], t_end) -> str:
    if not reason:
        return "-"
    return f"{reason} {t_end:.1f}" if t_end is not None and np.isfinite(t_end) else str(reason)


def run_dirs(d: str) -> list[str]:
    """d itself when it is a run folder (has bench_RX.bin), else every run under it, oldest first."""
    if os.path.exists(os.path.join(d, "bench_RX.bin")):
        return [os.path.normpath(d)]
    return bench_runs(d)


def print_run(rows: list[dict], info: dict, out: Callable = print) -> None:
    out(f"\n{info['run']}")
    out(f"{'walk':>4}  {'label':<5}  {'live end':<14}  {'offline end':<14}  {'active_s':>8}  {'e0':>7}  {'speed':>6}")
    for r in rows:
        sp = f"{r['f']['speed']:.2f}" if r["label"] == "u" and r["f"] else "-"
        e0 = f"{r['e0']:.3g}" if r["e0"] is not None else "-"
        out(f"{r['id']:>4}  {r['label'] or '-':<5}  {_fmt_end(r['live_reason'], r['live_t_end']):<14}  "
            f"{_fmt_end(r['reason'], r['t_end']):<14}  {r['active_s']:>8.2f}  {e0:>7}  {sp:>6}")
    reasons = dict(collections.Counter(r["reason"] or "no baseline" for r in rows))
    live_reasons = dict(collections.Counter(r["live_reason"] or "-" for r in rows))
    te = np.array([r["t_end"] for r in rows if np.isfinite(r["t_end"])])
    out(f"walks {len(rows)}, seen {sum(r['seen'] for r in rows)}; offline ends {reasons}; live ends {live_reasons}")
    if len(te):
        out(f"offline end after the click: min {te.min():.1f}  median {np.median(te):.1f}  max {te.max():.1f} s")
    out(f"false walks (synthetic clicks every 1 s in quiet spans): {info['false']} of {info['probes']}")
    e0 = np.array(info["e0"], float)
    if len(e0):
        out(f"room e0 over the run: min {e0.min():.3g}  median {np.median(e0):.3g}  max {e0.max():.3g}")
    for rd in info.get("rounds", []):                      # (only runs recorded with the reuse print anything here)
        if rd["reused"]:
            out(f"player {rd['index']}: Uzayr's learning walks {', '.join(str(w) for w in rd['reused'])} reused "
                f"from before")
        if rd["redo"]:
            out(f"player {rd['index']}: Uzayr redid his learning walks during the game")


def bench_check(cfg: dict, d: str, max_s: Optional[float] = None, out: Callable = print) -> int:
    """`bench-check DIR [--max-s S]`: walk detection on recorded runs (`live --record`), recomputed offline through
    the live chain and WalkWindow with the runs' own room measurements (bench_detect): one row per recorded walk,
    its live and offline end, then the totals, synthetic false walks in the quiet spans between walks, and the
    room level over the run. DIR is one run folder or a folder of runs. Prints only: nothing is written anywhere.

    Speeds are printed for walks labelled U (Uzayr's) only; a visitor's walk numbers are never shown. A run in which
    Uzayr's learning walks were reused for a later player, or redone (Redo my walks), says which (bench_rounds)."""
    cfg = copy.deepcopy(cfg)
    if max_s is not None:
        cfg["walk"]["max_s"] = float(max_s)
    runs = run_dirs(d)
    if not runs:
        out(f"No recorded runs (bench_RX.bin) in or under {d}")
        return 1
    out(f"bench-check: {len(runs)} run(s), walk.max_s {cfg['walk']['max_s']}")
    for r in runs:
        rows, info = bench_detect(cfg, r, probes=True)
        print_run(rows, info, out)
    return 0
