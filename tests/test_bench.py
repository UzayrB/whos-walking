"""eval --bench on a synthetic bench session built from Room B walks, in the exact on-disk format live --record
writes (bench_RX.bin via the capture file layout, events.jsonl beside it); the room level of a Measure room capture
someone walked through; and the bench tools rebuilding the pools of each game walk from a recording (also one the
app itself writes)."""

import json
import os

import numpy as np
import pytest

import openday
from openday import bench, chain, recordings

from helpers import _app, _type, _events, _player, _learn, _game


def _write_bin(path, frames):
    openday.use_code()
    from pipeline.io.Capture.capture_logger import FILE_MAGIC, FRAME_DTYPE, read_capture_file
    hdr = json.dumps({"node_id": "RX", "dtype": [list(d) for d in FRAME_DTYPE.descr]}).encode()
    with open(path, "wb") as f:
        f.write(FILE_MAGIC + np.uint32(len(hdr)).tobytes() + hdr + frames.astype(FRAME_DTYPE).tobytes())
    assert len(read_capture_file(path)[1]) == len(frames)


def _run(cfg, d, walks, wid0, seconds=None):
    """seconds: {i: j}: walk i is the second look of walk j (classifier.second_look's "second_of")."""
    seconds = seconds or {}
    st, ev = recordings.Stitcher(cfg), []
    idle = recordings.IdleSource("P4", 1)
    idle.take(st, 3.0)
    ev.append({"kind": "baseline_start", "t_mono": st.t_end + 0.2, "secs": 30.0})   # a Measure room capture
    idle.take(st, 31.0)
    for i, (name, day, block, p, lab) in enumerate(walks):
        click = recordings.walk_piece(st, p, recordings.load_rx2(name, day, block), cfg)
        ev.append({"kind": "walk", "id": wid0 + i, "t_click": click, "seen": True, "label": None, "f": None,
                   **({"second_of": wid0 + seconds[i]} if i in seconds else {})})
        if isinstance(lab, tuple):                                 # (label, source): the app's learning turns
            ev.append({"kind": "label", "walk_id": wid0 + i, "label": lab[0], "source": lab[1]})
        elif lab is not None:                                      # None: an unlabelled game walk (no event)
            ev.append({"kind": "label", "walk_id": wid0 + i, "label": lab})
        idle.take(st, 2.0)
    os.makedirs(d)
    _write_bin(os.path.join(d, "bench_RX.bin"), st.frames())
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(e) for e in ev) + "\n")


@pytest.mark.needs_code
def test_bench_report(cfg, tmp_path):
    S = lambda n, d, b: [(n, d, b, p) for p in recordings.passes(n, d, b, "S")]  # noqa: E731
    u1 = S("P4", 1, "slow") + S("P4", 1, "fast")
    _run(cfg, str(tmp_path / "20261101-100000"), [(*w, "u") for w in u1], 1)
    uz, vi = S("P4", 1, "natural") + S("P4", 2, "natural"), S("P3", 1, "natural") + S("P3", 2, "natural")
    enrol = [x for pair in zip([(*w, "v") for w in vi[:5]], [(*w, "u") for w in uz[:5]]) for x in pair]
    rng = np.random.default_rng(5)
    rest = {"u": [(*w, "u") for w in uz[5:]], "v": [(*w, "v") for w in vi[5:]]}
    tests = []
    while len(tests) < 20 and (rest["u"] or rest["v"]):
        k = "u" if (rng.random() < 0.5 and rest["u"]) or not rest["v"] else "v"
        tests.append(rest[k].pop(0))
    _run(cfg, str(tmp_path / "20261101-114500"), enrol + tests, 100)
    out = bench.run_bench(cfg, str(tmp_path), log=lambda s: None)
    text = open(out).read()
    first = text.splitlines()[0]
    assert ("PASS" in first or "FAIL" in first) and "of 20" in first
    for part in ("(a) stale", "(b) alternating", "(c) Do the two"):
        assert part in text
    assert "Walk ends: as shipped" in text
    os.remove(out)


@pytest.mark.needs_code
@pytest.mark.needs_model
def test_bench_report_scores_the_shipped_model(cfg_shipped, tmp_path):
    """The same synthetic session scored with the shipped model: its rows appear and every walk got a gait set."""
    cfg = cfg_shipped
    if cfg["classifier"]["kind"] == "m0":
        return
    S = lambda n, d, b: [(n, d, b, p) for p in recordings.passes(n, d, b, "S")]  # noqa: E731
    uz, vi = S("P4", 1, "natural"), S("P3", 1, "natural")
    enrol = [x for pair in zip([(*w, "u") for w in uz[:5]], [(*w, "v") for w in vi[:5]]) for x in pair]
    tests = [x for pair in zip([(*w, "v") for w in vi[5:]], [(*w, "u") for w in uz[5:]]) for x in pair]
    # the first test walk a close call walked once more (the same walker: its pass again), as the app records it
    walks_ = enrol + tests[:1] + [tests[0]] + tests[1:]
    _run(cfg, str(tmp_path / "20261101-114500"), walks_, 100, seconds={len(enrol) + 1: len(enrol)})
    ws, _ = bench.bench_walks(cfg, str(tmp_path / "20261101-114500"))
    assert ws and all("leg_torso" in w["f"] for w in ws)
    assert np.mean([np.isfinite(w["f"]["leg_torso"]) for w in ws]) > 0.9
    cfg["eval"]["bench_tests"] = len(tests)
    out = bench.run_bench(cfg, str(tmp_path), log=lambda s: None)
    text = open(out).read()
    k = cfg["classifier"]["kind"]
    assert f"shipped model {k}" in text
    assert f"tests: the {len(tests)} walks" in text                # (the second look took no test place)
    assert "1 test walks were close calls walked once more, each scored on both walks averaged" in text
    assert f"| (b) 2 + 5 (the app's round: the friend's 2, Uzayr's 5), shipped model {k} |" in text
    assert f"| (b) 2 + 2 (the older round), shipped model {k} |" in text and "2 + 2 (the app's round)" not in text
    os.remove(out)


@pytest.mark.needs_code
def test_bench_walks_apply_the_operator_stop(cfg, tmp_path):
    """A recorded "stop" press (the operator's "they've stopped") ends the offline walk there too, as live; every
    scored walk carries its end reason, end and arrival times."""
    ps = recordings.passes("P4", 1, "natural", "S")[:2]
    d = str(tmp_path / "20261101-120000")
    _run(cfg, d, [("P4", 1, "natural", p, "u") for p in ps], 1)
    ev = [json.loads(x) for x in open(os.path.join(d, "events.jsonl")) if x.strip()]
    click = next(e["t_click"] for e in ev if e["kind"] == "walk" and e["id"] == 2)
    with open(os.path.join(d, "events.jsonl"), "a") as f:
        f.write(json.dumps({"kind": "stop", "walk_id": 2, "t_mono": click + 2.0, "t_dev": None}) + "\n")
    ws, _ = bench.bench_walks(cfg, d)
    by = {w["id"]: w for w in ws}
    assert set(by) == {1, 2}
    assert by[2]["reason"] == "operator" and 2.0 - 1e-6 <= by[2]["t_end"] < 2.0 + 0.06
    assert by[1]["reason"] != "operator" and by[1]["t_end"] > 2.0
    for w in ws:
        assert 0.0 < w["t_arrive"] <= w["t_end"] and w["active_s"] > 0


@pytest.mark.needs_code
def test_bench_detect_takes_the_quiet_part_of_a_room_measurement_someone_walked_through(cfg, tmp_path):
    """Two walks pass during the 30 s Measure room capture: bench_detect gives the walk after it the room level the
    app now takes (chain.room_baseline: the capture's quiet part, far below the whole capture's), and bench-check
    says the measurement was disturbed."""
    d = str(tmp_path / "20261101-160000")
    st, ev = recordings.Stitcher(cfg), []
    idle = recordings.IdleSource("P4", 1)
    idle.take(st, 3.0)
    t0 = st.t_end + 0.2
    ev.append({"kind": "baseline_start", "t_mono": t0, "secs": 30.0})
    idle.take(st, 8.0)
    fr, ps = recordings.load_rx2("P4", 1, "natural"), recordings.passes("P4", 1, "natural", "S")
    for p in ps[:2]:                                               # someone walks while the room is measured
        recordings.walk_piece(st, p, fr, cfg)
    idle.take(st, t0 + 31.0 - st.t_end)
    click = recordings.walk_piece(st, ps[5], fr, cfg)
    ev.append({"kind": "walk", "id": 1, "t_click": click, "seen": True, "label": "u", "f": None})
    idle.take(st, 2.0)
    os.makedirs(d)
    _write_bin(os.path.join(d, "bench_RX.bin"), st.frames())
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(e) for e in ev) + "\n")
    bc = bench.bench_columns(cfg, d)
    rows, info = bench.bench_detect(cfg, d, bc=bc)
    t_dev, k = bench._to_dev(bc, t0)                               # the capture's columns, as bench_detect cuts them
    m = (bc["allc"].t >= t_dev) & (bc["allc"].t <= t_dev + 30.0) & (bc["ep"] == k)
    cap = chain.Columns(bc["allc"].t[m], bc["allc"].power[m], bc["allc"].gap[m])
    args = (cfg, bc["ch"].freqs, bc["ch"].spec.variant, bc["ch"].spec.nfft)
    b, whole = chain.room_baseline(cap, *args), chain.make_baseline(cap, *args)
    assert b.disturbed and info["disturbed"] == [b.d] and b.d >= float(cfg["baseline"]["quiet_flag_d"])
    assert rows[0]["e0"] == b.e0 == info["e0"][0] and b.e0 < whole.e0 / 2 and rows[0]["seen"]
    lines: list = []
    assert bench.bench_check(cfg, d, out=lines.append) == 0
    assert f"room measured while someone moved (d {b.d:.1f}): its quiet part used" in lines


@pytest.mark.needs_code
def test_bench_walks_take_turn_labels_and_skip_unlabelled_game_walks(cfg, tmp_path):
    """The app's recordings: learning walks carry a label event with source "turn" (no click), a game walk
    is labelled only if It was ... was clicked. Turn labels count as ground truth; an unlabelled walk is skipped
    as "no label", nothing else changes."""
    ps = recordings.passes("P4", 1, "natural", "S")[:3]
    d = str(tmp_path / "20261101-130000")
    _run(cfg, d, [("P4", 1, "natural", ps[0], ("u", "turn")), ("P4", 1, "natural", ps[1], None),
                  ("P4", 1, "natural", ps[2], "u")], 1)
    ws, skipped = bench.bench_walks(cfg, d)
    assert [w["id"] for w in ws] == [1, 3] and all(w["label"] == "u" for w in ws)
    assert skipped["no label"] == 1
    rows, _info = bench.bench_detect(cfg, d)
    assert [r["label"] for r in rows] == ["u", None, "u"]


@pytest.mark.needs_code
def test_bench_tools_rebuild_the_pools_of_a_reused_round(cfg, tmp_path):
    """A recording where the second player's round reused Uzayr's walks: a uzayr_reused event names
    his earlier walk ids. bench_detect / bench_walks give each game walk the pools it was decided on, and
    bench-check says which round reused them (a run without such events prints as before)."""
    ps = recordings.passes("P4", 1, "natural", "S")[:7]
    d = str(tmp_path / "20261101-150000")
    labs = [("v", "turn"), ("v", "turn"), ("u", "turn"), ("u", "turn"), "v", ("v", "turn"), "u"]
    _run(cfg, d, [("P4", 1, "natural", p, lab) for p, lab in zip(ps, labs)], 1)
    path = os.path.join(d, "events.jsonl")
    purpose = {1: "enrol_visitor", 2: "enrol_visitor", 3: "enrol_uzayr", 4: "enrol_uzayr", 5: "test",
               6: "enrol_visitor", 7: "test"}
    ev = []
    for e in (json.loads(x) for x in open(path) if x.strip()):
        if e["kind"] == "walk":
            e["purpose"] = purpose[e["id"]]
            if e["id"] in (1, 6):
                ev.append({"kind": "new_visitor", "index": 1 if e["id"] == 1 else 2, "t_mono": e["t_click"] - 5.0})
            if e["id"] == 6:
                ev.append({"kind": "uzayr_reused", "index": 2, "t_mono": e["t_click"] - 5.0, "age_min": 1.5,
                           "walk_ids": [3, 4]})
        ev.append(e)
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(e) for e in ev) + "\n")
    rows, info = bench.bench_detect(cfg, d)
    assert [r["reused"] for r in info["rounds"]] == [[], [3, 4]]
    by = {r["id"]: r for r in rows}
    assert by[5]["pools"] == {"u": [3, 4], "v": [1, 2]} and by[7]["pools"] == {"u": [3, 4], "v": [6]}
    assert by[3]["pools"] is None and by[3]["purpose"] == "enrol_uzayr"
    ws, _ = bench.bench_walks(cfg, d)
    assert {w["id"]: w["pools"] for w in ws if w["purpose"] == "test"} == {5: by[5]["pools"], 7: by[7]["pools"]}
    lines: list = []
    assert bench.bench_check(cfg, d, out=lines.append) == 0
    assert "player 2: Uzayr's learning walks 3, 4 reused from before" in lines
    assert not any("redid" in x for x in lines)


def test_bench_rounds_rebuild_older_recordings_and_the_new_events():
    """bench_rounds: a learning walk joins its pool from its own walk event (older recordings have no "turn" label
    events), a relabelled learning walk moves (older versions of the app), Undo walk's discard removes one;
    uzayr_dropped (Measure room while learning) empties Uzayr's set; uzayr_kept (Undo walk right after Redo my walks)
    brings it back."""
    def walk(i, purpose, label=None, seen=True):
        return {"kind": "walk", "id": i, "purpose": purpose, "seen": seen, "label": label}

    old = [{"kind": "new_visitor", "index": 1}, walk(1, "enrol_uzayr", "u"), walk(2, "enrol_visitor", "v"),
           walk(3, "enrol_uzayr", "u"), {"kind": "label", "walk_id": 3, "label": "v"},   # an It was ... click
           walk(4, "enrol_visitor", "v"), walk(5, "enrol_visitor", None, seen=False),
           walk(6, "enrol_uzayr", "u"), {"kind": "discard", "walk_id": 6}, walk(7, "test"),
           {"kind": "label", "walk_id": 7, "label": "u"}]                                # a game walk: no pool
    rounds, pools = bench.bench_rounds(old)
    assert pools == {7: {"u": [1], "v": [2, 3, 4]}} and rounds[0]["games"] == [7]
    new = [{"kind": "new_visitor", "index": 1}, walk(1, "enrol_visitor", "v"), walk(2, "enrol_visitor", "v"),
           walk(3, "enrol_uzayr", "u"), walk(4, "enrol_uzayr", "u"), walk(5, "test"),
           {"kind": "uzayr_redo", "index": 1}, {"kind": "uzayr_kept", "index": 1, "walk_ids": [3, 4]}, walk(6, "test"),
           {"kind": "new_visitor", "index": 2}, {"kind": "uzayr_reused", "index": 2, "walk_ids": [3, 4]},
           walk(7, "enrol_visitor", "v"), {"kind": "uzayr_dropped", "index": 2, "why": "room"},
           walk(8, "enrol_visitor", "v"), walk(9, "enrol_uzayr", "u"), walk(10, "enrol_uzayr", "u"), walk(11, "test")]
    for e in list(new):
        if e["kind"] == "walk" and e["label"]:                     # the app's turn label after each
            new.insert(new.index(e) + 1, {"kind": "label", "walk_id": e["id"], "label": e["label"], "source": "turn"})
    rounds, pools = bench.bench_rounds(new)
    assert pools[5] == pools[6] == {"u": [3, 4], "v": [1, 2]} and pools[11] == {"u": [9, 10], "v": [7, 8]}
    assert [(r["reused"], r["redo"]) for r in rounds] == [([], False), ([], False)]


@pytest.mark.shipped
def test_bench_rounds_rebuild_the_pools_from_a_recording(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    t = _game(app, t)
    v1 = [w["walk_id"] for w in app.visitor.walks]
    u1 = [w["walk_id"] for w in app.uz_fresh.walks]
    _player(app, "Si")
    t, _ = _learn(app, t)
    t = _game(app, t)
    g2 = app.last["id"]
    v2 = [w["walk_id"] for w in app.visitor.walks]
    _type(app, ["ctrl+u"])
    t, _ = _learn(app, t)
    _game(app, t)
    g3 = app.last["id"]
    u3 = [w["walk_id"] for w in app.uz_fresh.walks]
    app.close()
    rounds, pools = bench.bench_rounds(_events(tmp_path))
    assert [r["reused"] for r in rounds] == [[], u1] and [r["redo"] for r in rounds] == [False, True]
    assert pools[rounds[0]["games"][0]] == {"u": u1, "v": v1}
    assert pools[g2] == {"u": u1, "v": v2} and pools[g3] == {"u": u3, "v": v2}
    assert rounds[1]["games"] == [g2, g3]


def test_bench_rounds_on_the_home_run_with_players():
    """The 8 Nov home run with players (recorded before the reuse): every game walk gets the pools it was decided
    on (read only)."""
    path = os.path.join(openday.OPENDAY, "bench", "solo_20261108", "20261108-165425", "events.jsonl")
    if not os.path.exists(path):
        pytest.skip("home recording not on this machine")
    rounds, pools = bench.bench_rounds([json.loads(x) for x in open(path) if x.strip()])
    assert len(pools) == 19 and all(p["u"] and p["v"] for p in pools.values())
    assert {(tuple(p["u"]), tuple(p["v"])) for p in pools.values()} == {((12, 14, 16), (13, 15, 17)),
                                                                        ((34, 37, 39), (35, 36, 38, 40))}


def test_bench_rebuilds_a_second_look_from_its_walks(cfg):
    """bench.second_looks: a game walk with "second_of" and its close call, both labelled the same, decided again
    against the pools they were decided on (bench_rounds) on the average of both walks, and the close call alone;
    pairs whose pool walks or close call are missing, or labelled apart, are left out."""
    names = cfg["features"]["names"]
    rng = np.random.default_rng(3)

    def w(i, label, x, purpose="test", pools=None, second_of=None):
        f = dict(zip(names, x))
        return {"id": i, "label": label, "f": f, "x": np.array(x, float), "purpose": purpose, "pools": pools,
                "second_of": second_of}

    u0, v0 = np.array([0.9, 105.0, 2.0, 0.30]), np.array([1.2, 120.0, 2.6, 0.45])
    sd = np.array([0.03, 2.0, 0.05, 0.01])
    ws = [w(i, "u", u0 + rng.normal(0, 1, 4) * sd, "enrol_uzayr") for i in (1, 2, 3)]
    ws += [w(i, "v", v0 + rng.normal(0, 1, 4) * sd, "enrol_visitor") for i in (4, 5)]
    pools = {"u": [1, 2, 3], "v": [4, 5]}
    mid = (u0 + v0) / 2
    ws += [w(6, "v", mid, pools=pools), w(7, "v", v0 + 0.2 * (v0 - u0), pools=pools, second_of=6),
           w(8, "u", mid, pools=pools), w(9, "v", v0, pools=pools, second_of=8),          # labelled apart: out
           w(10, "u", u0, pools={"u": [1, 99], "v": [4]}, second_of=6)]                   # a pool walk missing: out
    out = bench.second_looks(cfg, ws)
    assert [(x["first"], x["second"], x["label"], x["both"]) for x in out] == [(6, 7, "v", "visitor")]
    from openday import models
    both = models.average_walks([ws[5]["f"], ws[6]["f"]])
    P = {"uzayr": np.array([x["x"] for x in ws[:3]]), "visitor": np.array([x["x"] for x in ws[3:5]])}
    use = cfg["classifier"]["use"]
    d = models.decide_m0(np.array([both[n] for n in use]), P, use, cfg)
    assert out[0]["margin"] == pytest.approx(d.margin)


def test_bench_rounds_link_a_second_look_to_its_close_call():
    """A second look is a game walk with "second_of": the same pools as its close call, and the round says which."""
    ev = [{"kind": "new_visitor", "index": 1},
          {"kind": "walk", "id": 1, "purpose": "enrol_visitor", "seen": True, "label": "v"},
          {"kind": "walk", "id": 2, "purpose": "enrol_uzayr", "seen": True, "label": "u"},
          {"kind": "walk", "id": 3, "purpose": "test", "seen": True, "label": None, "second_look": True},
          {"kind": "walk", "id": 4, "purpose": "test", "seen": True, "label": None, "second_of": 3},
          {"kind": "label", "walk_id": 4, "label": "v", "counted": "right"},
          {"kind": "label", "walk_id": 3, "label": "v", "counted": None, "both_with": 4}]
    rounds, pools = bench.bench_rounds(ev)
    assert rounds[0]["seconds"] == {4: 3} and rounds[0]["games"] == [3, 4]
    assert pools[3] == pools[4] == {"u": [2], "v": [1]}


def test_bench_check_lists_the_second_looks_only_when_there_are_some():
    """bench-check's run summary names each second look (close call -> the walk after it); a recording without any
    prints exactly what it did before."""
    info = {"run": "r", "e0": [1.0], "probes": 0, "false": 0,
            "rounds": [{"index": 1, "reused": [], "redo": False, "games": [3, 4], "seconds": {4: 3}}]}
    lines: list = []
    bench.print_run([], info, lines.append)
    assert "player 1: second looks (close call -> the walk after it): 3 -> 4" in lines
    plain: list = []
    info["rounds"][0]["seconds"] = {}
    bench.print_run([], info, plain.append)
    assert not any("second look" in x for x in plain) and len(plain) == len(lines) - 1


def test_bench_split_leaves_second_looks_out_and_scores_a_close_call_as_shown(cfg):
    """eval --bench's split (bench.split_session): a second look ("second_of") takes no pool, test or matched place
    of its own; bench.as_shown scores a test walk that was a close call walked once more (labelled the same, same
    run) on both walks' readings averaged, as the round showed it; a pair labelled apart, or a walk of another run
    with the same id, is not paired."""
    names = cfg["features"]["names"]

    def w(i, label, x, second_of=None, run="r1"):
        return {"id": i, "label": label, "f": dict(zip(names, map(float, x))), "x": np.array(x, float),
                "second_of": second_of, "run": run}

    u, v = [0.9, 105.0, 2.0, 0.30], [1.2, 120.0, 2.6, 0.45]
    post = [w(1, "v", v), w(2, "v", v), w(3, "u", u), w(4, "u", u),            # the pool, 2 + 2
            w(5, "u", [1.1, 112.0, 2.3, 0.40]), w(6, "u", [0.7, 100.0, 1.9, 0.30], second_of=5),   # a close call, again
            w(7, "v", v), w(8, "v", v),                                         # the other tests
            w(9, "u", u), w(10, "v", v, second_of=9),                           # matched; a "second look" labelled apart
            w(8, "v", [2.0, 150.0, 3.0, 0.6], second_of=8, run="r2")]          # another run's walk 8 (ids restart)
    eu, ev_, tests, matched = bench.split_session(post, 2, 3)
    assert [x["id"] for x in eu] == [3, 4] and [x["id"] for x in ev_] == [1, 2]
    assert [x["id"] for x in tests] == [5, 7, 8] and [x["id"] for x in matched] == [9]
    shown, n = bench.as_shown(cfg, tests, post)
    assert n == 1 and [x["id"] for x in shown] == [5, 7, 8] and shown[0]["both_with"] == 6
    from openday import models
    both = models.average_walks([post[4]["f"], post[5]["f"]])
    assert shown[0]["f"] == both and shown[0]["x"] == pytest.approx([both[k] for k in names])
    assert shown[0]["x"][0] == pytest.approx(0.9) and tests[0]["x"][0] == pytest.approx(1.1)   # (the input untouched)
    assert shown[1] is tests[1] and shown[2] is tests[2]
    shown_m, n_m = bench.as_shown(cfg, matched, post)
    assert n_m == 0 and shown_m[0] is matched[0]
    old = [x for x in post if x["run"] == "r1"]                                # without second looks: as before
    for x in old:
        x["second_of"] = None
    eu, ev_, tests, matched = bench.split_session(old, 2, 3)
    assert [x["id"] for x in tests] == [5, 6, 7] and [x["id"] for x in matched] == [8, 9, 10]
    assert bench.as_shown(cfg, tests, old) == (tests, 0)
