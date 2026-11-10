"""eval --bench on a synthetic bench session built from Room B walks, in the exact on-disk format live --record
writes (bench_RX.bin via the capture file layout, events.jsonl beside it); and the bench tools rebuilding the pools
of each game walk from a recording (also one the app itself writes)."""

import json
import os

import numpy as np
import pytest

import openday
from openday import bench, recordings

from helpers import _app, _type, _events, _player, _learn, _game


def _write_bin(path, frames):
    openday.use_code()
    from pipeline.io.Capture.capture_logger import FILE_MAGIC, FRAME_DTYPE, read_capture_file
    hdr = json.dumps({"node_id": "RX", "dtype": [list(d) for d in FRAME_DTYPE.descr]}).encode()
    with open(path, "wb") as f:
        f.write(FILE_MAGIC + np.uint32(len(hdr)).tobytes() + hdr + frames.astype(FRAME_DTYPE).tobytes())
    assert len(read_capture_file(path)[1]) == len(frames)


def _run(cfg, d, walks, wid0):
    st, ev = recordings.Stitcher(cfg), []
    idle = recordings.IdleSource("P4", 1)
    idle.take(st, 3.0)
    ev.append({"kind": "baseline_start", "t_mono": st.t_end + 0.2, "secs": 30.0})   # a Measure room capture
    idle.take(st, 31.0)
    for i, (name, day, block, p, lab) in enumerate(walks):
        click = recordings.walk_piece(st, p, recordings.load_rx2(name, day, block), cfg)
        ev.append({"kind": "walk", "id": wid0 + i, "t_click": click, "seen": True, "label": None, "f": None})
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
    for part in ("(a) stale", "(b) alternating", "(c) Enrolment size", "(d) Do the two", "(e) Re-fitted tau"):
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
    _run(cfg, str(tmp_path / "20261101-114500"), enrol + tests, 100)
    ws, _ = bench.bench_walks(cfg, str(tmp_path / "20261101-114500"))
    assert ws and all("leg_torso" in w["f"] for w in ws)
    assert np.mean([np.isfinite(w["f"]["leg_torso"]) for w in ws]) > 0.9
    cfg["eval"]["bench_tests"] = len(tests)
    out = bench.run_bench(cfg, str(tmp_path), log=lambda s: None)
    text = open(out).read()
    assert f"shipped model {cfg['classifier']['kind']}" in text
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
