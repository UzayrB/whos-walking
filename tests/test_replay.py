"""Replays of thesis recordings (openday/recordings.py and app.run_replay), headless: the stitched stream is
continuous at 150 packets a second, scripted rounds run the shipped model and leave no visitor name on disk, a
refused step sends the replay off script; and a round on the app leaves no name anywhere. Most need the thesis code
and recordings (needs_code)."""

import json
import os
import subprocess
import threading
import time

import numpy as np
import pytest
import yaml

import openday
from openday import app as A, recordings
from openday.app import NAME_ENTRY, RESULT

from helpers import _app, _type, _walk, _events


TYPED = ("Zanele", "Mpho")                                       # typed visitor names: not participants


@pytest.mark.needs_code
def test_stitching_is_contiguous(cfg):
    plan = recordings.build_plan(os.path.join(openday.SCRIPTS, "demo_two_people.yaml"), cfg)
    fr = plan.frames
    assert np.all(np.diff(fr["pkt_id"]) == 1)
    assert np.all(np.diff(fr["t_mono"]) >= 0)
    assert np.all(np.diff(fr["local_timestamp"].astype(np.int64)) > 0)
    assert abs(len(fr) / (fr["t_mono"][-1] - fr["t_mono"][0]) - 150) < 3
    clicks = [k for k in plan.keys if k[1] == "space"]
    assert len(clicks) >= len(plan.walks)
    ids = [(w["t_click"]) for w in plan.walks]
    assert len(set(ids)) == len(ids)                                # no pass reused


class Counter:
    def __init__(self):
        self.n, self.t, self.stop = 0, [], threading.Event()

    def handle(self, rec):
        self.n += 1
        self.t.append(rec.t_mono)

    def tick(self, now):
        pass

    def close(self):
        pass


@pytest.mark.needs_code
def test_logger_sees_150_per_second_no_drops(cfg):
    st = recordings.Stitcher(cfg)
    recordings.IdleSource("P4", 1).take(st, 5.0)
    plan = recordings.Plan(st.frames(), [])
    c = Counter()
    rp = recordings.Replay(plan, cfg, 1.0, lambda k, t: None)
    openday.use_code()
    from pipeline.io.Capture.capture_logger import run_capture
    states = {}
    rp.start()
    run_capture(rp.nodes, duration=rp.duration, extra_consumers=[c], quiet_monitor=True, states_out=states)
    s = states["RX"]
    assert c.n == len(plan.frames) and s.queue_dropped == 0 and s.counters.malformed == 0
    rate = (c.n - 1) / (c.t[-1] - c.t[0])
    assert 140 < rate < 160, rate


@pytest.mark.needs_code
def test_no_visitor_name_on_disk_after_replay_shipped(cfg_shipped, tmp_path):
    """A scripted replay with two visitors leaves neither name anywhere in state/, logs/ or reports/. The shipped
    model and round: each visitor's learning walk first, then Uzayr's; the second visitor's round reuses Uzayr's
    walk (taken a minute before), so only the visitor walks."""
    cfg = cfg_shipped
    cfg["pools"].update(n_enrol=1, n_enrol_uzayr=1)
    cfg["baseline"]["live_s"] = 3.0
    # gap_s 3.0 (was 1.5): a replay walk can now end "stopped" up to ~8 s after its click (walk.max_s 9), and the
    # second visitor's name must not be typed into a still-running walk
    script = {"uzayr_standin": "P4", "baseline": {"name": "P4", "day": 1}, "seed": 4, "gap_s": 3.0,
              "steps": [{"action": "baseline"}, {"action": "new_visitor", "name": TYPED[0]},
                        {"walk": "P3", "day": 1, "block": "natural"},
                        {"walk": "P4", "day": 1, "block": "natural"},
                        {"walk": "P3", "day": 1, "block": "natural"},
                        {"action": "new_visitor", "name": TYPED[1]},
                        {"walk": "P5", "day": 1, "block": "natural"},
                        {"walk": "P4", "day": 1, "block": "natural"}]}
    p = tmp_path / "two_visitors_c3.yaml"
    p.write_text(yaml.safe_dump(script))
    A.run_replay(cfg, str(p), speed=8.0, headless=True)
    _no_name_on_disk()
    # the app tallies nothing and a replay clicks no It was ...: the summary scores every answer against the
    # script's known walker, right or wrong (the answer always names someone: never "not sure")
    out = sorted(f for f in os.listdir(openday.LOGS) if f.startswith("replay_two_visitors_c3_") and f.endswith(".json"))
    summ = json.load(open(os.path.join(openday.LOGS, out[-1])))
    assert summ["tally"]["right"] == summ["tally"]["wrong"] == summ["tally"]["not_sure"] == 0
    dec = [w for w in summ["walks"] if w["decision"] and w.get("shown", True)]   # (the answers shown)
    assert summ["decisions"] == len(dec) == summ["decisions_right"] + summ["decisions_wrong"]
    for w in dec:
        assert w["scored"] == ("right" if w["decision"] == w["role"] else "wrong")
    if all(w["seen"] for w in summ["walks"]):                     # the second round reused Uzayr's walk
        assert [w["purpose"] for w in summ["walks"]] == ["enrol_visitor", "enrol_uzayr", "test", "enrol_visitor",
                                                          "test"]


@pytest.mark.needs_code
def test_redo_waits_for_the_answer_at_any_speed(cfg_shipped):
    """The scripted Redo my walks (Ctrl+U) comes once the answer is up: ui.reading_s of real time is reading_s x
    --speed of stream time, so the wait grows with the speed (at --speed 1 the plan is as before)."""
    p = os.path.join(openday.SCRIPTS, "demo_three_visitors.yaml")
    read = float(cfg_shipped["ui"]["reading_s"])
    with open(p, encoding="utf-8") as f:
        gap = float(yaml.safe_load(f)["gap_s"])
    for speed in (1.0, 4.0, 8.0):
        plan = recordings.build_plan(p, cfg_shipped, speed=speed)
        (cu,) = [t for t, k in plan.keys if k == "ctrl+u"]
        before = max(w["t_click"] for w in plan.walks if w["t_click"] < cu)   # the game walk before it
        after = [w for w in plan.walks if w["t_click"] > cu]
        assert cu - before >= gap + read * speed + 1.5, speed
        assert [w["expect"] for w in after[:2]] == ["enrol_uzayr"] * 2 and plan.players == 3


@pytest.mark.needs_code
def test_a_refused_step_sends_the_replay_off_script(cfg_shipped, tmp_path, monkeypatch):
    """A scripted step the app refuses (here Redo my walks pressed during a walk) sends the replay off script: it
    says so in its summary and returns 1, instead of carrying on silently."""
    cfg = cfg_shipped
    cfg["pools"].update(n_enrol=1, n_enrol_uzayr=1)
    cfg["baseline"]["live_s"] = 3.0
    script = {"uzayr_standin": "P4", "baseline": {"name": "P4", "day": 1}, "seed": 4, "gap_s": 3.0,
              "steps": [{"action": "baseline"}, {"action": "new_visitor", "name": TYPED[0]},
                        {"walk": "P3", "day": 1, "block": "natural"},
                        {"walk": "P4", "day": 1, "block": "natural"},
                        {"walk": "P3", "day": 1, "block": "natural"},
                        {"action": "enrol_uzayr"},
                        {"walk": "P4", "day": 1, "block": "natural"}]}
    p = tmp_path / "refused.yaml"
    p.write_text(yaml.safe_dump(script))
    build = recordings.build_plan

    def during_a_walk(path, c, speed=1.0):                       # Ctrl+U half a second into the third walk
        plan = build(path, c, speed=speed)
        click = sorted(t for t, k in plan.keys if k == "space")[2]
        plan.keys = sorted((click + 0.5 if k == "ctrl+u" else t, k) for t, k in plan.keys)
        return plan

    monkeypatch.setattr(recordings, "build_plan", during_a_walk)
    assert A.run_replay(cfg, str(p), speed=8.0, headless=True) == 1
    out = sorted(f for f in os.listdir(openday.LOGS) if f.startswith("replay_refused_") and f.endswith(".json"))
    summ = json.load(open(os.path.join(openday.LOGS, out[-1])))
    assert summ["off_script"] and "Redo my walks" in summ["off_script"][0]
    assert summ["walks"][-1]["purpose"] == "test"                 # (his learning walk became a game walk)
    _no_name_on_disk()


def _no_name_on_disk():
    for d in (openday.STATE, openday.LOGS, openday.REPORTS):
        out = subprocess.run(["grep", "-rlE", "|".join(TYPED), d], capture_output=True, text=True)
        assert out.stdout == "", f"visitor name found in {out.stdout}"


@pytest.mark.needs_code
def test_recorder_gets_unsigned_timestamps(monkeypatch):
    """Past 35.8 min the firmware prints the counter negative; the recorder must get it back unsigned."""
    openday.use_code()
    import pipeline.io.Capture.capture_logger as CL

    class Rec:
        local_timestamp = -927187572

    monkeypatch.setattr(CL, "parse_line", lambda *a, **k: Rec())
    A.unsigned_timestamps()
    assert CL.parse_line("x").local_timestamp == (1 << 32) - 927187572
    A.unsigned_timestamps()                                   # applying it twice wraps only once
    assert getattr(CL.parse_line, "openday_unsigned", False)


@pytest.mark.needs_code
@pytest.mark.needs_model
def test_interleaved_enrolment_takes_turns_and_uses_fresh_walks(cfg_shipped, tmp_path, monkeypatch):
    """Interleaved enrolment (C3) in the round: after New player the visitor's learning walks, then Uzayr's; tests
    are judged against Uzayr's fresh walks; the next player's round reuses them (memory only), and quitting wipes
    them."""
    cfg = cfg_shipped
    cfg["pools"].update(n_enrol=2, n_enrol_uzayr=2)
    cfg["baseline"]["live_s"] = 3.0
    seen = {}
    orig = A.App.close

    def close(self):
        seen["purposes"] = [w["purpose"] for w in self.walk_log if w["seen"]]
        seen["decisions"] = [w["decision"] for w in self.walk_log if w["purpose"] == "test"]
        seen["kept"], seen["kind"] = len(self.uz_fresh), (self.model.kind if self.model else "m0")
        seen["reused"], seen["app"] = self.uz_reused, self
        return orig(self)

    monkeypatch.setattr(A.App, "close", close)
    steps = [{"action": "baseline"}, {"action": "new_visitor", "name": "Zanele"}]
    for w in ("P3", "P3", "P4", "P4"):
        steps.append({"walk": w, "day": 1, "block": "natural"})
    steps += [{"walk": "P3", "day": 1, "block": "natural"},
              {"walk": "P4", "day": 1, "block": "natural"},
              {"action": "new_visitor", "name": "Mpho"}]
    p = tmp_path / "c3.yaml"
    # gap_s 3.0 (was 1.5): a replay walk can now end "stopped" up to ~8 s after its click (walk.max_s 9), so the
    # next key must not land inside it
    p.write_text(yaml.safe_dump({"uzayr_standin": "P4", "baseline": {"name": "P4", "day": 1}, "seed": 5,
                                 "gap_s": 3.0, "steps": steps}))
    A.run_replay(cfg, str(p), speed=8.0, headless=True)
    assert seen["purposes"] == ["enrol_visitor", "enrol_visitor", "enrol_uzayr", "enrol_uzayr", "test", "test"]
    assert all(d is not None for d in seen["decisions"]) and seen["kind"] == cfg["classifier"]["kind"]
    assert seen["kept"] == 2 and seen["reused"]                   # the next player reuses them ...
    assert len(seen["app"].uz_fresh) == 0                          # ... and quitting wipes them


# ---- privacy and the replay's own config -----------------------------------------------------------------------------

@pytest.mark.alternate_round
def test_no_name_on_disk_after_a_round(cfg, tmp_path):
    names = ("Thandiwe", "Bongani")
    log = tmp_path / "run.log"
    app = _app(cfg, tmp_path, record=True, log=str(log))
    _type(app, ["n", *names[0], "enter"])
    for i in range(6):
        _walk(app, 100.0 + 20 * i, label="u" if i % 2 == 0 else "v")
    _walk(app, 300.0)
    app.state = RESULT
    _type(app, ["v"])
    app.view(time.monotonic())
    _type(app, ["n", *names[1], "enter"])
    _type(app, ["ctrl+q", "escape", "ctrl+n"])
    app.close()
    assert app.name is None and len(app.visitor) == 0
    for root, _, files in os.walk(tmp_path):
        for f in files:
            text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
            assert not any(n in text for n in names), f
    keys = [e for e in _events(tmp_path) if e["kind"] == "key" and e["state"] == NAME_ENTRY]
    assert keys and all(e["key"] == "<name entry>" for e in keys)


@pytest.mark.alternate_round
def test_replay_works_on_a_copy_of_the_config(cfg, monkeypatch):
    from openday import app as A, recordings
    got = {}

    class Stop(Exception):
        pass

    def plan(script, c, speed=1.0):
        got["cfg"] = c
        raise Stop

    monkeypatch.setattr(recordings, "build_plan", plan)
    with pytest.raises(Stop):
        A.run_replay(cfg, "none.yaml", headless=True)
    assert got["cfg"] is not cfg
    assert got["cfg"]["baseline"]["delay_s"] == 0.0 and got["cfg"]["walk"]["after_end_lock_s"] == 0.0
    assert cfg["baseline"]["delay_s"] > 0 and cfg["walk"]["after_end_lock_s"] > 0


def test_a_replay_tallies_only_the_answers_shown():
    """A close call answered from both walks never showed its own answer ("walk once more"): the replay leaves it out
    of right / wrong and counts it apart (how its own answer would have scored); the answer from both walks counts,
    scored against the script's walker of that second walk; a missed second look leaves the first walk's answer
    shown, so that one counts. Without second looks the tally is as before."""
    from openday import models
    from openday.app import replay_rows, replay_tally

    def dec(label, margin):
        return models.Decision(label=label, margin=margin, not_sure=margin < 0.1, dist={}, contrib={})

    def walk(i, d, seen=True, **kw):
        return {"id": i, "purpose": "test", "seen": seen, "decision": d, "f": {"speed": 1.0} if seen else None,
                "reason": "arrived", "active_s": 3.0, **kw}

    truths = [{"role": r, "gt_speed": 1.0} for r in ("uzayr", "uzayr", "visitor", "visitor", "visitor", "uzayr")]
    seen = [walk(1, dec("visitor", 0.04)),                                   # a close call (wrong on its own) ...
            walk(2, dec("uzayr", 0.2), second_of=1, both=True),              # ... right from both walks
            walk(3, dec("uzayr", 0.03)),                                     # a close call ...
            walk(4, None, seen=False, second_of=3),                          # ... its second walk missed: it shows
            walk(5, dec("visitor", 0.3)), walk(6, dec("visitor", 0.25))]
    res = replay_rows(truths, seen)
    assert [r.get("shown", True) for r in res] == [False, True, True, True, True, True]
    assert [r["scored"] for r in res] == ["wrong", "right", "wrong", None, "right", "wrong"]
    t = replay_tally(res)
    assert (t["decisions"], t["decisions_right"], t["decisions_wrong"]) == (4, 2, 2)
    assert t["second_looks"] == 2 and t["close_calls_walked_again"] == {"walks": 1, "right_alone": 0,
                                                                         "wrong_alone": 1}
    assert t["decisions_close_calls"] == 1                                    # (walk 3, shown: its own answer)
    plain = replay_tally(replay_rows(truths[4:], seen[4:]))
    assert plain == {"decisions": 2, "decisions_right": 1, "decisions_wrong": 1, "decisions_close_calls": 0}
