"""The app (openday/app.py), headless, driven the way the operator drives it: what each click and key does, whose
turn it is, the light, what the card and the operator's line say, and the 3D picture inside the app.

Most tests play the round with a fixed answer (helpers._app), so they check the flow, not the model. A test marked
`shipped` plays the round as shipped (the visitor's two learning walks first, then Uzayr's, his kept across players
while fresh); one marked `alternate_round` plays the older alternating round, which can still be chosen
(pools.enrol_order alternate: three learning walks each, Uzayr first, his walks wiped with each player)."""

import collections
import json
import os
import time

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from openday import bench, chain, models, screen, walks
from openday.app import (App, RunLog, BUSY, DECIDING, ENROL_UZAYR, ENROL_VISITOR, IDLE, NAME_ENTRY, NEEDS_BASELINE,
                         READY, RESULT, WALKING)
from openday.chain import Columns

from helpers import (HOP, cols, _app, _type, _feed, _walk, _to_game, _to_game_rest, _events, _cards, _labels, _light,
                     _click, _player, _learn, _game, _age, _now, _ridge_app, _bar_heights)


# ---- the main button: Start walk / They've stopped / Next walk, guarded against double presses -----------------------

@pytest.mark.alternate_round
def test_stop_press_ends_the_walk_operator_and_is_logged(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)
    _type(app, ["n", "L", "e", "enter"])
    app.on_key_now("space", 100.0)
    _feed(app, [30.0] * 40, 100.0)                                 # still walking at 2.1 s
    stop_min = float(cfg["walk"]["stop_min_s"])
    app.on_key_now("space", 100.0 + stop_min - 0.5)                # a double press: ignored
    assert app.state == WALKING and app.walk["w"].t_stop is None
    app.on_key_now("space", 100.0 + stop_min + 0.4)                # "They've stopped": its column is already in
    assert app.walk is None and app.state == ENROL_VISITOR
    rec = app.walk_log[-1]
    assert rec["reason"] == "operator" and rec["seen"]
    ev = [e for e in _events(tmp_path) if e["kind"] == "stop"]
    assert len(ev) == 1 and ev[0]["walk_id"] == rec["id"] and abs(ev[0]["t_dev"] - (100.0 + stop_min + 0.4)) < 1e-9


@pytest.mark.alternate_round
def test_start_is_locked_just_after_a_walk_ends(cfg, tmp_path):
    """A late They've stopped click (the walk already ended by itself) starts nothing: the main button is grey for
    walk.after_end_lock_s after the end, and a refused click says so on the operator's line."""
    app = _app(cfg, tmp_path)
    _type(app, ["n", "L", "e", "enter"])
    _walk(app, 100.0)
    end = app.last_end_dev
    lock = float(cfg["walk"]["after_end_lock_s"])
    assert lock >= 2.5                                             # walking back 3 m to the mark takes longer
    vm = app.view(end + 0.3)
    assert vm["buttons"]["start_walk"]["look"] == "off"
    app.rate_hist = collections.deque([(end - 1.0, 0), (end + 0.3, 150)])
    assert app.health(end + 0.3)[:2] == ("grey", "Walk back to the start")    # the light agrees
    app.on_key_now("space", end + lock - 0.3)                      # a late second click
    assert app.walk is None and app.state == ENROL_VISITOR
    assert app.view(end + lock - 0.2)["hint"] == ("Ignored: that walk had already ended by itself.",
                                                  screen.STATUS["amber"])
    assert app.view(end + lock + 0.1)["buttons"]["start_walk"]["look"] == "next"
    app.on_key_now("space", end + lock + 0.1)
    assert app.state == WALKING


@pytest.mark.alternate_round
def test_a_late_click_keeps_the_answer_on_screen(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    t = _to_game(app)
    _walk(app, t)
    app._timers(app.timer + 0.01)
    assert app.state == RESULT
    app.on_key_now("space", app.last_end_dev + 1.5)               # They've stopped, 1.5 s after the walk ended
    assert app.walk is None and app.state == RESULT and app.view(time.monotonic())["card"]["big"] == "Le"


@pytest.mark.alternate_round
def test_lock_ignores_a_new_device_clock_epoch(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    app.last_end_dev = 5000.0                                      # from before a receiver reboot
    app.on_key_now("space", 10.0)
    assert app.state == WALKING


@pytest.mark.alternate_round
def test_main_button_labels_and_looks(cfg, tmp_path):
    app = _app(cfg, tmp_path, baseline=False)
    b = app.view(time.monotonic())["buttons"]["start_walk"]
    assert (b["label"], b["look"], b["key"]) == ("Measure room", "next", "space")
    app = _app(cfg, tmp_path)
    assert app.view(time.monotonic())["buttons"]["start_walk"]["look"] == "on"          # WELCOME: New player is next
    _type(app, ["n", "L", "e", "enter"])
    assert app.view(time.monotonic())["buttons"]["start_walk"]["label"] == "Start walk"
    app.on_key_now("space", 100.0)
    early = app.view(100.5)["buttons"]["start_walk"]
    late = app.view(100.0 + float(cfg["walk"]["stop_min_s"]) + 0.1)["buttons"]["start_walk"]
    assert early["label"] == late["label"] == "They've stopped" and early["look"] == "off" and late["look"] == "next"
    app = _app(cfg, tmp_path / "game")
    t = _to_game(app)
    _walk(app, t)
    assert app.state == DECIDING and app.view(time.monotonic())["buttons"]["start_walk"]["look"] == "off"
    app.state = RESULT
    b = app.view(time.monotonic())["buttons"]["start_walk"]
    assert (b["label"], b["look"]) == ("Next walk", "next")


# ---- keys: the quit box, the name box, the clicker's back key and Ctrl+U ---------------------------------------------

@pytest.mark.alternate_round
def test_buttons_drive_the_same_steps_as_keys(cfg, tmp_path):
    """A click sends its button's key, and the app takes the same step as when that key is pressed: the main button
    (Space) measures the room in set-up and starts a walk later, New player (N) opens the name box, OK sends Enter,
    and in the quit box Keep going sends Escape and Yes, quit sends Y."""
    def same(name, key):
        assert _click(a, name) == [key], name                      # the click on one app ...
        b.on_key_now(key, time.monotonic())                        # ... the key on its twin
        look = lambda x: (x.state, x.name, bool(x.quit_until), x.stop.is_set(), x.walk is not None)  # noqa: E731
        assert look(a) == look(b), name
        return a.state

    a, b = _app(cfg, tmp_path / "a", baseline=False), _app(cfg, tmp_path / "b", baseline=False)
    assert same("start_walk", "space") == BUSY                     # set-up: Measure room
    a, b = _app(cfg, tmp_path / "c"), _app(cfg, tmp_path / "d")
    assert same("new_visitor", "n") == NAME_ENTRY
    _type(a, ["L", "e"])
    _type(b, ["L", "e"])
    assert same("ov_a", "enter") == ENROL_VISITOR and a.name == "Le"   # OK in the name box
    assert same("start_walk", "space") == WALKING
    same("quit", "ctrl+q")
    assert a.quit_until and not a.stop.is_set()
    same("ov_b", "escape")                                         # Keep going
    assert not a.quit_until and not a.stop.is_set()
    same("quit", "ctrl+q")
    same("ov_a", "y")                                              # Yes, quit
    assert a.stop.is_set()


@pytest.mark.alternate_round
def test_end_game_button_wipes_the_player(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    assert app.view(time.monotonic())["buttons"]["wipe_visitor"]["look"] == "off"   # dim: no player yet
    _to_game(app)
    b = app.view(time.monotonic())["buttons"]["wipe_visitor"]
    assert b["look"] == "on" and b["label"] == "End game"
    _type(app, [b["key"]])                                         # asks first, like Quit
    vm = app.view(time.monotonic())
    assert app.name == "Le" and vm["overlay"] == ("End the game?", "Le's name and walks will be forgotten.")
    assert vm["buttons"]["ov_a"]["label"] == "Yes, end it"          # everything behind the box: dim, in place
    assert all(v["look"] == "off" for k, v in vm["buttons"].items() if k not in ("ov_a", "ov_b"))
    assert vm["hint"][0].startswith("Click Yes, end it")
    _type(app, ["space"])                                          # any other key only closes the box
    assert app.name == "Le" and app.state == READY and app.walk is None
    assert app.view(time.monotonic())["overlay"] is None
    _type(app, [b["key"], vm["buttons"]["ov_a"]["key"]])
    assert app.name is None and len(app.visitor) == 0 and app.state == IDLE
    assert app.view(time.monotonic())["hint"][0] == "Player wiped."


@pytest.mark.alternate_round
def test_quit_box_swallows_keys(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    _type(app, ["n", "L", "e", "enter"])
    _type(app, ["ctrl+q"])
    assert app.view(time.monotonic())["overlay"][0] == "Quit?"
    assert app.view(time.monotonic())["hint"][0].startswith("Click Yes, quit")
    app.on_key_now("pagedown", 100.0)                              # a Next while the box shows: closes it, nothing else
    assert app.walk is None and app.state == ENROL_VISITOR and app.view(time.monotonic())["overlay"] is None
    app.on_key_now("space", 101.0)
    assert app.state == WALKING
    _type(app, ["escape"])
    _type(app, ["ctrl+q", "y"])
    assert app.stop.is_set()


@pytest.mark.alternate_round
def test_name_box_next_and_back_keys(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    _type(app, ["n"])
    assert app.state == NAME_ENTRY
    _type(app, ["pagedown", "right", "down"])                      # nothing typed: Next does nothing
    assert app.state == NAME_ENTRY and app.typed == ""
    _type(app, ["L", "e", "pageup", "left", "space"])              # back keys never edit; space types a space
    assert app.typed == "Le "
    _type(app, ["pagedown"])                                       # Next confirms once something is typed
    assert app.state == ENROL_VISITOR and app.name == "Le"
    app.visitor.add({"speed": 1.1, "steps": 100.0, "swing": 2.0, "bounce": 0.3})   # (one of Le's walks)
    _type(app, ["n", "enter"])                                     # Enter with nothing typed: Guest, Le's walks go
    assert app.state == ENROL_VISITOR and app.name == "Guest" and len(app.visitor) == 0


@pytest.mark.alternate_round
def test_back_key_never_throws_away_an_answer(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    _type(app, ["n", "L", "e", "enter"])
    _walk(app, 100.0)
    assert app.done["uzayr"] == 1
    _type(app, ["pageup"])                                         # in LEARNING it redoes the turn
    assert app.done["uzayr"] == 0 and app.last is None and "Same turn again" in app.view(time.monotonic())["card"]["sub"]
    app = _app(cfg, tmp_path / "h")
    t = _to_game(app)
    last = app.last
    _type(app, ["pageup", "left"])                                 # GAME: only a hint
    assert app.last is last and app.state == READY and app.visitor_ready
    assert "Undo walk" in app.view(time.monotonic())["hint"][0]
    _walk(app, t)
    app.state = RESULT
    rec = app.last
    _type(app, ["pageup", "left"])                                 # ANSWER: the back key keeps it
    assert app.last is rec and app.state == RESULT
    _type(app, ["backspace"])                                      # Backspace / Undo walk does remove it
    assert app.last is None and app.state == READY
    assert len(app.visitor) == len(app.uz_fresh) == 3              # (a game walk was in no pool)
    assert app.view(time.monotonic())["card"]["sub"].startswith("Last walk removed")


@pytest.mark.alternate_round
def test_m_and_ctrl_u_do_nothing(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    for setup in ([], ["n", "L", "e", "enter"]):
        _type(app, setup)
        now = time.monotonic()
        before, st = app.view(now)["card"], app.state
        _type(app, ["m", "ctrl+u", "M"])
        assert app.view(now)["card"] == before and app.state == st and not app.uz_enrolling


# ---- the card: turns, notices, dots, who -----------------------------------------------------------------------------

@pytest.mark.alternate_round
def test_missed_turn_keeps_the_turn_and_the_dots(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    _type(app, ["n", "L", "e", "enter"])
    _walk(app, 100.0, seen=False)
    assert app.walk_log[-1]["reason"] == "nothing" and app.state == ENROL_VISITOR
    card = app.view(time.monotonic())["card"]
    assert card["big"] == "Uzayr's turn" and card["sub"].startswith("Missed that one. Same turn again")
    assert card["dots"]["rows"][0] == ("Uzayr", screen.UZAYR, ["now", "next", "next"])
    assert card["dots"]["rows"][1][0] == "Le"
    _walk(app, 120.0, seen=False)
    assert "Missed twice" in app.view(time.monotonic())["hint"][0]
    _walk(app, 140.0)
    card = app.view(time.monotonic())["card"]
    assert card["big"] == "Le's turn" and card["sub"] == "Got it!" and app.misses == 0
    assert card["dots"]["rows"][0][2] == ["done", "next", "next"] and card["dots"]["rows"][1][2][0] == "now"
    assert "Missed twice" not in app.view(time.monotonic())["hint"][0]


@pytest.mark.alternate_round
def test_missed_twice_lights_amber_keeps_undo_and_goes_with_the_player(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    _type(app, ["n", "L", "e", "enter"])
    _walk(app, 100.0)                                              # Uzayr's turn, seen
    _walk(app, 120.0, seen=False)
    _walk(app, 140.0, seen=False)
    assert app.misses == 2 and _light(app) == ("amber", "Missed twice")
    assert app.view(time.monotonic())["buttons"]["discard"]["look"] == "on"   # a miss keeps Undo walk
    _type(app, ["backspace"])                                      # it removes Uzayr's turn (spoiled, say)
    assert app.done["uzayr"] == 0 and app.last is None
    _type(app, ["n", "S", "i", "enter"])                           # the next player starts clean
    assert app.misses == 0 and _light(app) == ("green", "Ready")
    assert "Missed twice" not in app.view(time.monotonic())["hint"][0]


@pytest.mark.alternate_round
def test_light_says_measure_again_and_never_label_the_last_walk(cfg, tmp_path):
    r = _app(cfg, tmp_path / "r", record=True)
    _type(r, ["n", "L", "e", "enter"])
    _walk(r, 100.0)
    assert _light(r) == ("green", "Ready")                         # a recording waits for no label
    t = _to_game_rest(r, 120.0, 5)
    _walk(r, t)
    r.state = RESULT
    assert _light(r) == ("green", "Ready")
    a = _app(cfg, tmp_path / "a")
    a._hint("Room was busy: click Measure room again.", "amber", 15.0, light="Measure the room again")
    assert _light(a) == ("amber", "Measure the room again")


@pytest.mark.alternate_round
def test_who_is_playing(cfg, tmp_path):
    """The top line says who is playing ("Uzayr vs {V}"), nothing before a name."""
    app = _app(cfg, tmp_path)
    vm = app.view(time.monotonic())
    assert vm["who"] == ""
    _type(app, ["n", "L", "e", "enter"])
    assert app.view(time.monotonic())["who"] == "Uzayr vs Le"
    _to_game_rest(app)
    assert app.view(time.monotonic())["who"] == "Uzayr vs Le"


@pytest.mark.alternate_round
def test_got_you_both_and_thanks_notices(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)                         # Thanks: only in a test recording
    t = _to_game(app)
    card = app.view(time.monotonic())["card"]
    assert card["big"] == "Who's walking?" and card["sub"].startswith("Got you both!")
    _walk(app, t)
    assert app.view(time.monotonic())["card"]["big"] == "Thinking …"
    app.state = RESULT
    card = app.view(time.monotonic())["card"]
    assert card["big"] == "Le" and not card.get("sub") and "conf" in card   # nothing under the name: it is bigger
    _type(app, ["u"])
    card = app.view(time.monotonic())["card"]
    # against the answer: not green
    assert card["sub"] == "Thanks: that was Uzayr." and card["sub_color"] == screen.INK2
    assert card["big"] == "Le"                                                # the answer stays on screen
    _walk(app, t + 20)
    app.state = RESULT
    _type(app, ["v"])
    assert app.view(time.monotonic())["card"]["sub_color"] == screen.STATUS["green"]


@pytest.mark.alternate_round
def test_pulls_only_with_the_answer(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    t = _to_game(app)
    _walk(app, t)
    assert app.state == DECIDING                                   # "Thinking": nothing gives the answer away
    assert not any(tl.get("pull") for tl in app.view(time.monotonic())["tiles"].values())
    app.state = RESULT
    assert app.view(time.monotonic())["tiles"]["speed"]["pull"] == "like Le"


@pytest.mark.alternate_round
def test_learning_hint_on_uzayrs_own_turn(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    _type(app, ["n", "L", "e", "enter"])
    assert app.view(time.monotonic())["hint"][0] == "Click Start walk, then walk to the X yourself."
    _walk(app, 100.0)
    assert app.view(time.monotonic())["hint"][0] == "Click Start walk as Le sets off."


# ---- labels: learning goes by turn; on the day no It was ... at all; in a test recording optional and final ----------

@pytest.mark.alternate_round
def test_learning_needs_no_label_in_a_recording_and_writes_the_turn(cfg, tmp_path):
    """Every learning walk is labelled by its turn, also under --record: no It was ... buttons, Start walk is
    never held back, and the label event a click used to write is written with source "turn"."""
    app = _app(cfg, tmp_path, record=True)
    _type(app, ["n", "L", "e", "enter"])
    for i in range(6):
        b = app.view(time.monotonic())["buttons"]
        assert "confirm_uzayr" not in b and "confirm_visitor" not in b, i
        _walk(app, 100.0 + 20 * i)                                 # (asserts the walk starts: nothing to label)
    assert app.state == READY and app.visitor_ready and len(app.visitor) == len(app.uz_fresh) == 3
    walks = [e for e in _events(tmp_path) if e["kind"] == "walk"]
    labs = _labels(tmp_path)
    assert [e["walk_id"] for e in labs] == [e["id"] for e in walks]
    assert [e["label"] for e in labs] == ["u", "v"] * 3 and all(e["source"] == "turn" for e in labs)
    assert [e["label"] for e in walks] == ["u", "v"] * 3        # (the walk event carries it too)
    _type(app, ["u", "v"])                                         # U / V on a learning walk: a hint, nothing else
    assert len(_labels(tmp_path)) == 6 and app.view(time.monotonic())["hint"][0].startswith("Learning walks go by turn")


@pytest.mark.alternate_round
def test_a_wrong_turn_is_redone_with_undo_walk(cfg, tmp_path):
    """The visitor walked on Uzayr's turn: Undo walk removes that walk (a discard event, so the bench tools
    skip it) and the same turn is walked again."""
    app = _app(cfg, tmp_path, record=True)
    _type(app, ["n", "L", "e", "enter"])
    _walk(app, 100.0)
    wid = app.last["id"]
    _type(app, ["backspace"])
    assert app.done["uzayr"] == 0 and len(app.uz_fresh) == 0 and app.view(time.monotonic())["card"]["big"] == "Uzayr's turn"
    assert any(e["kind"] == "discard" and e["walk_id"] == wid for e in _events(tmp_path))
    _walk(app, 120.0)
    assert app.done["uzayr"] == 1 and _labels(tmp_path)[-1]["walk_id"] == app.last["id"]


@pytest.mark.alternate_round
def test_on_the_day_no_it_was_buttons_and_u_v_do_nothing(cfg, tmp_path):
    """Without --record there is no It was ... anywhere, U / V change nothing, game walks join no pool and
    nothing is tallied; the hint no longer mentions It was."""
    app = _app(cfg, tmp_path)
    _walk(app, 50.0)                                               # a speed-only walk on the welcome screen
    assert app.state == IDLE and not {"confirm_uzayr", "confirm_visitor"} & set(app.view(time.monotonic())["buttons"])
    t = _to_game(app)
    for k in range(3):
        _walk(app, t + 20 * k)
        app._timers(app.timer + 0.01)
        assert app.state == RESULT
        vm = app.view(time.monotonic())
        assert not {"confirm_uzayr", "confirm_visitor"} & set(vm["buttons"]), vm["buttons"]
        assert "It was" not in vm["hint"][0] and vm["hint"][0] == "Click Next walk as they set off."
        rec, card = app.last, vm["card"]
        _type(app, ["u", "v", "U", "V"])
        assert rec["label"] is None and not rec.get("final") and app.notice is None
        assert app.view(time.monotonic())["card"] == card
    assert len(app.uz_fresh) == len(app.visitor) == 3
    assert (app.tally.doc["right"], app.tally.doc["wrong"], app.tally.doc["not_sure"]) == (0, 0, 0)


@pytest.mark.alternate_round
def test_in_a_recording_it_was_is_optional_one_click_and_final(cfg, tmp_path):
    """On a game answer It was Uzayr / It was {V} show but never hold up the next walk; one click writes
    the label event (counted right or wrong) and thanks; both buttons go dim; a second click or key only says
    so; the walk joins no pool and nothing is tallied."""
    app = _app(cfg, tmp_path, record=True)
    t = _to_game(app)
    _walk(app, t)
    app._timers(app.timer + 0.01)
    b = app.view(time.monotonic())["buttons"]
    assert b["confirm_uzayr"]["look"] == b["confirm_visitor"]["look"] == "on" and b["start_walk"]["look"] == "next"
    assert b["confirm_visitor"]["label"] == "It was Le" and b["new_visitor"]["look"] == "on"
    n0 = len(_labels(tmp_path))
    unlabelled = app.last
    _walk(app, t + 20)                                             # Next walk without a label: fine
    app._timers(app.timer + 0.01)
    assert unlabelled["label"] is None and len(_labels(tmp_path)) == n0
    rec = app.last
    _type(app, ["v"])                                              # DEC says visitor: right
    assert rec["label"] == "v" and rec["final"] and rec["pool"] is None and rec["entry"] is None
    lab = _labels(tmp_path)[-1]
    assert (lab["walk_id"], lab["label"], lab["counted"]) == (rec["id"], "v", "right") and "source" not in lab
    vm = app.view(time.monotonic())
    assert vm["card"]["sub"] == "Thanks: that was Le." and vm["card"]["big"] == "Le"
    assert vm["buttons"]["confirm_uzayr"]["look"] == vm["buttons"]["confirm_visitor"]["look"] == "dim"
    assert vm["buttons"]["confirm_visitor"]["label"] == "It was Le \u2713"    # the one clicked is ticked
    assert vm["buttons"]["confirm_uzayr"]["label"] == "It was Uzayr"
    assert vm["buttons"]["start_walk"]["look"] == "next"
    _type(app, ["u", "v"])                                         # final: nothing changes but a hint
    assert rec["label"] == "v" and len(_labels(tmp_path)) == n0 + 1
    assert app.view(time.monotonic())["hint"][0] == "Already noted. Undo walk removes the walk."
    assert len(app.uz_fresh) == len(app.visitor) == 3
    assert (app.tally.doc["right"], app.tally.doc["wrong"], app.tally.doc["not_sure"]) == (0, 0, 0)
    _walk(app, t + 40)
    app._timers(app.timer + 0.01)
    _type(app, ["u"])                                              # against the answer: counted wrong
    assert _labels(tmp_path)[-1]["counted"] == "wrong"
    assert app.view(time.monotonic())["card"]["sub_color"] == screen.INK2


@pytest.mark.alternate_round
def test_a_dim_it_was_button_click_only_says_already_noted(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)
    t = _to_game(app)
    _walk(app, t)
    app._timers(app.timer + 0.01)
    app.screen.render(app.view(time.monotonic()))

    def click(name):
        x0, y0, x1, y1 = app.screen.buttons[name]["box"]
        canvas = app.screen.fig.canvas
        px, py = app.screen.fig.transFigure.transform(((x0 + x1) / 2, (y0 + y1) / 2))
        canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", canvas, px, py, button=1))
        while not app.keys.empty():
            k, tk = app.keys.get_nowait()
            app.on_key_now(k, tk)
        app.screen.render(app.view(time.monotonic()))

    click("confirm_uzayr")
    assert app.last["label"] == "u" and len(_labels(tmp_path)) == 7
    app.screen._last_click = ("", -1e9)                            # (past the double-click window)
    click("confirm_visitor")
    assert app.last["label"] == "u" and len(_labels(tmp_path)) == 7
    assert app.view(time.monotonic())["hint"][0] == "Already noted. Undo walk removes the walk."


@pytest.mark.alternate_round
def test_undo_walk_after_a_label_removes_the_walk(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)
    t = _to_game(app)
    _walk(app, t)
    app.state = RESULT
    rec = app.last
    _type(app, ["v", "backspace"])
    assert app.last is None and app.state == READY
    assert any(e["kind"] == "discard" and e["walk_id"] == rec["id"] for e in _events(tmp_path))
    assert len(app.uz_fresh) == len(app.visitor) == 3
    assert (app.tally.doc["right"], app.tally.doc["wrong"], app.tally.doc["not_sure"]) == (0, 0, 0)


@pytest.mark.alternate_round
def test_speed_only_walk_in_a_recording_offers_it_was_uzayr_once(cfg, tmp_path):
    """A walk with no player: on the day no It was; in a recording It was Uzayr only, optional, final."""
    day = _app(cfg, tmp_path / "day")
    _walk(day, 50.0)
    assert "confirm_uzayr" not in day.view(time.monotonic())["buttons"]
    app = _app(cfg, tmp_path / "rec", record=True)
    _walk(app, 50.0)
    assert app.state == IDLE and app.last["purpose"] == "free"
    b = app.view(time.monotonic())["buttons"]
    assert b["confirm_uzayr"]["look"] == "on" and "confirm_visitor" not in b and b["start_walk"]["look"] == "on"
    _type(app, ["v"])                                              # no player: V does nothing
    assert app.last["label"] is None
    _type(app, ["u"])
    lab = [e for e in _events(tmp_path / "rec") if e["kind"] == "label"]
    assert len(lab) == 1 and lab[0]["label"] == "u" and lab[0]["counted"] is None
    assert app.view(time.monotonic())["buttons"]["confirm_uzayr"]["look"] == "dim"
    _type(app, ["u"])
    assert len([e for e in _events(tmp_path / "rec") if e["kind"] == "label"]) == 1
    assert len(app.uz_fresh) == 0                                  # no pool


@pytest.mark.alternate_round
def test_it_was_hides_after_a_missed_walk(cfg, tmp_path):
    """A walk that was missed came after the speed-only answer: It was Uzayr no longer offers to label the walk
    before it (a click would have labelled the older walk)."""
    app = _app(cfg, tmp_path, record=True)
    _walk(app, 50.0)
    assert "confirm_uzayr" in app.view(time.monotonic())["buttons"]
    _walk(app, 70.0, seen=False)
    assert app.state == IDLE and "confirm_uzayr" not in app.view(time.monotonic())["buttons"]
    _type(app, ["u"])
    assert not [e for e in _events(tmp_path) if e["kind"] == "label"]


@pytest.mark.alternate_round
def test_undo_walk_key_does_nothing_while_thinking(cfg, tmp_path):
    """While "Thinking ..." Undo walk is dim, and Backspace and the clicker's back key do nothing either."""
    app = _app(cfg, tmp_path)
    t = _to_game(app)
    _walk(app, t)
    assert app.state == DECIDING
    rec = app.last
    _type(app, ["backspace", cfg["keys"]["clicker_back"][0]])
    assert app.state == DECIDING and app.last is rec and app.hint_ev is None
    app._timers(app.timer + 0.01)
    assert app.state == RESULT
    _type(app, ["backspace"])                                      # with the answer shown: Undo walk works
    assert app.last is None and app.state == READY


@pytest.mark.alternate_round
def test_last_walk_bar_colour_only_for_learning_walks(cfg, tmp_path):
    """The Last walk bars take the colour of whose turn it was in learning; a game walk's take the last-walk colour
    (screen.THIS, yellow), also after an It was ... click in a test recording (so a recording looks like the day)."""
    app = _app(cfg, tmp_path, record=True)
    _type(app, ["n", "L", "e", "enter"])
    _walk(app, 100.0)
    assert app.view(time.monotonic())["tiles"]["speed"]["this_color"] == screen.UZAYR
    _walk(app, 120.0)
    assert app.view(time.monotonic())["tiles"]["speed"]["this_color"] == screen.VISITOR
    _to_game_rest(app, 140.0, 4)
    _walk(app, 300.0)
    app._timers(app.timer + 0.01)
    _type(app, ["v"])
    assert app.last["label"] == "v" and app.view(time.monotonic())["tiles"]["speed"]["this_color"] == screen.THIS


@pytest.mark.alternate_round
def test_a_close_call_still_names_the_nearer_person(cfg, tmp_path):
    """Below tau the card still names the nearer person, with How sure; the margin and not_sure stay in the log
    and the events; a label counts right or wrong, never not sure."""
    close = models.Decision(label="uzayr", margin=0.01, not_sure=True, dist={},
                            contrib={"speed": 0.1, "steps": 0.0, "swing": 0.0, "bounce": 0.0}, why=["speed"])
    log = tmp_path / "run.log"
    app = _app(cfg, tmp_path, record=True, log=str(log))
    app._decide = lambda f: close
    t = _to_game(app)
    _walk(app, t)
    app._timers(app.timer + 0.01)
    card = app.view(time.monotonic())["card"]
    assert card["big"] == "Uzayr" and "conf" in card and card["conf_text"].startswith("How sure: ")
    assert not any("can't tell" in x.lower() or "too close" in x.lower() for x in _cards(app.view(time.monotonic())))
    _type(app, ["v"])
    assert _labels(tmp_path)[-1]["counted"] == "wrong"
    ev = [e for e in _events(tmp_path) if e["kind"] == "walk"][-1]
    assert ev["decision"]["not_sure"] is True and ev["decision"]["margin"] == pytest.approx(0.01)
    assert "margin 0.010 not sure" in open(log).read()


# ---- the room measurement: the countdown, the room level, someone moving, and the 1.3x check -------------------------

def _capture_cols(app, t0, t1, scale, seed, loud=None):
    """Columns of the room into the running capture; loud = (from, to, x): someone moving then, x the power."""
    rng = np.random.default_rng(seed)
    t = np.arange(t0, t1, HOP)
    p = scale * rng.exponential(1.0, (len(t), len(app.chain.freqs)))
    if loud is not None:
        p[(t >= loud[0]) & (t < loud[1])] *= loud[2]
    app.capture["cols"].append(Columns(t, p, np.zeros(len(t), bool)))


@pytest.mark.alternate_round
def test_setup_next_measures_after_a_countdown(cfg, tmp_path):
    app = _app(cfg, tmp_path, baseline=False, record=True)
    assert app.state == NEEDS_BASELINE
    app.on_key_now("space", 50.0)                                  # the main button in SET-UP
    d, secs = float(cfg["baseline"]["delay_s"]), float(cfg["baseline"]["bench_s"])
    cap = app.capture
    assert app.state == BUSY and cap["t0"] == 50.0 + d and cap["t1"] == 50.0 + d + secs
    ev = [e for e in _events(tmp_path) if e["kind"] == "baseline_start"]
    assert ev[-1]["t_mono"] == 50.0 + d and ev[-1]["t_dev"] == 50.0 + d and ev[-1]["secs"] == secs
    assert app.view(time.monotonic())["card"]["title"].startswith("Step back from the boards")
    _capture_cols(app, 50.0, 53.0, 1.0, 0)
    assert app.view(time.monotonic())["card"]["title"].endswith("\u2026 3 s")            # 2.01 s left
    _capture_cols(app, 53.0, 50.0 + d + 2.0, 1.0, 1)
    assert app.view(time.monotonic())["card"]["title"].startswith("Hold still: measuring the room")
    _capture_cols(app, 50.0 + d + 2.0, 50.0 + d + secs + 0.5, 1.0, 2)
    app._timers(time.monotonic())
    assert app.state == IDLE and app.chain.baseline is not None
    assert app.view(time.monotonic())["hint"][0] == f"Room level {app.chain.baseline.e0:.2f}."
    e0 = app.chain.baseline.e0
    app.on_key_now("ctrl+b", 200.0)                                # again, with the room busy (3x the energy)
    _capture_cols(app, 200.0, 200.0 + d + secs + 0.5, 3.0, 3)
    app._timers(time.monotonic())
    assert app.chain.baseline.e0 > 1.3 * e0 and app.quiet_e0 == e0
    hint = app.view(time.monotonic())["hint"]
    assert hint[0].startswith("Room was busy") and hint[1] == screen.STATUS["amber"]
    assert app.hint_ev[3] == "Measure the room again"             # the light says it too


MOVED = "Someone moved: using the quiet part. Measure room again when everyone is still."


@pytest.mark.alternate_round
def test_the_room_level_after_each_measurement(cfg, tmp_path):
    """After Measure room the operator's line gives the room level, green, and the run's quietest level when an
    earlier measurement was quieter. Someone moving while it measured (3 s of the 30): amber for 20 s, the light
    says "Measure the room again", the quiet part is the room level, and it never counts as the run's quietest.
    The baseline_end event says which it was."""
    app = _app(cfg, tmp_path, baseline=False, record=True)
    d, secs = float(cfg["baseline"]["delay_s"]), float(cfg["baseline"]["bench_s"])
    green, amber = screen.STATUS["green"], screen.STATUS["amber"]

    def measure(t, scale, seed, loud=None):
        app.on_key_now("ctrl+b", t)
        _capture_cols(app, t, t + d + secs + 0.5, scale, seed, loud)
        app._timers(time.monotonic())
        return app.chain.baseline, app.view(time.monotonic())["hint"]

    b1, hint = measure(50.0, 1.0, 0)
    assert hint == (f"Room level {b1.e0:.2f}.", green) and _light(app) == ("green", "Ready")
    b2, hint = measure(200.0, 1.2, 1)                              # a little busier (under 1.3x): its quietest too
    assert b1.e0 < b2.e0 < 1.3 * b1.e0
    assert hint == (f"Room level {b2.e0:.2f} (quietest today {b1.e0:.2f}).", green) and _light(app)[0] == "green"
    b3, hint = measure(400.0, 1.0, 2, loud=(400.0 + d + 10.0, 400.0 + d + 13.0, 50.0))
    assert b3.disturbed and b3.d >= float(cfg["baseline"]["quiet_flag_d"]) and b3.e0 < 1.3 * b1.e0
    assert hint == (MOVED, amber) and _light(app) == ("amber", "Measure the room again")
    assert app.hint_ev[2] - time.monotonic() > 19.0 and app.quiet_e0 == b1.e0
    b4, hint = measure(600.0, 0.8, 3)                              # the quietest yet: just its number
    assert hint == (f"Room level {b4.e0:.2f}.", green) and app.quiet_e0 == b4.e0
    ev = [e for e in _events(tmp_path) if e["kind"] == "baseline_end"]
    assert [(e["e0"], e["disturbed"], e["d"]) for e in ev] == [(b.e0, b.disturbed, b.d) for b in (b1, b2, b3, b4)]
    assert [e["disturbed"] for e in ev] == [False, False, True, False]


@pytest.mark.alternate_round
def test_a_room_measurement_someone_moved_in_is_not_called_busy(cfg, tmp_path):
    """Someone moved while measuring and even the quiet part is over 1.3x the run's quietest: the line still says
    someone moved (it already asks to measure again), not "Room was busy"."""
    app = _app(cfg, tmp_path, baseline=False, record=True)
    d, secs = float(cfg["baseline"]["delay_s"]), float(cfg["baseline"]["bench_s"])
    app.on_key_now("ctrl+b", 50.0)
    _capture_cols(app, 50.0, 50.0 + d + secs + 0.5, 1.0, 0)
    app._timers(time.monotonic())
    e0 = app.chain.baseline.e0
    app.on_key_now("ctrl+b", 200.0)
    _capture_cols(app, 200.0, 200.0 + d + secs + 0.5, 2.0, 1, loud=(200.0 + d + 5.0, 200.0 + d + 8.0, 50.0))
    app._timers(time.monotonic())
    assert app.chain.baseline.disturbed and app.chain.baseline.e0 > 1.3 * e0 and app.quiet_e0 == e0
    assert app.view(time.monotonic())["hint"] == (MOVED, screen.STATUS["amber"])
    assert app.hint_ev[3] == "Measure the room again"


@pytest.mark.alternate_round
def test_a_room_measurement_saved_by_the_older_app_is_offered(cfg, tmp_path):
    """state/baseline_live.json as the app wrote it before disturbed and d existed: loaded at start-up as a still
    room's measurement and offered by Use last measurement."""
    n = int(cfg["dsp"]["nfft"]) // 2 + 1
    with open(tmp_path / "baseline_live.json", "w", encoding="utf-8") as f:
        json.dump({"floor": [1.0] * n, "e0": 0.7, "n_cols": 187, "variant": cfg["dsp"]["variant"],
                   "nfft": int(cfg["dsp"]["nfft"]), "created": time.time() - 120}, f)
    app = App(cfg, state_dir=str(tmp_path), log=RunLog(None), headless=True)
    b = app.cached
    assert b is not None and b.e0 == 0.7 and not b.disturbed and b.d == 0.0
    _type(app, [app.view(time.monotonic())["buttons"]["reuse_baseline"]["key"]])
    assert app.chain.baseline is b and app.quiet_e0 == 0.7
    assert app.view(time.monotonic())["hint"][0] == "Using the last measurement."


@pytest.mark.alternate_round
def test_setup_reuses_the_cached_measurement_by_button(cfg, tmp_path):
    app = _app(cfg, tmp_path, baseline=False)
    app.cached = chain.Baseline(np.ones(len(app.chain.freqs)), 2.0, 100, app.chain.spec.variant,
                              app.chain.spec.nfft, created=time.time() - 120)
    b = app.view(time.monotonic())["buttons"]["reuse_baseline"]
    assert b["look"] == "on" and b["key"] == "enter" and "2 min" in b["label"]
    _type(app, [b["key"]])
    assert app.chain.baseline is app.cached and app.state == IDLE
    assert app.view(time.monotonic())["hint"][0] == "Using the last measurement."


@pytest.mark.alternate_round
def test_a_measurement_that_got_too_old_is_not_offered(cfg, tmp_path):
    app = _app(cfg, tmp_path, baseline=False)
    old = 60.0 * (float(cfg["baseline"]["max_reuse_min"]) + 1)   # fresh at start-up, too old by now
    app.cached = chain.Baseline(np.ones(len(app.chain.freqs)), 2.0, 100, app.chain.spec.variant,
                              app.chain.spec.nfft, created=time.time() - old)
    assert "reuse_baseline" not in app.view(time.monotonic())["buttons"]
    _type(app, ["enter"])
    assert app.chain.baseline is None and app.state == NEEDS_BASELINE


@pytest.mark.alternate_round
def test_state_repairs(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    for st in (WALKING, BUSY, RESULT, DECIDING):
        app.state, app.walk, app.capture, app.last = st, None, None, None
        app._timers(time.monotonic())
        assert app.state == IDLE, st


# ---- the light -------------------------------------------------------------------------------------------------------

@pytest.mark.alternate_round
def test_light(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    now = time.monotonic()
    app.last_pkt_wall = now
    app.rate_hist = collections.deque([(now - 1.0, 0), (now, 150)])
    assert app.health(now)[:2] == ("green", "Ready")
    for _ in range(40):                                            # someone moving near the boards: 5x e0
        app.ring.append((np.zeros(app.n_bins, np.float32), np.nan, 5.0))
    app.last_end_wall = now - 2.0
    assert app.health(now)[:2] == ("grey", "Walk back to the start")
    app.last_end_wall, app.busy_since = None, now - 1.0
    assert app.health(now)[:2] == ("green", "Ready")
    app.busy_since = now - float(cfg["health"]["busy_hold_s"]) - 0.1
    assert app.health(now)[:2] == ("amber", "Wait for green")
    app.state = WALKING
    assert app.health(now)[:2] == ("grey", "Watching")
    app.state = IDLE
    app.last_pkt_wall = now - 5.0
    assert app.health(now)[0] == "red"


# ---- the order: the visitor's walks first, then Uzayr's; alternate still selectable ----------------------------------

def _n(cfg):
    return int(cfg["pools"]["n_enrol"])


@pytest.mark.shipped
def test_shipped_round_is_two_each_visitor_first(cfg):
    assert (cfg["pools"]["n_enrol"], cfg["pools"]["enrol_order"]) == (2, "visitor_first")
    assert cfg["pools"]["uzayr_reuse_min"] == 25                   # (the age study)


@pytest.mark.shipped
def test_visitor_walks_first_then_uzayr_then_the_game(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)
    _player(app)
    card = app.view(time.monotonic())["card"]
    assert card["big"] == "Le's turn" and card["big_color"] == screen.VISITOR
    assert card["dots"]["rows"] == [("Le", screen.VISITOR, ["now", "next"]), ("Uzayr", screen.UZAYR, ["next", "next"])]
    assert app.view(time.monotonic())["hint"][0] == "Click Start walk as Le sets off."
    _walk(app, 100.0)
    assert app.view(time.monotonic())["card"]["sub"] == "Got it! Once more."     # the card says what comes next
    _walk(app, 120.0)
    card = app.view(time.monotonic())["card"]
    assert card["sub"] == "Got it! Now Uzayr walks twice."
    assert card["big"] == "Uzayr's turn" and card["big_color"] == screen.UZAYR
    assert card["dots"]["rows"][0][2] == ["done", "done"] and card["dots"]["rows"][1][2] == ["now", "next"]
    assert app.view(time.monotonic())["hint"][0] == "Click Start walk, then walk to the X yourself."
    _walk(app, 140.0)
    assert app.view(time.monotonic())["card"]["sub"] == "Got it! Once more."
    t, who = _learn(app, 160.0)
    assert who == ["uzayr"] and app.state == READY and app.visitor_ready
    assert app.view(time.monotonic())["card"]["sub"] == "Got you both! Now the guessing game."
    assert len(app.visitor) == len(app.uz_fresh) == 2
    labs = _labels(tmp_path)
    assert [e["label"] for e in labs] == ["v", "v", "u", "u"] and all(e["source"] == "turn" for e in labs)
    _game(app, t)
    assert len(app.uz_fresh) == len(app.visitor) == 2              # a game walk joins no pool


@pytest.mark.shipped
def test_alternate_order_keeps_the_old_turns(cfg, tmp_path):
    cfg["pools"]["enrol_order"] = "alternate"
    app = _app(cfg, tmp_path)
    _player(app)
    assert app.view(time.monotonic())["card"]["dots"]["rows"][0][0] == "Uzayr"   # Uzayr's row on top, as before
    _, who = _learn(app, 100.0)
    assert who == ["uzayr", "visitor"] * _n(cfg) and app.state == READY


@pytest.mark.shipped
def test_a_bad_order_is_refused(cfg, tmp_path):
    cfg["pools"]["enrol_order"] = "uzayr_first"
    with pytest.raises(ValueError):
        _app(cfg, tmp_path)


@pytest.mark.shipped
def test_undo_walk_redoes_uzayrs_turn_too(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)
    _player(app)
    t, _ = _learn(app, 100.0)
    wid = app.last["id"]
    assert app.last["label"] == "u" and app.state == READY
    _type(app, ["backspace"])                                      # Uzayr's last walk was spoiled
    assert app.state == ENROL_VISITOR and app._turn() == "uzayr" and app.done["uzayr"] == _n(cfg) - 1
    assert len(app.uz_fresh) == _n(cfg) - 1 and app.view(time.monotonic())["card"]["big"] == "Uzayr's turn"
    assert any(e["kind"] == "discard" and e["walk_id"] == wid for e in _events(tmp_path))
    _walk(app, t)
    assert app.state == READY and app.view(time.monotonic())["card"]["sub"].startswith("Got you both!")


# ---- Uzayr's walks kept across players while fresh -------------------------------------------------------------------

@pytest.mark.shipped
def test_fresh_walks_are_reused_so_only_the_visitor_walks(cfg, tmp_path):
    log = tmp_path / "run.log"
    app = _app(cfg, tmp_path, record=True, log=str(log))
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    t = _game(app, t)
    uz_ids = [w["walk_id"] for w in app.uz_fresh.walks]
    entries = [w["id"] for w in app.uz_fresh.walks]
    _age(app, 6.0)
    assert app.view(time.monotonic())["hint"][0] == "Click Next walk as they set off. Uzayr's walks: 6 min old."
    _player(app, "Si")
    assert app.uz_reused and len(app.visitor) == 0 and [w["id"] for w in app.uz_fresh.walks] == entries
    card = app.view(time.monotonic())["card"]
    assert card["big"] == "Si's turn"
    assert card["dots"]["rows"] == [("Si", screen.VISITOR, ["now", "next"]), ("Uzayr", screen.UZAYR, ["done", "done"])]
    t, who = _learn(app, t)
    assert who == ["visitor"] * _n(cfg) and app.state == READY and app.visitor_ready
    assert app.view(time.monotonic())["card"]["sub"] == "Got it! Now the guessing game."
    _game(app, t)
    assert [w["id"] for w in app.uz_fresh.walks] == entries        # the game walks never join his pool
    ev = [e for e in _events(tmp_path) if e["kind"] == "uzayr_reused"]
    assert len(ev) == 1 and ev[0]["walk_ids"] == uz_ids and ev[0]["index"] == app.vidx
    assert 5.9 < ev[0]["age_min"] < 6.2
    assert "Uzayr's walks reused (6." in log.read_text() and "Si" not in log.read_text()


@pytest.mark.shipped
@pytest.mark.parametrize("why", ["old", "measured", "used last", "restart", "half"])
def test_walks_are_not_reused_when_stale(cfg, tmp_path, why):
    """Too old, taken before a room measurement (Measure room, Use last measurement), lost with an app restart,
    or not a whole set: the next player's round asks for Uzayr's walks again, after the visitor's."""
    app = _app(cfg, tmp_path)
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    if why == "old":
        _age(app, float(cfg["pools"]["uzayr_reuse_min"]) + 0.1)
    elif why in ("measured", "used last"):
        n = len(app.chain.freqs)
        b = chain.Baseline(np.ones(n), 1.0, 100, app.chain.spec.variant, app.chain.spec.nfft, created=time.time())
        app.prev = app.state                                       # (as Measure room / the set-up button do)
        app._set_baseline(b, "captured" if why == "measured" else "reused")
        assert app._uz_note() == " Uzayr walks next round."
    elif why == "restart":
        app.close()
        app = _app(cfg, tmp_path)                                  # a new app run: nothing kept
        assert len(app.uz_fresh) == 0 and app.done["uzayr"] == 0
    else:
        _type(app, ["backspace"])                                  # his last walk undone, then a new player
    _player(app, "Si")
    assert not app.uz_reused and len(app.uz_fresh) == 0 and app.done["uzayr"] == 0
    _, who = _learn(app, t)
    assert who == ["visitor"] * _n(cfg) + ["uzayr"] * _n(cfg)
    assert app.view(time.monotonic())["card"]["sub"].startswith("Got you both!")


def _measure(app):
    """Measure room (or Use last measurement) finished, from the state the app is in."""
    n = len(app.chain.freqs)
    app.prev = app.state
    app._set_baseline(chain.Baseline(np.ones(n), 1.0, 100, app.chain.spec.variant, app.chain.spec.nfft), "captured")


@pytest.mark.shipped
def test_measure_room_while_the_visitor_learns_asks_for_his_walks(cfg, tmp_path):
    """Reused walks, then Measure room before the game: this round asks for Uzayr's walks after the visitor's (and
    the recording says so, for the bench tools)."""
    app = _app(cfg, tmp_path, record=True)
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    _player(app, "Si")
    assert app.uz_reused
    _walk(app, t)
    _measure(app)
    assert not app.uz_reused and app.done["uzayr"] == 0 and app.state == ENROL_VISITOR
    _, who = _learn(app, t + 20)
    assert who == ["visitor"] * (_n(cfg) - 1) + ["uzayr"] * _n(cfg)
    assert [e["why"] for e in _events(tmp_path) if e["kind"] == "uzayr_dropped"] == ["room"]
    rounds, _ = bench.bench_rounds(_events(tmp_path))
    assert rounds[1]["reused"] == []                               # (not "reused" any more)


@pytest.mark.shipped
@pytest.mark.parametrize("when", ["his turns", "redo, no player", "redo, game"])
def test_measure_room_halfway_through_his_walks_takes_both_again(cfg, tmp_path, when):
    """Measure room after one of his two walks (his turns in a fresh round, or Redo my walks): that walk is from
    the old room, so he walks both again; the new set is then reused by the next player."""
    app = _app(cfg, tmp_path, record=True)
    t = 100.0
    if when == "his turns":
        _player(app, "Le")
        for _ in range(_n(cfg) + 1):                               # the visitor's walks, then one of his
            _walk(app, t)
            t += 20.0
    else:
        if when == "redo, game":
            _player(app, "Le")
            t, _ = _learn(app, t)
            t = _game(app, t)
        _type(app, ["ctrl+u"])
        _walk(app, t)
        t += 20.0
    assert app.done["uzayr"] == 1 and app.last["label"] == "u"
    _measure(app)
    assert app.done["uzayr"] == 0 and len(app.uz_fresh) == 0 and app.last is None
    assert app._turn() == "uzayr" and app.state in (ENROL_VISITOR, ENROL_UZAYR)
    t, who = _learn(app, t)
    assert who == ["uzayr"] * _n(cfg)
    assert app._uz_reusable() and app._uz_note() == " Uzayr's walks: 1 min old."
    ids = [w["walk_id"] for w in app.uz_fresh.walks]
    _player(app, "Si")
    assert app.uz_reused
    t, _ = _learn(app, t)
    t = _game(app, t)
    _, pools = bench.bench_rounds(_events(tmp_path))
    assert pools[app.last["id"]]["u"] == ids                       # the bench tools agree with the app


@pytest.mark.shipped
def test_reuse_off_he_walks_every_round(cfg, tmp_path):
    cfg["pools"]["uzayr_reuse_min"] = 0
    app = _app(cfg, tmp_path)
    assert "enrol_uzayr" not in app.view(time.monotonic())["buttons"] and "enrol_uzayr" not in app.screen.buttons
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    _type(app, ["ctrl+u"])                                         # nothing to redo
    assert app.state == READY and len(app.uz_fresh) == _n(cfg)
    _player(app, "Si")
    assert len(app.uz_fresh) == 0 and not app.uz_reused
    _, who = _learn(app, t)
    assert who.count("uzayr") == _n(cfg)
    assert "Uzayr's walks" not in _now(app)["hint"][0] and "next round" not in _now(app)["hint"][0]


# ---- Redo my walks ---------------------------------------------------------------------------------------------------

@pytest.mark.shipped
def test_redo_my_walks_in_the_game(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    t = _game(app, t)
    old = [w["walk_id"] for w in app.uz_fresh.walks]
    b = app.view(time.monotonic())["buttons"]["enrol_uzayr"]
    assert (b["label"], b["look"], b["key"]) == ("Redo my walks", "on", "ctrl+u")
    assert _click(app, "enrol_uzayr") == ["ctrl+u"]
    assert app.state == ENROL_UZAYR and len(app.uz_fresh) == 0 and app.last is None and len(app.visitor) == _n(cfg)
    vm = app.view(time.monotonic())
    assert vm["card"]["big"] == "Uzayr's turn" and vm["buttons"]["enrol_uzayr"]["look"] == "off"
    assert vm["card"]["dots"]["rows"] == [("Le", screen.VISITOR, ["done", "done"]),
                                          ("Uzayr", screen.UZAYR, ["now", "next"])]
    assert vm["who"] == "Uzayr vs Le"
    assert vm["hint"][0] == "Click Start walk as you set off. Undo walk keeps the old ones."   # (till he walks)
    assert vm["card"]["sub"] == "Uzayr walks twice, then the game carries on."   # the visitor knows it goes on
    _walk(app, t)
    assert app.view(time.monotonic())["card"]["sub"] == "Got it! Once more."
    assert _now(app)["hint"][0] == "Click Start walk, then walk to the X yourself."
    assert _now(app)["card"]["sub"] == "Uzayr walks once more, then the game carries on."
    _type(app, ["backspace"])                                      # Undo walk works on his redo too
    assert app.state == ENROL_UZAYR and app.done["uzayr"] == 0
    t, who = _learn(app, t + 20)
    assert who == ["uzayr"] * _n(cfg) and app.state == READY
    assert app.view(time.monotonic())["card"]["sub"] == "Got it! Back to the game."
    new = [w["walk_id"] for w in app.uz_fresh.walks]
    assert len(new) == _n(cfg) and not set(new) & set(old)
    ev = [e for e in _events(tmp_path) if e["kind"] == "uzayr_redo"]
    assert len(ev) == 1 and ev[0]["index"] == app.vidx
    _player(app, "Si")
    assert app.uz_reused and [w["walk_id"] for w in app.uz_fresh.walks] == new


@pytest.mark.shipped
def test_redo_my_walks_with_no_player(cfg, tmp_path):
    """Before the next visitor: Uzayr's turns only (one row of dots), then WELCOME, and the next player's round
    reuses them. Undo walk on his walk gives his turn back."""
    app = _app(cfg, tmp_path)
    assert app.view(time.monotonic())["buttons"]["enrol_uzayr"]["look"] == "on"
    assert _now(app)["hint"][0] == "Click New player to start. Uzayr walks next round."
    _type(app, ["ctrl+u"])
    vm = app.view(time.monotonic())
    assert app.state == ENROL_UZAYR and vm["card"]["dots"]["rows"] == [("Uzayr", screen.UZAYR, ["now", "next"])]
    assert vm["card"]["sub"] == "Uzayr walks twice, then on to the next player."
    assert vm["who"] == "" and vm["buttons"]["wipe_visitor"]["look"] == "off"
    assert vm["hint"][0] == "Click Start walk as you set off. Undo walk cancels this."   # (he had no walks)
    t, who = _learn(app, 100.0)
    assert who == ["uzayr"] * _n(cfg) and app.state == IDLE
    assert app.view(time.monotonic())["card"]["sub"] == "Got it! Ready for the next player."
    _type(app, ["backspace"])
    assert app.state == ENROL_UZAYR and app.done["uzayr"] == _n(cfg) - 1
    _walk(app, t)
    assert app.state == IDLE and _now(app)["hint"][0] == "Click New player to start. Uzayr's walks: 1 min old."
    _player(app, "Le")
    assert app.uz_reused
    _, who = _learn(app, t + 20)
    assert who == ["visitor"] * _n(cfg)


@pytest.mark.shipped
@pytest.mark.parametrize("how", ["undo walk", "ctrl+u again", "clicker back"])
def test_redo_by_mistake_keeps_the_old_walks_until_he_walks(cfg, tmp_path, how):
    """Redo my walks clicked by mistake: until his first new walk counts, Undo walk (or Ctrl+U again, or the
    clicker's back key) brings his old walks back from memory and the game goes on; a missed walk changes nothing.
    The recording says so (uzayr_kept), so the bench tools rebuild the old pools."""
    app = _app(cfg, tmp_path, record=True)
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    t = _game(app, t)
    old = [dict(w) for w in app.uz_fresh.walks]
    _type(app, ["ctrl+u"])
    assert app.state == ENROL_UZAYR and len(app.uz_fresh) == 0
    assert app.view(time.monotonic())["buttons"]["discard"]["look"] == "on"   # Undo walk can take it back
    _walk(app, t, seen=False)                                      # a missed walk: nothing counted yet
    t += 20.0
    assert app.state == ENROL_UZAYR and app.done["uzayr"] == 0
    assert _now(app)["hint"][0] == "Click Start walk as you set off. Undo walk keeps the old ones."
    _type(app, {"undo walk": ["backspace"], "ctrl+u again": ["ctrl+u"], "clicker back": ["pageup"]}[how])
    assert app.state == READY and app.visitor_ready and app.uz_fresh.walks == old and app.done["uzayr"] == _n(cfg)
    assert not app.uz_redo and app.uz_held is None
    vm = app.view(time.monotonic())
    assert vm["card"]["sub"] == "Back to the game." and vm["hint"][0] == "Kept the old walks."
    assert vm["buttons"]["discard"]["look"] == "off"
    t = _game(app, t)
    g = app.last["id"]
    kept = [e for e in _events(tmp_path) if e["kind"] == "uzayr_kept"]
    assert len(kept) == 1 and kept[0]["walk_ids"] == [w["walk_id"] for w in old]
    rounds, pools = bench.bench_rounds(_events(tmp_path))
    assert pools[g]["u"] == [w["walk_id"] for w in old] and rounds[0]["redo"] is False
    _player(app, "Si")                                             # and the next player reuses them
    assert app.uz_reused and app.uz_fresh.walks == old


@pytest.mark.shipped
def test_redo_keeps_nothing_once_he_has_walked(cfg, tmp_path):
    """His first new walk counted: the old walks are gone for good (memory freed); Undo walk now redoes that turn
    only. With no player and no walks before, Undo walk right after Redo just goes back to the welcome screen."""
    app = _app(cfg, tmp_path)
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    t = _game(app, t)
    _type(app, ["ctrl+u"])
    _walk(app, t)
    assert app.uz_held is None and app.done["uzayr"] == 1
    _type(app, ["backspace"])
    assert app.state == ENROL_UZAYR and app.done["uzayr"] == 0 and len(app.uz_fresh) == 0
    _type(app, ["backspace"])                                      # nothing left to bring back
    assert app.state == ENROL_UZAYR and len(app.uz_fresh) == 0
    b = _app(cfg, tmp_path / "b")
    _type(b, ["ctrl+u", "backspace"])
    assert b.state == IDLE and b.uz_held is None and not b.uz_enrolling
    assert b.view(time.monotonic())["hint"][0] == "Redo my walks cancelled."


@pytest.mark.shipped
def test_redo_is_dim_and_refused_with_no_signal(cfg, tmp_path):
    """The light's red "No signal": his walks could not be taken, so Redo my walks is dim and Ctrl+U only says
    why; his walks stay."""
    app = _app(cfg, tmp_path)
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    _game(app, t)
    app.last_pkt_wall = time.monotonic() - 60.0                    # no packets for a minute
    assert app.view(time.monotonic())["buttons"]["enrol_uzayr"]["look"] == "off"
    _type(app, ["ctrl+u"])
    assert app.state == RESULT and len(app.uz_fresh) == _n(cfg) and app.uz_held is None
    assert app.view(time.monotonic())["hint"] == ("No signal: check RX2's USB and the TX charger.",
                                                   screen.STATUS["red"])


@pytest.mark.shipped
def test_redo_only_between_walks(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    _player(app, "Le")
    assert app.view(time.monotonic())["buttons"]["enrol_uzayr"]["look"] == "off"   # learning: dim
    _type(app, ["ctrl+u"])
    assert app.state == ENROL_VISITOR and app._turn() == "visitor"
    assert app.view(time.monotonic())["hint"][0] == "Not now: finish this step first."
    app.on_key_now("space", 100.0)
    assert app.view(time.monotonic())["buttons"]["enrol_uzayr"]["look"] == "off"   # walking: dim
    _type(app, ["ctrl+u"])
    assert app.state == WALKING


@pytest.mark.shipped
def test_undo_his_last_redo_walk_in_the_game_keeps_the_redo_card(cfg, tmp_path):
    """Redo my walks in a game, both his new walks, then Undo walk: his turn again, and the card still says the
    game carries on after it (not the plain "Stand on the back mark ..."); his walk again: back to the game."""
    app = _app(cfg, tmp_path)
    _player(app)
    t, _ = _learn(app, 100.0)
    t = _game(app, t)
    _type(app, ["ctrl+u"])
    t, who = _learn(app, t)
    assert who == ["uzayr"] * _n(cfg) and app.state == READY
    _type(app, ["backspace"])
    assert app.state == ENROL_UZAYR and app.done["uzayr"] == _n(cfg) - 1 and not app.visitor_ready
    card = _now(app)["card"]
    assert card["big"] == "Uzayr's turn" and card["sub"] == "Uzayr walks once more, then the game carries on."
    assert card["dots"]["rows"][1] == ("Uzayr", screen.UZAYR, ["done", "now"])
    _walk(app, t)
    assert app.state == READY and app.view(time.monotonic())["card"]["sub"] == "Got it! Back to the game."
    _type(app, ["ctrl+n", "y"])                                    # (the next player reuses the new walks)
    _player(app, "Si")
    assert app.uz_reused


# ---- what the operator sees: the age, the light ----------------------------------------------------------------------

@pytest.mark.shipped
def test_his_line_says_how_old_his_walks_are_and_the_light_stays_green(cfg, tmp_path):
    """The age of Uzayr's walks (whole minutes) is on his line only, in the third person (the crowd can read it);
    the light stays green "Ready" throughout, also just before and after they expire (amber means wait)."""
    app = _app(cfg, tmp_path)
    _player(app, "Le")
    t, _ = _learn(app, 100.0)
    _game(app, t)
    reuse = float(cfg["pools"]["uzayr_reuse_min"])
    app.last_end_dev = None                                        # (past the walk-back lock)
    _age(app, 6.5)
    assert _light(app) == ("green", "Ready")
    assert _now(app)["hint"][0] == "Click Next walk as they set off. Uzayr's walks: 6 min old."
    _age(app, reuse - 0.2)
    assert _light(app) == ("green", "Ready")
    assert _now(app)["hint"][0] == f"Click Next walk as they set off. Uzayr's walks: {reuse - 1:.0f} min old."
    assert f"your walks {reuse - 0.2:.1f} min" in app._uz_details()
    _age(app, reuse + 0.5)                                         # expired: still green
    assert _light(app) == ("green", "Ready")
    assert _now(app)["hint"][0] == "Click Next walk as they set off. Uzayr walks next round."
    _type(app, ["ctrl+n", "y"])
    assert app.state == IDLE and _now(app)["hint"][0] == "Click New player to start. Uzayr walks next round."


# ---- privacy, and the run log on Quit --------------------------------------------------------------------------------

@pytest.mark.shipped
@pytest.mark.parametrize("record", [False, True])
def test_visitor_wiped_uzayr_kept_in_memory_only(cfg, tmp_path, record):
    """New player and End game wipe the visitor only; Uzayr's kept walks stay in memory (no pool file in state/;
    on the day nothing but the tally and the log, whose lines about his own walks are as before) and Quit wipes
    them too. No name anywhere, also in a recording."""
    names = ("Thandiwe", "Bongani", "Ayanda")
    log = tmp_path / "run.log"
    app = _app(cfg, tmp_path, record=record, log=str(log))
    _player(app, names[0])
    t, _ = _learn(app, 100.0)
    t = _game(app, t)
    uz = [dict(w) for w in app.uz_fresh.walks]
    _player(app, names[1])
    assert app.name == names[1] and len(app.visitor) == 0 and app.uz_fresh.walks == uz
    t, _ = _learn(app, t)
    t = _game(app, t)
    _type(app, ["ctrl+n", "y"])                                    # End game
    assert app.name is None and len(app.visitor) == 0 and app.walk_log == [] and app.uz_fresh.walks == uz
    _player(app, names[2])
    _learn(app, t)
    _type(app, ["ctrl+q", "y"])
    app.close()
    assert app.name is None and len(app.visitor) == 0 and len(app.uz_fresh) == 0 and app.done["uzayr"] == 0
    assert sorted(os.listdir(tmp_path / "state")) == ["tally.json"]   # no pool file of any kind
    if not record:
        assert sorted(os.listdir(tmp_path)) == ["run.log", "state"]
        assert " uzayr speed " not in log.read_text()              # on the day his walk numbers stay in memory
    speeds = [repr(float(w["f"]["speed"])) for w in uz]
    for root, _, files in os.walk(tmp_path):
        for f in files:
            text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
            assert not any(n in text for n in names), f
            if not record or f not in ("run.log", "events.jsonl"):   # (a recording's walk events carry features)
                assert not any(s in text for s in speeds), f


@pytest.mark.shipped
@pytest.mark.parametrize("record", [False, True])
def test_quit_logs_the_last_walk_once(cfg, tmp_path, record):
    """Quit straight after Uzayr's last learning walk: the run log has that walk once (it had it twice: close()
    logged it, then wipe_visitor() again)."""
    log = tmp_path / "run.log"
    app = _app(cfg, tmp_path, record=record, log=str(log))
    _player(app)
    _learn(app, 100.0)
    assert app.last is not None and app.last.get("label") == "u" and app.last.get("seen")
    wid = app.last["id"]
    app.close()
    lines = [s for s in log.read_text().splitlines() if f" walk {wid} uzayr" in s]
    assert len(lines) == 1, lines


# ---- the default cfg, no fixed answer: the name box, a stuck walk, the operator's stop, the gait window --------------

def _plain_app(cfg, tmp_path):
    """An App with the room measured and nothing else: no packets yet, and it decides each walk itself."""
    app = App(cfg, state_dir=str(tmp_path), log=RunLog(None), headless=True, baseline_cache=False)
    n = len(app.chain.freqs)
    app.chain.baseline = chain.Baseline(np.ones(n), 1.0, 100, app.chain.spec.variant, app.chain.spec.nfft)
    app.state = IDLE
    return app


def test_name_box_ignores_hotkeys_and_escape_cancels(cfg, tmp_path):
    app = _plain_app(cfg, tmp_path)
    _type(app, ["n", "u", "v", "m", "space", "b", "-", "'", "1", "backspace", "escape"])
    assert app.name is None and app.state == IDLE
    _type(app, ["n", "shift+a", "A", "enter"])
    assert app.name == "AA"


def test_stuck_walk_is_closed(cfg, tmp_path):
    import time
    app = _plain_app(cfg, tmp_path)
    app.last_pkt_wall = time.monotonic()
    app.chain.clock.blocks.append((0, 0.0))
    app.start_walk(time.monotonic() - 20.0)                        # no columns ever arrive
    assert app.walk is not None
    app._timers(time.monotonic())
    assert app.walk is None and app.state == IDLE


def test_operator_stop_ends_the_walk(cfg, tmp_path):
    import time
    app = _plain_app(cfg, tmp_path)                                      # e0 = 1: threshold active_k
    app.last_pkt_wall = time.monotonic()
    app.chain.clock.blocks.append((0, 0.0))                        # host time = device time
    app.start_walk(100.0)
    assert app.state == "WALKING"
    app.walk["w"].add(cols([30.0] * 60, 100.0))              # 3.2 s of walking, below the arrival level
    assert not app.request_stop(100.0 + float(cfg["walk"]["stop_min_s"]) - 0.3)   # a double press: ignored
    assert app.state == "WALKING"
    assert app.request_stop(102.5)                                 # its columns are already in: ends at once
    assert app.walk is None and app.state == IDLE
    rec = app.walk_log[-1]
    assert rec["reason"] == "operator" and rec["seen"] and 2.5 <= rec["t_end"] < 2.5 + HOP   # at the press
    assert app.last_end_dev == rec["t_end_dev"]
    assert not app.request_stop(103.0)                             # no walk running


def test_gait_window_closes_at_the_arrival(cfg, tmp_path, monkeypatch):
    import openday.app as A
    from openday.walks import WalkWindow
    got = {}
    monkeypatch.setattr(A.walks, "walk_gait", lambda rt, ra, t0, t1, c: got.update(t0=t0, t1=t1) or {})
    app = _plain_app(cfg, tmp_path)
    e = list(np.geomspace(8.0, 400.0, 75)) + [100.0] * 60          # climbs to the X, then stands there
    w = WalkWindow(cfg, 0.0, float(cfg["dsp"]["active_k"]), HOP)
    w.add(cols(e))
    assert w.reason == "arrived"
    app.n_walks += 1
    app.walk = {"w": w, "purpose": "free", "t_click": 0.0, "t_dev": 0.0, "from": app.state, "id": app.n_walks}
    app.state = "WALKING"
    app._walk_ended()
    arr = w.arrays()
    assert got["t0"] == 0.0 and got["t1"] == w.t_arrive < arr["t"][arr["active"]].max()
    rec = app.walk_log[-1]
    assert rec["t_arrive"] == w.t_arrive and rec["t_end"] == w.t_end_col and rec["reason"] == "arrived"


# ---- the screen's own wait (walk.screen_hold_s): the same walk and answer, shown a little later --------------------

ARRIVE = list(np.geomspace(8.0, 400.0, 75)) + [100.0] * 60       # climbs to the X (peak at 3.9 s), then stands there
CREEP = list(np.geomspace(8.0, 400.0, 75)) + [v for j in range(10) for v in [400.0 + 20 * (j + 1)] * 12]
                                                                 # (... then a new small peak every 0.64 s)


def _feed_to_answer(app, e, t0):
    """The walk's columns one at a time until the model's window has ended (its answer made); returns the columns
    not fed yet and the time of the next one."""
    for i, x in enumerate(e):
        _feed(app, [x], t0 + i * HOP)
        if app.walk is None or app.walk["w"].ended:
            return e[i + 1:], t0 + (i + 1) * HOP
    return [], t0 + len(e) * HOP


def _same_numbers(f, g):
    return f.keys() == g.keys() and np.array_equal([f[k] for k in f], [g[k] for k in f], equal_nan=True)


def test_screen_wait_keeps_the_walk_and_the_answer(cfg, tmp_path):
    """The same columns with the screen's wait on (as shipped) and off: the model's window, its arrival, the numbers
    and the answer (the real nearest-centre one) are the same; only the screen ends later."""
    assert float(cfg["walk"]["screen_hold_s"]) > float(cfg["walk"]["arrive_hold_s"])
    out = []
    for hold in (cfg["walk"]["screen_hold_s"], 0.0):
        cfg["walk"]["screen_hold_s"] = hold
        app = _app(cfg, tmp_path / str(hold))
        del app._decide                                            # (helpers._app's fixed answer)
        t = _to_game(app)
        app.on_key_now("space", t)
        w = app.walk["w"]
        _feed(app, ARRIVE, t)
        assert app.walk is None and app.state == DECIDING
        out.append((w, app.walk_log[-1]))
    (w1, r1), (w0, r0) = out
    assert (w1.reason, w1.t_end_col, w1.t_arrive, w1.n_active) == (w0.reason, w0.t_end_col, w0.t_arrive, w0.n_active)
    for k, v in w1.arrays().items():
        assert np.array_equal(v, w0.arrays()[k], equal_nan=True), k
    for k in ("seen", "reason", "t_end", "t_end_dev", "t_arrive", "active_s"):
        assert r1[k] == r0[k], k
    assert _same_numbers(r1["f"], r0["f"])
    d1, d0 = r1["decision"], r0["decision"]
    assert (d1.label, d1.margin, d1.not_sure, d1.contrib) == (d0.label, d0.margin, d0.not_sure, d0.contrib)
    assert r1["reason"] == "arrived" and r0["screen_t_end"] == r0["t_end"] < r1["screen_t_end"]


def test_screen_ends_after_the_walk_and_at_most_the_extra_wait_later(cfg, tmp_path):
    """The screen ends no sooner than the model's window and at most walk.screen_max_extra_s after it: standing
    still, screen_hold_s after the peak; with new small peaks all along, at the cap."""
    hold, extra = float(cfg["walk"]["screen_hold_s"]), float(cfg["walk"]["screen_max_extra_s"])
    app = _app(cfg, tmp_path)
    for i, e in enumerate((ARRIVE, CREEP)):
        t = 100.0 + 20 * i
        app.on_key_now("space", t)
        _feed(app, e, t)
        assert app.walk is None
        r = app.walk_log[-1]
        assert r["reason"] == "arrived" and r["t_end"] <= r["screen_t_end"] <= r["t_end"] + extra
        assert app.last_end_dev == pytest.approx(t + r["screen_t_end"])   # the after-end lock counts from there
    still, creep = app.walk_log[-2:]
    assert hold <= still["screen_t_end"] - still["t_arrive"] < hold + HOP
    assert creep["screen_t_end"] >= creep["t_end"] + extra - HOP


def test_they_have_stopped_during_the_screen_wait_shows_the_answer_at_once(cfg, tmp_path):
    app = _app(cfg, tmp_path, record=True)
    t = _to_game(app)
    app.on_key_now("space", t)
    rest, t_next = _feed_to_answer(app, ARRIVE, t)
    _feed(app, rest[:3], t_next)                                   # the answer is made; the screen still waits
    assert app.state == WALKING and app.walk["w"].reason == "arrived"
    vm = app.view(time.monotonic())
    assert vm["card"]["big"] == "Walk to the X" and vm["buttons"]["start_walk"]["label"] == "They've stopped"
    press = t_next + 3 * HOP + 0.05
    app.on_key_now("space", press)                                 # They've stopped
    assert app.walk is None and app.state == DECIDING
    r = app.walk_log[-1]
    assert r["reason"] == "arrived" and r["t_end"] < r["screen_t_end"] == pytest.approx(press - t)
    assert app.last_end_dev == press
    ev = [e for e in _events(tmp_path) if e.get("id") == r["id"] or e.get("walk_id") == r["id"]]
    assert [e["kind"] for e in ev] == ["walk"]                     # (no stop event: the walk itself was whole)


def test_cancel_walk_during_the_screen_wait_throws_the_walk_away(cfg, tmp_path):
    """Cancel walk while the screen still waits: as a cancel during the walk. Nothing counted, no answer, and
    the same events (a discard, no walk event); a learning turn is walked again."""
    app = _app(cfg, tmp_path, record=True)
    _player(app)
    app.on_key_now("space", 100.0)
    _feed_to_answer(app, ARRIVE, 100.0)
    assert app.state == WALKING and app.walk["w"].ended
    wid = app.walk["id"]
    _type(app, ["backspace"])                                      # Cancel walk
    assert app.walk is None and app.state == ENROL_VISITOR and app.done == {"uzayr": 0, "visitor": 0}
    assert len(app.visitor) == 0 and app.walk_log == [] and app.last is None
    assert app.view(time.monotonic())["card"]["sub"] == "Walk cancelled. Same turn again."
    t = _to_game_rest(app, 120.0)                                  # then into the game
    last = app.last
    app.on_key_now("space", t)
    _feed_to_answer(app, ARRIVE, t)
    assert app.state == WALKING and app.walk["w"].ended
    gid = app.walk["id"]
    _type(app, ["pageup"])                                         # (the clicker's back key cancels it too)
    assert app.walk is None and app.state == READY and app.last is last
    kinds = {i: [e["kind"] for e in _events(tmp_path) if e.get("id") == i or e.get("walk_id") == i]
             for i in (wid, gid)}
    assert kinds == {wid: ["discard"], gid: ["discard"]}


@pytest.mark.parametrize("hold", [None, 0.0, 0.5])
def test_screen_wait_off_ends_the_walk_on_screen_with_the_model(cfg, tmp_path, hold):
    """walk.screen_hold_s absent, 0 or no longer than arrive_hold_s: no screen window, and the walk ends on screen
    on the very column the model's window ends on, as before the screen wait."""
    if hold is None:
        del cfg["walk"]["screen_hold_s"]
    else:
        cfg["walk"]["screen_hold_s"] = hold
    app = _app(cfg, tmp_path, record=True)
    app.on_key_now("space", 100.0)
    assert app.walk["screen"] is None
    w = app.walk["w"]
    _feed_to_answer(app, ARRIVE, 100.0)
    assert app.walk is None and w.reason == "arrived"
    r = app.walk_log[-1]
    assert r["screen_t_end"] == r["t_end"] and app.last_end_dev == w.t_end_col == r["t_end_dev"]
    assert [e for e in _events(tmp_path) if e["kind"] == "walk"][-1]["screen_t_end"] == r["t_end"]


def test_walk_event_keeps_the_models_end(cfg, tmp_path):
    """The walk event of a walk the screen waited for: reason, t_end, t_arrive and the numbers are the model's
    window (what the bench tools re-run and compare with), plus screen_t_end."""
    app = _app(cfg, tmp_path, record=True)
    app.on_key_now("space", 100.0)
    _feed(app, ARRIVE, 100.0)
    ev = [e for e in _events(tmp_path) if e["kind"] == "walk"]
    assert len(ev) == 1
    e = ev[0]
    w = walks.WalkWindow(cfg, 100.0, float(cfg["dsp"]["active_k"]), HOP)   # the model's window on its own
    w.add(cols(ARRIVE, 100.0))
    assert e["reason"] == w.reason == "arrived" and e["t_end_dev"] == w.t_end_col
    assert e["t_end"] == w.t_end_col - 100.0 and e["t_arrive"] == w.t_arrive - 100.0
    assert e["active_s"] == w.active_s
    f = walks.walk_features(w.arrays(), HOP, cfg)
    assert all(e["f"][k] == f[k] for k in f)
    assert e["screen_t_end"] > e["t_end"]
    assert set(e) == {"kind", "id", "purpose", "t_click", "t_dev", "seen", "reason", "f", "label", "active_s",
                      "t_end_dev", "t_end", "t_arrive", "screen_t_end"}


def test_signal_lost_during_the_screen_wait_still_shows_the_answer(cfg, tmp_path):
    """The answer was made before the signal went: a lost signal, or no more columns, during the screen's wait
    shows it instead of dropping a whole walk."""
    for how in ("lost", "no columns"):
        d = tmp_path / how.replace(" ", "_")
        d.mkdir()
        app = _app(cfg, d, record=True)
        app.on_key_now("space", 100.0)
        _feed_to_answer(app, ARRIVE, 100.0)
        assert app.state == WALKING
        if how == "lost":
            app._abort("Signal lost")
        else:
            app._timers(100.0 + float(cfg["walk"]["max_s"]) + float(cfg["walk"]["guard_s"]) + 0.1)
        assert app.walk is None and app.walk_log[-1]["reason"] == "arrived"
        kinds = [e["kind"] for e in _events(d)]
        assert "walk" in kinds and "discard" not in kinds


@pytest.mark.parametrize("after", [-HOP / 2, 0.3])
def test_they_have_stopped_before_the_walk_ends_shows_the_answer_with_it(cfg, tmp_path, after):
    """They've stopped pressed while the walk is still open, for a moment its columns have not reached yet. The walk
    then ends "arrived" (on the press's own column, or just before the press): the screen does not wait, and the
    after-end lock counts from the press (or from the walk's end, when that is later)."""
    t = 100.0
    alone = walks.WalkWindow(cfg, t, float(cfg["dsp"]["active_k"]), HOP)   # where the model's window ends
    alone.add(cols(ARRIVE, t))
    end, n = alone.t_end_col, len(alone.t) - 1                     # (ARRIVE[n] is the column that ends it)
    press = end + after
    app = _app(cfg, tmp_path, record=True)
    app.on_key_now("space", t)
    _feed(app, ARRIVE[:n], t)
    app.on_key_now("space", press)                                 # They've stopped: taken, its column not in yet
    assert app.state == WALKING and app.walk["w"].t_stop == app.walk["screen"].t_stop == press
    _feed(app, ARRIVE[n:n + 1], end)
    assert app.walk is None and app.state == IDLE                  # shown on the column that ended the walk
    r = app.walk_log[-1]
    assert r["reason"] == "arrived" and r["t_end_dev"] == end and r["t_arrive"] == alone.t_arrive - t
    assert app.last_end_dev == max(press, end) and r["screen_t_end"] == pytest.approx(max(press, end) - t)
    kinds = [e["kind"] for e in _events(tmp_path) if e.get("id") == r["id"] or e.get("walk_id") == r["id"]]
    assert kinds == ["stop", "walk"]


@pytest.mark.parametrize("fails", ["the numbers", "the answer"])
def test_a_fault_in_the_answer_drops_the_walk_at_once(cfg, tmp_path, monkeypatch, fails):
    """An error while the walk's numbers or its answer are made: the walk is dropped at once, as before the screen
    wait (tick logs the error), and the screen goes back to resting; it never waits on Walk to the X."""
    app = _app(cfg, tmp_path, record=True)
    t = _to_game(app)

    def broken(*a, **k):
        raise RuntimeError("broken")

    if fails == "the numbers":
        monkeypatch.setattr(walks, "walk_features", broken)
    else:
        app._decide = broken
    app.on_key_now("space", t)
    wid = app.walk["id"]
    with pytest.raises(RuntimeError):
        _feed(app, ARRIVE, t)
    assert app.walk is None
    app._timers(time.monotonic())
    assert app.state == READY
    assert not [e for e in _events(tmp_path) if e.get("id") == wid or e.get("walk_id") == wid]


def test_quit_during_the_screen_wait_still_records_the_walk(cfg, tmp_path):
    """Quit while only the screen still waits: the answer is made, so the walk is logged and recorded as any walk
    before the files close. A walk still going on is dropped on quit, as always."""
    log = tmp_path / "run.log"
    app = _app(cfg, tmp_path, record=True, log=str(log))
    t = _to_game(app)
    app.on_key_now("space", t)
    _feed_to_answer(app, ARRIVE, t)
    assert app.state == WALKING and app.walk["w"].ended
    wid = app.walk["id"]
    app.close()
    ev = [e for e in _events(tmp_path) if e.get("id") == wid or e.get("walk_id") == wid]
    assert [e["kind"] for e in ev] == ["walk"]
    assert ev[0]["reason"] == "arrived" and ev[0]["decision"]["label"] == "visitor"   # (helpers._app's answer)
    lines = open(log).read().splitlines()
    assert f"walk {wid} test seen ended arrived" in lines[-2] and lines[-1].endswith(" quit")
    d = tmp_path / "going"
    d.mkdir()
    app = _app(cfg, d, record=True)
    app.on_key_now("space", 100.0)
    _feed(app, ARRIVE[:40], 100.0)                                 # still walking
    wid = app.walk["id"]
    app.close()
    assert not [e for e in _events(d) if e.get("id") == wid or e.get("walk_id") == wid]


def test_screen_wait_ends_when_the_columns_stop(cfg, tmp_path):
    """No columns after the answer was made (the packets stopped): the screen does not wait for the walk's own guard
    (walk.max_s + guard_s after the click). It shows the answer screen_max_extra_s + guard_s after it was made."""
    wc = cfg["walk"]
    app = _app(cfg, tmp_path, record=True)
    t = time.monotonic()                                           # (host time = device time here)
    app.on_key_now("space", t)
    _feed_to_answer(app, ARRIVE, t)
    made = time.monotonic()                                        # (the answer was made between t and now)
    assert app.state == WALKING and app.walk["w"].ended
    wait = float(wc["screen_max_extra_s"]) + float(wc["guard_s"])
    assert made + wait + 0.1 - t < float(wc["max_s"]) + float(wc["guard_s"])
    app._timers(t + wait - 0.1)
    assert app.state == WALKING
    app._timers(made + wait + 0.1)
    assert app.walk is None and app.state == IDLE and app.walk_log[-1]["reason"] == "arrived"
    assert [e["kind"] for e in _events(tmp_path)].count("walk") == 1


@pytest.mark.parametrize("hold", [0.0, 1.0])
def test_a_no_signal_walk_logs_as_before_the_screen_wait(cfg, tmp_path, hold):
    """A walk closed for "no signal" (no columns past walk.max_s + guard_s) ends on screen with the model: no
    end time (t_end and screen_t_end None) and no "on screen" in the run log, with the screen's wait off or on."""
    cfg["walk"]["screen_hold_s"] = hold
    log = tmp_path / "run.log"
    app = _app(cfg, tmp_path, record=True, log=str(log))
    app.on_key_now("space", 100.0)
    _feed(app, [30.0] * 40, 100.0)                                 # seen, still walking, then nothing more comes
    app._timers(100.0 + float(cfg["walk"]["max_s"]) + float(cfg["walk"]["guard_s"]) + 0.1)
    assert app.walk is None
    r = app.walk_log[-1]
    assert r["seen"] and r["reason"] == "no signal" and r["t_end"] is None and r["screen_t_end"] is None
    line = [x for x in open(log) if f"walk {r['id']} free seen ended no signal" in x]
    assert len(line) == 1 and "on screen" not in line[0]
    assert [e for e in _events(tmp_path) if e["kind"] == "walk"][-1]["screen_t_end"] is None


# ---- the 3D picture in the app: fed by the packets, display only, a fault never stops the show -----------------------

def _landscape(m, k=0.0):
    """A room's shape across m subcarriers, bent by k (0 = nobody moving)."""
    sc = np.linspace(0.0, 1.0, m)
    return 1.0 + 0.3 * np.sin(2 * np.pi * (1.3 * sc + 0.15)) + k * np.sin(2 * np.pi * (2.2 * sc - k))


def _packets(n, t0, k_of_t=lambda t: 0.0, seed=0):
    """n raw CSI packets at 150 a second from host time t0: (local_timestamp us, t_mono, int8 data (n, 256)) with
    each subcarrier's amplitude following _landscape (all 128 bins; the chain keeps its live ones)."""
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n) / 150.0
    lt = np.round(t * 1e6).astype(np.int64)
    data = np.zeros((n, 256), np.int8)
    for i in range(n):
        amp = 40.0 * _landscape(128, k_of_t(t[i])) + rng.normal(0, 0.3, 128)
        ph = rng.uniform(0, 2 * np.pi, 128)
        data[i, 0::2] = np.clip(np.round(amp * np.sin(ph)), -127, 127)
        data[i, 1::2] = np.clip(np.round(amp * np.cos(ph)), -127, 127)
    return lt, t, data


def _feed_packets(app, lt, t, data, batch=8):
    for i in range(0, len(lt), batch):
        app.pending.extend(zip(lt[i:i + batch], t[i:i + batch], data[i:i + batch]))
        app._process()


@pytest.mark.shipped
def test_the_app_draws_the_live_signal_and_restarts_clean(cfg, tmp_path):
    app = _ridge_app(cfg, tmp_path)
    lt, t, data = _packets(1050, 100.0)                            # 7 s of a still room
    _feed_packets(app, lt, t, data)
    z, act = app.view(200.0)["ridge"]
    assert z.shape == (app.ridge.n, app.ridge.points) and act.shape == (app.ridge.n,)
    assert np.isfinite(z).all() and z.mean(axis=1) == pytest.approx(1.0)   # shapes, mean 1
    app.screen.render(app.view(200.0))
    assert len(app.screen.ridge_pc.get_paths()) == 25 * app.ridge.points   # a bar per subcarrier, every third ridge
    app.last_pkt_wall = time.monotonic() - 10.0                    # the receiver went silent: no frozen ridges
    vm = app.view(time.monotonic())
    assert vm["silent"] and len(vm["ridge"][0]) == 0
    app.screen.render(vm)                                          # and no empty frame under the warning
    assert not app.screen.ridge_frame.get_visible() and not any(t.get_visible() for t in app.screen.ridge_labels)
    app.last_pkt_wall = 1e12
    app.screen.render(app.view(200.0))
    assert app.screen.ridge_frame.get_visible() and all(t.get_visible() for t in app.screen.ridge_labels)
    lt2, t2, data2 = _packets(40, 108.0, seed=1)
    _feed_packets(app, lt2 - 50_000_000, t2, data2)                        # the board rebooted: its counter went back
    assert app.chain.reboots == 1
    assert len(app.ridge.heights()) < 10                           # only ridges from after the restart
    app.close()


@pytest.mark.shipped
def test_display_only(cfg, tmp_path):
    """The same packets give the same waterfall, the same raw samples for the gait features and the same chain
    state with the ridges fed or not fed."""
    lt, t, data = _packets(600, 100.0, k_of_t=lambda s: 0.15 * np.sin(2 * np.pi * 1.8 * s) if s > 102 else 0.0)
    out = {}
    for on in (True, False):
        app = _ridge_app(cfg, tmp_path / str(on))
        if not on:
            app.ridge.push = lambda x: None                        # the ridges get nothing
        _feed_packets(app, lt, t, data)
        assert (len(app.ridge.heights()) > 0) == on
        tr, raw = app.chain.raw.window(0.0, 1e9)
        out[on] = ([np.asarray(r[0]) for r in app.ring], [r[1:] for r in app.ring], tr, raw)
        app.close()
    a, b = out[True], out[False]
    assert len(a[0]) == len(b[0]) and all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
    assert a[1] == pytest.approx(b[1], nan_ok=True)
    assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])


@pytest.mark.shipped
def test_a_ridge_fault_never_stops_the_walk_or_the_screen(cfg, tmp_path, monkeypatch):
    app = _ridge_app(cfg, tmp_path)

    def boom(*a, **k):
        raise RuntimeError("test")

    monkeypatch.setattr(app.ridge, "push", boom)
    lt, t, data = _packets(450, 100.0)                             # (the waterfall's first column needs ~2 s)
    _feed_packets(app, lt, t, data)
    assert len(app.ring) > 0                                       # the waterfall still filled
    monkeypatch.setattr(app.screen, "_draw_ridge", boom)
    app.screen.render(app.view(200.0))                             # and the screen still draws
    assert len(app.screen.ridge_pc.get_paths()) == 0
    app.close()


@pytest.mark.shipped
def test_a_ridge_fault_is_logged_and_the_frame_still_drawn(cfg, tmp_path, monkeypatch):
    log = tmp_path / "run.log"
    app = _ridge_app(cfg, tmp_path, log=str(log))
    lt, t, data = _packets(450, 100.0)
    _feed_packets(app, lt, t, data)

    def boom(*a, **k):
        raise RuntimeError("ridge test fault")

    drawn = []
    real = app.screen.render
    monkeypatch.setattr(app.screen, "render", lambda vm: (drawn.append(vm), real(vm)))
    monkeypatch.setattr(app.ridge, "heights", boom)                # in the view
    app.next_draw = 0.0
    app._maybe_draw(time.monotonic())
    assert len(drawn) == 1 and len(drawn[0]["ridge"][0]) == 0 and drawn[0]["tiles"]
    monkeypatch.undo()
    monkeypatch.setattr(app.screen, "_draw_ridge", boom)           # in the drawing
    app._ridge_err_at = -1e9
    app.next_draw = 0.0
    app._maybe_draw(time.monotonic())
    assert app.screen.ridge_error is None                          # handed to the log
    app.close()
    assert open(log).read().count("ridge error RuntimeError: ridge test fault") == 2


@pytest.mark.shipped
def test_movement_through_the_app_flat_when_still_tall_when_walking(cfg, tmp_path):
    """Live: the waterfall's own energy over the room's stands the ridges up; a still room leaves them low."""
    app = _ridge_app(cfg, tmp_path)
    lt, t, data = _packets(750, 100.0)                             # 5 s of a still room
    _feed_packets(app, lt, t, data)

    def drawn():
        app.screen.render(app.view(200.0))
        return float(_bar_heights(app.screen, app.ridge.points)[-1].mean())

    calm = drawn()
    walk = lambda s: 0.2 * np.sin(2 * np.pi * 1.8 * s)             # noqa: E731
    lt, t, data = _packets(450, 105.0, k_of_t=walk, seed=1)
    _feed_packets(app, lt, t, data)
    busy = drawn()
    assert calm < screen.RIDGE_BASE + 0.06 and busy > calm + 0.1
    app.close()


@pytest.mark.shipped
def test_after_a_restart_movement_ridges_wait_for_their_own_fit_and_columns(cfg, tmp_path):
    app = _ridge_app(cfg, tmp_path)
    walk = lambda s: 0.2 * np.sin(2 * np.pi * 1.8 * s)             # noqa: E731
    lt, t, data = _packets(600, 100.0, k_of_t=walk)
    _feed_packets(app, lt, t, data)
    _, act = app.view(200.0)["ridge"]
    assert act.max() > 6.0
    lt2, t2, data2 = _packets(150, 106.0, seed=2)                  # a still room, after a reboot
    _feed_packets(app, lt2 - 50_000_000, t2, data2)
    assert app.chain.reboots == 1 and len(app.view(300.0)["ridge"][0]) == 0   # no ridge before its own fit
    lt3, t3, data3 = _packets(300, 107.0, seed=3)
    _feed_packets(app, lt3 - 50_000_000, t3, data3)
    z, act = app.view(300.0)["ridge"]
    assert len(z) > 10 and app.ridge_cols == len(z)                # one ridge per new column
    assert act.max() == pytest.approx(screen.RIDGE_E_MIN)              # lit by the new (still: no energy) columns
                                                                   # only, at the least a column counts for
    app.close()


@pytest.mark.shipped
def test_movement_a_lone_busy_column_barely_lifts_a_ridge_and_a_run_stands_it_up(cfg, tmp_path):
    """movement: a ridge's movement is the geometric mean of its own column's energy over the room's and the
    RIDGE_TRAIL - 1 columns before it (each at least RIDGE_E_MIN; before this restart's first column, that one
    again, but never above the room's own level), so a still room's lone busy column barely counts (the first after
    a restart too) and a walk's run of them stands the ridges up."""
    app = _ridge_app(cfg, tmp_path)                                      # (the room's energy: 1)
    n, m = app.ridge.n, app.ridge.points
    assert screen.RIDGE_TRAIL == 9
    for _ in range(n):
        app.ridge.add_shape(np.ones(m))

    def act_for(es):
        app.ring.clear()
        for e in es:
            app.ring.append((np.zeros(app.n_bins, np.float32), np.nan, float(e)))
        app.ridge_cols = len(es)
        return app._ridge_view(False)[1]

    es = np.ones(100)
    es[-1] = 50.0                                                  # one busy column, in front
    act = act_for(es)
    assert act[-1] == pytest.approx(50.0 ** (1 / 9)) and act[:-1] == pytest.approx(np.ones(n - 1))
    assert act[-1] < 1.6
    es = np.ones(100)
    es[-12:] = 20.0                                                # a walk's run of them
    act = act_for(es)
    assert act[-4:] == pytest.approx(np.full(4, 20.0))                # (its own and the 8 before: all busy)
    assert act[-12] == pytest.approx(20.0 ** (1 / 9)) and act[-13] == pytest.approx(1.0)   # (each sees only its
    act = act_for(np.full(5, 4.0))                                 # own column and the ones before it)
    assert act[-5:] == pytest.approx(4.0 ** (np.arange(1, 6) / 9)) and not act[:-5].any()   # (padded with the
    act = act_for([83.5, 55.6, 2.3, 0.3, 0.0, 0.0])                # room's level: a busy first column after a
    assert act[-6] == pytest.approx(83.5 ** (1 / 9)) and act[-6:].max() < 4.0   # restart barely counts either)
    act = act_for([0.0, 0.0, 0.0])                                 # (a dead one pads low)
    assert act[-3:] == pytest.approx(np.full(3, screen.RIDGE_E_MIN))
    assert act_for(np.zeros(20))[-1] == pytest.approx(screen.RIDGE_E_MIN)
    app.close()
