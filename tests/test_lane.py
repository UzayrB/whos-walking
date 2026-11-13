"""The lane above the 3D picture (openday/lane.py, drawn by screen.py): where the walker is, from START to the X.

The estimate never goes backwards, stands at START until someone moves, never reaches the X by itself, and goes onto
the X only at the app's own end of a seen walk; it starts again with every walk (a second look and a free walk too),
and a missed walk, Cancel walk or a lost signal never put it on the X. After a walk it stays on the X without a blink,
but never in the name box, while the room is measured or after End game. It changes nothing in a walk or its answer,
a fault in it never stops a walk or loses a frame, and `lane.show: false` gives the screen exactly as before. Its
band fits beside everything else at every screen size, in every state of a walk, and its words never talk about
pace."""

import copy
import time

import numpy as np
import pytest

import overlap
from openday import chain, lane, screen
from openday.app import AGAIN, BUSY, LANE_FADE_S, LANE_HOLD_S, NAME_ENTRY, NEEDS_BASELINE, RESULT, WALKING
from openday.chain import Columns

from helpers import HOP, SEEN, _app, _feed, _player, _type, _walk

SIZES = {"MacBook": (14.70, 9.24), "16:9": (16.0, 9.0), "Dell": (19.20, 10.80), "MacBook 1280": (12.80, 8.00),
         "Window": (11.6, 6.525)}


def _cols(energy, f50, t0=0.0, gap=None):
    e = np.asarray(energy, float)
    f = np.broadcast_to(np.asarray(f50, float), e.shape).copy()
    t = t0 + np.arange(len(e)) * HOP
    g = np.zeros(len(e), bool) if gap is None else np.asarray(gap, bool)
    return Columns(t, None, g, None, e, f, 2 * f)


def _lane(cfg, length=3.2):
    c = copy.deepcopy(cfg)
    c["lane"]["length_m"] = length
    return lane.Lane(c, 0.0, 1.0, HOP)


# ---- the estimate ---------------------------------------------------------------------------------------------------

def test_the_lane_adds_up_the_speed_tiles_readout(cfg):
    """A walker seen at a steady speed: the distance is that speed times the time seen, over the lane's length; the
    columns before the click only fill the readout; a still room (under dsp.active_k x the room level) and gaps add
    nothing; a column over MAX_MPS is left out."""
    assert cfg["lane"]["show"] is True and cfg["lane"]["length_m"] > 0
    v20 = float(chain.speed_of(20.0, cfg))                       # (the helpers' columns: f50 20 Hz)
    ln = _lane(cfg, 3.2)
    ln.add(_cols([30.0] * 10, 20.0, t0=-10 * HOP))               # before the click: nothing walked yet
    assert ln.dist == 0.0 and ln.frac == 0.0
    ln.add(_cols([30.0] * 20, 20.0, t0=0.0))
    assert ln.dist == pytest.approx(20 * HOP * v20)
    assert ln.frac == pytest.approx(ln.dist / 3.2)
    still = _lane(cfg)
    still.add(_cols([1.0] * 100, 20.0))                          # the room level, nobody walking
    still.add(_cols([30.0] * 50, 20.0, gap=[True] * 50, t0=100 * HOP))   # gaps
    still.add(_cols([30.0] * 50, 60.0, t0=150 * HOP))            # 3.9 m/s: over MAX_MPS
    assert 3.0 * v20 > lane.MAX_MPS
    assert still.dist == 0.0 and still.frac == 0.0 and still.words() == lane.WORDS[0]


def test_the_marker_never_goes_back_and_never_reaches_the_x_by_itself(cfg):
    """Speeds that change, dropouts, a walk far longer than the lane: the marker only moves forward, follows the
    distance up to NEAR, slows down after it and never passes TOP (never the X) until arrive(); then it is on the X,
    says so, and stays."""
    rng = np.random.default_rng(1)
    ln = _lane(cfg, 3.2)
    shown = [ln.frac]
    for k in range(200):                                          # about 30 s of columns: 10 m and more
        e = rng.choice([0.5, 3.0, 40.0], size=3, p=[0.2, 0.1, 0.7])
        ln.add(_cols(e, rng.uniform(5.0, 40.0, 3), t0=3 * k * HOP))
        shown.append(ln.frac)
        assert ln.frac <= lane.TOP < 1.0
    assert np.all(np.diff(shown) >= 0.0)
    assert ln.walked > 2.0 and ln.frac > lane.NEAR and ln.words() == "Nearly there"
    part = _lane(cfg, 3.2)
    part.add(_cols([30.0] * 20, 20.0))
    assert part.frac == pytest.approx(part.walked) and part.frac < lane.NEAR and part.words() == "Walking \u2026"
    ln.arrive()
    assert ln.frac == 1.0 and ln.words() == "On the X"
    ln.add(_cols([30.0] * 20, 20.0, t0=1000.0))                  # (nothing after the end moves it)
    assert ln.frac == 1.0


def test_the_lanes_words_never_talk_about_pace():
    words = " ".join(lane.WORDS).lower()
    assert lane.WORDS == ("Walking \u2026", "Nearly there", "On the X")
    for w in ("fast", "slow", "quick", "hurry", "speed", "pace", "m/s", "run", "faster", "slower"):
        assert w not in words, w


# ---- in the app: one walk at a time, ends, misses, cancels, the second look -----------------------------------------

def _lv(app, now=None):
    return app.view(time.monotonic() if now is None else now)["lane"]


def test_each_walk_starts_at_start_and_ends_on_the_x(cfg, tmp_path):
    """A seen walk: the marker starts at START, only moves forward as columns come, is on the X when the walk ends
    ("On the X" in the walker's colour), stays there for LANE_HOLD_S fading over the last LANE_FADE_S, then rests.
    The next walk starts again from START."""
    app = _app(cfg, tmp_path)
    _player(app)
    app.on_key_now("space", 100.0)
    assert app.state == WALKING
    lv = _lv(app)
    assert lv["frac"] == 0.0 and lv["text"] == "Walking \u2026" and lv["colour"] == screen.VISITOR
    seen = []
    for k in range(0, 60, 4):                                     # the walk, a few columns at a time
        _feed(app, SEEN[k:k + 4], 100.0 + k * HOP)
        seen.append(_lv(app)["frac"])
    assert np.all(np.diff(seen) >= 0.0) and 0.0 < seen[0] < seen[-1] < 1.0
    _feed(app, SEEN[60:], 100.0 + 60 * HOP)                      # still: the walk ends "stopped"
    assert app.walk is None and app.last["reason"] == "stopped"
    now = time.monotonic()
    lv = _lv(app, now)
    assert lv["frac"] == 1.0 and lv["text"] == "On the X" and lv["colour"] == screen.VISITOR and lv["fade"] == 1.0
    lv = _lv(app, now + LANE_HOLD_S - LANE_FADE_S / 2)
    assert lv["frac"] == 1.0 and 0.0 < lv["fade"] < 1.0
    lv = _lv(app, now + LANE_HOLD_S + 0.1)
    assert lv["frac"] is None and lv["text"] == ""
    app.on_key_now("space", 120.0)                               # the next walk: from START again
    assert _lv(app)["frac"] == 0.0 and app.lane_last is None


def test_no_blink_in_the_frame_drawn_as_the_walk_ends(cfg, tmp_path):
    """Live, a frame's time is taken before the tick's work: in the tick a walk ends, that time is a moment BEFORE
    the walk's end is noted. The marker is on the X in that frame (no one-frame blink to rest), at full strength;
    an end noted a little after the frame's time counts the same."""
    app = _app(cfg, tmp_path)
    app.on_key_now("space", 100.0)
    _feed(app, SEEN[:60], 100.0)
    now = time.monotonic()                                       # the frame's time, taken first ...
    _feed(app, SEEN[60:], 100.0 + 60 * HOP)                      # ... then the walk ends in the same tick
    assert app.walk is None and app.lane_last[2] >= now
    lv = _lv(app, now)
    assert lv["frac"] == 1.0 and lv["text"] == "On the X" and lv["fade"] == 1.0
    ln, colour, at = app.lane_last
    app.lane_last = (ln, colour, now + 0.05)                     # (an end noted 50 ms after the frame's time)
    lv = _lv(app, now)
    assert lv["frac"] == 1.0 and lv["fade"] == 1.0
    assert _lv(app, now + 0.05 + LANE_HOLD_S - LANE_FADE_S / 2)["fade"] == pytest.approx(0.5)
    assert _lv(app, now + 0.05 + LANE_HOLD_S + 0.01)["frac"] is None


def test_never_on_the_x_in_the_name_box_while_measuring_or_after_end_game(cfg, tmp_path):
    """Right after a walk the marker is on the X; New player (the name box), Measure room (the countdown and the
    measuring) and End game each put the lane at rest at once, and stay so after the name box is cancelled. In
    those states the lane is at rest whatever the last walk was, and before the room is measured too."""
    app = _app(cfg, tmp_path)
    _walk(app, 60.0)
    assert _lv(app)["frac"] == 1.0
    _type(app, ["n"])                                            # New player: the name box
    assert app.state == NAME_ENTRY and _lv(app)["frac"] is None and app.lane_last is None
    _type(app, ["escape"])
    assert _lv(app)["frac"] is None
    _walk(app, 80.0)
    assert _lv(app)["frac"] == 1.0
    _type(app, ["ctrl+b"], 90.0)                                 # Measure room
    assert app.state == BUSY and _lv(app)["frac"] is None and app.lane_last is None
    for st in (NAME_ENTRY, BUSY):                                # (the state alone: at rest)
        b = _app(cfg, tmp_path / st)
        _walk(b, 60.0)
        b.state = st
        assert b.lane_last is not None and _lv(b)["frac"] is None
    c = _app(cfg, tmp_path / "game")
    _player(c)
    _walk(c, 60.0)
    assert _lv(c)["frac"] == 1.0
    _type(c, ["ctrl+n"])
    _type(c, ["y"])                                              # End game
    assert c.name is None and _lv(c)["frac"] is None
    d = _app(cfg, tmp_path / "setup", baseline=False)
    d.state = NEEDS_BASELINE
    assert _lv(d)["frac"] is None


def test_a_missed_walk_a_cancel_and_a_lost_signal_never_reach_the_x(cfg, tmp_path):
    """None of them puts the marker on the X: the lane goes back to rest (no marker). Undo walk right after a walk
    takes its marker off the X too."""
    app = _app(cfg, tmp_path)
    _walk(app, 60.0)
    assert _lv(app)["frac"] == 1.0
    _type(app, ["backspace"])                                    # Undo walk
    assert _lv(app)["frac"] is None
    _walk(app, 100.0, seen=False)                                # Missed that one
    assert not app.walk_log[-1]["seen"]
    assert _lv(app)["frac"] is None
    app.on_key_now("space", 120.0)
    _feed(app, SEEN[:30], 120.0)
    assert 0.0 < _lv(app)["frac"] < 1.0
    _type(app, ["backspace"])                                    # Cancel walk
    assert app.walk is None and _lv(app)["frac"] is None
    app.on_key_now("space", 140.0)
    _feed(app, SEEN[:30], 140.0)
    app._abort("Signal lost")                                    # a lost signal
    assert app.walk is None and _lv(app)["frac"] is None


@pytest.mark.shipped
def test_the_second_look_and_a_free_walk_each_get_their_own_lane(cfg, tmp_path):
    """A close call: "walk once more", and the second walk starts from START and ends on the X; a free walk (no
    player) too, in the last walk's yellow; a learning walk of Uzayr's in his colour."""
    from openday import models
    app = _app(cfg, tmp_path)
    _walk(app, 50.0)                                             # a free walk: no player
    assert app.walk_log[-1]["purpose"] == "free"
    assert _lv(app)["frac"] == 1.0 and _lv(app)["colour"] == screen.THIS
    _player(app)
    t = 100.0
    colours = []
    for _ in range(12):                                          # the visitor's learning walks, then Uzayr's
        if app.state == "READY":
            break
        app.on_key_now("space", t)
        colours.append(_lv(app)["colour"])
        _feed(app, SEEN, t)
        t += 20.0
    assert app.state == "READY" and set(colours) == {screen.VISITOR, screen.UZAYR}
    close = models.Decision(label="uzayr", margin=0.01, not_sure=True, dist={}, contrib={"speed": 0.01}, why=["speed"])
    app._decide = lambda f: close
    _walk(app, t)
    app._timers(app.timer + 0.01)
    assert app.state == AGAIN
    app.on_key_now("space", t + 20.0)
    assert app.walk.get("second_of") is not None and _lv(app)["frac"] == 0.0
    _feed(app, SEEN, t + 20.0)
    assert app.last.get("both") and _lv(app)["frac"] == 1.0
    app._timers(app.timer + 0.01)
    assert app.state == RESULT


@pytest.mark.parametrize("ending", ["operator", "arrived"])
def test_the_marker_goes_onto_the_x_when_the_walk_ends_not_when_the_answer_shows(cfg, tmp_path, ending):
    """They've stopped, or an arrival (the screen then waits walk.screen_hold_s more for the answer): the marker
    is on the X from the walk's own end, while the screen still waits."""
    app = _app(cfg, tmp_path)
    app.on_key_now("space", 100.0)
    if ending == "operator":
        _feed(app, [30.0] * 50, 100.0)
        app.request_stop(100.0 + 40 * HOP)
        assert app.walk is None and app.last["reason"] == "operator"
    else:
        rise = [30.0 * 1.06 ** k for k in range(60)] + [30.0 * 1.06 ** 59] * 12
        _feed(app, rise, 100.0)
        assert app.walk is not None and app.walk["w"].ended and app.walk["w"].reason == "arrived"
        assert app.state == WALKING and _lv(app)["frac"] == 1.0 and _lv(app)["text"] == "On the X"
    _feed(app, [1.0] * 60, 100.0 + 90 * HOP)
    assert app.walk is None and _lv(app)["frac"] == 1.0


def test_the_lane_changes_nothing_in_a_walk_and_a_fault_in_it_stops_nothing(cfg, tmp_path, monkeypatch):
    """The same walks with the lane on, off and failing give the same walks and answers; a failing lane is logged
    and dropped (the walk carries on), and the screen shows the lane at rest."""
    logs = {}

    def run(name, show, fail=False):
        c = copy.deepcopy(cfg)
        c["lane"]["show"] = show
        log = tmp_path / f"{name}.log"
        app = _app(c, tmp_path / name, log=str(log))
        del app._decide                                          # (the real nearest-centre answer, m0)
        with monkeypatch.context() as m:
            if fail:
                m.setattr(lane.Lane, "add", lambda self, c: 1 / 0)
            _player(app)
            t = 100.0
            for i in range(9):                                   # three learning walks each, then the game
                _walk(app, t, seen=i != 7)
                app._timers(app.timer + 0.01)
                t += 20.0
        logs[name] = log.read_text()
        keep = ("id", "purpose", "seen", "reason", "t_end", "t_arrive", "active_s", "f", "label", "pool")
        rows = [{k: r[k] for k in keep} for r in app.walk_log]
        dec = [(r["decision"].label, r["decision"].margin) for r in app.walk_log if r.get("decision") is not None]
        return repr(rows) + repr(dec), len(dec), app
    on, n_on, a_on = run("on", True)
    off, _, a_off = run("off", False)
    bad, _, a_bad = run("bad", True, fail=True)
    assert on == off == bad and n_on >= 2
    assert a_off.view(time.monotonic())["lane"] is None
    assert a_bad.view(time.monotonic())["lane"]["frac"] is None
    assert "lane error ZeroDivisionError" in logs["bad"] and "lane" not in logs["on"] and "lane" not in logs["off"]
    strip = lambda s: [ln[9:] for ln in s.splitlines()]          # noqa: E731  (without the clock)
    assert strip(logs["on"]) == strip(logs["off"])


# ---- on the screen ------------------------------------------------------------------------------------------------

def test_the_lane_sits_above_the_3d_picture_level_with_the_top_buttons(cfg, tmp_path):
    """With the lane: its band is the top row of small buttons' height, at the graphs' left and as wide as them; the
    3D picture starts under it at the second row's top. Without it (lane.show false): no band and the 3D picture as
    before, up to the graphs' top."""
    x, y, w, h = screen.LANE_BOX
    rows = screen.top_boxes(True)
    top = [b for a, _, b in rows if a == "wipe_visitor"][0]
    second = [b for a, _, b in rows if a == "debug"][0]
    assert (x, y, w, h) == pytest.approx((screen.GRAPH_X0, top[1], screen.GRAPH_X1 - screen.GRAPH_X0, top[3]))
    assert y + h == pytest.approx(screen.GRAPH_Y1)
    rb = screen.ridge_box(True)
    assert rb[1] + rb[3] == pytest.approx(second[1] + second[3]) and rb[1] == screen.RIDGE_BOX[1]
    assert screen.ridge_box(False) == screen.RIDGE_BOX
    app = _app(cfg, tmp_path / "on")
    p = app.screen.rax.get_position()
    assert (p.x0, p.y0, p.width, p.height) == pytest.approx(rb) and app.screen.lane_on
    c = copy.deepcopy(cfg)
    c["lane"]["show"] = False
    off = _app(c, tmp_path / "off")
    p = off.screen.rax.get_position()
    assert (p.x0, p.y0, p.width, p.height) == pytest.approx(screen.RIDGE_BOX)
    assert not off.screen.lane_on and not hasattr(off.screen, "lane_box")
    assert off.view(time.monotonic())["lane"] is None


@pytest.mark.parametrize("size", list(SIZES))
def test_the_lane_fits_at_every_size_in_every_state(cfg, tmp_path, size):
    """At rest, part-way, nearly there, on the X and fading, in each walker's colour, with a long name and a
    recording's box on the top line: nothing overlaps (tests/overlap.py), the words stay inside the band, the dot
    stays on its track between START and the X, and the main screen's other parts keep their places."""
    w, h = SIZES[size]
    found = []
    for rec in (False, True):
        app = _app(cfg, tmp_path / f"{size}{rec}", record=rec)
        _type(app, ["n", *"Thandiwe Mokoena", "enter"])
        sc = app.screen
        sc.fig.set_dpi(100)
        sc.fig.set_size_inches(w, h, forward=False)
        sc._rescale()
        for frac, text, colour, fade in ((None, "", None, 0.0), (0.0, "Walking \u2026", screen.VISITOR, 1.0),
                                         (0.5, "Walking \u2026", screen.UZAYR, 1.0),
                                         (0.93, "Nearly there", screen.THIS, 1.0), (1.0, "On the X", screen.THIS, 1.0),
                                         (1.0, "On the X", screen.VISITOR, 0.4)):
            vm = app.view(time.monotonic())
            vm["lane"] = {"frac": frac, "text": text, "colour": colour, "fade": fade}
            sc.render(vm)
            probs = overlap.check(sc)
            found += [dict(p, state=f"{size} rec={rec} {frac} {text}") for p in probs
                      if p["kind"] != "d" or size in ("MacBook", "16:9")]
            r = sc.fig.canvas.get_renderer()
            band = sc.lane_box.get_window_extent(r)
            if text:
                assert sc.lane_words.get_text() == text
                we = sc.lane_words.get_window_extent(r)
                assert band.x0 <= we.x0 and we.x1 <= band.x1 + 0.5, (size, text)
            if frac is not None:
                de = sc.lane_dot.get_window_extent(r)
                assert band.y0 - 0.5 <= de.y0 and de.y1 <= band.y1 + 0.5, size
                xd = sc.lane_dot.get_xdata()[0]
                assert xd == pytest.approx(sc.lane_x0 + frac * (sc.lane_x1 - sc.lane_x0))
            assert sc.lane_x0 < sc.lane_x1 and sc.lane_x1 < screen.GRAPH_X1
            assert sc.lane_dot.get_visible() == (frac is not None)
        app.close()
    assert not found, "\n" + overlap.fmt(found)


def test_a_fault_drawing_the_lane_never_loses_a_frame(cfg, tmp_path):
    """Like the 3D picture: a fault drawing the lane hides its dot and fill for that frame, the rest of the frame is
    drawn, the app's draw loop draws the frame and logs the fault, and the next good frame shows the dot again."""
    log = tmp_path / "run.log"
    app = _app(cfg, tmp_path, log=str(log))
    app.on_key_now("space", 100.0)
    _feed(app, SEEN[:20], 100.0)
    sc = app.screen
    bad = {"frac": 0.5, "text": "Walking \u2026", "colour": "not-a-colour", "fade": 1.0}
    vm = app.view(time.monotonic())
    vm["lane"] = bad
    sc.render(vm)                                                # (no exception)
    assert sc.lane_error is not None and not sc.lane_dot.get_visible() and not sc.lane_fill.get_visible()
    assert sc.big.get_text() == "Walk to the X" and sc.buttons["start_walk"]["text"].get_text() == "They've stopped"
    sc.lane_error = None
    app._lane_view = lambda now: dict(bad)
    n = len(app.stats["draw_s"])
    app.next_draw = 0.0
    app._maybe_draw(time.monotonic())
    assert len(app.stats["draw_s"]) == n + 1 and sc.lane_error is None
    assert "lane error" in log.read_text()
    del app._lane_view
    app.next_draw = 0.0
    app._maybe_draw(time.monotonic())
    assert sc.lane_dot.get_visible() and sc.lane_error is None
    vm = app.view(time.monotonic())
    vm["lane"] = dict(bad, colour=screen.THIS, fade=1.2)         # (a fade out of range is held to 0..1)
    sc.render(vm)
    assert sc.lane_error is None and sc.lane_dot.get_alpha() == 1.0


def test_drawing_the_lane_costs_little(cfg, tmp_path):
    """A frame with the lane part-way takes about as long to draw as one at rest (headless, the MacBook's size): the
    best of three alternating runs each, with a generous margin, so a busy machine does not fail it."""
    app = _app(cfg, tmp_path)
    sc = app.screen
    sc.fig.set_dpi(100)
    sc.fig.set_size_inches(*SIZES["MacBook"], forward=False)
    sc._rescale()
    vm = app.view(time.monotonic())

    def frames(lv, n=6):
        ts = []
        for k in range(n):
            vm["lane"] = dict(lv, frac=None if lv["frac"] is None else lv["frac"] + 0.01 * k)
            t0 = time.perf_counter()
            sc.render(vm)
            sc.fig.canvas.draw()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts[1:]))
    rest, moving = [], []
    for _ in range(3):
        rest.append(frames({"frac": None, "text": "", "colour": None, "fade": 0.0}))
        moving.append(frames({"frac": 0.3, "text": "Walking \u2026", "colour": screen.THIS, "fade": 1.0}))
    assert min(moving) < 2.0 * min(rest) + 0.02, (rest, moving)
