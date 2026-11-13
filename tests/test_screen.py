"""The screen (openday/screen.py), headless: the main button never moves and is the biggest, the rows and the
top-right group are fixed and even, the operator's line fits two lines, the card never talks to the operator, and
nothing overlaps in any state at the MacBook's and the Dell's shapes; every screen of `python -m openday screens`
checked by tests/overlap.py at four sizes; the screenshots tool itself; the 3D picture's frame and bars.

A test marked `alternate_round` tours the older alternating round's states, one marked `shipped` the round as
shipped (see tests/test_app.py)."""

import collections
import os
import subprocess
import time

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.image import imread

import openday
import overlap
from openday import chain, screen, screenshots
from openday import screenshots as S
from openday.app import RESULT

from helpers import (EMPTY, _app, _type, _feed, _walk, _to_game, _cards, _player, _learn, _game, _age, _now,
                     _ridge_app, _bar_heights)


MAIN_ROW = ("start_walk", "discard", "confirm_uzayr", "confirm_visitor", "new_visitor", "reuse_baseline")
CORNER = ("baseline", "debug", "quit", "wipe_visitor", "enrol_uzayr")   # the top-right group (no Full screen)
FORBIDDEN = ("click", "press", "ctrl", "key", "button", "u /", "enter", "tally", "score", "right today",
             "counts as a miss", "clicker")


# ---- the main button, the rows and the operator's line in every state of a round -------------------------------------

def _tour(cfg, tmp_path):
    """(name, view model) for every live state of a round, with the screen drawn for each."""
    out = []
    a = _app(cfg, tmp_path / "a", baseline=False)
    a.cached = chain.Baseline(np.ones(len(a.chain.freqs)), 1.0, 100, a.chain.spec.variant, a.chain.spec.nfft,
                            created=time.time() - 300)
    out.append(("SET-UP", a, a.view(time.monotonic())))
    a.on_key_now("space", 50.0)
    out.append(("MEASURING", a, a.view(time.monotonic())))
    app = _app(cfg, tmp_path / "b")
    out.append(("WELCOME", app, app.view(time.monotonic())))
    _type(app, ["n", "L", "e"])
    out.append(("NAME", app, app.view(time.monotonic())))
    _type(app, ["enter"])
    out.append(("LEARNING", app, app.view(time.monotonic())))
    app.on_key_now("space", 100.0)
    out.append(("WALKING early", app, app.view(100.2)))
    out.append(("WALKING", app, app.view(103.0)))
    _feed(app, EMPTY, 100.0)
    out.append(("LEARNING missed", app, app.view(time.monotonic())))
    _walk(app, 120.0)
    out.append(("LEARNING got it", app, app.view(time.monotonic())))
    app.on_key_now("space", 140.0)
    _type(app, ["pageup"], 141.0)
    out.append(("LEARNING cancelled", app, app.view(time.monotonic())))
    for i in range(5):
        _walk(app, 160.0 + 20 * i)
    out.append(("GAME got you both", app, app.view(time.monotonic())))
    out.append(("GAME", app, app.view(time.monotonic() + 10)))
    _walk(app, 300.0)
    out.append(("THINKING", app, app.view(time.monotonic())))
    app.state = RESULT
    out.append(("ANSWER", app, app.view(time.monotonic())))
    _type(app, ["v"])                                              # on the day: does nothing
    out.append(("ANSWER after V", app, app.view(time.monotonic())))
    _type(app, ["backspace"])
    out.append(("GAME removed", app, app.view(time.monotonic())))
    _type(app, ["ctrl+n"])
    out.append(("END GAME box", app, app.view(time.monotonic())))
    _type(app, ["escape", "ctrl+q"])
    out.append(("QUIT", app, app.view(time.monotonic())))
    w = _app(cfg, tmp_path / "w")
    _walk(w, 100.0)
    out.append(("WELCOME speed", w, w.view(time.monotonic())))
    r = _app(cfg, tmp_path / "c", record=True)
    _type(r, ["n", "L", "e", "enter"])
    _walk(r, 100.0)
    out.append(("RECORD learning", r, r.view(time.monotonic())))
    t = 120.0
    for i in range(5):
        _walk(r, t + 20 * i)
    _walk(r, 300.0)
    r.state = RESULT
    out.append(("RECORD answer", r, r.view(time.monotonic())))
    _type(r, ["v"])
    out.append(("RECORD noted", r, r.view(time.monotonic())))
    s = _app(cfg, tmp_path / "s", record=True)
    _walk(s, 100.0)
    out.append(("RECORD speed", s, s.view(time.monotonic())))
    for name, ap, vm in out:
        ap.screen.render(vm)
    return out


@pytest.mark.alternate_round
def test_main_button_never_moves_and_is_the_biggest(cfg, tmp_path):
    box = screen.SHOW_MAIN[2]
    area = box[2] * box[3]
    for name, app, vm in _tour(cfg, tmp_path):
        app.screen.render(vm)
        bs = app.screen.buttons
        main = bs["start_walk"]
        if main["look"] != "hidden":
            r = main["rect"]
            assert (r.get_x(), r.get_y(), r.get_width(), r.get_height()) == box, name
        x0, y0, x1, y1 = main["hit"]
        for other, b in bs.items():
            if other == "start_walk" or b["look"] == "hidden":
                continue
            ox0, oy0, ox1, oy1 = b["box"]
            assert (ox1 - ox0) * (oy1 - oy0) < area, (name, other)
            assert ox1 <= x0 or ox0 >= x1 or oy1 <= y0 or oy0 >= y1, (name, other)   # never where the main sits


@pytest.mark.alternate_round
def test_at_most_five_main_buttons_and_one_bright(cfg, tmp_path):
    one_next = {"SET-UP", "WELCOME", "LEARNING", "WALKING", "GAME", "ANSWER", "LEARNING missed", "LEARNING got it",
                "GAME removed", "ANSWER after V", "RECORD learning", "RECORD answer", "RECORD noted"}
    for name, app, vm in _tour(cfg, tmp_path):
        b = vm["buttons"]
        shown = [k for k in MAIN_ROW if b.get(k, {"look": "hidden"})["look"] != "hidden"]
        assert len(shown) <= 5, (name, shown)
        assert set(b) - set(MAIN_ROW) - set(CORNER) <= {"ov_a", "ov_b"}, name
        if name in one_next:
            assert [k for k, v in b.items() if v["look"] == "next"] in (["start_walk"], ["new_visitor"]), (name, b)


@pytest.mark.alternate_round
def test_hints_fit_two_lines_at_full_size(cfg, tmp_path):
    """The operator's line is 20 pt, to be read from the back mark: every state's hint fits two lines unshrunk."""
    for name, app, vm in _tour(cfg, tmp_path):
        app.screen.render(vm)
        h = app.screen.hint
        base = h._base_size * app.screen.fig.get_figheight() / screen.BASE_H
        assert h._base_size >= 20 and h.get_fontsize() == pytest.approx(base), (name, h.get_text())
        assert h.get_text().count("\n") <= 1, (name, h.get_text())


@pytest.mark.alternate_round
def test_card_never_talks_to_the_operator_and_shows_no_score(cfg, tmp_path):
    for name, app, vm in _tour(cfg, tmp_path):
        for text in _cards(vm):
            low = text.lower()
            assert not any(w in low for w in FORBIDDEN), (name, text)
    app = _app(cfg, tmp_path / "d")
    assert "click" in app.view(time.monotonic())["hint"][0].lower()          # the operator's line: mouse wording


@pytest.mark.alternate_round
def test_mouse_main_button_reaches_the_screen_edge_and_small_buttons_ignore_a_double_click(cfg, tmp_path):
    app = _app(cfg, tmp_path)
    t = _to_game(app)
    _walk(app, t)
    app.state = RESULT
    app.screen.render(app.view(time.monotonic()))

    def click(fx, fy):
        canvas = app.screen.fig.canvas
        px, py = app.screen.fig.transFigure.transform((fx, fy))
        canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", canvas, px, py, button=1))
        sent = []
        while not app.keys.empty():
            sent.append(app.keys.get_nowait()[0])
        return sent

    x, y, w, h = screen.SHOW_MAIN[2]
    assert click(0.999, y + h / 2) == ["space"]                    # the pointer thrown to the right edge still hits
    x0, y0, x1, y1 = app.screen.buttons["discard"]["box"]
    mid = ((x0 + x1) / 2, (y0 + y1) / 2)
    assert click(*mid) == ["backspace"]                            # Undo walk sends Backspace
    assert click(*mid) == []                                       # its double-click: ignored
    assert "confirm_uzayr" not in app.screen.buttons               # on the day there is no It was ... at all


@pytest.mark.alternate_round
@pytest.mark.parametrize("size", [(16, 9), (14.7, 9.56)])
def test_hint_two_even_lines(cfg, tmp_path, size):
    """A hint too long for one line is split into two lines: after a sentence (": ", or ". " before a capital) when
    both lines fit, at full size or one step smaller (screen.HINT_SENTENCE_FIT), else into two lines of about equal
    length at full size (the longer line as short as it can be). The age of Uzayr's walks goes on a line of its
    own, also in a test recording on the MacBook (a step smaller there)."""
    for record in (True, False):
        app = _app(cfg, tmp_path / f"r{record}", record=record)
        sc = app.screen
        sc.fig.set_size_inches(*size)
        sc._rescale()
        r = sc.fig.canvas.get_renderer()
        x_max = sc.fig.transFigure.transform((sc.hint._fit_x, 0))[0]
        full = 20 * size[1] / screen.BASE_H
        texts = ("Click Start walk as Lerato sets off.", "Click New player to start a game.",
                 "One of them walks: click Start walk as they set off.",
                 "Click They've stopped when they stand still (optional).",
                 "Click Start walk as Bartholo. sets off.",
                 "Click Next walk as they set off. Uzayr's walks: 6 min old.",
                 "Click New player to start. Uzayr walks next round.")
        for text in texts if record else texts[4:]:
            sc._hint_text = None
            sc._set_hint(text)
            lines = sc.hint.get_text().split("\n")
            fs = sc.hint.get_fontsize()
            if len(lines) == 1:                                    # (it fits one line: on the day, wide screen)
                assert sc.hint.get_window_extent(r).x1 <= x_max and not record, (text, lines)
                assert fs == pytest.approx(full), (record, text)
                continue
            assert len(lines) == 2 and " ".join(lines) == text, lines
            words = text.split()

            def measure(at):                                       # every split at this size: its longer line,
                widths, ends = [], []                              # and whether it is a sentence end that fits
                sc.hint.set_fontsize(at)
                for k in range(1, len(words)):
                    w = []
                    for part in (" ".join(words[:k]), " ".join(words[k:])):
                        sc.hint.set_text(part)
                        w.append(sc.hint.get_window_extent(r))
                    widths.append(max(e.width for e in w))
                    end = words[k - 1][-1] == ":" or (words[k - 1][-1] == "." and words[k][0].isupper())
                    if end and len(words) - k >= 2 and max(e.x1 for e in w) <= x_max:
                        ends.append(k - 1)
                return widths, ends

            got = len(lines[0].split()) - 1
            widths, ends = measure(full)
            smaller = measure(full * 0.92) if not ends else None
            sc.hint.set_fontsize(fs)
            sc.hint.set_text("\n".join(lines))
            if ends:                                               # a whole sentence a line, full size
                assert fs == pytest.approx(full), (record, text)
                assert got in ends and widths[got] == pytest.approx(min(widths[i] for i in ends)), (text, lines)
            elif smaller[1]:                                       # ... or one step smaller
                assert fs == pytest.approx(full * 0.92) and full * 0.92 >= screen.HINT_SENTENCE_FIT * full, (text, fs)
                assert got in smaller[1], (text, lines)
            else:                                                  # no other split has a shorter longer line
                assert fs == pytest.approx(full), (record, text)
                assert widths[got] == pytest.approx(min(widths)), (text, lines)
            assert not lines[0].endswith("Bartholo."), lines      # a name cut short is no sentence end
            if "Uzayr" in text:
                assert lines[1].startswith("Uzayr"), lines


# ---- the layout: fixed slots, one top-right group, nothing overlaps --------------------------------------------------

def _overlap(a, b, pad=0.0):
    return not (a[2] <= b[0] + pad or b[2] <= a[0] + pad or a[3] <= b[1] + pad or b[3] <= a[1] + pad)


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("redo", [True, False])
def test_button_boxes_are_fixed_even_and_never_overlap(record, redo):
    """The main button's box is unchanged; the row is equal slots with equal gaps, vertically centred on
    the main button, the last gap the gap to the main button; Undo walk and New player beside it in both modes
    (a recording's It was ... slots left of them); the top-right group is identical buttons in two rows, each
    evenly spaced and right-aligned, the top one at the graphs' top, exactly as wide as the tiles when full: Redo
    my walks (with pools.uzayr_reuse_min) then End game and Measure room, under them Details and Quit (last), no
    Full screen, the others in the same places with or without Redo my walks; no two boxes overlap (Use last
    measurement shares the Undo walk and New player slots, never shown with them)."""
    main = screen.SHOW_MAIN[2]
    assert main == (0.78, 0.259, 0.20, 0.075)
    row = screen.show_row(record)
    assert [a for a, _, _ in row] == (["confirm_visitor", "confirm_uzayr", "discard", "new_visitor"] if record
                                      else ["discard", "new_visitor"])
    day = {a: b for a, _, b in screen.show_row(False)}                 # Undo walk and New player: the same places in a
    assert all(b == day[a] for a, _, b in row if a in day)         # test recording as on the day
    xs = [b[0] for _, _, b in row] + [main[0]]
    assert len({(b[2], b[3]) for _, _, b in row}) == 1
    assert np.allclose(np.diff(xs), screen.ROW_W + screen.ROW_GAP)
    for _, _, b in row:
        assert b[1] + b[3] / 2 == pytest.approx(main[1] + main[3] / 2)
    top = screen.top_boxes(redo)
    assert [a for a, _, _ in top] == (["enrol_uzayr"] if redo else []) + ["wipe_visitor", "baseline", "debug", "quit"]
    plain = {a: b for a, _, b in screen.top_boxes(False)}
    assert all(b == plain[a] for a, _, b in top if a in plain)      # Redo my walks moves none of the others
    assert len({(b[2], b[3]) for _, _, b in top}) == 1
    rows = [[b for a, _, b in top if a in dict(r)] for r in screen.TOP_ROWS]
    assert [len(r) for r in rows] == [2 + redo, 2]
    for r in rows:                                                 # each row: one height, even, right-aligned
        assert len({b[1] for b in r}) == 1
        assert np.allclose(np.diff([b[0] for b in r]), screen.TOP_W + screen.TOP_GAP)
        assert r[-1][0] + r[-1][2] == pytest.approx(screen.TOP_X1)
    assert rows[0][0][1] + screen.TOP_H == pytest.approx(screen.GRAPH_Y1)   # the top row's top is the graphs' top
    assert rows[1][0][1] + screen.TOP_H + screen.TOP_GAP == pytest.approx(rows[0][0][1])
    assert rows[1][0][1] > screen.TILE_Y1                              # above the tiles
    if redo:                                                       # the full row is exactly the tiles' width
        assert rows[0][0][0] == pytest.approx(screen.TILE_X0)
    boxes = {a: (b[0], b[1], b[0] + b[2], b[1] + b[3]) for a, _, b in row + top + [screen.SHOW_MAIN]}
    names = sorted(boxes)
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            assert not _overlap(boxes[p], boxes[q]), (p, q)
    (_, _, e), = screen.show_extra(record)
    extra = (e[0], e[1], e[0] + e[2], e[1] + e[3])
    assert extra[0] == boxes["discard"][0] and extra[2] == boxes["new_visitor"][2]
    assert screen.hint_x1(record) < boxes["discard"][0]


@pytest.mark.alternate_round
@pytest.mark.parametrize("size", [(16, 9), (14.7, 9.56)])          # a 16:9 monitor and the MacBook's 1.54:1
def test_nothing_overlaps_on_screen(cfg, tmp_path, size):
    """In every state of a round, drawn at both screen shapes: no two shown buttons overlap, the operator's
    hint (at most two lines, never one word alone on a line) ends left of the row, the light's words stay on the
    top line with air above the top-right group, the light's dot ends before its words, "Uzayr vs {V}" ends before the
    group, "like {V}" never touches a tile's name, the row's text keeps one size all run and the two It was ...
    buttons share one size."""
    sizes: dict = {}
    for name, app, vm in _tour(cfg, tmp_path):
        _screen_checks(name, app, vm, size, sizes)
    for k, v in sizes.items():                                     # the row's text never changes size in a run
        assert len(v) == 1, (size, v)


def _screen_checks(name, app, vm, size, sizes):
    """test_nothing_overlaps_on_screen's checks for one state, drawn at one screen shape (also used by
    test_new_states_fit_on_screen)."""
    sc = app.screen
    sc.fig.set_size_inches(*size)
    sc._rescale()
    sc._hint_text = None
    sc.render(vm)
    r = sc.fig.canvas.get_renderer()
    inv = sc.fig.transFigure.inverted()

    def ext(t):
        (x0, y0), (x1, y1) = inv.transform(t.get_window_extent(r).get_points())
        return (x0, y0, x1, y1)

    shown = {n: b["box"] for n, b in sc.buttons.items() if b["look"] != "hidden" and n not in ("ov_a", "ov_b")}
    ns = sorted(shown)
    for i, p in enumerate(ns):
        for q in ns[i + 1:]:
            assert not _overlap(shown[p], shown[q]), (name, size, p, q)
    for n, box in shown.items():                               # every label inside its own button
        tb = ext(sc.buttons[n]["text"])
        assert box[0] - 1e-3 <= tb[0] and tb[2] <= box[2] + 1e-3, (name, size, n)
    if sc.hint.get_text():
        h = ext(sc.hint)
        assert sc.hint.get_text().count("\n") <= 1, (name, size)
        for n, box in shown.items():
            assert not _overlap(h, box), (name, size, n, sc.hint.get_text())
    light = ext(sc.light_txt)
    who = ext(sc.who) if sc.who.get_text() else None
    for n in sc.top:
        if n in shown:
            assert not _overlap(light, shown[n]), (name, size, n)
            assert who is None or not _overlap(who, shown[n]), (name, size, n)
            if sc.light_txt.get_text():                        # air between the light's words and the group
                assert light[1] >= shown[n][3] + 0.006, (name, size, n, light[1])
    if sc.light_txt.get_text():                                # the light's words start clear of its dot
        assert ext(sc.light)[2] + 0.004 <= light[0], (name, size)
    if sc.light_txt.get_text():                                # the light's words end in the top-right corner
        assert light[2] == pytest.approx(screen.TOP_X1, abs=2e-3), (name, size, light)
    if who is not None:                                        # "Uzayr vs {V}" ends before the light's dot
        assert who[2] < ext(sc.light)[0], (name, size)
    words = sc.hint.get_text().split()
    if "\n" in sc.hint.get_text() and len(words) >= 4:        # two lines of about equal length
        assert all(len(line.split()) >= 2 for line in sc.hint.get_text().split("\n")), sc.hint.get_text()
    for tn, tl in sc.tiles.items():                            # "like {V}" never touches the tile's name
        if tl["pull"].get_text():
            assert ext(tl["pull"])[0] > ext(tl["label"])[2] + 0.004, (name, size, tn)
    row = {n: sc.buttons[n]["text"].get_fontsize() for n in ("discard", "new_visitor") if n in shown}
    sizes.setdefault(id(app), set()).update(round(v, 3) for v in row.values())
    pair = [sc.buttons[n]["text"].get_fontsize() for n in ("confirm_uzayr", "confirm_visitor") if n in shown]
    assert len(set(round(v, 3) for v in pair)) <= 1, (name, size, pair)   # the two It was ... one size
    assert all(v <= min(row.values()) + 1e-6 for v in pair) if row else True


# ---- the round as shipped: the light's dot, every new state fits, the operator's line splits after a sentence --------

@pytest.mark.shipped
@pytest.mark.parametrize("size", [(16, 9), (14.7, 9.56)])
def test_the_light_dot_keeps_air_above_the_buttons_and_before_its_words(cfg, tmp_path, size):
    """The dot's ink (measured in the pixels) ends above the top-right group with air, and before its words."""
    app = _app(cfg, tmp_path)
    sc = app.screen
    sc.fig.set_size_inches(*size)
    sc.fig.set_dpi(120)
    sc._rescale()
    app.last_pkt_wall, app.rate_hist = time.monotonic() + 3600, collections.deque([(time.monotonic() - 1, 0),
                                                                                    (time.monotonic(), 150)])
    sc.render(_now(app))
    sc.fig.canvas.draw()
    img = np.asarray(sc.fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    h, w = img.shape[:2]
    green = np.array([int(screen.STATUS["green"][i:i + 2], 16) for i in (1, 3, 5)])
    ink = np.abs(img - green).sum(axis=2) < 60
    ink[int(0.15 * h):] = False
    ys, xs = np.nonzero(ink)
    assert len(ys) > 100 and ys.min() > 0                          # the whole dot, not cut at the top
    bottom, right = 1.0 - (ys.max() + 1) / h, (xs.max() + 1) / w
    top = [sc.buttons[n]["box"] for n in sc.top if sc.buttons[n]["look"] != "hidden"]
    assert top and bottom >= max(b[3] for b in top) + 0.006, (size, bottom)
    r = sc.fig.canvas.get_renderer()
    words = sc.fig.transFigure.inverted().transform((sc.light_txt.get_window_extent(r).x0, 0))[0]
    assert sc.light_txt.get_text() == "Ready" and words >= right + 0.004, (size, right, words)


def _tour_new(cfg, tmp_path, record):
    out = []
    app = _app(cfg, tmp_path / f"t{record}", record=record)
    out.append(("WELCOME", app, _now(app)))
    _player(app, "Lerato")
    out.append(("LEARNING visitor", app, app.view(time.monotonic())))
    t, _ = _learn(app, 100.0)
    out.append(("GOT YOU BOTH", app, app.view(time.monotonic())))
    t = _game(app, t)
    _age(app, float(cfg["pools"]["uzayr_reuse_min"]) - 1.0)
    out.append(("ANSWER soon", app, _now(app)))
    _type(app, ["ctrl+u", "backspace"])                            # Redo my walks taken back
    out.append(("REDO kept", app, app.view(time.monotonic())))
    _player(app, "Thandiwe")
    out.append(("REUSE learning", app, _now(app)))
    t, _ = _learn(app, t)
    out.append(("REUSE got it", app, app.view(time.monotonic())))
    t = _game(app, t)
    _type(app, ["ctrl+u"])
    out.append(("REDO", app, _now(app)))
    t, _ = _learn(app, t)
    out.append(("REDO done", app, app.view(time.monotonic())))
    _type(app, ["ctrl+n", "y", "ctrl+u"])
    out.append(("REDO no player", app, _now(app)))
    for name, ap, vm in out:
        ap.screen.render(vm)
    return out


@pytest.mark.shipped
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("size", [(16, 9), (14.7, 9.56)])
def test_new_states_fit_on_screen(cfg, tmp_path, record, size):
    sizes: dict = {}
    for name, app, vm in _tour_new(cfg, tmp_path, record):
        for text in _cards(vm):
            assert not any(w in text.lower() for w in FORBIDDEN), (name, text)
        b = vm["buttons"]
        assert len([k for k in MAIN_ROW if b.get(k, {"look": "hidden"})["look"] != "hidden"]) <= 5, name
        assert set(b) - set(MAIN_ROW) - set(CORNER) <= {"ov_a", "ov_b"}, name
        assert [k for k, v in b.items() if v["look"] == "next"] in (["start_walk"], ["new_visitor"]), (name, b)
        _screen_checks(name, app, vm, size, sizes)
        _hint_full_or_sentences(app, size, name)
    for v in sizes.values():
        assert len(v) == 1


def _tour_second_look(cfg, tmp_path, record):
    """The round as shipped through the new states: Uzayr's third of five walks (five dots), a close call's "walk
    once more", a cancelled second walk, Thinking on both walks, the answer from both walks (its tiles "Both walks";
    noted, in a recording), a missed second walk, Redo my walks part-way ("Uzayr walks 4 more times")."""
    from openday import models
    from openday.app import AGAIN
    close = models.Decision("visitor", 0.05, True, {}, {"speed": -0.3, "steps": 0.1, "swing": -0.2, "bounce": 0.0},
                            why=["speed"])
    clear = models.Decision("visitor", 0.25, False, {}, {"speed": -0.4, "steps": -0.1, "swing": -0.3, "bounce": 0.0},
                            why=["speed", "leg swing"])
    out = []
    app = _app(cfg, tmp_path / f"s{record}", record=record)
    _player(app, "Thandiwe Mokoena")
    t = 100.0
    while app._turn() == "visitor" or app.done["uzayr"] < 2:
        _walk(app, t)
        t += 20.0
    out.append(("UZAYR walk 3 of 5", app, _now(app)))
    t, _ = _learn(app, t)
    app._decide = lambda f: close
    _walk(app, t)
    app._timers(app.timer + 0.01)
    assert app.state == AGAIN
    out.append(("WALK ONCE MORE", app, _now(app)))
    app.on_key_now("space", t + 20.0)
    out.append(("WALKING second look", app, app.view(t + 23.0)))
    _type(app, ["backspace"])
    out.append(("SECOND LOOK cancelled", app, app.view(time.monotonic())))
    app._decide = lambda f: clear
    _walk(app, t + 40.0)
    out.append(("THINKING both walks", app, app.view(time.monotonic())))
    app._timers(app.timer + 0.01)
    out.append(("ANSWER from both walks", app, _now(app)))
    if record:
        _type(app, ["v"])
        out.append(("ANSWER from both walks, noted", app, app.view(time.monotonic())))
    app._decide = lambda f: close
    _walk(app, t + 60.0)
    app._timers(app.timer + 0.01)
    _walk(app, t + 80.0, seen=False)
    app._timers(app.timer + 0.01)
    out.append(("SECOND LOOK missed", app, app.view(time.monotonic())))
    app._timers(app.timer + 0.01)
    _type(app, ["ctrl+u"])
    _walk(app, t + 100.0)
    out.append(("REDO 4 more times", app, _now(app)))
    for name, ap, vm in out:
        ap.screen.render(vm)
    return out


@pytest.mark.shipped
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("size", [(14.70, 9.24), (16, 9), (14.7, 9.56)])    # the MacBook full screen first
def test_the_second_look_and_his_five_walks_fit_on_screen(cfg, tmp_path, record, size):
    """"Walk once more!" (its headline as big as the card allows: full size at 16:9, one fit step on the MacBook)
    over "Too close to call: the same walker again.", Thinking on both walks, the answer "(from both walks)" with its tiles' "Both walks", "Walk 3 of 5: to the X, then stand
    still." with five dots, "Uzayr walks 4 more times": the card never talks to the operator, one bright button, the
    operator's line two lines at full size clear of the buttons, nothing overlapping (tests/overlap.py) at the
    MacBook's full screen."""
    sizes: dict = {}
    for name, app, vm in _tour_second_look(cfg, tmp_path, record):
        for text in _cards(vm):
            assert not any(w in text.lower() for w in FORBIDDEN), (name, text)
        b = vm["buttons"]
        assert [k for k, v in b.items() if v["look"] == "next"] in (["start_walk"], ["new_visitor"], []), (name, b)
        _screen_checks(name, app, vm, size, sizes)
        _hint_full_or_sentences(app, size, name)
        sc = app.screen
        sc.fig.set_dpi(DPI)
        found = [p for p in overlap.check(sc) if p["kind"] != "d" or size == (14.70, 9.24)]
        assert not found, name + "\n" + overlap.fmt(found)
        if name == "WALK ONCE MORE":                       # the headline is the action, as big as the card allows
            assert sc.big.get_text() == "Walk once more!"  # (full size on a 16:9 screen; on the MacBook's own,
            steps = {(16, 9): 0, (14.70, 9.24): 1, (14.7, 9.56): 2}[size]   # 6 % too wide: one fit step, 0.92)
            assert sc.big.get_fontsize() == pytest.approx(0.92 ** steps * screen.BIG_PT * size[1] / screen.BASE_H), size
    for v in sizes.values():
        assert len(v) == 1
    vms = {n: vm for n, _a, vm in _tour_second_look(cfg, tmp_path / "again", record)}
    card = {n: vm["card"] for n, vm in vms.items()}
    assert card["UZAYR walk 3 of 5"]["sub"] == "Walk 3 of 5: to the X, then stand still."
    assert len(card["UZAYR walk 3 of 5"]["dots"]["rows"][1][2]) == 5
    assert (card["WALK ONCE MORE"]["big"], card["WALK ONCE MORE"]["sub"]) == ("Walk once more!",
                                                                              "Too close to call: the same walker again.")
    assert card["THINKING both walks"]["sub"] == "Comparing both walks with Uzayr's and Thandiwe Mokoena's"
    assert card["ANSWER from both walks"]["sub"] == "(from both walks)"
    assert all(t["bars"][0][0] == "Both walks" for t in vms["ANSWER from both walks"]["tiles"].values())
    assert card["REDO 4 more times"]["sub"] == "Uzayr walks 4 more times, then the game carries on."


def _hint_full_or_sentences(app, size, name):
    """Every hint at most two lines at full size, or (a test recording on a narrow screen) a whole sentence a
    line at most one step smaller (screen.HINT_SENTENCE_FIT)."""
    h = app.screen.hint
    full = h._base_size * size[1] / screen.BASE_H
    lines = h.get_text().split("\n")
    assert len(lines) <= 2, (name, lines)
    if h.get_fontsize() != pytest.approx(full):
        assert h.get_fontsize() >= screen.HINT_SENTENCE_FIT * full - 1e-6, (name, h.get_fontsize(), lines)
        assert len(lines) == 2 and (lines[0][-1] == ":" or (lines[0][-1] == "." and lines[1][0].isupper())), \
            (name, lines)


def _drawn_hint(app, vm, size):
    """vm drawn at one screen shape (the checks of test_nothing_overlaps_on_screen): the hint's lines."""
    _screen_checks("hint", app, vm, size, {})
    _hint_full_or_sentences(app, size, vm["hint"][0])
    return app.screen.hint.get_text().split("\n")


@pytest.mark.shipped
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("size", [(16, 9), (14.7, 9.56)])
def test_a_long_name_cut_short_is_no_sentence_end(cfg, tmp_path, record, size):
    """A name over 9 letters is cut to 8 and a "." ("Bartholo.", as on the dots): the hint never splits there as if
    a sentence ended ("Click Start walk as Bartholo." / "sets off."); full size, at most two lines, clear of the
    buttons."""
    app = _app(cfg, tmp_path, record=record)
    _player(app, "Bartholomew")
    vm = app.view(time.monotonic())
    assert vm["hint"][0] == "Click Start walk as Bartholo. sets off."
    lines = _drawn_hint(app, vm, size)
    assert " ".join(lines) == vm["hint"][0] and not lines[0].endswith("Bartholo."), lines
    assert app.screen.hint.get_fontsize() == pytest.approx(app.screen.hint._base_size * size[1] / screen.BASE_H)
    assert all(len(line.split()) >= 2 for line in lines), lines


@pytest.mark.shipped
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("size", [(16, 9), (14.7, 9.56)])
def test_the_answer_line_splits_after_the_sentence(cfg, tmp_path, record, size):
    """On an answer, "Click Next walk as they set off. Uzayr's walks: 1 min old.": one line, or each sentence on
    its own line; in a test recording on the MacBook a step smaller for that (it used to split "... as they set" /
    "off. Uzayr's walks: ..."). Never three lines, never on a button."""
    app = _app(cfg, tmp_path, record=record)
    _player(app)
    t, _ = _learn(app, 100.0)
    _game(app, t)
    app.last_end_dev = None                                        # (past the walk-back lock)
    vm = _now(app)
    text = "Click Next walk as they set off. Uzayr's walks: 1 min old."
    assert vm["hint"][0] == text
    lines = _drawn_hint(app, vm, size)
    assert lines in ([text], ["Click Next walk as they set off.", "Uzayr's walks: 1 min old."]), lines
    if record and size == (14.7, 9.56):
        h = app.screen.hint
        assert len(lines) == 2 and h.get_fontsize() < h._base_size * size[1] / screen.BASE_H


@pytest.mark.shipped
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("size", [(16, 9), (14.7, 9.56)])
def test_after_redo_his_line_says_undo_walk_keeps_the_old_ones(cfg, tmp_path, record, size):
    """Right after Redo my walks, until his first new walk counts, his line also says how to take it back, in two
    lines at most (a whole sentence a line), at both screen shapes, on the day and in a test recording."""
    app = _app(cfg, tmp_path, record=record)
    _player(app)
    t, _ = _learn(app, 100.0)
    t = _game(app, t)
    _type(app, ["ctrl+u"])
    assert app._can_unredo()
    vm = _now(app)
    text = "Click Start walk as you set off. Undo walk keeps the old ones."
    assert vm["hint"][0] == text
    lines = _drawn_hint(app, vm, size)
    assert lines in ([text], ["Click Start walk as you set off.", "Undo walk keeps the old ones."]), lines
    _walk(app, t)                                                  # his first new walk: no way back now
    assert not app._can_unredo() and _now(app)["hint"][0] == "Click Start walk, then walk to the X yourself."


@pytest.mark.shipped
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("size", [(16, 9), (14.7, 9.56)])
def test_the_lines_after_measure_room_fit(cfg, tmp_path, record, size):
    """After Measure room: the room level beside the run's quietest on one line, and "Someone moved: ..." a whole
    sentence a line, both at full size and clear of the buttons, at both screen shapes, on the day and in a test
    recording."""
    app = _app(cfg, tmp_path, record=record)
    n, v, nfft = len(app.chain.freqs), app.chain.spec.variant, app.chain.spec.nfft
    full = app.screen.hint._base_size * size[1] / screen.BASE_H
    app.quiet_e0 = 11.44
    app._set_baseline(chain.Baseline(np.ones(n), 26.31, 100, v, nfft), "captured")
    vm = app.view(time.monotonic())
    assert vm["hint"][0] == "Room level 26.31 (quietest today 11.44)."
    assert _drawn_hint(app, vm, size) == [vm["hint"][0]] and app.screen.hint.get_fontsize() == pytest.approx(full)
    app._set_baseline(chain.Baseline(np.ones(n), 0.76, 100, v, nfft, disturbed=True, d=35.8), "captured")
    vm = app.view(time.monotonic())
    first, second = "Someone moved: using the quiet part.", "Measure room again when everyone is still."
    assert vm["hint"][0] == f"{first} {second}"
    assert _drawn_hint(app, vm, size) == [first, second] and app.screen.hint.get_fontsize() == pytest.approx(full)


@pytest.mark.shipped
@pytest.mark.parametrize("size", [(14.70, 9.24), (16, 9)])
def test_the_lines_when_todays_level_is_kept_fit(cfg, tmp_path, size):
    """Measure room not clearly quieter than today's level (on the day): "Busier than earlier: kept the earlier room
    level 26.31." and its two siblings, "Quieter than earlier: room level 26.31." and a measurement dropped with
    today's level in use, at full size in one line or split after the ":", clear of the buttons, on the MacBook's
    full screen and a 16:9 screen."""
    app = _app(cfg, tmp_path)
    n, v, nfft = len(app.chain.freqs), app.chain.spec.variant, app.chain.spec.nfft
    full = app.screen.hint._base_size * size[1] / screen.BASE_H
    level = chain.Baseline(np.ones(n), 26.31, 100, v, nfft)
    app.chain.baseline = level
    for b, why in ((chain.Baseline(np.ones(n), 99.99, 100, v, nfft), "Busier than earlier"),
                   (chain.Baseline(np.ones(n), 27.0, 100, v, nfft), "About the same as earlier"),
                   (chain.Baseline(np.ones(n), 0.76, 100, v, nfft, disturbed=True, d=35.8), "Someone moved")):
        app.prev = app.state
        app._keep_level(b, level, float(cfg["baseline"]["recheck_ratio"]))
        vm = app.view(time.monotonic())
        assert vm["hint"][0] == f"{why}: kept the earlier room level 26.31."
        assert _drawn_hint(app, vm, size) in ([vm["hint"][0]], [f"{why}:", "kept the earlier room level 26.31."])
        assert app.screen.hint.get_fontsize() == pytest.approx(full) and app.chain.baseline is level
    for text in ("Quieter than earlier: room level 26.31.", "Receiver restarted: kept the earlier room level 26.31.",
                 "Receiver silent: kept the earlier room level 26.31."):   # (a replaced level; a dropped measurement)
        app._hint(text, "green")
        vm = app.view(time.monotonic())
        assert vm["hint"][0] == text
        first, rest = text.split(": ", 1)
        assert _drawn_hint(app, vm, size) in ([text], [f"{first}:", rest])
        assert app.screen.hint.get_fontsize() == pytest.approx(full)


# ---- every screen at every size: nothing overlaps (tests/overlap.py measures the letters themselves) -----------------

# Every screen of `python -m openday screens` and the hard cases (long names, the longest "Told apart by", extreme
# tile values, the open boxes over a game, Details, a running walker), drawn at the MacBook Air's full screen (14.70 x
# 9.24 in: 1470 x 924 points below the notch; the Open Day screen), a 16:9 monitor (the Dell, 19.20 x 10.80 in; not
# used on Open Day), the MacBook's "larger text" setting (12.80 x 8.00 in) and the window (screen.WINDOW_IN: Ctrl+F,
# and the moment before full screen). At the MacBook and the Dell nothing overlaps, sticks out, is cut off or even
# touches (closer than 4 points); at the smaller two nothing overlaps, sticks out or is cut off.

SIZES = {"MacBook": (14.70, 9.24), "Dell": (19.20, 10.80), "MacBook 1280": (12.80, 8.00), "Window": (11.6, 6.525)}
MAIN = ("MacBook", "Dell")                                 # the Open Day screen (the MacBook's own) and a 16:9 monitor
DPI = 100                                                  # one pixel = one screen point (the layout follows inches)
LONG = ("Mary-Jane O'Neil", "WWWWWWWWWWWWWWWW", "Thandiwe Mokoena")   # ui.name_max_len is 16


@pytest.fixture
def _start_clear_of_the_lock(monkeypatch):
    """screenshots.start with the after-walk lock cleared first: the checks take time, and a walk must not start
    inside the lock (a test artefact)."""
    orig = S.start

    def start(a, ago=3.0):
        a.last_end_dev = None
        orig(a, ago)

    monkeypatch.setattr(S, "start", start)


def _look(a, state, now, found, card: bool = False) -> None:
    """Draw the app's screen as it is at `now` at every size and keep what the checker finds (card: the big text
    touching the line under it counts at every size, the window too)."""
    sc = a.screen
    for size, (w, h) in SIZES.items():
        sc.fig.set_dpi(DPI)
        sc.fig.set_size_inches(w, h, forward=False)
        sc._rescale()
        sc.render(a.view(now))
        for p in overlap.check(sc):
            if size in MAIN or p["kind"] != "d" or (card and {p["a"], p["b"]} == {"card.big", "card.sub"}):
                found.append(dict(p, state=f"{state} @ {size}"))


class _Check(S._Shots):
    """screenshots.py's own rounds, each shot checked at every size instead of saved."""

    def __init__(self, cfg, tmp, found):
        super().__init__(cfg, str(tmp), str(tmp), DPI)
        self.found = found

    def shot(self, a, name, notice=False, now=None, silent=False):
        if now is None:
            now = self.t_act if notice else time.monotonic() + S.LATER
        S.signal(a, silent=silent)
        _look(a, name, now, self.found)
        return ""

    def pixels(self, a, now, hide=()):                    # lights.png: each light's words in the top bar
        _look(a, "light: " + a.view(now)["light_text"], now, self.found)
        return np.zeros((924, 1470, 3), np.uint8)


@pytest.mark.usefixtures("_start_clear_of_the_lock")
def test_every_screen_fits(cfg_shipped, tmp_path):
    found: list = []
    sh = _Check(cfg_shipped, tmp_path, found)
    try:
        for part in (S._main_path, S._others, S._reuse, S._notices):
            part(sh)
            sh.close()
        S._lights(sh, str(tmp_path / "lights.png"))
    finally:
        sh.close()
    assert not found, "\n" + overlap.fmt(found)


def _answer(a, why, sure):
    S.walk(a, (1.13, 121, 2.4, 0.41), decision=S._dec("visitor", 0.15, {"speed": -0.45, "steps": -0.2,
                                                                        "swing": -0.35, "bounce": -0.1}, why))
    a._timers(a.timer + 0.01)
    assert a.state == RESULT
    if a.model is not None:
        a.model.how_sure = lambda m: sure


@pytest.mark.usefixtures("_start_clear_of_the_lock")
@pytest.mark.parametrize("record", [False, True])
def test_long_names_and_extremes_fit(cfg_shipped, tmp_path, record):
    """A 16-letter name (the longest the name box takes) through a whole round, the longest "Told apart by", How
    sure 100 %, every tile at its extreme (bounce 100 %), Details on, the End game, Quit and name boxes over a
    game, and a running walker's speed line (4.3 m/s)."""
    found: list = []
    sh = _Check(cfg_shipped, tmp_path, found)
    try:
        for name in LONG:
            a = sh.app(record=record)
            S.keys(a, ["n"] + list(name))
            sh.shot(a, f"{name}: name box", now=time.monotonic())
            S.keys(a, ["enter"])
            sh.shot(a, f"{name}: learning")
            a.debug = True
            sh.shot(a, f"{name}: learning, Details")
            a.debug = False
            S.walk(a, S.VISITOR[0])
            sh.shot(a, f"{name}: got it", now=time.monotonic())
            S.learn(a)
            _answer(a, ["movement strength", "stride rhythm"], 1.0)
            sh.shot(a, f"{name}: answer, longest why")
            a.last["f"] = dict(a.last["f"], speed=9.99, steps=199.0, swing=9.9, bounce=1.0)
            for w in a.visitor.walks:
                w["f"] = dict(w["f"], speed=9.99, steps=199.0, swing=9.9, bounce=1.0)
            sh.shot(a, f"{name}: tiles at their extremes")
            for keys, what in ((["ctrl+n"], "End game box"), (["ctrl+q"], "Quit box"),
                               (["n"] + list(name), "name box over a game")):
                S.keys(a, keys)
                sh.shot(a, f"{name}: {what}", now=time.monotonic())
                S.keys(a, ["escape"])
            S.start(a, ago=3.0)
            S.echo(a, "walking", speed=4.3)
            sh.shot(a, f"{name}: a runner")
            a.discard()
    finally:
        sh.close()
    assert not found, "\n" + overlap.fmt(found)


@pytest.mark.usefixtures("_start_clear_of_the_lock")
def test_the_card_and_the_tiles_stay_readable(cfg_shipped, tmp_path):
    """The crowd's card and tiles (the narrow tiles, the shorter card): a full-size big text with a
    descender over the line under it keeps 4 points clear of that line at every size, the window too ("Ty's turn";
    "Who's walking?" over "Uzayr or Émile: ...", whose É reaches above an "l"; in a recording "Uzayr" with "Thanks:
    that was Uzayr." or "... Émile." under it), and an accent stays inside the card ("Chloé", "Émile" with a notice
    under the name, an "Émile" answer); an answer's name is centred in the card at whatever size it is fitted to (a
    long name shrinks, it does not sink), and "Ágnes" (accent and descender: matplotlib's box already holds the
    accent) is not shrunk for it; on the MacBook the line under the big text is drawn at 22 pt, and "like Lerato" on
    the tiles keeps at least 0.8 of its size, "like Thandiwe" (8 letters) at least 0.65 (as in the wider tiles)."""
    from openday import screen
    found: list = []
    sh = _Check(cfg_shipped, tmp_path, found)
    try:
        a = sh.app()
        S.keys(a, ["n", "T", "y", "enter"])
        assert a.view(time.monotonic())["card"]["big"] == "Ty's turn"
        _look(a, "Ty's turn", time.monotonic() + S.LATER, found, card=True)
        sc = a.screen
        sc.fig.set_size_inches(*SIZES["MacBook"], forward=False)
        sc._rescale()
        sc.render(a.view(time.monotonic() + S.LATER))
        assert sc.sub.get_text() and sc.sub.get_fontsize() >= 22 * SIZES["MacBook"][1] / screen.BASE_H - 1e-6
        for name in ("Chloé", "Émile"):                    # an accent above the card's top: the name shrinks
            e = sh.app(record=True)
            S.keys(e, ["n"] + list(name) + ["enter"])
            S.learn(e)
            _answer(e, ["speed"], 0.8)
            S.keys(e, ["v"])                               # It was {name}: a notice under the name
            assert e.view(time.monotonic())["card"]["sub"] == f"Thanks: that was {name}."
            _look(e, f"{name}, Thanks under it", time.monotonic(), found, card=True)
        g = sh.app(record=True)                            # the visitor's É in the line under the big text
        S.keys(g, ["n"] + list("Émile") + ["enter"])
        S.learn(g)
        vm = g.view(time.monotonic() + S.LATER)
        assert vm["card"]["big"] == "Who's walking?" and vm["card"]["sub"].startswith("Uzayr or Émile:")
        _look(g, "Who's walking? over Émile", time.monotonic() + S.LATER, found, card=True)
        S.walk(g, (0.9, 112, 1.9, 0.31), decision=S._dec("uzayr", 0.15, {"speed": 0.45}, ["speed"]))
        g._timers(g.timer + 0.01)
        S.keys(g, ["v"])                                   # It was Émile, on an answer "Uzayr"
        vm = g.view(time.monotonic())
        assert vm["card"]["big"] == "Uzayr" and vm["card"]["sub"] == "Thanks: that was Émile."
        _look(g, "Uzayr over Thanks: that was Émile.", time.monotonic(), found, card=True)
        b = sh.app(record=True)
        S.keys(b, ["n"] + list("Lerato") + ["enter"])
        S.learn(b)
        S.walk(b, (0.9, 112, 1.9, 0.31), decision=S._dec("uzayr", 0.15, {"speed": 0.45, "steps": 0.2}, ["speed"]))
        b._timers(b.timer + 0.01)
        S.keys(b, ["u"])                                   # It was Uzayr: a notice under the answer
        vm = b.view(time.monotonic())
        assert vm["card"]["big"] == "Uzayr" and vm["card"]["sub"] == "Thanks: that was Uzayr."
        _look(b, "Uzayr, Thanks under it", time.monotonic(), found, card=True)
        for name in ("Lerato", "Uzayr", "Thandiwe Mokoena", "WWWWWWWWWWWWWWWW", "Émile", "Ágnes"):
            c = sh.app()
            S.keys(c, ["n"] + list(name) + ["enter"])
            S.learn(c)
            _answer(c, ["speed"], 0.8)
            if name == "Uzayr":                            # (the answer naming Uzayr: a descender, full size)
                c.last["decision"] = S._dec("uzayr", 0.15, {"speed": 0.45}, ["speed"])
            sc = c.screen
            for size, (w, h) in SIZES.items():
                sc.fig.set_dpi(DPI)
                sc.fig.set_size_inches(w, h, forward=False)
                sc._rescale()
                sc.render(c.view(time.monotonic() + S.LATER))
                sc.fig.canvas.draw()
                r = sc.fig.canvas.get_renderer()
                card, big = sc.card.get_window_extent(r), sc.big.get_window_extent(r)
                assert not sc.sub.get_text()
                assert abs((big.y0 + big.y1) / 2 - (card.y0 + card.y1) / 2) <= 0.02 * card.height, (name, size)
                ink_top = big.y1 + screen._rise(sc.big.get_text(), sc.big.get_fontproperties(), r) \
                    * sc.big.get_fontsize() * DPI / 72.0          # (an accent above the box counted)
                pad = screen.CARD_PAD * h / screen.BASE_H * DPI / 72.0
                assert ink_top <= card.y1 - 0.9 * pad and big.y0 >= card.y0 + 0.9 * pad, (name, size)
                if name == "Ágnes":                        # (its box already holds the accent: two 0.92 steps
                    assert sc.big.get_fontsize() >= 0.84 * screen.BIG_ALONE_PT * h / screen.BASE_H, size   # at most)
                if size == "MacBook" and name in ("Lerato", "Thandiwe Mokoena"):
                    full, short = 15 * h / screen.BASE_H, name.split()[0]
                    pulls = [t["pull"] for t in sc.tiles.values() if t["pull"].get_text()]
                    assert len(pulls) == 4 and all(p.get_text() == f"like {short}" for p in pulls)
                    floor = 0.8 if short == "Lerato" else 0.65
                    assert min(p.get_fontsize() for p in pulls) >= floor * full, [p.get_fontsize() for p in pulls]
            _look(c, f"{name}: answer", time.monotonic() + S.LATER, found)
    finally:
        sh.close()
    assert not found, "\n" + overlap.fmt(found)


@pytest.mark.usefixtures("_start_clear_of_the_lock")
def test_long_names_are_cut_cleanly(cfg_shipped, tmp_path):
    """A name too long for its place: its first word, else the part before a hyphen, else cut with a "." (never
    "Nomvula-." or "Thandiwe ."), the same short form on the dots, the tiles, the hint and the It was button; the
    top line keeps the whole name."""
    sh = _Check(cfg_shipped, tmp_path, [])
    try:
        a = sh.app()
        cut = {}
        for name in ("Thandiwe Mokoena", "Nomvula-Precious", "Bartholomew", "Mary-Jane O'Neil", "Lerato"):
            a.name = name
            cut[name] = (a._vname(13), a._vname(10), a._vname(9), a._vname(8))
        assert cut["Thandiwe Mokoena"] == ("Thandiwe",) * 4
        assert cut["Nomvula-Precious"] == ("Nomvula",) * 4
        assert cut["Bartholomew"] == ("Bartholomew", "Bartholom.", "Bartholo.", "Barthol.")
        assert cut["Mary-Jane O'Neil"] == ("Mary-Jane", "Mary-Jane", "Mary-Jane", "Mary")
        assert cut["Lerato"] == ("Lerato",) * 4
        a.name = None
        S.keys(a, ["n"] + list("Thandiwe Mokoena") + ["enter"])
        vm = a.view(time.monotonic())
        assert vm["card"]["big"] == "Thandiwe's turn" and vm["who"] == "Uzayr vs Thandiwe Mokoena"
        assert vm["card"]["dots"]["rows"][0][0] == "Thandiwe" and vm["hint"][0] == "Click Start walk as Thandiwe sets off."
        S.learn(a)
        _answer(a, ["speed"], 0.9)
        vm = a.view(time.monotonic())
        assert vm["card"]["big"] == "Thandiwe" and vm["tiles"]["speed"]["bars"][2][0] == "Thandiwe"
    finally:
        sh.close()


def test_a_window_fits_every_macbook_setting():
    """Out of full screen (Ctrl+F) the window is screen.WINDOW_IN: the macOS backend opens it at x = 100 pt with its
    whole size (1 in = 100 pt), so it fits the MacBook's narrowest setting (1280 x 800 pt below the menu bar)."""
    from openday import screen
    w, h = screen.WINDOW_IN
    assert 100 + 100 * w <= 1280 and 100 * h + 28 <= 800 - 32 and w / h == pytest.approx(16 / 9)


# ---- python -m openday screens: a picture of every screen, stale pictures removed, nothing left behind ---------------

COMMON = ["04_welcome", "05_name_box", "06_learning_lerato_turn", "07_walking_learning", "08_notice_got_it",
          "09_learning_uzayr_turn", "10_notice_missed", "11_notice_last_walk_removed", "12_notice_walk_cancelled",
          "13_notice_got_you_both", "14_game", "15_walking_game", "16_thinking", "17_answer_lerato",
          "18_answer_uzayr", "19_close_call", "20_notice_lost_signal", "21_quit_box", "22_welcome_speed",
          "23_no_signal", "24_recording_learning", "25_recording_answer", "26_recording_noted", "27_recording_speed",
          "28_reuse_learning", "29_notice_reuse_got_it", "30_walks_expire_soon", "31_redo_my_walks",
          "32_notice_now_uzayr_walks", "33_notice_redo_undone", "34_notice_redo_back_to_game",
          "35_notice_redo_next_player", "36_walk_once_more", "37_answer_both_walks", "flow", "lights"]
FIRST = ["01_setup", "02_measuring_countdown", "03_measuring"]


def test_screens_every_picture_and_nothing_left_behind(cfg_shipped, tmp_path):
    out = tmp_path / "screens"
    out.mkdir()
    (out / "09_old_name.png").write_bytes(b"an old screenshot")   # a stale one of ours: removed
    (out / "holiday.png").write_bytes(b"someone's picture")       # not ours: kept (screens --out ~/Desktop)
    (out / "notes.txt").write_text("not a picture")               # kept
    state_before = sorted(os.listdir(openday.STATE))
    files = screenshots.run(cfg_shipped, str(out), dpi=30)
    names = sorted(os.listdir(out))
    want = sorted(n + ".png" for n in FIRST + COMMON)
    assert names == sorted(want + ["holiday.png", "notes.txt"])
    assert sorted(os.path.basename(f) for f in files) == want
    h, w = imread(str(out / "04_welcome.png")).shape[:2]
    assert (w, h) == (480, 270)                                    # 16 x 9 in at the given dpi
    assert sorted(os.listdir(openday.STATE)) == state_before         # the temporary app folders are gone
    for d in (openday.STATE, openday.LOGS):                            # the demo names are only in the pictures
        for name in (screenshots.NAME, screenshots.NAME2):
            found = subprocess.run(["grep", "-rl", name, d], capture_output=True, text=True).stdout
            assert found == "", found


# ---- the 3D picture: its labelled frame, the bars' heights and colours -----------------------------------------------

@pytest.mark.shipped
def test_the_frame_names_the_subcarriers_and_the_time(cfg, tmp_path):
    """The ridges stand on a labelled frame above the waterfall: subcarrier numbers under the front edge at the
    right places, the seconds back along the right edge (now in front, 4 s back), the amplitude change up the left;
    the oldest seconds' label stays inside the box; the caption top right, like the waterfall's."""
    up = "Amplitude change"
    app = _ridge_app(cfg, tmp_path)
    sc = app.screen
    box, rbox = sc.ax.get_position(), sc.rax.get_position()        # the two halves: the ridges above the waterfall
    assert (box.x0, box.y0, box.width, box.height) == pytest.approx(screen.WF_LOW)
    assert (rbox.x0, rbox.y0, rbox.width, rbox.height) == pytest.approx(screen.RIDGE_BOX)
    cap = sc.ridge_caption
    assert cap.get_text() == "What the Wi-Fi sees: each subcarrier's amplitude"
    assert sc.caption.get_text() == "Turned into speed: bright = movement"
    assert cap.get_ha() == sc.caption.get_ha() == "right"
    assert cap.get_position() == pytest.approx(sc.caption.get_position())
    texts = {t.get_text(): t for t in sc.ridge_labels}
    secs = ("now", "-1 s", "-2 s", "-3 s", "-4 s")
    assert set(texts) == {"-20", "-10", "0", "+10", "+20", "Subcarrier", *secs, up}
    m = app.ridge.points
    sc._draw_ridge(np.zeros((app.ridge.n, m)), np.zeros(app.ridge.n))
    x_front = sc._ridge_x[-1]                                      # the front ridge's points, left to right
    for num in (-20, -10, 10, 20):
        assert texts[f"{num:+d}"].get_position()[0] == pytest.approx(x_front[list(sc.ridge_sc).index(num)])
    i0, i1 = list(sc.ridge_sc).index(-1), list(sc.ridge_sc).index(1)
    assert x_front[i0] < texts["0"].get_position()[0] < x_front[i1]   # 0 (no subcarrier) between -1 and +1
    ys = [texts[k].get_position()[1] for k in secs]
    xs = [texts[k].get_position()[0] for k in secs]
    assert np.all(np.diff(ys) > 0) and np.all(np.diff(xs) > 0)     # back in time = further back (right and up)
    assert sc._ridge_x[0, -1] > 0.9                                # the oldest reach into the box's right side
    sc.fig.canvas.draw()
    box = sc.rax.get_window_extent()
    assert texts["-4 s"].get_window_extent().x1 < box.x1 - 2       # (the MacBook's shape too: test_every_screen_fits)
    assert texts[up].get_rotation() == 90
    assert all(t.get_position()[1] < screen.RF for k, t in texts.items() if k[-1].isdigit() and "s" not in k)
    app.close()


@pytest.mark.shipped
def test_movement_lies_flat_when_still_and_stands_up_with_movement(cfg, tmp_path):
    """A ridge's height: flat (RIDGE_BASE over its floor) at RIDGE_ACT[0] of the room's movement or less (a still
    room, about 1x, included), tallest at RIDGE_ACT[1] or more (never out of the box, never under its floor);
    taller = brighter; +27 and +28 (the band's noisiest edge) no taller than +26."""
    app = _ridge_app(cfg, tmp_path)
    sc = app.screen
    n, m = app.ridge.n, app.ridge.points

    def front(z, act):
        sc._draw_ridge(z, act)
        return _bar_heights(sc, m)[-1], sc.ridge_pc.get_facecolor()[-m:, :3].sum(axis=1).mean()

    ones = np.ones((n, m))
    h, _ = front(ones, np.full(n, 0.0))                            # room not measured
    assert h == pytest.approx(np.full(m, screen.RIDGE_BASE))
    h, _ = front(ones, np.full(n, screen.RIDGE_ACT[0]))
    assert h == pytest.approx(np.full(m, screen.RIDGE_BASE))
    still, dim = front(ones, np.full(n, 1.0))                     # a still room: flat
    walk, bright = front(ones, np.full(n, 6.0))
    top, _ = front(ones, np.full(n, screen.RIDGE_ACT[1]))
    u = 1.0 / screen.RIDGE_SOFT
    assert top == pytest.approx(np.full(m, screen.RIDGE_BASE + screen.RIDGE_TALL * u / np.sqrt(1 + u * u)))
    assert front(ones, np.full(n, 1e6))[0] == pytest.approx(top)  # (no taller past RIDGE_ACT[1])
    assert still == pytest.approx(np.full(m, screen.RIDGE_BASE))
    assert walk.min() > screen.RIDGE_BASE + 0.15 and walk.max() < top.min()
    assert bright > dim + 0.5
    z = ones.copy()
    z[:, 10:20] = 3.0
    h, _ = front(z, np.full(n, 6.0))
    assert h[10:20].min() > h[:8].max() + 0.1                      # the shape shows
    z = np.full((n, m), 1e6)                                       # the biggest stay inside the box
    sc._draw_ridge(z, np.full(n, 1e6))
    ys = np.concatenate([p.vertices[:, 1] for p in sc.ridge_pc.get_paths()])
    assert ys.max() < 1.0
    h, _ = front(np.full((n, m), -5.0), np.full(n, 100.0))         # never under its floor
    assert h == pytest.approx(np.full(m, screen.RIDGE_BASE))
    z = ones.copy()
    z[:, -2:] = 9.0
    h, _ = front(z, np.full(n, 6.0))
    assert h[-2:] == pytest.approx(h[-3:-2].repeat(2))
    app.close()


@pytest.mark.shipped
def test_movement_each_subcarrier_is_its_own_bar_coloured_by_height(cfg, tmp_path):
    """movement: every RIDGE_EVERY-th ridge (the newest always) is a row of bars, one per subcarrier, standing apart
    (each centred on its subcarrier, RIDGE_BAR of the space between two wide, with a thin dark edge), back to front;
    a bar's colour runs along RIDGE_CMAP with its height (flat: purple, tallest: yellow), older rows fade."""
    from matplotlib import colormaps
    app = _ridge_app(cfg, tmp_path)
    sc = app.screen
    n, m = app.ridge.n, app.ridge.points
    z = np.ones((n, m))
    z[:, 30:] = 3.0                                                # the upper subcarriers stand taller
    act = np.full(n, 10.0)
    act[:-6] = 0.0                                                 # only the newest are moving
    sc._draw_ridge(z, act)
    v = np.array([p.vertices[:4] for p in sc.ridge_pc.get_paths()])
    rows = np.arange(n - 1, -1, -screen.RIDGE_EVERY)[::-1]
    assert screen.RIDGE_EVERY == 3 and len(v) == len(rows) * m == 25 * m
    v = v.reshape(len(rows), m, 4, 2)
    x0, x1 = v[..., 0, 0], v[..., 1, 0]
    assert np.all(x1[:, :-1] < x0[:, 1:])                          # each its own: a gap to the next
    pitch = screen.RW / (sc.ridge_sc[-1] - sc.ridge_sc[0])             # neighbours: one subcarrier number apart
    assert x1 - x0 == pytest.approx(np.full(x0.shape, screen.RIDGE_BAR * pitch))
    i0, i1 = list(sc.ridge_sc).index(-1), list(sc.ridge_sc).index(1)
    assert x0[-1, i1] - x1[-1, i0] > pitch                         # no bar at 0: a wider gap there
    from matplotlib.path import Path
    assert all(p.codes is not None and p.codes[-1] == Path.CLOSEPOLY for p in sc.ridge_pc.get_paths())   # closed
    assert (x0 + x1) / 2 == pytest.approx(sc._ridge_x[rows])       # on its subcarrier (and its ridge's place)
    assert v[..., 0, 1] == pytest.approx(np.broadcast_to(sc._ridge_floor[rows, None], (len(rows), m)))
    assert np.all(np.diff(sc._ridge_floor[rows]) < 0)              # back to front: a row hides the ones behind
    face = sc.ridge_pc.get_facecolor().reshape(len(rows), m, 4)
    assert (face[..., 3] == 1.0).all()
    edge = np.asarray(sc.ridge_pc.get_edgecolor())
    assert len(edge) and (edge == sc._bg_rgba).all()               # a thin dark edge between overlapping bars
    assert np.asarray(sc.ridge_pc.get_linewidth()) == pytest.approx(screen.RIDGE_EDGE)
    h = v[..., 2, 1] - v[..., 0, 1]
    front = face[-1]
    assert h[-1, 40] > h[-1, 10] + 0.05                            # taller ...
    pl = colormaps["plasma"]
    flat, tall = np.asarray(pl(screen.RIDGE_COL0)), front[40]
    assert front[10][0] > flat[0] + 0.3 and tall[1] > front[10][1] + 0.2   # ... a new colour: purple, pink, orange
    assert face[0, 10, :3] == pytest.approx(sc._bg_rgba[:3] + screen.RIDGE_AGE[0] * (flat[:3] - sc._bg_rgba[:3]),
                                            abs=0.02)              # the oldest row: flat purple, faded
    assert h[0] == pytest.approx(np.full(m, screen.RIDGE_BASE))
    app.close()


@pytest.mark.shipped
def test_the_show_flow_colours_match_the_bars(cfg, tmp_path):
    """The speed picture runs from the background through plasma's purple, pink and orange to its yellow, like the
    bars."""
    from matplotlib.colors import to_rgba
    app = _ridge_app(cfg, tmp_path)
    assert app.screen.cmap(0.0) == pytest.approx(to_rgba(screen.BG))
    assert app.screen.cmap(1.0) == pytest.approx(to_rgba("#f6e826"))
    app.close()
