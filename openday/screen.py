"""The matplotlib screen: one dark full-screen window, readable from 3 m. It draws a view model and decides nothing.
The view is the plain dictionary App.view builds (app.py); `Screen.render` draws it.

This file, in order:
  1. Colours and text sizes.
  2. The layout, in figure coordinates (every size follows the window height, so it is the same at any shape), and
     the helpers that place the operator's buttons (show_row, show_extra, hint_x1, top_boxes, redo_button), the
     card's text (card_baselines) and the lane above the 3D picture (ridge_box).
  3. Key names: norm_key spells matplotlib's key names as the config does.
  4. The window: Screen builds the figure once, then fits text, places buttons, draws the 3D ridges and the lane,
     and renders each view.

Driven from the capture loop's tick (as the thesis's live screen does): `draw_idle()` then `flush_events()`. Every
matplotlib key shortcut is switched off before the figure exists, so typing a name cannot save, quit or rescale.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from openday import chain, lane

# --- 1. Colours and text sizes ---

BG, PANEL, INK, INK2, MUTED = "#101317", "#1a1d22", "#f0efec", "#c3c2b7", "#7d7c75"
# the players and the last walk, in the graphs' purple-orange-yellow (matplotlib's "plasma", RIDGE_CMAP, SHOW_RAMP):
# Uzayr purple (plasma's own purple is too dark to read as text on BG: this one is lightened to about the old blue's
# contrast, 4.9:1 on PANEL), the visitor plasma's orange (0.78), the last walk plasma's yellow (0.96). Each pair stays
# apart for colour-blind eyes too (OKLab difference 13 or more), and the tiles name each row as well.
UZAYR, VISITOR, THIS = "#b768e0", "#fb9f3a", "#f6e826"
STATUS = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b", "grey": MUTED}
# the speed picture, matching the movement bars (RIDGE_CMAP): dark, then matplotlib's "plasma" (purple, pink, orange,
# yellow) at 0.30, 0.55, 0.78 and 0.96; the second step is plasma's 0.10 a little above the background, so the still
# room's noise stays dim
SHOW_RAMP = ["#101317", "#260c53", "#8f0da4", "#d6556d", "#fb9f3a", "#f6e826"]
BTN = "#2b3038"                                                              # an available button; the next step is INK
BASE_H = 9.0                                                                 # font sizes are set for a 9 in tall figure
MIN_FIT = 0.4                                                                # long text shrinks to at most this share
HINT_SENTENCE_FIT = 0.9          # the operator's hint may shrink this far (one step) to keep a whole sentence a line

# --- 2. The layout and the helpers that place the buttons ---

# A window (Ctrl+F, --windowed, and the moment before full screen) opens at this size: the macOS backend puts it at
# x = 100 pt and keeps its whole size, so 1160 pt wide fits every MacBook setting (1280 pt "larger text" too); full
# screen then resizes it. Headless renders (screens, tests) stay 16 x 9 in.
WINDOW_IN = (11.6, 6.525)

# The layout, in figure coordinates; every size follows the window height, so the layout is the same at any shape.
# Under the top line ("Uzayr vs {V}" at the left, the light in the top-right corner) two columns: the graphs on the
# left (the 3D ridges, then the waterfall), and on the right the operator's small buttons over the four tiles,
# exactly as wide as the tiles. The graphs' top is the buttons' top. Across the bottom: the operator's row and the
# card for the crowd.
LINE1_Y = 0.962
TILE_W, TILE_GAP = 0.15, 0.015                         # the tiles: two columns, the right one ends at 0.98
TILE_X0 = 0.98 - 2 * TILE_W - TILE_GAP
GRAPH_X0, GRAPH_X1 = 0.06, TILE_X0 - 0.03              # the graphs' left (also the hint's and the card's) and right
AXIS_TITLE_X = 0.032                                   # both graphs' upright titles, in one column left of them
# The operator works a mouse, often from the back mark: the main button (start_walk: Start walk / They've stopped /
# Next walk / Measure room) is the biggest and never moves, and its hit area runs to the screen's right edge, so the
# pointer can stay parked on it. (action, label, (x, y, w, h)); a click sends the key the app gives the button
# (else the action's first key). It sits just above the card (CARD_BOX).
CARD_BOX = (GRAPH_X0, 0.03, 0.92, 0.22)
SHOW_MAIN = ("start_walk", "Start walk", (0.78, 0.259, 0.20, 0.075))
# The operator's row left of it: fixed slots of one size and one gap (the last gap is the gap to the main button),
# vertically centred on the main button; the operator's hint sits left of the row. See show_row.
ROW_W, ROW_H, ROW_GAP = 0.095, 0.046, 0.010
_ROW_Y = SHOW_MAIN[2][1] + (SHOW_MAIN[2][3] - ROW_H) / 2
# Top right: the small buttons over the tiles, in rows of identical buttons (three slots a row, as wide as the
# tiles together), each row right-aligned (Quit at the far right): Redo my walks, End game, Measure room on top
# (their top is the graphs' top), Details and Quit under them. Redo my walks (Uzayr's learning walks again, sends
# Ctrl+U) only with pools.uzayr_reuse_min, on the left so the others keep their places; End game asks first and
# wipes the player. No Full screen button: the app opens full screen (Ctrl+F still toggles it).
TOP_ROWS = ([("enrol_uzayr", "Redo my walks"), ("wipe_visitor", "End game"), ("baseline", "Measure room")],
            [("debug", "Details"), ("quit", "Quit")])
TOP_H, TOP_GAP, TOP_Y, TOP_X1, TOP_PT = 0.040, 0.008, 0.885, 0.98, 14
TOP_W = (TOP_X1 - TILE_X0 - 2 * TOP_GAP) / 3
GRAPH_Y1 = TOP_Y + TOP_H                               # the graphs' top: the top row's top
# the light in the top-right corner: its words end at TOP_X1 (never starting left of LIGHT_X0) and its dot sits
# LIGHT_GAP left of them, a little bigger and higher than the line, with air above the buttons
LIGHT_X0, LIGHT_Y, LIGHT_PT, LIGHT_GAP = 0.70, LINE1_Y + 0.005, 34, 0.006
# the RECORDING box (bench sessions): on the top line, between "Uzayr vs {V}" and the light
REC_BOX, REC_PT = (0.30, LINE1_Y - 0.017, 0.355, 0.034), 11
# The middle band. The tiles run from TILE_Y0 (above the main button by a gap) to under the buttons, in two rows;
# the gap between the two graphs is centred on the gap between the two tile rows (TILE_MID), so the open box's top
# edge runs through both. The waterfall's bottom sits above TILE_Y0 by its time labels, which scale with the window
# too, so labels, the operator's hint (centred on the row) and the card never meet.
TILE_Y0 = SHOW_MAIN[2][1] + SHOW_MAIN[2][3] + 0.016
TILE_Y1 = TOP_Y - TOP_H - TOP_GAP - 0.015
TILE_H = (TILE_Y1 - TILE_Y0 - 0.01) / 2
TILE_MID = TILE_Y0 + TILE_H + 0.005
_WF_Y0, _GW = TILE_Y0 + 0.03, GRAPH_X1 - GRAPH_X0
# The graphs' space cut in two: the 3D ridges on top (the larger part: they are what catches the eye) and the
# waterfall below.
RIDGE_BOX = (GRAPH_X0, TILE_MID + 0.01, _GW, GRAPH_Y1 - TILE_MID - 0.01)
WF_LOW = (GRAPH_X0, _WF_Y0, _GW, TILE_MID - 0.01 - _WF_Y0)
# inside the ridge box (0..1 each way), drawn like a 3D plot's frame: a floor whose front edge (at height RF, under
# the front ridge, RX0 to RX0 + RW) carries the subcarrier numbers below it and its title under them; its right
# edge runs back in time (each older ridge sits further right and up, the oldest RDX / RDY away: ui.ridge_s, 4 s,
# reaches into the box's right side), with the seconds beside it; an upright 2 * RH tall at the front-left corner (the
# box's left edge, right above the waterfall's axis) stands for the amplitude's change (RIDGE_UP), its title in the
# waterfall's title's column.
# The caption sits in the top-right corner, like the waterfall's; only the oldest, faintest ridges can reach up
# there, so it has no backing (that would cut them off in a straight line).
RX0, RF, RW, RDX, RDY, RH = 0.012, 0.21, 0.66, 0.25, 0.32, 0.225
# The lane (lane.show; openday/lane.py decides where the marker is): a band at the top of the graphs' column, level
# with the top row of small buttons (LANE_BOX), START at its left, then the track to the X (a cross, like the tape on
# the floor) and after it the words (lane.WORDS, the widest kept free). The marker is a dot on the track, and the track
# behind it fills in the same colour (LANE_FILL_ALPHA); at rest (no walk) only START, the track and the X, dim. With
# the lane the 3D picture starts under the band, at the second row's top (ridge_box): it is that much shorter. Sizes
# in points at BASE_H (they follow the window height, as the texts do).
LANE_BOX = (GRAPH_X0, TOP_Y, _GW, TOP_H)
LANE_PT, LANE_DOT_PT, LANE_X_PT, LANE_X_LW, LANE_TRACK_PT, LANE_GAP_PT = 20, 21, 16, 3.5, 6, 7
LANE_FILL_ALPHA = 0.5
AXIS_PT, TICK_PT = 14, 12                              # the graphs' axis titles and their numbers
RIDGE_CAPTION = "What the Wi-Fi sees: each subcarrier's amplitude"
RIDGE_UP = "Amplitude change"                          # the upright's title
# A ridge is its wobble's shape across the subcarriers (mean 1, chain.Ridge), stood up by the room's movement (act:
# app.py takes the waterfall's energy over the room's for the ridge's own column and the
# RIDGE_TRAIL - 1 before it, 0.5 s, as a geometric mean, each column at least RIDGE_E_MIN: one busy column in a still
# room barely lifts a ridge, a walk's run of them stands it up). RIDGE_ACT[0] or less lies flat, RIDGE_ACT[1] or more
# stands tallest (a log scale between). On the owner's home run that left a still room flat (stray stand-ups about
# 3 a minute, from 39 with the old centred 3-column mean), a walk at about half to all of the height, rising about
# 0.15 s after the waterfall brightens. Each bar stands RIDGE_BASE above its floor plus RIDGE_TALL * squash(height /
# RIDGE_SOFT): the tallest bend over and stay inside the box.
# Drawn as bars, so each subcarrier's value stands on its own: one bar per subcarrier (RIDGE_BAR of the space between
# two neighbouring subcarriers wide, a thin dark edge of RIDGE_EDGE points between overlapping ones) for every RIDGE_EVERY-th ridge
# (the newest always: a row every 0.16 s), coloured by height along RIDGE_CMAP from RIDGE_COL0 (flat: purple) to the
# end (tallest: yellow), each row faded toward the background with age (RIDGE_AGE: a + (1 - a) * (1 - age) ** p).
# (No bar at subcarrier 0: Wi-Fi leaves it empty, so each row has a gap there.)
RIDGE_ACT, RIDGE_BASE, RIDGE_TALL, RIDGE_SOFT = (1.2, 20.0), 0.02, 0.40, 1.2
RIDGE_TRAIL, RIDGE_E_MIN = 9, 0.01
RIDGE_EVERY, RIDGE_BAR, RIDGE_EDGE, RIDGE_CMAP, RIDGE_COL0, RIDGE_AGE = 3, 0.55, 0.6, "plasma", 0.12, (0.25, 1.4)
RIDGE_TICKS = (-20, -10, 0, 10, 20)                    # subcarrier numbers on the floor's front edge
HINT_Y = SHOW_MAIN[2][1] + SHOW_MAIN[2][3] / 2
# The open box (name, Quit, End game): it covers exactly the band of the lower tiles (TILE_Y0 to the middle of the
# gaps above them and beside them), centred on the screen, so its edges cut through no text, no tile row and no
# button; its texts and two buttons inside.
_OV_X1 = TILE_X0 + TILE_W + TILE_GAP / 2
OVERLAY_BOX = (1.0 - _OV_X1, TILE_Y0, 2 * _OV_X1 - 1.0, TILE_MID - TILE_Y0)
OVERLAY_TITLE_Y, OVERLAY_SUB_Y = TILE_MID - 0.045, TILE_MID - 0.122
OVERLAY_BTN = (TILE_Y0 + 0.02, 0.052)
# The card: the big text, with the crowd's instruction line (SUB_PT: it also carries the notices, "Got it! Once
# more.") under it, its descenders CARD_GAP clear of that line's capitals; with no line under it (an
# answer) the big text is bigger (BIG_ALONE_PT) and centred, whatever size it is fitted to.
BIG_PT, BIG_ALONE_PT, SUB_PT, CARD_PAD, CARD_GAP = 100, 124, 22, 6.0, 7.0   # (CARD_GAP: 4 screen points even in
# the window, WINDOW_IN; the big text keeps CARD_PAD inside the card's top and bottom too, an accent included:
# it shrinks if it must, _fit_y0 / _fit_y1; a letter of the line's own above an "l", a visitor's É, lifts it)
CARD_PT = CARD_BOX[3] * BASE_H * 72.0                  # the card's height in points at BASE_H
# the learning dots (card coordinates): each row's name at DOTS_NAME_X, its dots from DOTS_X; the big text beside
# them ends at BIG_DOTS_X1 (further right than beside the answer's bar, 0.60: a turn's name gets more room)
DOTS_NAME_X, DOTS_X, BIG_DOTS_X1 = 0.66, 0.78, 0.64
# up to 3 dots in a row: DOTS_W wide, DOTS_STEP apart; a row of 4 or 5 (pools.n_enrol_uzayr) takes the same room with
# narrower dots (DOTS_W5, DOTS_STEP5: the fifth ends where the third does), in both rows
DOTS_W, DOTS_STEP, DOTS_W5, DOTS_STEP5, DOTS_MAX = 0.05, 0.07, 0.03, 0.04, 5
# the tiles' comparison rows (tile coordinates): the name, its value right-aligned at VALUE_X, then the bar ("Last
# walk" and a 4-character value keep their full size in the narrower tiles, on the MacBook's screen too)
VALUE_X, BAR_X, BAR_W = 0.60, 0.63, 0.31


def _slot(k: int) -> tuple:
    """The k-th slot left of the main button (1 = beside it) as (x, y, w, h)."""
    return (SHOW_MAIN[2][0] - k * (ROW_W + ROW_GAP), _ROW_Y, ROW_W, ROW_H)


def show_row(record: bool) -> list:
    """The operator's row, left to right, as (action, label, (x, y, w, h)); every slot is fixed for the whole run.
    Undo walk and New player sit beside the main button (always there, dim when they can't act), in the same
    places on the day and in a test recording. In a test recording (--record) two more slots LEFT of Undo walk are
    kept for It was {V} / It was Uzayr (It was Uzayr next to Undo walk): filled only on a game answer (It was
    Uzayr only on a speed-only answer), empty otherwise, so nothing else moves and no gap opens in the row."""
    acts = ([("confirm_visitor", "It was visitor"), ("confirm_uzayr", "It was Uzayr")] if record else []) \
        + [("discard", "Undo walk"), ("new_visitor", "New player")]
    return [(a, label, _slot(len(acts) - i)) for i, (a, label) in enumerate(acts)]


# every label a row button can show: the row is sized once from these, so its text never changes size in a run
ROW_LABELS = {"discard": ("Undo walk", "Cancel walk"), "new_visitor": ("New player",),
              "confirm_uzayr": ("It was Uzayr",)}


def show_extra(record: bool) -> list:
    """Set-up only (a recent measurement; never in --record): Use last measurement, in the
    two slots Undo walk and New player use later (neither can act before the room is measured)."""
    row = {a: box for a, _, box in show_row(record)}
    x0, y, w, h = row["discard"]
    x1 = row["new_visitor"][0] + row["new_visitor"][2]
    return [("reuse_baseline", "Use last measurement", (x0, y, x1 - x0, h))]


def hint_x1(record: bool) -> float:
    """The operator's hint ends here: left of the row, never on a button (record: the row's two It was ... slots
    are shown; while they are empty the hint may use their room)."""
    return show_row(record)[0][2][0] - 0.012


def top_boxes(redo: bool = False) -> list:
    """The top-right group as (action, label, (x, y, w, h)): TOP_ROWS from the top down, each row right-aligned
    at TOP_X1. Redo my walks only with redo."""
    out = []
    for k, row in enumerate(TOP_ROWS):
        items = [(a, label) for a, label in row if redo or a != "enrol_uzayr"]
        n, y = len(items), TOP_Y - k * (TOP_H + TOP_GAP)
        out += [(a, label, (TOP_X1 - (n - i) * TOP_W - (n - 1 - i) * TOP_GAP, y, TOP_W, TOP_H))
                for i, (a, label) in enumerate(items)]
    return out


_RISE: dict = {}


def _rise(text: str, prop, renderer, box: bool = True) -> float:
    """How far (in em) a line's letters reach above matplotlib's box for it (box), or above an "l" (not box: the
    line top card_baselines assumes). The box is the taller of the text's own letters and "lp" (the font's ascent
    and descent), set on the deeper of the two descents, so it stops at an "l" unless the text both reaches higher
    and lower than "lp". Over the box: "Émile" 0.17 (its É), "Chloé" 0.04, "Ágnes" (the g takes the box down to the
    accent) and plain text 0; over an "l": "Ágnes" 0.17 too. The first line only; cached by text and font."""
    text = text.split("\n")[0]
    if not text.strip():
        return 0.0
    key = (text, prop.get_weight(), tuple(prop.get_family()), prop.get_style())
    if key not in _RISE:
        p = prop.copy()
        p.set_size(100.0)
        _, h, d = renderer.get_text_width_height_descent(text, p, ismath=False)
        _, hl, dl = renderer.get_text_width_height_descent("lp", p, ismath=False)
        em = renderer.points_to_pixels(100.0)
        _RISE[key] = (max(0.0, (h - d) - (max(h, hl) - max(d, dl))) / em, max(0.0, (h - d) - (hl - dl)) / em)
    return _RISE[key][0 if box else 1]


def card_baselines() -> dict:
    """The card's baselines (card coordinates): "sub" (its descenders CARD_PAD above the bottom) and "big"
    over it (its descenders CARD_GAP above the line's capitals). From DejaVu Sans' ascender ("l", 0.76 em) and
    descenders ("y", 0.216; "p" in the regular weight, 0.208); every size follows the window height, so these hold
    at any window size. (An answer, with no line under it, is centred instead: Screen.render.)"""
    h = CARD_PT
    sub = CARD_PAD + 0.208 * SUB_PT
    big = sub + 0.76 * SUB_PT + CARD_GAP + 0.216 * BIG_PT
    return {"sub": sub / h, "big": big / h}


def ridge_box(with_lane: bool) -> tuple:
    """The 3D picture's box: from above the waterfall to the graphs' top (RIDGE_BOX), or, with the lane above it,
    to the second row of small buttons' top."""
    if not with_lane:
        return RIDGE_BOX
    x, y, w, _ = RIDGE_BOX
    return (x, y, w, TOP_Y - TOP_GAP - y)


def redo_button(cfg: dict) -> bool:
    """Uzayr's learning walks are kept across players (pools.uzayr_reuse_min > 0): Redo my walks."""
    return float(cfg["pools"].get("uzayr_reuse_min", 0) or 0) > 0


# --- 3. Key names ---


def norm_key(k: Optional[str]) -> str:
    """matplotlib key names -> the config's names: ' ' -> space, shift+a -> A, control -> ctrl, return -> enter."""
    if not k:
        return ""
    if k == " ":
        return "space"
    k = k.replace("control+", "ctrl+")
    if k == "return":
        return "enter"
    if k.startswith("shift+") and len(k) == 7 and k[6].isalpha():
        return k[6].upper()
    return k


# --- 4. The window ---


class Screen:
    """The window. `on_key(name, t_mono)` gets every key press; `on_close()` the window closing."""

    def __init__(self, cfg: dict, on_key: Callable[[str, float], None], on_close: Callable[[], None],
                 headless: bool = False, fullscreen: bool = True, record: bool = False) -> None:
        import matplotlib
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Rectangle
        for k in list(plt.rcParams):
            if k.startswith("keymap."):
                plt.rcParams[k] = []
        plt.rcParams["toolbar"] = "None"
        self.plt, self.cfg, self.headless = plt, cfg, headless
        self.u = cfg["ui"]
        self.fig = plt.figure(figsize=(16, 9) if headless else WINDOW_IN, facecolor=BG)
        if self.fig.canvas.manager is not None and not headless:
            self.fig.canvas.manager.set_window_title("Who's walking?")
        self._on_key = on_key
        self.fig.canvas.mpl_connect("key_press_event", lambda e: on_key(norm_key(e.key), time.monotonic()))
        self.fig.canvas.mpl_connect("button_press_event", self._click)
        self.fig.canvas.mpl_connect("close_event", lambda e: on_close())
        self.fig.canvas.mpl_connect("resize_event", lambda e: self._rescale())
        self._texts: list = []
        T = self._text
        self.redo = redo_button(cfg)                       # Redo my walks in the top-right group
        self.record = record                               # a test recording: the row keeps two It was ... slots
        self._last_click: tuple = ("", -1e9)

        # top bar: one line ("Uzayr vs {V}", the light in the corner)
        self.who = T(0.02, LINE1_Y, "", 20, color=INK2, fit_x=REC_BOX[0] - 0.01)
        self.light = T(TOP_X1, LIGHT_Y, "\u25cf", LIGHT_PT, color=STATUS["green"], ha="center")   # (x: _place_light)
        self.light_txt = T(TOP_X1, LINE1_Y, "", 20, ha="right", fit_x=TOP_X1)
        self.light_txt._fit_x0 = LIGHT_X0
        rx, ry, rw, rh = REC_BOX
        self.rec = Rectangle((rx, ry), rw, rh, transform=self.fig.transFigure, color=STATUS["red"], visible=False)
        self.fig.patches.append(self.rec)
        self.rec_txt = T(rx + rw / 2, ry + rh / 2, "RECORDING: raw radio data is being saved", REC_PT, weight="bold",
                         ha="center", visible=False, fit_x=rx + rw - 0.008)

        # waterfall (the lower part, the ridges above it)
        self.ax = self.fig.add_axes(list(WF_LOW), facecolor=BG)
        self.n_cols = int(round(self.u["waterfall_s"] / chain.hop_s(self.cfg)))
        self.cmap = LinearSegmentedColormap.from_list("echo", SHOW_RAMP)
        self.mesh = None                                   # a QuadMesh: much cheaper to draw than a resampled image
        self.set_speed_axis(float(self.u["speed_max_mps"]), 2)
        import matplotlib.patheffects as pe
        (self.line,) = self.ax.plot([], [], color=INK, lw=2.5, zorder=3,
                                    path_effects=[pe.Stroke(linewidth=4.5, foreground=BG), pe.Normal()])
        self.ax.set_xlim(-self.u["waterfall_s"], 0)
        self.ax.set_ylim(0, float(self.u["speed_max_mps"]))
        self.ax.set_ylabel("Speed (m/s)", color=INK2, fontsize=AXIS_PT)
        from matplotlib.transforms import blended_transform_factory   # the title in a fixed column (the ridges' too)
        self.ax.yaxis.set_label_coords(AXIS_TITLE_X, 0.5,
                                       transform=blended_transform_factory(self.fig.transFigure, self.ax.transAxes))
        self.ax.yaxis.label.set_va("center")
        # (under the ridges: top right, where the open box covers it whole)
        self.caption = self._text(0.988, 0.965, "Turned into speed: bright = movement", 13, color=INK2,
                                  va="top", ha="right", ax=self.ax, zorder=4)
        # a dark backing: a fast walker's speed line passes behind the caption, never through its letters
        self.caption.set_bbox(dict(facecolor=BG, alpha=0.85, edgecolor="none", pad=3))
        ticks = list(range(-int(self.u["waterfall_s"]), 1))
        self.ax.set_xticks(ticks, [f"{t} s" if t else "now" for t in ticks])
        self.ax.tick_params(colors=INK2, labelsize=TICK_PT)
        for sp in self.ax.spines.values():
            sp.set_color(PANEL)
        self.ridge_error: Optional[Exception] = None       # the last fault drawing the ridges, for the app's log
        self.lane_error: Optional[Exception] = None        # the same for the lane
        self.lane_on = lane.enabled(cfg)                   # the lane above the 3D picture (lane.show)
        self.ridge_box = ridge_box(self.lane_on)
        self._make_ridge(cfg, self.ridge_box)
        if self.lane_on:
            self._make_lane()
        # (in the middle of the ridges' box, above the open box's top edge)
        sx = (GRAPH_X0 + GRAPH_X1) / 2
        sy = self.ridge_box[1] + self.ridge_box[3] / 2
        self.silent = T(sx, sy, "", 26, color=STATUS["red"], ha="center", weight="bold", fit_x=GRAPH_X1 - 0.01)

        # tiles
        self.tiles = {}
        x1, y1 = TILE_X0 + TILE_W + TILE_GAP, TILE_Y0 + TILE_H + 0.01
        spots = {"speed": (TILE_X0, y1), "steps": (x1, y1), "swing": (TILE_X0, TILE_Y0), "bounce": (x1, TILE_Y0)}
        for name, (x, y) in spots.items():
            ax = self.fig.add_axes([x, y, TILE_W, TILE_H], facecolor=PANEL)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            t = {"ax": ax, "label": self._text(0.06, 0.86, "", 15, color=INK2, ax=ax),
                 "value": self._text(0.06, 0.58, "", 34, weight="bold", ax=ax, fit_x=0.96),
                 "pull": self._text(0.94, 0.86, "", 15, color=INK2, ha="right", ax=ax, fit_x=1.0),
                 "bars": []}
            # "like {name}" stays clear of the tile's label (from half a letter after it)
            t["pull"]._fit_after, t["pull"]._min_fit = t["label"], 0.25
            for i, col in enumerate((THIS, UZAYR, VISITOR)):
                yb = 0.34 - i * 0.12
                bg = Rectangle((BAR_X, yb), BAR_W, 0.08, color=BG)
                fg = Rectangle((BAR_X, yb), 0.0, 0.08, color=col)
                fg.set_linewidth(0)                        # an empty bar draws nothing (no stray edge line)
                ax.add_patch(bg)
                ax.add_patch(fg)
                t.setdefault("bgs", []).append(bg)
                val = self._text(VALUE_X, yb + 0.04, "", 11, color=INK, ha="right", ax=ax)   # (made first: the
                lab = self._text(0.06, yb + 0.04, "", 11, color=INK2, ax=ax, fit_x=VALUE_X)  # name is fitted after it)
                lab._fit_before = val                      # the row's name ends half a letter before its value
                t["bars"].append((fg, lab, val))
            t["bar_w"] = BAR_W
            self.tiles[name] = t

        # result card: the big text and a small instruction line for the crowd
        self.card = self.fig.add_axes(list(CARD_BOX), facecolor=PANEL)
        self.card.set_xlim(0, 1)
        self.card.set_ylim(0, 1)
        self.card.set_xticks([])
        self.card.set_yticks([])
        for sp in self.card.spines.values():
            sp.set_visible(False)
        self.title = self._text(0.03, 0.58, "", 36, weight="bold", va="center", ax=self.card, fit_x=0.97)
        # baselines, so the descenders are where they are drawn
        self._card_y, self._lift = card_baselines(), ("", 0.0)
        self.big = self._text(0.03, self._card_y["big"], "", BIG_PT, weight="bold", va="baseline", ax=self.card,
                              fit_x=0.60)
        self.big._min_fit = 0.25                           # a long name always ends before the dots and the bar
        self.big._fit_y0, self.big._fit_y1 = CARD_PAD / CARD_PT, 1.0 - CARD_PAD / CARD_PT   # (an accent too)
        self.sub = self._text(0.03, self._card_y["sub"], "", SUB_PT, color=INK, va="baseline", ax=self.card,
                              fit_x=0.97)
        self.conf_bg = Rectangle((0.62, 0.50), 0.34, 0.15, color=BG, visible=False)
        self.conf_fg = Rectangle((0.62, 0.50), 0.0, 0.15, color=UZAYR, visible=False)
        self.card.add_patch(self.conf_bg)
        self.card.add_patch(self.conf_fg)
        self.conf_txt = self._text(0.62, 0.80, "", 22, color=INK2, ax=self.card, fit_x=0.97)
        self.why = self._text(0.62, 0.25, "", 22, color=INK, va="center", ax=self.card, fit_x=0.97)
        self.why._raw, self.why._wrap_at = "", 0.8         # "Told apart by:" on a line of its own rather than small
        # the learning walks as two rows of up to DOTS_MAX named dots (in the order walked), done / now / to come
        self.track = []
        self.dot_names = []
        for i in range(2 * DOTS_MAX):
            row, j = divmod(i, DOTS_MAX)
            r = Rectangle((DOTS_X + j * DOTS_STEP, 0.60 if row == 0 else 0.25), DOTS_W, 0.22, color=BG, visible=False)
            self.card.add_patch(r)
            self.track.append(r)
        for y, col in ((0.71, UZAYR), (0.36, VISITOR)):
            self.dot_names.append(self._text(DOTS_NAME_X, y, "", 18, color=col, weight="bold", ax=self.card,
                                             fit_x=DOTS_X - 0.005))

        # overlays
        ox, oy, ow, oh = OVERLAY_BOX
        self.overlay_bg = Rectangle((ox, oy), ow, oh, transform=self.fig.transFigure, color=PANEL,
                                    ec=INK2, lw=2, visible=False, zorder=10)
        self.fig.patches.append(self.overlay_bg)
        self.overlay = T(0.5, OVERLAY_TITLE_Y, "", 36, weight="bold", ha="center", zorder=11, fit_x=0.79)
        # the sub may be a long sentence (the name box): then two lines rather than small print
        self.overlay_sub = T(0.5, OVERLAY_SUB_Y, "", 16, color=INK2, ha="center", zorder=11, fit_x=0.79)
        self.overlay_sub._raw, self.overlay_sub._wrap_at = "", 0.8
        # Details: shrinks to fit the screen's width (the operator's numbers come first)
        self.debug = T(0.06, 0.005, "", 10, color=MUTED, family="monospace", va="bottom", fit_x=0.98)
        # the operator's hint: left of the buttons, at most two lines, never on the card; 20 pt so it can be read
        # from the back mark (the app keeps every hint short enough for two lines at that size, or a step smaller
        # for a whole sentence a line: HINT_SENTENCE_FIT)
        self.hint = T(0.06, HINT_Y, "", 20, color=INK2, fit_x=hint_x1(record))
        self._hint_text = None

        # buttons: the main button, the row left of it, the top-right group, and two inside the overlay box (name
        # entry, quit). Plain figure shapes, not axes, so they cost almost nothing to draw; the app decides which
        # can act.
        self.buttons: dict = {}
        a, label, box = SHOW_MAIN
        self._button(a, label, box, cfg["keys"][a][0], size=24, hit=(box[0], box[1], 1.0, box[1] + box[3]))
        self.buttons[a]["text"].set_fontweight("bold")
        for a, label, box in show_row(record) + show_extra(record):
            self._button(a, label, box, cfg["keys"][a][0], size=16)
        for a, label, box in top_boxes(self.redo):
            self._button(a, label, box, cfg["keys"][a][0], size=TOP_PT)
        self.row = [a for a, _, _ in show_row(record)]     # the small buttons, evened to one size
        self.top = [a for a, _, _ in top_boxes(self.redo)]   # the top-right group, one size
        by, bh = OVERLAY_BTN
        self._button("ov_a", "", (0.36, by, 0.13, bh), "", zorder=12, size=16)
        self._button("ov_b", "", (0.51, by, 0.13, bh), "", zorder=12, size=16)
        self._rescale()
        if not headless:
            plt.ion()
            plt.show(block=False)
            if fullscreen:
                self.toggle_fullscreen()

    # -- helpers --------------------------------------------------------------------------------
    def _text(self, x, y, s, size, ax=None, fit_x=None, **kw):
        """A text whose size follows the window height; with fit_x (in its own axes' or the figure's coordinates)
        it shrinks, down to MIN_FIT of its size, until its right edge is inside fit_x."""
        kw.setdefault("color", INK)
        kw.setdefault("va", "center")
        t = (ax.text(x, y, s, transform=ax.transAxes, **kw) if ax is not None else self.fig.text(x, y, s, **kw))
        t._base_size, t._fit_x, t._fit_ax = size, fit_x, ax
        t.set_fontsize(size)
        self._texts.append(t)
        return t

    def _fit(self, t) -> None:
        """Size t for this window (and shrink it to fit, see _text). Optional, per text: _fit_x0 / _fit_after (start
        right of x / half a letter right of another text), _fit_before (end half a letter before another text),
        _fit_y1 (its top inside y, in the same coordinates as fit_x: the card's big text keeps an accent inside
        the card), _min_fit (its own floor instead of MIN_FIT), and _raw with _wrap_at (a text that would shrink below
        _wrap_at of its size is set on two lines, broken after a ": " or between two sentences, where the longer
        line is shortest)."""
        full = t._base_size * self.fig.get_figheight() / BASE_H
        t.set_fontsize(full)
        raw = getattr(t, "_raw", None)
        if raw is not None:                                # (a text that may wrap: from its one-line form)
            t.set_text(raw)
        if t._fit_x is None or not t.get_text() or not t.get_visible():
            return
        tr = t._fit_ax.transAxes if t._fit_ax is not None else self.fig.transFigure
        x_max = tr.transform((t._fit_x, 0))[0]
        x_min = tr.transform((t._fit_x0, 0))[0] if getattr(t, "_fit_x0", None) is not None else -np.inf
        y_max = tr.transform((0, t._fit_y1))[1] if getattr(t, "_fit_y1", None) is not None else np.inf
        y_min = tr.transform((0, t._fit_y0))[1] if getattr(t, "_fit_y0", None) is not None else -np.inf
        r = self.fig.canvas.get_renderer()
        em = 0.5 * self.fig.dpi / 72.0
        after = getattr(t, "_fit_after", None)             # starts at least half a letter right of this text
        if after is not None and after.get_text() and after.get_visible():
            x_min = max(x_min, after.get_window_extent(r).x1 + em * after.get_fontsize())
        before = getattr(t, "_fit_before", None)           # ends at least half a letter left of this text
        if before is not None and before.get_text() and before.get_visible():
            x_max = min(x_max, before.get_window_extent(r).x0 - em * before.get_fontsize())
        floor = full * getattr(t, "_min_fit", MIN_FIT)
        # (matplotlib's box stops at the font's ascent: a letter drawn above it, an accent (É, é), counts too)
        rise = _rise(t.get_text(), t.get_fontproperties(), r) * self.fig.dpi / 72.0 if y_max < np.inf else 0.0

        def shrink(size: float) -> None:
            t.set_fontsize(size)
            while size > floor and ((e := t.get_window_extent(r)).x1 > x_max or e.x0 < x_min
                                    or e.y1 + rise * size > y_max or e.y0 < y_min):
                size *= 0.92
                t.set_fontsize(size)

        shrink(full)
        wrap = getattr(t, "_wrap_at", None)
        cuts = [i + 1 for i in range(len(raw or "") - 2)
                if raw[i + 1] == " " and (raw[i] == ":" or (raw[i] == "." and raw[i + 2].isupper()))]
        if wrap is not None and cuts and t.get_fontsize() < wrap * full - 1e-9:
            t.set_fontsize(full)

            def wide(k: int) -> float:
                t.set_text(raw[:k] + "\n" + raw[k + 1:])
                return t.get_window_extent(r).width

            k = min(cuts, key=wide)
            t.set_text(raw[:k] + "\n" + raw[k + 1:])
            shrink(full)

    def _rescale(self) -> None:
        s = self.fig.get_figheight() / BASE_H              # the waterfall's numbers follow the window height too
        self.ax.tick_params(labelsize=TICK_PT * s, length=3.5 * s, width=0.8 * s, pad=3.5 * s)
        self.ax.yaxis.label.set_fontsize(AXIS_PT * s)
        for t in self._texts:
            self._fit(t)
        self._even_tiles()
        self._place_light()
        if getattr(self, "lane_on", False):
            self._place_lane()
        if getattr(self, "row", None):                     # the row's one size for this window size
            bs = self.buttons
            self._row_size = min(self._label_size(bs[a]["text"], label) for a in self.row if a in ROW_LABELS
                                 for label in ROW_LABELS[a])
        self._even_row()
        raw, self._hint_text = self._hint_text, None
        self._set_hint(raw or "")

    def _place_light(self) -> None:
        """The light's words end at the top-right corner and its dot sits LIGHT_GAP left of them; the dot's size
        follows the window height (on the MacBook's narrower screen it is wider in figure terms). Again whenever the
        words change."""
        r = self.fig.canvas.get_renderer()
        inv = self.fig.transFigure.inverted()
        self._fit(self.light_txt)
        x0 = inv.transform((self.light_txt.get_window_extent(r).x0, 0))[0] if self.light_txt.get_text() else TOP_X1
        half = self.light.get_window_extent(r).width / 2 / max(self.fig.bbox.width, 1.0)
        self.light.set_x(x0 - LIGHT_GAP - half)

    def _label_size(self, t, label: str) -> float:
        """The size t (a button's text) would be fitted to with this label; t keeps its own text and visibility."""
        text, vis = t.get_text(), t.get_visible()
        t.set_text(label)
        t.set_visible(True)                                # (a hidden text has no extent to measure)
        self._fit(t)
        size = t.get_fontsize()
        t.set_text(text)
        t.set_visible(vis)
        self._fit(t)
        return size

    def _even_row(self) -> None:
        """Every label in the button row at one size, so the row reads as one: one size for the whole run (from every
        label its buttons can show, ROW_LABELS), and the two It was ... buttons share theirs (a long name shrinks
        both). The same for the top-right group."""
        bs = getattr(self, "buttons", {})
        if getattr(self, "_row_size", None) is not None:
            size = self._row_size
            pair = [bs[a]["text"] for a in ("confirm_uzayr", "confirm_visitor") if a in self.row]
            shown = [t for t in pair if t.get_visible() and t.get_text()]   # (each was just fitted on its own)
            both = min([size] + [t.get_fontsize() for t in shown])
            for a in self.row:
                t = bs[a]["text"]
                t.set_fontsize(both if t in pair else size)
        top = [bs[a]["text"] for a in getattr(self, "top", [])]
        if top:                                            # every label fits at this size: the group is one size
            size = min(t._base_size * self.fig.get_figheight() / BASE_H for t in top)
            r = self.fig.canvas.get_renderer()
            for t in top:
                t.set_fontsize(size)
            while size > MIN_FIT * top[0]._base_size * self.fig.get_figheight() / BASE_H and any(
                    t.get_text() and t.get_window_extent(r).x1 > self.fig.transFigure.transform((t._fit_x, 0))[0]
                    for t in top):
                size *= 0.92
                for t in top:
                    t.set_fontsize(size)

    def _set(self, artist, text: str) -> None:
        if artist is self.hint:
            return self._set_hint(text)
        raw = getattr(artist, "_raw", None)                # (a text that may wrap: compared in its one-line form)
        if (artist.get_text() if raw is None else raw) != text:
            if raw is not None:
                artist._raw = text
            artist.set_text(text)
            self._fit(artist)

    def _even_tiles(self) -> None:
        """The four tiles read as one: their "like ..." notes at one size, and each comparison row's name ("Last
        walk", "Uzayr", the visitor) at one size in all four (the smallest any was fitted to)."""
        tiles = list(self.tiles.values())
        for group in [[t["pull"] for t in tiles]] + [[t["bars"][i][1] for t in tiles] for i in range(3)]:
            shown = [x for x in group if x.get_text() and x.get_visible()]
            if shown:
                size = min(x.get_fontsize() for x in shown)
                for x in shown:
                    x.set_fontsize(size)

    def _set_hint(self, text: str) -> None:
        """The operator's hint left of the buttons: one line, else two lines split after a sentence (": ", or ". "
        before a capital: a name cut short by App._vname also ends in ".") when both fit, at full size or a step
        smaller (HINT_SENTENCE_FIT), else two lines of about equal length (never one word alone on the second line),
        shrinking only if even the best two lines don't fit."""
        if text == self._hint_text:
            return
        self._hint_text = text
        t = self.hint
        base = t._base_size * self.fig.get_figheight() / BASE_H
        size, words = base, text.split()
        x_max = self.fig.transFigure.transform((t._fit_x, 0))[0]
        r = self.fig.canvas.get_renderer()

        def right(s: str) -> float:
            t.set_text(s)
            return t.get_window_extent(r).x1

        def wide(ab: tuple) -> float:
            return max(right(ab[0]), right(ab[1]))

        while words:
            t.set_fontsize(size)
            if right(" ".join(words)) <= x_max:
                t.set_text(" ".join(words))
                return
            splits = [(" ".join(words[:k]), " ".join(words[k:])) for k in range(1, len(words))]
            ends = [ab for ab in splits if len(ab[1].split()) >= 2
                    and (ab[0][-1] == ":" or (ab[0][-1] == "." and ab[1][0].isupper()))]
            s = size                                       # a whole sentence a line: easier to read, so worth a
            while ends and s >= min(size, base * HINT_SENTENCE_FIT) - 1e-9:   # slightly smaller size
                t.set_fontsize(s)
                fit = [ab for ab in ends if wide(ab) <= x_max]
                if fit:
                    t.set_text("\n".join(min(fit, key=wide)))
                    return
                s *= 0.92
            t.set_fontsize(size)
            best = min(splits, key=wide, default=None)
            if best is not None and wide(best) <= x_max:
                t.set_text("\n".join(best))
                return
            if size <= base * MIN_FIT:
                break
            size *= 0.92
        while True:                                        # at the smallest size: greedy lines, as many as it takes
            t.set_fontsize(size)
            lines, cur = [], ""
            for w in words:
                cand = f"{cur} {w}" if cur else w
                t.set_text(cand)
                if cur and t.get_window_extent(r).x1 > x_max:
                    lines.append(cur)
                    cur = w
                else:
                    cur = cand
            if cur:
                lines.append(cur)
            t.set_text("\n".join(lines))
            if not lines or size <= base * MIN_FIT or (len(lines) <= 2 and t.get_window_extent(r).x1 <= x_max):
                return
            size *= 0.92

    def _button(self, name: str, label: str, box: tuple, key: str, zorder: int = 5, size: float = 13,
                hit: Optional[tuple] = None) -> None:
        from matplotlib.patches import Rectangle
        x, y, w, h = box
        rect = Rectangle((x, y), w, h, transform=self.fig.transFigure, color=BTN, zorder=zorder, visible=False)
        self.fig.patches.append(rect)
        txt = self._text(x + w / 2, y + h / 2, label, size, ha="center", zorder=zorder + 1, fit_x=x + w - 0.004,
                         visible=False)
        self.buttons[name] = {"rect": rect, "text": txt, "box": (x, y, x + w, y + h), "key": key, "on": False,
                              "look": "hidden", "hit": hit or (x, y, x + w, y + h)}

    def _click(self, event) -> None:
        """A left click on an active button sends that button's key, exactly as a key press would. The same small
        button clicked again within ui.double_click_s is a double-click and is ignored (the main button has the
        app's own guards: walk.stop_min_s and walk.after_end_lock_s)."""
        if event.x is None or event.y is None or getattr(event, "button", 1) != 1:
            return
        fx, fy = self.fig.transFigure.inverted().transform((event.x, event.y))
        now = time.monotonic()
        for name, b in reversed(list(self.buttons.items())):   # the overlay's buttons were made last: they win
            x0, y0, x1, y1 = b["hit"]
            if b["on"] and x0 <= fx <= x1 and y0 <= fy <= y1:
                if name != SHOW_MAIN[0]:
                    last, at = self._last_click
                    if last == name and now - at < float(self.u.get("double_click_s", 0.8)):
                        return
                self._last_click = (name, now)
                self._on_key(b["key"], now)
                return

    def _render_buttons(self, specs: dict) -> None:
        """specs: name -> {"look": "off" | "on" | "next" | "dim" | "hidden", optional "label" and "key"}; missing =
        hidden. "dim" is drawn exactly like "off" but a click still sends its key, so the app can say
        why it does nothing (an It was ... button already used for this walk)."""
        looks = {"off": (BG, MUTED, "normal"), "dim": (BG, MUTED, "normal"), "on": (BTN, INK, "normal"),
                 "next": (INK, BG, "normal")}
        edges = {"off": MUTED, "dim": MUTED, "on": BTN, "next": INK}   # an off button keeps its outline
        changed = False
        for name, b in self.buttons.items():
            spec = specs.get(name, {"look": "hidden"})
            look = spec["look"]
            if spec.get("key"):
                b["key"] = spec["key"]
            b["on"] = look in ("on", "next", "dim")
            if look != b["look"]:
                b["look"], changed = look, True
                for a in (b["rect"], b["text"]):
                    a.set_visible(look != "hidden")
                if look in looks:
                    face, ink, weight = looks[look]
                    b["rect"].set_facecolor(face)
                    b["rect"].set_edgecolor(edges[look])
                    b["rect"].set_linewidth(1.5)
                    b["text"].set_color(ink)
                    b["text"].set_fontweight("bold" if name == SHOW_MAIN[0] else weight)
            label = spec.get("label")
            if label is not None and label != b["text"].get_text():
                b["text"].set_text(label)
                changed = True
        if changed:
            for a in self.row:
                self._fit(self.buttons[a]["text"])
            others = [n for n in self.buttons if n not in self.row and n not in self.top]
            for name in others:
                self._fit(self.buttons[name]["text"])
            self._even_row()

    def toggle_fullscreen(self) -> None:
        m = self.fig.canvas.manager
        if m is not None and hasattr(m, "full_screen_toggle"):
            try:
                m.full_screen_toggle()
            except Exception:  # noqa: BLE001 - some backends cannot
                pass

    def set_speed_axis(self, top_mps: float, n_bins: int) -> None:
        """The waterfall rows are spectrogram bins from 0 Hz; their top edge in m/s."""
        if self.mesh is not None:
            self.mesh.remove()
        X, Y = np.meshgrid(np.linspace(-self.u["waterfall_s"], 0, self.n_cols + 1), np.linspace(0, top_mps, n_bins + 1))
        self.mesh = self.ax.pcolormesh(X, Y, np.zeros((n_bins, self.n_cols)), cmap=self.cmap, vmin=0.0,
                                       vmax=float(self.u["db_range"]), shading="flat", antialiased=False, zorder=1)
        self.ax.set_xlim(-self.u["waterfall_s"], 0)
        self.ax.set_ylim(0, float(self.u["speed_max_mps"]))
        # no 0 (it met "-6 s" in the corner), no top label; one a metre a second (one digit each, so they keep clear
        # of the titles' column, AXIS_TITLE_X)
        self.ax.set_yticks(np.arange(1.0, float(self.u["speed_max_mps"]) - 0.25, 1.0))

    # -- the ridges ------------------------------------------------------------------------------
    def _make_ridge(self, cfg: dict, box: tuple = RIDGE_BOX) -> None:
        """The ridge box (box: ridge_box): one PolyCollection, the rows of bars drawn in order from the back so a
        single artist hides what is behind. Under them the 3D frame (floor edges, ticks, the upright) and its
        labels."""
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.colors import to_rgba
        self.rax = self.fig.add_axes(list(box), facecolor=BG)
        self.rax.set_xlim(0, 1)
        self.rax.set_ylim(0, 1)
        self.rax.set_xticks([])
        self.rax.set_yticks([])
        for sp in self.rax.spines.values():
            sp.set_color(PANEL)
        u, d = cfg["ui"], cfg["dsp"]
        fs = float(cfg["radio"]["fs_hz"])
        hop = int(d["nperseg"]) - int(d["noverlap"])
        self.ridge_n = max(2, int(round(float(u.get("ridge_s", 2.0)) * fs / hop)))
        sc = chain.subcarriers(chain.live_bins(cfg))[0]    # the subcarrier numbers, left to right (-26..+28)
        self.ridge_sc = sc
        self._ridge_u = (sc - sc[0]) / float(sc[-1] - sc[0])   # each one's place across a ridge (0..1)
        self.ridge_pc = PolyCollection([], closed=False, zorder=2, joinstyle="round", capstyle="round")
        self.rax.add_collection(self.ridge_pc)
        # the frame, a little outside the ridges: the floor's front edge (subcarriers), its right edge (time, from
        # now in front to the oldest ridge at the back) and the upright (RIDGE_UP: the amplitude's change) on the
        # box's left edge, right above the waterfall's axis (not clipped: the box's edge would cut the line in half)
        from matplotlib.transforms import blended_transform_factory
        e, tk = RX0, 0.03
        xr = RX0 + RW + e
        frame = [[(RX0 - e, RF), (xr, RF)], [(xr, RF), (xr + RDX, RF + RDY)], [(RX0 - e, RF), (RX0 - e, RF + 2 * RH)]]
        self.ridge_labels = []
        for num in RIDGE_TICKS:
            x = RX0 + RW * (num - sc[0]) / float(sc[-1] - sc[0])
            frame.append([(x, RF), (x, RF - tk)])
            self.ridge_labels.append(self._text(x, RF - tk - 0.012, f"{num:+d}" if num else "0", TICK_PT, color=INK2,
                                                ha="center", va="top", ax=self.rax))
        # each axis' title in line with it: "Subcarrier" centred under its numbers, RIDGE_UP along the
        # upright in the waterfall's title's column (AXIS_TITLE_X), so the two upright titles stand in one line
        under = (0.97 * TICK_PT + 3.0) / (box[3] * BASE_H * 72.0)   # a line of numbers and a gap, box units
        self.ridge_labels.append(self._text(RX0 + RW / 2, RF - tk - 0.012 - under, "Subcarrier", AXIS_PT,
                                            color=INK2, ha="center", va="top", ax=self.rax))
        oldest = (self.ridge_n - 1) * hop / fs             # seconds back to the oldest ridge
        for sec in range(int(np.floor(oldest + 0.1)) + 1):
            dd = min(sec / oldest, 1.0)
            x, y = xr + RDX * dd, RF + RDY * dd
            frame.append([(x, y), (x + 0.012, y)])
            self.ridge_labels.append(self._text(x + 0.02, y, "now" if sec == 0 else f"-{sec} s", TICK_PT,
                                                color=INK2, ax=self.rax))
        self.ridge_ytitle = self._text(AXIS_TITLE_X, RF + RH, RIDGE_UP, AXIS_PT, color=INK2, ha="center",
                                       va="center", rotation=90, rotation_mode="anchor",
                                       transform=blended_transform_factory(self.fig.transFigure, self.rax.transAxes))
        self.ridge_labels.append(self.ridge_ytitle)        # (outside the box, like the waterfall's own title)
        self.ridge_frame = LineCollection(frame, colors=MUTED, linewidths=1.0, zorder=1.5, capstyle="butt")
        self.ridge_frame.set_clip_on(False)
        self.rax.add_collection(self.ridge_frame)
        # top right, like the waterfall's caption (no backing: only the oldest, faintest ridges reach up there, and a
        # backing would cut them off in a straight line)
        self.ridge_caption = self._text(0.988, 0.965, RIDGE_CAPTION, 13, color=INK2, va="top", ha="right",
                                        ax=self.rax, zorder=4)
        self._ridge_x: Optional[np.ndarray] = None
        self._bg_rgba = np.asarray(to_rgba(BG))
        from matplotlib import colormaps
        self.ridge_cmap = colormaps[RIDGE_CMAP]

    def _ridge_geometry(self, m: int) -> None:
        """Where each ridge's points go (m points a ridge)."""
        u = self._ridge_u if len(self._ridge_u) == m else np.linspace(0.0, 1.0, m)
        dd = np.linspace(1.0, 0.0, self.ridge_n)           # 1 = the oldest place, 0 = the front
        self._ridge_x = RX0 + RW * u[None, :] + RDX * dd[:, None]
        self._ridge_floor, self._ridge_age = RF + RDY * dd, dd

    def _draw_ridge(self, z: np.ndarray, act: np.ndarray) -> None:
        """z: (rows, points) the ridges, oldest first (chain.Ridge.heights), each one's shape (mean 1); act: how much
        the room moved for each (the waterfall's energy over the room's, app.py; 0 = not measured). The newest ridge
        is in front, the oldest ridge_n - 1 places back. Every RIDGE_EVERY-th ridge (the newest kept) is a row of
        bars, one per subcarrier, back to front (a row hides the ones behind it); each bar
        RIDGE_BASE + RIDGE_TALL * squash(up(act) * z / RIDGE_SOFT) tall: flat (RIDGE_BASE over its floor) at
        RIDGE_ACT[0] or less, tallest at RIDGE_ACT[1] or more, coloured by that height (RIDGE_CMAP), its row faded
        with age (RIDGE_AGE); +27, +28 no taller than +26."""
        z = np.asarray(z, float)
        n = min(len(z), self.ridge_n)
        if n == 0 or z.ndim != 2 or z.shape[1] < 2:
            self.ridge_pc.set_verts([])
            return
        z, act = z[-n:], np.asarray(act, float)[-n:]
        m = z.shape[1]
        if self._ridge_x is None or self._ridge_x.shape[1] != m:
            self._ridge_geometry(m)
        pos = np.arange(self.ridge_n - n, self.ridge_n)    # the newest ridge takes the front place
        xs, age, fl = self._ridge_x[pos], self._ridge_age[pos], self._ridge_floor[pos]
        a0, a1 = RIDGE_ACT
        up = np.clip(np.log(np.maximum(act, 1e-9) / a0) / np.log(a1 / a0), 0.0, 1.0)
        z = np.maximum(z, 0.0)
        if m > 3:                                          # +27, +28: one estimate each at the band's edge, they
            z = z.copy()                                   # wobble most even in a still room (about 2x): no taller
            z[:, -2:] = np.minimum(z[:, -2:], z[:, -3:-2])   # than +26 (chain.Ridge already caps them before
                                                           # smoothing; this keeps made-up shapes to it too)
        rows = np.arange(n - 1, -1, -RIDGE_EVERY)[::-1]    # back to front, the newest last
        u = up[rows, None] * z[rows] / RIDGE_SOFT
        tall = u / np.sqrt(1.0 + u * u)                    # (rows, m), 0..1
        x = xs[rows]
        floor = np.broadcast_to(fl[rows, None], x.shape)
        top = floor + RIDGE_BASE + RIDGE_TALL * tall
        u = self._ridge_u if len(self._ridge_u) == m else np.linspace(0.0, 1.0, m)
        half = RIDGE_BAR * RW * float(np.min(np.diff(u))) / 2.0   # (neighbours: one subcarrier number apart)
        verts = np.empty(x.shape + (4, 2))
        verts[..., 0, 0] = verts[..., 3, 0] = x - half
        verts[..., 1, 0] = verts[..., 2, 0] = x + half
        verts[..., 0, 1] = verts[..., 1, 1] = floor
        verts[..., 2, 1] = verts[..., 3, 1] = top
        col = np.asarray(self.ridge_cmap(RIDGE_COL0 + (1.0 - RIDGE_COL0) * tall), float)
        a, p = RIDGE_AGE
        fade = (a + (1.0 - a) * (1.0 - age[rows]) ** p)[:, None, None]
        col[..., :3] = self._bg_rgba[:3] + fade * (col[..., :3] - self._bg_rgba[:3])   # opaque, toward the background
        col[..., 3] = 1.0
        self.ridge_pc.set_verts(verts.reshape(-1, 4, 2), closed=True)
        self.ridge_pc.set_facecolor(col.reshape(-1, 4))
        self.ridge_pc.set_edgecolor(self._bg_rgba)
        self.ridge_pc.set_linewidth(RIDGE_EDGE)

    # -- the lane ---------------------------------------------------------------------------------
    def _make_lane(self) -> None:
        """The lane's band (LANE_BOX): START, the track, the X, the marker and the track filled behind it, the words.
        Plain figure shapes, like the buttons; placed for the window by _place_lane, drawn by _render_lane."""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        x, y, w, h = LANE_BOX
        self.lane_y = y + h / 2
        self.lane_x0 = self.lane_x1 = x                    # the track's two ends (_place_lane)
        self.lane_box = Rectangle((x, y), w, h, transform=self.fig.transFigure, visible=False)   # its place only
        self.lane_start = self._text(x, self.lane_y, "START", LANE_PT, color=MUTED, weight="bold",
                                     fit_x=x + 0.25 * w)
        self.lane_words = self._text(x, self.lane_y, "", LANE_PT, color=INK, fit_x=x + w)
        self.lane_track = Rectangle((x, y), 0.0, 0.0, transform=self.fig.transFigure, color=BTN, lw=0, zorder=2)
        self.lane_fill = Rectangle((x, y), 0.0, 0.0, transform=self.fig.transFigure, color=THIS, lw=0, zorder=2.1,
                                   visible=False)
        self.lane_x = Line2D([x], [self.lane_y], marker="x", ls="none", color=MUTED, transform=self.fig.transFigure,
                             zorder=2.2)
        self.lane_dot = Line2D([x], [self.lane_y], marker="o", ls="none", color=THIS, markeredgecolor=BG,
                               transform=self.fig.transFigure, zorder=2.3, visible=False)
        self.fig.patches.extend([self.lane_box, self.lane_track, self.lane_fill])
        for ln in (self.lane_x, self.lane_dot):
            self.fig.add_artist(ln)
        self._lane_shown: Optional[tuple] = None           # (frac, colour, fade) last drawn

    def _place_lane(self) -> None:
        """The track from START (its words, then a gap and the dot's half) to the X, which ends where the widest of
        lane.WORDS still fits after it, inside the band; the dot, the X and the track sized for this window."""
        r = self.fig.canvas.get_renderer()
        s = self.fig.get_figheight() / BASE_H
        fw, fh = self.fig.get_figwidth(), self.fig.get_figheight()
        fx = lambda p: p * s / 72.0 / fw                   # points (at BASE_H) -> figure units  # noqa: E731
        fy = lambda p: p * s / 72.0 / fh                   # noqa: E731
        inv = self.fig.transFigure.inverted()
        x, y, w, h = LANE_BOX
        t = self.lane_words
        text, size = t.get_text(), t.get_fontsize()
        t.set_fontsize(t._base_size * s)
        widest = 0.0
        for word in lane.WORDS:
            t.set_text(word)
            e = t.get_window_extent(r)
            widest = max(widest, (inv.transform((e.x1, 0))[0] - inv.transform((e.x0, 0))[0]))
        t.set_text(text)
        t.set_fontsize(size)
        start_x1 = inv.transform((self.lane_start.get_window_extent(r).x1, 0))[0]
        half = fx(max(LANE_DOT_PT, LANE_X_PT + LANE_X_LW) / 2)
        self.lane_x0 = start_x1 + fx(LANE_GAP_PT) + half
        self.lane_x1 = x + w - widest - fx(LANE_GAP_PT) - half
        th = fy(LANE_TRACK_PT)
        self.lane_track.set_bounds(self.lane_x0, self.lane_y - th / 2, self.lane_x1 - self.lane_x0, th)
        self.lane_fill.set_y(self.lane_y - th / 2)
        self.lane_fill.set_height(th)
        self.lane_fill.set_x(self.lane_x0)
        self.lane_x.set_data([self.lane_x1], [self.lane_y])
        self.lane_x.set_markersize(LANE_X_PT * s)
        self.lane_x.set_markeredgewidth(LANE_X_LW * s)
        self.lane_dot.set_markersize(LANE_DOT_PT * s)
        self.lane_dot.set_markeredgewidth(1.5 * s)
        t.set_x(self.lane_x1 + half + fx(LANE_GAP_PT))
        self._fit(t)                                       # (fitted again where it now starts)
        self._lane_shown = None                            # (drawn again for this size)

    def _render_lane(self, lv: Optional[dict]) -> None:
        """lv = {"frac": 0..1 or None (no marker), "text", "colour", "fade": 0..1} (App._lane_view). At rest (no
        marker) START and the X are dim; with a marker they are lit, the dot and the filled track in its colour, the
        words beside the X, all fading with "fade"."""
        lv = lv or {}
        frac, fade = lv.get("frac"), min(max(float(lv.get("fade") or 0.0), 0.0), 1.0)
        on = frac is not None and fade > 0.0
        state = (round(float(frac), 4) if on else None, lv.get("colour") if on else None, round(fade, 3) if on else 0)
        self._set(self.lane_words, (lv.get("text") or "") if on else "")
        if state == self._lane_shown:
            return
        self._lane_shown = state
        ink = INK2 if on else MUTED
        self.lane_start.set_color(ink)
        self.lane_x.set_color(ink)
        self.lane_words.set_alpha(fade if on else 1.0)
        for a in (self.lane_dot, self.lane_fill):
            a.set_visible(on)
        if not on:
            return
        colour = lv.get("colour") or THIS
        xd = self.lane_x0 + min(max(float(frac), 0.0), 1.0) * (self.lane_x1 - self.lane_x0)
        self.lane_dot.set_data([xd], [self.lane_y])
        self.lane_dot.set_markerfacecolor(colour)
        self.lane_dot.set_alpha(fade)
        self.lane_fill.set_width(xd - self.lane_x0)
        self.lane_fill.set_facecolor(colour)
        self.lane_fill.set_alpha(LANE_FILL_ALPHA * fade)

    def _lane_failed(self, exc: Exception) -> None:
        """A fault drawing the lane: no dot and no fill this frame (drawn afresh the next), the fault kept for the
        app's log (lane_error); the rest of the frame is drawn as usual."""
        self.lane_error = exc
        self._lane_shown = None
        for a in (self.lane_dot, self.lane_fill):
            a.set_visible(False)

    # -- drawing --------------------------------------------------------------------------------
    def render(self, vm: dict) -> None:
        """Update every artist from the view model (see App.view)."""
        if self.record:                                    # no It was ... buttons shown: the hint may use their slots
            b = vm.get("buttons", {})
            x1 = hint_x1(any(b.get(a, {}).get("look", "hidden") != "hidden"
                             for a in ("confirm_uzayr", "confirm_visitor")))
            if self.hint._fit_x != x1:
                self.hint._fit_x, self._hint_text = x1, None
        self._set(self.hint, (vm.get("hint") or ("", INK2))[0])
        self.hint.set_color((vm.get("hint") or ("", INK2))[1])
        self._set(self.who, vm["who"])
        self.light.set_color(STATUS[vm["light"]])
        if self.light_txt.get_text() != vm["light_text"]:
            self._set(self.light_txt, vm["light_text"])
            self._place_light()                            # (the dot sits just left of the words)
        if self.rec_txt.get_visible() != vm["recording"]:
            for a in (self.rec, self.rec_txt):
                a.set_visible(vm["recording"])
            self._fit(self.rec_txt)                        # (a hidden text is never fitted)
        img = vm.get("wf")
        if img is not None:
            self.mesh.set_array(np.asarray(img).ravel())
        lx, ly = vm.get("line", (np.zeros(0), np.zeros(0)))
        self.line.set_data(lx, ly)
        on = not vm["silent"]                              # receiver silent: no ridges, so no empty frame either
        if self.ridge_frame.get_visible() != on:           # (only the caption and the warning stay)
            for a in [self.ridge_frame] + self.ridge_labels:
                a.set_visible(on)
        try:                                               # (a fault in the ridges never stops the rest)
            self._draw_ridge(*vm.get("ridge", (np.zeros((0, 2)), np.zeros(0))))
        except Exception as exc:  # noqa: BLE001 - the app logs it (ridge_error)
            self.ridge_pc.set_verts([])
            self.ridge_error = exc
        self._set(self.silent, vm["silent"])
        if self.lane_on:                                   # the lane above the 3D picture (a fault there never
            try:                                           # stops the rest of the frame: the app logs it)
                self._render_lane(vm.get("lane"))
            except Exception as exc:  # noqa: BLE001
                self._lane_failed(exc)
        changed = False                                    # a row or a "like ..." note changed: even them again
        for name, t in self.tiles.items():
            tv = vm["tiles"].get(name, {})
            self._set(t["label"], tv.get("label", ""))
            self._set(t["value"], tv.get("value", ""))
            changed |= t["pull"].get_text() != tv.get("pull", "")
            self._set(t["pull"], tv.get("pull", ""))
            t["pull"].set_color(tv.get("pull_color", INK2))
            rows = list(tv.get("bars", []))
            for i, ((fg, lab, val), (txt, num, frac)) in enumerate(zip(t["bars"], rows)):
                fg.set_width(t["bar_w"] * frac if frac is not None and np.isfinite(frac) else 0.0)
                if i == 0:                                 # the last walk, in the colour of who walked it
                    fg.set_color(tv.get("this_color", THIS))
                changed |= lab.get_text() != txt or val.get_text() != num
                self._set(val, num)                        # (the value first: the name is fitted against it)
                self._set(lab, txt)
            for (fg, lab, val) in t["bars"][len(rows):]:
                fg.set_width(0.0)
                changed |= bool(lab.get_text() or val.get_text())
                self._set(val, "")
                self._set(lab, "")
            for i, bg in enumerate(t["bgs"]):              # no empty bar where there is nobody to compare with
                bg.set_visible(i < len(rows))
        if changed:
            for t in self.tiles.values():
                for _, lab, _ in t["bars"]:
                    self._fit(lab)
                self._fit(t["pull"])
            self._even_tiles()
        c = vm["card"]
        conf = c.get("conf")
        dots = c.get("dots")
        self._set(self.title, c.get("title", ""))
        # the big text: with nothing under it (an answer) bigger, and centred at whatever size it is fitted to; over a
        # line whose letters reach above an "l" (a visitor's É), up by as much (worked out when the line changes)
        sub = c.get("sub") or ""
        pt = BIG_PT if sub else BIG_ALONE_PT
        if sub != self._lift[0]:
            self._lift = (sub, SUB_PT / CARD_PT * _rise(sub, self.sub.get_fontproperties(),
                                                        self.fig.canvas.get_renderer(), box=False))
        y = self._card_y["big"] + self._lift[1] if sub else 0.5
        if self.big._base_size != pt or self.big.get_position()[1] != y:
            self.big._base_size = pt
            self.big.set_y(y)
            self.big.set_va("baseline" if sub else "center")
            self._fit(self.big)
        # clear of the answer's bar; beside a turn's dots (they sit further right); else the card's whole width
        fit = 0.60 if conf is not None else (BIG_DOTS_X1 if dots is not None else 0.97)
        if self.big._fit_x != fit:
            self.big._fit_x = fit
            self._fit(self.big)
        self._set(self.big, c.get("big", ""))
        self.big.set_color(c.get("big_color", INK))
        self._render_dots(dots)
        sub_fit = 0.60 if conf is not None else 0.97
        if self.sub._fit_x != sub_fit:
            self.sub._fit_x = sub_fit
            self._fit(self.sub)
        self._set(self.sub, c.get("sub", ""))
        self.sub.set_color(c.get("sub_color", INK))
        for a in (self.conf_bg, self.conf_fg):
            a.set_visible(conf is not None)
        if conf is not None:
            self.conf_fg.set_width(0.34 * max(0.0, min(1.0, conf)))
            self.conf_fg.set_color(c.get("conf_color", UZAYR))
        self._set(self.conf_txt, c.get("conf_text", "") if conf is not None else "")
        self._set(self.why, c.get("why", "") if conf is not None else "")
        ov = vm.get("overlay")
        self.overlay_bg.set_visible(ov is not None)
        self._set(self.overlay, ov[0] if ov else "")
        self._set(self.overlay_sub, ov[1] if ov else "")
        self._render_buttons(vm.get("buttons", {}))
        self._set(self.debug, vm.get("debug") or "")

    def _render_dots(self, dots: Optional[dict]) -> None:
        """dots = {"rows": [(name, colour, [state, ...]), (name, colour, [...])]} with states done / now / next, the
        first row on top, each name in its row's colour; None hides them. No letters."""
        rows = dots["rows"] if dots else []
        many = any(len(r[2]) > 3 for r in rows)            # 4 or 5 in a row: narrower dots, the same room
        w, step = (DOTS_W5, DOTS_STEP5) if many else (DOTS_W, DOTS_STEP)
        for k, r in enumerate(self.track):
            row, j = divmod(k, DOTS_MAX)
            r.set_x(DOTS_X + j * step)
            r.set_width(w)
            st = rows[row][2][j] if row < len(rows) and j < len(rows[row][2]) else None
            r.set_visible(st is not None)
            if st is not None:
                colour = rows[row][1]
                r.set_facecolor(colour)
                r.set_alpha({"done": 1.0, "now": 0.55, "next": 0.15}[st])
                r.set_linewidth(3 if st == "now" else 0)
                r.set_edgecolor(INK if st == "now" else colour)
        for k, t in enumerate(self.dot_names):
            self._set(t, rows[k][0] if k < len(rows) else "")
            if k < len(rows):
                t.set_color(rows[k][1])

    def draw(self) -> float:
        """Request the frame and run the window's event loop, as the thesis's live screen does from the capture
        loop's tick. Returns the seconds taken (the macOS backend may finish the render in the next `pump`)."""
        t0 = time.perf_counter()
        if self.headless:
            self.fig.canvas.draw()
        else:                                              # canvas.draw() then flush would render twice on macOS
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        return time.perf_counter() - t0

    def pump(self) -> None:
        """Run the window's event loop without drawing (keys still arrive between frames)."""
        if not self.headless:
            self.fig.canvas.flush_events()

    def close(self) -> None:
        try:
            self.plt.close(self.fig)
        except Exception:  # noqa: BLE001
            pass
