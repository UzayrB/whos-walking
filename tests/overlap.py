"""The overlap checker for the app's screen (openday/screen.py Screen), headless: used by test_screen.py.

    problems = check(screen)          # [] = clean

draws the figure, collects every visible thing on it and returns one dict per problem. Text is measured by its
ink (the letters drawn alone into a scratch canvas, cached), so "touching" means what the eye sees. Kinds:
  a  text over text
  b  a text sticking out of its own box (a button, a tile, the card, the open box), or the operator's hint past
     the end of its space or on more than two lines
  c  text or a patch over a box it does not belong to (the speed line through a text too), or the open box's
     edge cutting through something it does not cover whole
  d  touching: two unrelated things whose ink is under GAP screen points apart
  e  cut off by the edge of the screen
Allowed: a member inside its own box, a bar inside its background, the waterfall's caption in the waterfall, the
ridges' caption, their frame's labels and "Receiver silent" in the ridges' box (and its upright's title beside it),
tick labels beside their own waterfall, anything the open box covers whole, and the speed line behind a text on its
own dark backing.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox, IdentityTransform

GAP = 4.0             # "touching": closer than this, in screen points (1/100 in of the figure)
TOL = 0.5             # rounding slack, in display pixels

_INK: dict = {}


class El:
    """One visible thing: bbox / ink in display pixels, its owner box (or "fig")."""

    def __init__(self, name, kind, artist, bbox, ink=None, owner="fig", part_of=None, attached=None, z=0.0,
                 text="", pts=None, is_button=False, lines=1):
        self.name, self.kind, self.artist, self.bbox = name, kind, artist, bbox
        self.ink = ink if ink is not None else bbox
        self.owner, self.part_of, self.attached, self.z, self.text = owner, part_of, attached, z, text
        self.pts, self.is_button, self.lines = pts, is_button, lines


def text_ink(t, renderer, dpi: float) -> Optional[Bbox]:
    """The ink box of Text t in display px (None: no ink), from the text drawn alone at the same dpi."""
    bb = t.get_window_extent(renderer)
    if not t.get_text().strip():
        return None
    fp = t.get_fontproperties()
    key = (t.get_text(), tuple(fp.get_family()), fp.get_style(), str(fp.get_weight()), round(fp.get_size(), 4),
           t.get_ha(), t.get_va(), t._linespacing, round(dpi, 3))
    off = _INK.get(key)
    if off is None:
        bw, bh = max(bb.width, 1.0), max(bb.height, 1.0)
        pad = max(24.0, 0.6 * bh)
        f2 = Figure(figsize=((bw + 2 * pad) / dpi, (bh + 2 * pad) / dpi), dpi=dpi, facecolor="white")
        c2 = FigureCanvasAgg(f2)
        ha, va = t.get_ha(), t.get_va()
        x = {"left": pad, "center": pad + bw / 2, "right": pad + bw}[ha]
        y = {"bottom": pad, "top": pad + bh, "center": pad + bh / 2, "baseline": pad + 0.25 * bh,
             "center_baseline": pad + bh / 2}[va]
        t2 = f2.text(x, y, t.get_text(), fontproperties=fp.copy(), ha=ha, va=va, linespacing=t._linespacing,
                     multialignment=t._multialignment, color="black", transform=IdentityTransform())
        c2.draw()
        b2 = t2.get_window_extent(c2.get_renderer())
        img = np.asarray(c2.buffer_rgba())[:, :, :3]
        mask = img.min(axis=2) < 160
        if not mask.any():
            off = ()
        else:
            rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
            h2 = img.shape[0]
            off = (cols[0] - b2.x0, h2 - (rows[-1] + 1) - b2.y0, cols[-1] + 1 - b2.x1, h2 - rows[0] - b2.y1)
        _INK[key] = off
    if not off:
        return None
    return Bbox.from_extents(bb.x0 + off[0], bb.y0 + off[1], bb.x1 + off[2], bb.y1 + off[3])


def _patch(p, renderer, dpi) -> tuple:
    """(bbox, ink) of a Rectangle in display px: ink with half its edge line, cut to its clip box."""
    bb = p.get_window_extent(renderer)
    h = (p.get_linewidth() if p.get_edgecolor()[3] > 0 else 0.0) * dpi / 72.0 / 2.0
    ink = Bbox.from_extents(bb.x0 - h, bb.y0 - h, bb.x1 + h, bb.y1 + h)
    cb = p.get_clip_box()
    if p.get_clip_on() and cb is not None:
        ink = Bbox.intersection(ink, cb) or Bbox.from_extents(0, 0, 0, 0)
        bb = Bbox.intersection(bb, cb) or Bbox.from_extents(0, 0, 0, 0)
    return bb, ink


def collect(sc, r) -> list:
    """Every visible element of Screen sc (after a draw)."""
    dpi = sc.fig.dpi
    els: list = []

    def text(t, name, owner="fig", attached=None):
        if t is None or not t.get_visible() or not t.get_text().strip():
            return
        ink = text_ink(t, r, dpi)
        if ink is not None:
            els.append(El(name, "text", t, t.get_window_extent(r), ink, owner=owner, attached=attached,
                          z=t.get_zorder(), text=t.get_text(), lines=t.get_text().count("\n") + 1))

    def box(p, name, owner="fig", kind="box", part_of=None, is_button=False):
        bb, ink = _patch(p, r, dpi)
        els.append(El(name, kind, p, bb, ink, owner=owner, part_of=part_of, z=p.get_zorder(), is_button=is_button))

    for bn, b in sc.buttons.items():
        if b["rect"].get_visible():
            box(b["rect"], f"button:{bn}", owner="overlay" if bn in ("ov_a", "ov_b") else "fig", is_button=True)
            text(b["text"], f"button:{bn}.text", owner=f"button:{bn}")
    if sc.overlay_bg.get_visible():
        box(sc.overlay_bg, "overlay")
        text(sc.overlay, "overlay.title", owner="overlay")
        text(sc.overlay_sub, "overlay.sub", owner="overlay")
    if sc.rec.get_visible():
        box(sc.rec, "rec")
        text(sc.rec_txt, "rec.text", owner="rec")
    for tn, t in sc.tiles.items():
        bb = t["ax"].get_window_extent(r)
        own = f"tile:{tn}"
        els.append(El(own, "box", t["ax"], bb, bb))
        for k in ("label", "value", "pull"):
            text(t[k], f"{own}.{k}", owner=own)
        for i, ((fg, lab, val), bg) in enumerate(zip(t["bars"], t["bgs"])):
            if bg.get_visible():
                box(bg, f"{own}.bar_bg{i}", owner=own, kind="patch")
            if fg.get_visible() and fg.get_width() > 0:
                box(fg, f"{own}.bar{i}", owner=own, kind="patch", part_of=f"{own}.bar_bg{i}")
            text(lab, f"{own}.row_name{i}", owner=own)
            text(val, f"{own}.row_value{i}", owner=own)
    cb = sc.card.get_window_extent(r)
    els.append(El("card", "box", sc.card, cb, cb))
    for nm in ("title", "big", "sub", "conf_txt", "why"):
        text(getattr(sc, nm), f"card.{nm}", owner="card")
    for k, t in enumerate(sc.dot_names):
        text(t, f"card.dot_name{k}", owner="card")
    if sc.conf_bg.get_visible():
        box(sc.conf_bg, "card.conf_bg", owner="card", kind="patch")
    if sc.conf_fg.get_visible() and sc.conf_fg.get_width() > 0:
        box(sc.conf_fg, "card.conf_bar", owner="card", kind="patch", part_of="card.conf_bg")
    for k, p in enumerate(sc.track):
        if p.get_visible():
            box(p, f"card.dot{k}", owner="card", kind="patch")
    ax = sc.ax
    wb = ax.get_window_extent(r)
    els.append(El("wf", "box", ax, wb, wb))
    text(sc.caption, "wf.caption", owner="wf")
    rax = sc.rax                                           # the ridges' box above the waterfall
    rb = rax.get_window_extent(r)
    els.append(El("ridge", "box", rax, rb, rb))
    text(sc.ridge_caption, "ridge.caption", owner="ridge")
    for t in sc.ridge_labels:                              # the frame's subcarrier numbers, seconds and names;
        if t is sc.ridge_ytitle:                           # the upright's title left of the box, like wf.ylabel
            text(t, "ridge.ylabel", attached="ridge")
        else:
            text(t, f"ridge.label[{t.get_text()}]", owner="ridge")
    text(sc.silent, "ridge.silent", owner="ridge")
    x, y = (np.asarray(v, float) for v in sc.line.get_data())
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 2 and sc.line.get_visible():
        pts = sc.line.get_transform().transform(np.column_stack([x[ok], y[ok]]))
        pts = pts[(pts[:, 0] >= wb.x0) & (pts[:, 0] <= wb.x1) & (pts[:, 1] >= wb.y0) & (pts[:, 1] <= wb.y1)]
        if len(pts) >= 2:
            h = 4.5 * dpi / 72.0 / 2.0                 # the dark outline stroke
            lb = Bbox.from_extents(pts[:, 0].min() - h, pts[:, 1].min() - h, pts[:, 0].max() + h, pts[:, 1].max() + h)
            els.append(El("wf.line", "line", sc.line, lb, lb, owner="wf", z=sc.line.get_zorder(), pts=pts))
    for lab in ax.get_xticklabels():
        text(lab, f"wf.xtick[{lab.get_text()}]", attached="wf")
    for lab in ax.get_yticklabels():
        text(lab, f"wf.ytick[{lab.get_text()}]", attached="wf")
    text(ax.yaxis.label, "wf.ylabel", attached="wf")
    for nm in ("who", "light", "light_txt", "hint", "debug"):
        text(getattr(sc, nm, None), nm)
    return els


def _inter(a: Bbox, b: Bbox) -> tuple:
    return min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)


def _gap(a: Bbox, b: Bbox) -> float:
    return math.hypot(max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1)), max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1)))


def _out(outer: Bbox, inner: Bbox) -> float:
    """How far inner sticks out of outer (the worst side), in px; <= 0 inside."""
    return max(outer.x0 - inner.x0, outer.y0 - inner.y0, inner.x1 - outer.x1, inner.y1 - outer.y1)


def check(sc, gap: float = GAP) -> list:
    """Every problem on Screen sc as it is now: dicts with kind, a, b (names), text_a, text_b, amount (screen
    points: overlap depth, gap, or overhang) and note."""
    fig = sc.fig
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pt = fig.dpi / 100.0                                   # one screen point in display px
    els = collect(sc, r)
    by = {e.name: e for e in els}
    probs: list = []

    def prob(kind, a, b, px, note=""):
        probs.append({"kind": kind, "a": a.name, "b": b.name if b is not None else "", "text_a": a.text,
                      "text_b": b.text if b is not None else "", "amount": round(px / pt, 2), "note": note})

    screen = Bbox.from_extents(0, 0, fig.bbox.width, fig.bbox.height)
    for e in els:                                          # (e) the screen's edge
        over = _out(screen, e.ink if e.kind in ("text", "line") else e.bbox)
        if over > TOL:
            prob("e", e, None, over, "cut off by the screen's edge")
    for e in els:                                          # (b) inside its own box
        c = by.get(e.owner) if e.kind != "line" else None
        if c is not None and _out(c.bbox, e.ink) > TOL:
            prob("b", e, c, _out(c.bbox, e.ink), f"sticks out of {c.name}")
    h = by.get("hint")
    if h is not None:
        x_max = fig.transFigure.transform((sc.hint._fit_x, 0))[0]
        if h.ink.x1 > x_max + TOL:
            prob("b", h, None, h.ink.x1 - x_max, "the hint runs past its space")
        if h.lines > 2:
            prob("b", h, None, 0.0, f"the hint has {h.lines} lines")

    ov = by.get("overlay")

    def under(e) -> bool:                                  # drawn below the open box
        return ov is not None and e.owner != "overlay" and e is not ov and not e.name.startswith("button:ov_") \
            and e.z < ov.z

    def related(a, b) -> bool:
        if a.owner == b.name or b.owner == a.name or a.part_of == b.name or b.part_of == a.name:
            return True
        if (a.attached and a.attached == b.name) or (b.attached and b.attached == a.name):
            return True
        oa, ob = by.get(a.owner), by.get(b.owner)
        return (oa is not None and oa.owner == b.name) or (ob is not None and ob.owner == a.name)

    def top_level(e) -> bool:                              # members of different boxes meet through their boxes
        c = by.get(e.owner)
        return e.owner == "fig" or (c is not None and _out(c.bbox, e.ink) > TOL)

    def backed(t, line) -> bool:                           # a text on its own dark backing hides the line
        p = t.artist.get_bbox_patch() if t.kind == "text" else None
        return p is not None and p.get_visible() and (p.get_alpha() or 0) >= 0.8 and t.z > line.z

    for i, a in enumerate(els):
        for b in els[i + 1:]:
            if ov is not None and (a is ov or b is ov):    # the open box (over every text, tile rows too):
                x = b if a is ov else a                    # covers it whole, or its edge cuts through it
                if under(x) and (x.kind == "text" or x.is_button):
                    w, hh = _inter(ov.ink, x.ink)
                    if w > TOL and hh > TOL and _out(ov.ink, x.ink) > 0:
                        prob("c", ov, x, min(w, hh), "the open box's edge cuts through it")
                continue
            if related(a, b):
                continue
            same = a.owner == b.owner and a.owner != "fig"
            if not same and not (top_level(a) and top_level(b)):
                continue
            if ov is not None and ((under(a) and b.owner == "overlay") or (under(b) and a.owner == "overlay")):
                continue
            if under(a) and under(b):                      # hidden by the box where they meet: nobody sees it
                w, hh = _inter(a.ink, b.ink)
                if w > 0 and hh > 0:
                    meet = Bbox.from_extents(max(a.ink.x0, b.ink.x0), max(a.ink.y0, b.ink.y0),
                                             min(a.ink.x1, b.ink.x1), min(a.ink.y1, b.ink.y1))
                    if _out(ov.ink, meet) <= 0:
                        continue
                elif _out(ov.ink, a.ink) <= 0 or _out(ov.ink, b.ink) <= 0:
                    continue
            if a.kind == "line" or b.kind == "line":       # the speed line, by its points
                ln, x = (a, b) if a.kind == "line" else (b, a)
                if x.name == "wf" or backed(x, ln):
                    continue
                hw = 4.5 * fig.dpi / 72.0 / 2.0
                p = ln.pts
                if ((p[:, 0] >= x.ink.x0 - hw) & (p[:, 0] <= x.ink.x1 + hw) & (p[:, 1] >= x.ink.y0 - hw)
                        & (p[:, 1] <= x.ink.y1 + hw)).any():
                    prob("c", x, ln, hw, "the speed line runs through it")
                continue
            w, hh = _inter(a.ink, b.ink)
            if w > TOL and hh > TOL:
                prob("a" if a.kind == "text" and b.kind == "text" else "c", a, b, min(w, hh),
                     f"overlap {w / pt:.1f} x {hh / pt:.1f}")
            elif _gap(a.ink, b.ink) < gap * pt:
                prob("d", a, b, _gap(a.ink, b.ink), f"only {_gap(a.ink, b.ink) / pt:.1f} apart")
    return probs


def fmt(probs: list) -> str:
    return "\n".join(f"  {p['kind']} {p['a']} x {p['b']}: {p['note']} ({p['amount']}) | {p['text_a'][:40]!r} | "
                     f"{p['text_b'][:40]!r}" + (f" | {p['state']}" if p.get("state") else "") for p in probs)
