"""python -m openday screens: a picture of every screen, for the README and a visual check.
(This file is screenshots.py, so it is not confused with screen.py, the window itself.)

This file, in order:
  1. The demo players and the fixed answers (NAME, NAME2, UZAYR, VISITOR, ANSWER_V, ANSWER_U, CLOSE, CLOSE_V,
     BOTH_V), and the captions of the main path (FLOW, flow).
  2. An app with a pretend receiver: _Shots makes the apps and saves each shot; signal (pretend packets), echo
     (the waterfall) and ridges (the 3D picture) fill the graphs.
  3. Clicking and walking through the app: start, walk, walked (part of a walk: the lane), keys, learn, age.
  4. The two montages: flow.png and lights.png.
  5. Every screen, in order: run, then the main path, the screens off it, the next player (Uzayr's walks
     kept) and the notices.

It drives a headless App with the shipped configuration and the demo visitors "Lerato" and "Sipho" (the next player,
whose round reuses Uzayr's walks): no receiver and no serial port. Synthetic walks go through the app's own WalkWindow,
features and state machine; the answers are fixed decisions (the screens are under test here, not the model). Writes one
1920 x 1080 PNG per state and notice into reports/screens/, plus flow.png (the main path with captions) and lights.png
(each state of the light), and deletes the stale screenshots there: only names this command makes (NN_name.png,
flow.png, lights.png), so --out pointed at a folder of other pictures deletes none of them. The app's own files (tally,
bench events) go to a temporary folder under state/ that is removed afterwards, and nothing is logged, so the demo names
end up only in the pictures.
"""

from __future__ import annotations

import collections
import os
import re
import shutil
import tempfile
import time
from typing import Optional

import numpy as np

import openday
from openday import chain, models, screen
from openday.app import App, RunLog, AGAIN, BUSY, DECIDING, ENROL_UZAYR, ENROL_VISITOR, IDLE, RESULT, WALKING, _times

# --- 1. The demo players and the fixed answers ---

NAME = "Lerato"                                            # the demo visitor (not a real visitor)
NAME2 = "Sipho"                                            # the next demo player (Uzayr's walks reused)
DPI = 120                                                  # the 16 x 9 in figure -> 1920 x 1080
LATER = 30.0                                               # "clean" shots render this long after the action
# per-person walks (speed m/s, steps/min, leg swing f90/f50, bounce): Lerato walks faster, with more swing
UZAYR = [(0.84, 106, 1.9, 0.28), (0.88, 110, 1.8, 0.31), (0.86, 108, 2.0, 0.30)]
OURS = re.compile(r"^(\d\d_[a-z0-9_]+|flow|lights)\.png$")   # the only names a run may delete
VISITOR = [(1.10, 118, 2.4, 0.40), (1.15, 122, 2.3, 0.38), (1.12, 120, 2.5, 0.42)]


def _dec(label: str, margin: float, contrib: dict, why: list, not_sure: bool = False) -> models.Decision:
    return models.Decision(label=label, margin=margin, not_sure=not_sure, dist={}, contrib=contrib, why=why)


ANSWER_V = _dec("visitor", 0.15, {"speed": -0.45, "steps": -0.15, "swing": -0.35, "bounce": -0.02},
                ["speed", "leg swing"])
ANSWER_U = _dec("uzayr", 0.12, {"speed": 0.40, "steps": 0.25, "swing": 0.03, "bounce": 0.08},
                ["speed", "step rate"])
# a close call (margin below tau): the app still names the nearer person, less sure
CLOSE = _dec("uzayr", 0.03, {"speed": 0.06, "steps": 0.02, "swing": 0.01, "bounce": 0.0}, ["speed"], not_sure=True)
# the second look (classifier.second_look): a close call on Lerato's walk, then her answer from both walks
CLOSE_V = _dec("visitor", 0.05, {"speed": -0.08, "steps": -0.02, "swing": -0.03, "bounce": 0.0}, ["speed"],
               not_sure=True)
BOTH_V = _dec("visitor", 0.22, {"speed": -0.40, "steps": -0.10, "swing": -0.30, "bounce": -0.02},
              ["speed", "leg swing"])

# flow.png: the main path, in order: (file stem, caption), after the set-up panel (run adds it first).
FLOW = [("04_welcome", "2. Click New player"),
        ("05_name_box", "3. The visitor types their name, then OK"),
        ("06_learning_lerato_turn", "4. The visitor walks twice: Start walk"),
        ("09_learning_uzayr_turn", "5. Then you walk twice (skipped while yours are fresh)"),
        ("14_game", "6. The game: either of you walks"),
        ("17_answer_lerato", "7. The answer; then Next walk"),
        ("19_close_call", "8. A close call: still a name, less sure"),
        ("28_reuse_learning", "9. Next player: your walks are kept")]


def flow(cfg: dict) -> list:
    """FLOW for this configuration: how many learning walks each walks, and with classifier.second_look the close
    call's "walk once more" in place of a close-call answer."""
    nv = int(cfg["pools"]["n_enrol"])
    nu = int(cfg["pools"].get("n_enrol_uzayr", nv))
    out = []
    for stem, cap in FLOW:
        if stem == "06_learning_lerato_turn":
            cap = f"4. The visitor walks {_times(nv)}: Start walk"
        elif stem == "09_learning_uzayr_turn":
            cap = f"5. Then you walk {_times(nu)} (skipped while yours are fresh)"
        elif stem == "19_close_call" and cfg["classifier"].get("second_look"):
            stem, cap = "36_walk_once_more", "8. Too close to call: the same walker walks once more"
        out.append((stem, cap))
    return out


# --- 2. An app with a pretend receiver ---


class _Shots:
    """One run of the screenshots: the apps it made, the files it wrote, and the temporary state folder."""

    def __init__(self, cfg: dict, out: str, tmp: str, dpi: int) -> None:
        self.cfg, self.out, self.tmp, self.dpi = cfg, out, tmp, dpi
        self.files: list[str] = []
        self.apps: list[App] = []
        self.t_act = time.monotonic()

    # -- an app with a pretend receiver -----------------------------------------------------------
    def app(self, measured: bool = True, record: bool = False, cached: bool = False) -> App:
        sd = tempfile.mkdtemp(dir=self.tmp)
        a = App(self.cfg, state_dir=sd, log=RunLog(None), headless=True, baseline_cache=False,
                record_dir=sd if record else None)
        self.apps.append(a)
        a.screen.fig.set_dpi(self.dpi)
        a.screen._rescale()
        a.chain.clock.blocks.append((0, 0.0))              # host time = device time
        signal(a)
        if cached:                                         # a measurement from 7 minutes ago ("Use last ...")
            a.cached = _baseline(a, time.time() - 7 * 60)
        if measured:
            a._set_baseline(_baseline(a), "captured")
            a.hint_ev = None                               # ("Room measured.")
        echo(a, "quiet" if measured else "none")
        return a

    def act(self) -> None:
        """Mark the moment of an action: a notice shot renders at this moment."""
        self.t_act = time.monotonic()

    def shot(self, a: App, name: str, notice: bool = False, now: Optional[float] = None,
             silent: bool = False) -> str:
        """Render the app's screen into <out>/<name>.png, packets arriving (or, silent, none). A clean shot
        renders LATER s on (notices and event hints gone); a notice shot at the moment of the last action."""
        if now is None:
            now = self.t_act if notice else time.monotonic() + LATER
        signal(a, silent=silent)
        a.screen.render(a.view(now))
        path = os.path.join(self.out, name + ".png")
        a.screen.fig.savefig(path, dpi=self.dpi, facecolor=screen.BG)
        self.files.append(path)
        return path

    def pixels(self, a: App, now: float, hide: tuple = ()) -> np.ndarray:
        """The rendered screen as an RGB array, with the artists in `hide` left out of this one frame."""
        a.screen.render(a.view(now))
        shown = [x.get_visible() for x in hide]
        for x in hide:
            x.set_visible(False)
        a.screen.fig.canvas.draw()
        for x, v in zip(hide, shown):
            x.set_visible(v)
        return np.asarray(a.screen.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self) -> None:
        for a in self.apps:
            a.close()
        self.apps = []


def _baseline(a: App, created: Optional[float] = None) -> chain.Baseline:
    n = len(a.chain.freqs)
    return chain.Baseline(np.ones(n), 1.0, 100, a.chain.spec.variant, a.chain.spec.nfft,
                        created=time.time() if created is None else created)


def signal(a: App, rate: float = 150.0, silent: bool = False) -> None:
    """Pretend packets: arriving now at `rate` a second, or (silent) none for 5 s."""
    now = time.monotonic()
    a.last_pkt_wall = now - 5.0 if silent else now + 3600.0
    a.rate_hist = collections.deque([(now - 1.0, 0), (now, rate)])


def echo(a: App, kind: str, speed: float = 1.0, seed: int = 1, busy_tail_s: float = 0.0) -> None:
    """The waterfall's last 6 s: "none" (room not measured: black), "quiet" (nobody), "walked" (a walk that
    ended about a second ago) or "walking" (one under way for 3 s). busy_tail_s: the last seconds busy (someone
    moving near the boards: the light's grey and amber)."""
    a.ring.clear()
    rng = np.random.default_rng(seed)
    n = a.ring.maxlen
    s = np.asarray(chain.speed_of(a.chain.freqs[:a.n_bins], a.cfg), float)
    span = {"walked": (-5.0, -1.0), "walking": (-3.0, 1.0)}.get(kind)
    for i in range(n):
        t = (i - n + 1) * a.hop_s                          # seconds before now
        if kind == "none":
            a.ring.append((np.zeros(a.n_bins, np.float32), np.nan, 0.0))
            continue
        img = np.clip(rng.normal(0.8, 1.4, a.n_bins), 0, None)
        v, e = np.nan, 1.0
        if span is not None and span[0] <= t <= span[1]:
            p = (t - span[0]) / (span[1] - span[0])        # the walker comes closer: the echo grows
            step = np.sin(2 * np.pi * 1.85 * t)
            v = speed * (1.0 + 0.08 * step)
            amp = 13.0 + 13.0 * min(p, 1.0) + 3.0 * step
            img += amp * (np.exp(-0.5 * ((s - v) / (0.3 * v)) ** 2)
                          + 0.5 * np.exp(-0.5 * ((s - 1.8 * v) / (0.45 * v)) ** 2) * (0.8 + 0.4 * step))
            e = 20.0 + 200.0 * min(p, 1.0)
        if busy_tail_s and t > -busy_tail_s:
            img += 9.0 * np.exp(-0.5 * ((s - 0.5) / 0.3) ** 2) * rng.uniform(0.5, 1.0)
            e = 5.0
        a.ring.append((img.astype(np.float32), v, e))
    ridges(a, kind, seed)


def ridges(a: App, kind: str, seed: int = 1) -> None:
    """The ridges above the waterfall, made up to match echo(): each subcarrier's wobble, a little noise in a still
    room, a broad shape rolling across the subcarriers while somebody walks (echo's energy stands them up)."""
    a.ridge.reset()
    a.ridge_cols = len(a.ring)                             # (echo's columns all belong to these ridges)
    rng = np.random.default_rng(seed + 100)
    m = a.ridge.points
    sc = np.linspace(0.0, 1.0, m)
    span = {"walked": (-5.0, -1.0), "walking": (-3.0, 1.0)}.get(kind)
    for i in range(a.ridge.n):
        t = (i - a.ridge.n + 1) * a.hop_s
        w = 1.0 + rng.normal(0.0, 0.12, m)                 # a still room: the noise's own small wobble
        if span is not None and span[0] <= t <= span[1]:
            step = np.sin(2 * np.pi * 1.85 * t)
            w += (0.55 + 0.15 * step) * np.sin(2 * np.pi * (1.1 * sc - 1.3 * t)) \
                + 0.25 * np.sin(2 * np.pi * (2.6 * sc + 2.1 * t)) + 0.3 * (sc - 0.5) ** 2 * 4
        a.ridge.add_shape(np.maximum(w, 0.05))


def _columns(a: App, t0: float, f: Optional[tuple]) -> chain.Columns:
    """A walk's spectrogram columns from t0 (device time): 3.6 s of walking with the given (speed, steps, swing,
    bounce), the echo growing as the walker nears, then 1.2 s standing still. None: nobody walks (6 s)."""
    hop = a.hop_s
    if f is None:
        e = np.ones(int(6.0 / hop))
        t = t0 + (np.arange(len(e)) + 0.5) * hop
        return chain.Columns(t, None, np.zeros(len(e), bool), None, e, np.full(len(e), 5.0), np.full(len(e), 9.0))
    speed, steps, swing, bounce = f
    n_walk, n_still = int(3.6 / hop), int(1.2 / hop)
    tt = np.arange(n_walk) * hop
    ph = 2 * np.pi * steps / 60.0 * tt
    hz_per_mps = 1.0 / float(chain.speed_of(1.0, a.cfg))
    f50 = speed * hz_per_mps * (1.0 + np.sqrt(2.0) * bounce * np.sin(ph))
    e = 20.0 * np.exp(2.5 * tt / tt[-1]) * (1.0 + 0.6 * np.sin(ph))
    t = t0 + (np.arange(n_walk + n_still) + 0.5) * hop
    return chain.Columns(t, None, np.zeros(len(t), bool), None, np.concatenate([e, np.ones(n_still)]),
                       np.concatenate([f50, np.full(n_still, 5.0)]),
                       np.concatenate([swing * f50, np.full(n_still, 9.0)]))


# --- 3. Clicking and walking through the app ---


def start(a: App, ago: float = 3.0) -> None:
    """Start walk, as clicked `ago` s before now."""
    signal(a)
    a.start_walk(time.monotonic() - ago)
    assert a.state == WALKING, a.state


def walk(a: App, f: Optional[tuple], decision: Optional[models.Decision] = None, ago: float = 6.0) -> None:
    """One whole walk through the app: the click `ago` s before now, its columns, its end."""
    if decision is not None:
        a._decide = lambda feats, d=decision: d
    start(a, ago)
    a._feed_walk(_columns(a, a.walk["t_dev"], f))
    assert a.walk is None, "the synthetic walk did not end"


def walked(a: App, f: tuple, secs: Optional[float] = None, start: float = 0.0) -> None:
    """A walk's columns (those walk() feeds) from `start` s for `secs` s (None: to the end) into the walk under way,
    through the app's own path: part of a walk shows the walker part-way on the lane above the 3D picture, and the
    rest (start = the same secs) finishes the walk with every column fed once, as walk() does."""
    c = _columns(a, a.walk["t_dev"], f)
    i = int(round(start / a.hop_s))
    j = len(c.t) if secs is None else i + int(round(secs / a.hop_s))
    a._feed_walk(chain.Columns(c.t[i:j], None, c.gap[i:j], None, c.energy[i:j], c.f50[i:j], c.f90[i:j]))
    if secs is not None:
        assert a.walk is not None and not a.walk["w"].ended, "the walk ended part-way"


def keys(a: App, ks) -> None:
    for k in ks:
        a.on_key_now(k, time.monotonic())


def learn(a: App) -> None:
    """Every learning walk still to come, each walked by whoever's turn it is (no clicks, as on the day)."""
    while a.state in (ENROL_VISITOR, ENROL_UZAYR):
        who = a._turn() or "visitor"
        walk(a, (UZAYR if who == "uzayr" else VISITOR)[a.done[who] % 3])


def age(a: App, minutes: float) -> None:
    """Make Uzayr's learning walks `minutes` old (the oldest of them), the room measured before them."""
    ws = a.uz_fresh.walks
    shift = time.time() - 60.0 * minutes - min(w["t"] for w in ws)
    for w in ws:
        w["t"] += shift
    a.room_at = min(a.room_at, min(w["t"] for w in ws) - 1.0)


# --- 4. The two montages: flow.png and lights.png ---


def _montage(files: list, captions: list, path: str, cols: int = 4) -> None:
    """The screenshots in a grid, 22 inches wide, each with its caption above it, on the app's background."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.image import imread
    width_in = 22.0
    rows = int(np.ceil(len(files) / cols))
    pad, cap = 0.12, 0.42                                  # inches: gap between panels, caption band
    pw = (width_in - pad * (cols + 1)) / cols
    ph = pw * 9 / 16
    height = rows * (ph + cap + pad) + pad
    fig = Figure(figsize=(width_in, height), dpi=100, facecolor=screen.BG)
    FigureCanvasAgg(fig)
    for i, (f, c) in enumerate(zip(files, captions)):
        r, k = divmod(i, cols)
        x = pad + k * (pw + pad)
        y = height - (r + 1) * (ph + cap + pad)
        ax = fig.add_axes([x / width_in, y / height, pw / width_in, ph / height])
        ax.imshow(imread(f), interpolation="antialiased")
        ax.set_axis_off()
        fig.text(x / width_in, (y + ph + 0.08) / height, c, color=screen.INK, fontsize=17, va="bottom")
    fig.savefig(path, facecolor=screen.BG)


def _lights(sh: _Shots, path: str) -> None:
    """lights.png: the light's states, each cropped from the top bar of a real render, three to a row."""
    from matplotlib.image import imsave
    crops = []

    def grab(a: App, now: Optional[float] = None) -> None:
        top = tuple(x for n in a.screen.top for x in (a.screen.buttons[n]["rect"], a.screen.buttons[n]["text"]))
        img = sh.pixels(a, time.monotonic() if now is None else now, hide=top)   # (the buttons below the light)
        h, w = img.shape[:2]                               # (the top-right corner: the dot is never left of
        crops.append(img[int(0.003 * h):int(0.075 * h),
                         int((screen.LIGHT_X0 - 0.05) * w):int((screen.TOP_X1 + 0.008) * w)])
        # LIGHT_X0 by more than its width and a gap)

    a = sh.app()
    grab(a)                                                # green Ready
    start(a)
    echo(a, "walking")
    grab(a)                                                # grey Watching
    a.discard()
    keys(a, ["n"])
    grab(a)                                                # grey Typing a name
    keys(a, ["escape"])
    echo(a, "walked", busy_tail_s=2.0)
    a.last_end_wall = time.monotonic() - 2.0
    grab(a)                                                # grey Walk back to the start
    a.last_end_wall = time.monotonic() - 30.0
    a.busy_since = time.monotonic() - 5.0
    grab(a)                                                # amber Wait for green
    a.busy_since = None
    echo(a, "quiet")
    signal(a, rate=80.0)
    grab(a)                                                # red Weak signal
    signal(a, silent=True)
    grab(a)                                                # red No signal
    b = sh.app(measured=False)
    grab(b)                                                # amber Room not measured
    b.start_baseline(time.monotonic())
    grab(b)                                                # amber Measuring the room
    c = sh.app()
    c.misses = 2
    grab(c)                                                # amber Missed twice
    c.misses = 0
    c._hint("Room was busy: click Measure room again.", "amber", 15.0, light="Measure the room again")
    grab(c)                                                # amber Measure the room again
    order = [0, 1, 2, 3, 4, 9, 10, 7, 8, 6, 5]             # green, grey, amber, red
    crops = [crops[i] for i in order]                      # (the age of Uzayr's walks is on his line, not here)
    cols, gap = 3, 12
    rows = int(np.ceil(len(crops) / cols))
    ch, cw = crops[0].shape[:2]
    canvas = np.empty((rows * ch + (rows + 1) * gap, cols * cw + (cols + 1) * gap, 3), np.uint8)
    canvas[:] = [int(screen.BG[i:i + 2], 16) for i in (1, 3, 5)]
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        y, x = gap + r * (ch + gap), gap + k * (cw + gap)
        canvas[y:y + ch, x:x + cw] = c[:ch, :cw]
    imsave(path, canvas)
    sh.files.append(path)


# --- 5. Every screen, in order ---


def run(cfg: dict, out: Optional[str] = None, dpi: int = DPI) -> list:
    """Write every screenshot into `out` (reports/screens/), then delete the PNGs there that this run did not
    write. Returns the files written."""
    out = out or os.path.join(openday.REPORTS, "screens")
    os.makedirs(out, exist_ok=True)
    os.makedirs(openday.STATE, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".screens_", dir=openday.STATE)
    sh = _Shots(cfg, out, tmp, dpi)
    try:
        _main_path(sh)
        sh.close()                                         # (each app's window is closed as soon as it is done)
        _others(sh)
        sh.close()
        if screen.redo_button(cfg):                            # (the reuse pictures: only when Uzayr's walks are kept)
            _reuse(sh)
            sh.close()
        _notices(sh)                                       # 32-35 (after 31, so 01-31 keep their numbers)
        sh.close()
        first = ("01_setup", "1. Set-up: click Measure room")
        flow_ = [first] + [x for x in flow(cfg) if screen.redo_button(cfg) or not x[0].startswith("28_")]
        flow_png = os.path.join(out, "flow.png")
        _montage([os.path.join(out, f + ".png") for f, _ in flow_], [c for _, c in flow_], flow_png,
                 cols=3 if len(flow_) == 9 else 4)
        sh.files.append(flow_png)
        _lights(sh, os.path.join(out, "lights.png"))
    finally:
        sh.close()
        shutil.rmtree(tmp, ignore_errors=True)
    keep = {os.path.abspath(f) for f in sh.files}
    for f in sorted(os.listdir(out)):
        p = os.path.abspath(os.path.join(out, f))
        if OURS.match(f) and p not in keep:
            os.remove(p)                                   # a stale screenshot of ours; any other file stays
    return sh.files


def _main_path(sh: _Shots) -> None:
    """A whole round on one app: set-up, the name, the learning walks (the visitor's two, then Uzayr's, with a
    miss, an undo and a cancel), the game."""
    a = sh.app(measured=False, cached=True)
    sh.shot(a, "01_setup")
    a.start_baseline(time.monotonic())
    assert a.state == BUSY
    sh.shot(a, "02_measuring_countdown")
    c = a.capture
    c["cols"].append(chain.Columns(np.array([c["t0"] + 3.0]), np.zeros((1, len(a.chain.freqs))), np.zeros(1, bool)))
    sh.shot(a, "03_measuring")
    a.capture = None
    a._set_baseline(_baseline(a), "captured")
    a.hint_ev = None
    echo(a, "quiet")
    assert a.state == IDLE, a.state
    sh.shot(a, "04_welcome")
    keys(a, ["n"] + list(NAME))
    sh.shot(a, "05_name_box")
    keys(a, ["enter"])
    sh.shot(a, "06_learning_lerato_turn")                  # (pools.enrol_order visitor_first: Lerato first)
    first = a._turn() or "visitor"
    f0 = (UZAYR if first == "uzayr" else VISITOR)[0]
    start(a, ago=3.0)
    echo(a, "walking", speed=f0[0])
    walked(a, f0, 1.6)                                     # (the lane: part-way)
    sh.shot(a, "07_walking_learning")
    walked(a, f0, start=1.6)                               # (the rest of the walk)
    assert a.walk is None, "the synthetic walk did not end"
    sh.act()
    echo(a, "walked", speed=f0[0])
    sh.shot(a, "08_notice_got_it", notice=True)
    while a._turn() == first:                              # the rest of Lerato's walks: then Uzayr's turn
        walk(a, (UZAYR if first == "uzayr" else VISITOR)[a.done[first] % 3])
    sh.shot(a, "09_learning_uzayr_turn")
    walk(a, None)                                          # nobody walked
    sh.act()
    echo(a, "quiet", seed=2)
    sh.shot(a, "10_notice_missed", notice=True)
    walk(a, UZAYR[a.done["uzayr"] % 3])
    keys(a, ["backspace"])                                 # Undo walk: Uzayr's turn again
    sh.act()
    sh.shot(a, "11_notice_last_walk_removed", notice=True)
    start(a)
    keys(a, ["backspace"])                                 # Cancel walk (during the walk)
    sh.act()
    sh.shot(a, "12_notice_walk_cancelled", notice=True)
    learn(a)                                               # Uzayr's walks
    sh.act()
    echo(a, "walked", speed=UZAYR[1][0])
    sh.shot(a, "13_notice_got_you_both", notice=True)
    sh.shot(a, "14_game")
    start(a, ago=3.0)
    echo(a, "walking", speed=1.12)
    walked(a, (1.12, 120, 2.4, 0.40), 2.5)                 # (the lane: nearly there)
    sh.shot(a, "15_walking_game")
    a.discard()
    walk(a, (1.13, 121, 2.4, 0.41), decision=ANSWER_V)
    echo(a, "walked", speed=1.13)
    assert a.state == DECIDING, a.state
    sh.shot(a, "16_thinking")
    a._timers(a.timer + 0.01)
    assert a.state == RESULT, a.state
    sh.shot(a, "17_answer_lerato")
    walk(a, (0.85, 107, 1.9, 0.29), decision=ANSWER_U)
    a._timers(a.timer + 0.01)
    echo(a, "walked", speed=0.85, seed=3)
    sh.shot(a, "18_answer_uzayr")
    walk(a, (0.92, 110, 2.0, 0.33), decision=CLOSE)
    a._timers(a.timer + 0.01)
    echo(a, "walked", speed=0.92, seed=4)
    if a.second_look:                                      # too close to call: the same walker walks once more,
        assert a.state == AGAIN, a.state                   # and is still close on both walks: a name, less sure
        sh.shot(a, "36_walk_once_more")
        walk(a, (0.92, 110, 2.0, 0.33), decision=CLOSE)
        a._timers(a.timer + 0.01)
        assert a.state == RESULT and a.last.get("both"), a.state
    sh.shot(a, "19_close_call")
    keys(a, ["ctrl+q"])
    sh.act()
    sh.shot(a, "21_quit_box", notice=True)
    keys(a, ["escape"])
    start(a)
    a._abort("Signal lost")
    sh.act()
    echo(a, "quiet", seed=5)
    sh.shot(a, "20_notice_lost_signal", notice=True)
    if a.second_look:                                      # Lerato's close call, then her answer from both walks
        walk(a, (1.11, 119, 2.4, 0.40), decision=CLOSE_V)
        a._timers(a.timer + 0.01)
        assert a.state == AGAIN, a.state
        walk(a, (1.14, 122, 2.5, 0.41), decision=BOTH_V)
        a._timers(a.timer + 0.01)
        assert a.state == RESULT and a.last.get("both"), a.state
        echo(a, "walked", speed=1.14, seed=11)
        sh.shot(a, "37_answer_both_walks")


def _others(sh: _Shots) -> None:
    """Screens off the main path: a speed-only walk, no signal, and a test recording (--record: learning needs no
    clicks; on an answer, and on a speed-only walk, It was ... is optional, and dim once used)."""
    a = sh.app()
    walk(a, (1.04, 112, 2.2, 0.33))                        # Start walk on the welcome screen: speed only
    echo(a, "walked", speed=1.04)
    sh.shot(a, "22_welcome_speed")
    c = sh.app()
    sh.shot(c, "23_no_signal", silent=True)
    b = sh.app(record=True)                                # --record (the solo test): the turn labels the walk
    keys(b, ["n"] + list(NAME) + ["enter"])
    walk(b, VISITOR[0])
    echo(b, "walked", speed=VISITOR[0][0])
    sh.shot(b, "24_recording_learning")
    learn(b)                                               # the other turns: no clicks
    walk(b, (1.13, 121, 2.4, 0.41), decision=ANSWER_V)
    b._timers(b.timer + 0.01)
    echo(b, "walked", speed=1.13)
    assert b.state == RESULT, b.state
    sh.shot(b, "25_recording_answer")
    keys(b, ["v"])                                         # It was Lerato: both buttons dim, Thanks on the card
    sh.act()
    sh.shot(b, "26_recording_noted", now=sh.t_act + 2.0)
    d = sh.app(record=True)                                # a timed walk in the solo test: It was Uzayr only
    walk(d, (1.04, 112, 2.2, 0.33))
    echo(d, "walked", speed=1.04)
    sh.shot(d, "27_recording_speed")


def _reuse(sh: _Shots) -> None:
    """The next player within pools.uzayr_reuse_min: Uzayr's walks are kept, so only Sipho walks (his row already
    done), then the game at once; Uzayr's walks nearly too old (his line says how long is left; the light stays
    green); Redo my walks."""
    a = sh.app()
    keys(a, ["n"] + list(NAME) + ["enter"])
    learn(a)
    walk(a, (1.13, 121, 2.4, 0.41), decision=ANSWER_V)
    a._timers(a.timer + 0.01)
    age(a, 4.0)
    keys(a, ["n"] + list(NAME2) + ["enter"])
    assert a.uz_reused, "Uzayr's walks were not reused"
    echo(a, "quiet", seed=6)
    sh.shot(a, "28_reuse_learning")
    learn(a)                                               # Sipho's walks only
    sh.act()
    echo(a, "walked", speed=VISITOR[1][0])
    sh.shot(a, "29_notice_reuse_got_it", notice=True)
    walk(a, (0.85, 107, 1.9, 0.29), decision=ANSWER_U)
    a._timers(a.timer + 0.01)
    echo(a, "walked", speed=0.85, seed=7)
    age(a, a.reuse_min - 1.0)                              # a minute before they expire
    sh.shot(a, "30_walks_expire_soon")
    keys(a, ["ctrl+u"])                                    # Redo my walks (top right)
    assert a.state == ENROL_UZAYR, a.state
    echo(a, "quiet", seed=8)
    sh.shot(a, "31_redo_my_walks")


def _notices(sh: _Shots) -> None:
    """The notices with no picture above: the hand-over to Uzayr's turns (pools.enrol_order visitor_first) and,
    with Redo my walks, its three ends: Undo walk right after it, his last new walk in a game, and his last new
    walk with no player."""
    a = sh.app()
    keys(a, ["n"] + list(NAME) + ["enter"])
    if a.order == "visitor_first":
        while a._turn() == "visitor":
            walk(a, VISITOR[a.done["visitor"] % 3])
        sh.act()
        assert a.notice[0].startswith("Got it! Now Uzayr walks"), a.notice
        echo(a, "walked", speed=VISITOR[1][0])
        sh.shot(a, "32_notice_now_uzayr_walks", notice=True)
    if not screen.redo_button(sh.cfg):
        return
    learn(a)
    walk(a, (1.13, 121, 2.4, 0.41), decision=ANSWER_V)
    a._timers(a.timer + 0.01)
    assert a.state == RESULT, a.state
    keys(a, ["ctrl+u", "backspace"])                       # Redo my walks, then Undo walk: the old walks kept
    sh.act()
    assert a.notice[0] == "Back to the game.", a.notice
    echo(a, "quiet", seed=9)
    sh.shot(a, "33_notice_redo_undone", notice=True)
    keys(a, ["ctrl+u"])
    learn(a)                                               # his two new walks in the game
    sh.act()
    assert a.notice[0] == "Got it! Back to the game.", a.notice
    echo(a, "walked", speed=UZAYR[1][0])
    sh.shot(a, "34_notice_redo_back_to_game", notice=True)
    keys(a, ["ctrl+n", "y", "ctrl+u"])                     # End game, then Redo my walks with no player
    learn(a)
    sh.act()
    assert a.notice[0] == "Got it! Ready for the next player.", a.notice
    echo(a, "walked", speed=UZAYR[0][0], seed=10)
    sh.shot(a, "35_notice_redo_next_player", notice=True)


def main(cfg: dict, out: Optional[str] = None) -> int:
    t0 = time.time()
    files = run(cfg, out)
    where = os.path.dirname(files[0]) if files else out
    print(f"{len(files)} pictures in {where} ({time.time() - t0:.0f} s):")
    for f in files:
        print("  " + os.path.basename(f))
    return 0
