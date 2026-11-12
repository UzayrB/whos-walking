"""What the test files share: a fake receiver (packets), walk columns (cols), and drivers that play the app the way
the operator does (_app, _type, _walk, ...). _app gives every walk the same fixed answer, so those tests check the
flow, not the model."""

import collections
import json
import time

import numpy as np
from matplotlib.backend_bases import MouseEvent

from openday import chain, models
from openday.app import App, RunLog, ENROL_UZAYR, ENROL_VISITOR, IDLE, READY, RESULT, WALKING
from openday.chain import Columns


HOP = 8 / 150.0                                                    # one waterfall column: 8 packets at 150 a second


# ---- a fake receiver and walk columns --------------------------------------------------------------------------------

def packets(seconds: float, tone_hz: float = 0.0, depth: float = 0.25, noise: float = 0.02, seed: int = 0,
            fs: float = 150.0, lt0_us: int = 5_000_000):
    """Synthetic receiver packets, a Doppler-like tone on every subcarrier, with jittered timing and batched host
    stamps: (local_timestamp, t_mono, data int8 (n, 256)). Amplitude 40 * (1 + depth cos(2 pi f t + phase_k)
    + noise)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    dt = rng.uniform(0.8, 1.2, n) / fs
    t = np.cumsum(dt)
    lt = (lt0_us + t * 1e6).astype(np.int64)
    tm = 100.0 + np.ceil(t / 0.06) * 0.06 + 0.004                 # host stamps per ~60 ms serial read
    phase = rng.uniform(0, 2 * np.pi, 128)
    amp = 1.0 + noise * rng.standard_normal((n, 128))
    if tone_hz:
        amp += depth * np.cos(2 * np.pi * tone_hz * t[:, None] + phase[None, :])
    data = np.zeros((n, 128, 2), np.int8)
    data[:, :, 1] = np.clip(np.round(40 * amp), -128, 127).astype(np.int8)
    return lt, tm, data.reshape(n, 256)


def cols(energy, t0=0.0, f50=20.0):
    """Waterfall columns with this energy, one every HOP from t0, as the chain hands them to a walk."""
    e = np.asarray(energy, float)
    t = t0 + np.arange(len(e)) * HOP
    return Columns(t, None, np.zeros(len(e), bool), None, e, np.full(len(e), f50), np.full(len(e), 2 * f50))


# ---- the app played the way the operator plays it, with a fixed answer -----------------------------------------------

DEC = models.Decision(label="visitor", margin=0.4, not_sure=False, dist={},
                      contrib={"speed": -0.3, "steps": 0.2, "swing": -0.5, "bounce": 0.01}, why=["speed", "leg swing"])


def _app(cfg, tmp_path, baseline=True, record=False, log=None):
    app = App(cfg, state_dir=str(tmp_path / "state"), log=RunLog(log), headless=True, baseline_cache=False,
              record_dir=str(tmp_path) if record else None)
    app.last_pkt_wall = 1e12                                       # packets keep arriving
    app.chain.clock.blocks.append((0, 0.0))                        # host time = device time
    if baseline:
        n = len(app.chain.freqs)
        app.chain.baseline = chain.Baseline(np.ones(n), 1.0, 100, app.chain.spec.variant, app.chain.spec.nfft)
        app.state = IDLE
    app._decide = lambda f: DEC                                   # a fixed answer: the flow is under test here
    return app


def _type(app, keys, t=0.0):
    for k in keys:
        app.on_key_now(k, t)


def _feed(app, e, t0):
    """Columns into the running walk, as _process does."""
    if app.walk is not None:
        app._feed_walk(cols(e, t0))


SEEN = [30.0] * 60 + [1.0] * 20                                  # 3.2 s of walking, then still: ends "stopped"
EMPTY = [1.0] * 110                                                # nobody: ends "nothing" at give_up_s


def _walk(app, t, seen=True, label=None):
    """Next at host time t, the walk's columns, then (bench runs) its label."""
    app.on_key_now("space", t)
    assert app.state == WALKING, app.state
    _feed(app, SEEN if seen else EMPTY, t)
    assert app.walk is None
    if label:
        _type(app, [label], t + 8.0)


def _to_game(app, t=100.0):
    """New player "Le", then the six turns (Uzayr first): no clicks, also in a test recording."""
    _type(app, ["n", "L", "e", "enter"])
    for i in range(6):
        _walk(app, t + 20 * i)
    assert app.state == READY and app.visitor_ready
    return t + 120


def _to_game_rest(app, t=100.0, n=6):
    for i in range(n):
        _walk(app, t + 20 * i)
    return t + 20 * n


def _events(tmp_path):
    return [json.loads(x) for x in open(tmp_path / "events.jsonl")]


def _cards(vm):
    c = vm["card"]
    return [str(c.get(k, "")) for k in ("big", "title", "sub", "why", "conf_text")]


def _labels(tmp_path):
    return [e for e in _events(tmp_path) if e["kind"] == "label"]


def _light(app):
    now = time.monotonic()
    app.last_pkt_wall, app.rate_hist = now, collections.deque([(now - 1.0, 0), (now, 150)])
    return app.health(now)[:2]


def _click(app, name):
    """Draw the screen as the app would, click the middle of a button, then run the keys it queued."""
    app.screen.render(app.view(time.monotonic()))
    x0, y0, x1, y1 = app.screen.buttons[name]["box"]
    canvas = app.screen.fig.canvas
    px, py = app.screen.fig.transFigure.transform(((x0 + x1) / 2, (y0 + y1) / 2))
    canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", canvas, px, py, button=1))
    sent = []
    while not app.keys.empty():
        key, t = app.keys.get_nowait()
        sent.append(key)
        app.on_key_now(key, t)
    return sent


# ---- the round as shipped: a player, the learning walks still to come, a game walk -----------------------------------

def _player(app, name="Le"):
    _type(app, ["n", *name, "enter"])
    assert app.name == name and app.state == ENROL_VISITOR


def _learn(app, t):
    """Every learning walk still to come; returns the time after them and whose walks they were."""
    who = []
    while app.state in (ENROL_VISITOR, ENROL_UZAYR):
        who.append(app._turn())
        _walk(app, t)
        t += 20.0
    return t, who


def _game(app, t):
    _walk(app, t)
    app._timers(app.timer + 0.01)
    assert app.state == RESULT
    return t + 20.0


def _age(app, minutes):
    """Make Uzayr's learning walks `minutes` old (the oldest of them)."""
    shift = time.time() - 60.0 * minutes - min(w["t"] for w in app.uz_fresh.walks)
    for w in app.uz_fresh.walks:
        w["t"] += shift


def _now(app):
    return app.view(time.monotonic() + 30.0)                      # (past any notice, event hint or lock)


# ---- an app fed raw packets (the 3D picture's tests) -----------------------------------------------------------------

def _ridge_app(cfg, tmp_path, log=None):
    """An App with the room measured and packets arriving, for the tests that feed it raw packets."""
    app = App(cfg, state_dir=str(tmp_path / "state"), log=RunLog(log), headless=True, baseline_cache=False)
    app.last_pkt_wall = 1e12                                       # packets keep arriving
    n = len(app.chain.freqs)
    app.chain.baseline = chain.Baseline(np.ones(n), 1.0, 100, app.chain.spec.variant, app.chain.spec.nfft)
    app.state = IDLE
    return app


def _bar_heights(sc, m):
    """The bars' heights over their floors, (rows, subcarriers), back row first."""
    v = np.array([p.vertices[:4] for p in sc.ridge_pc.get_paths()])
    return (v[:, 2, 1] - v[:, 0, 1]).reshape(-1, m)
