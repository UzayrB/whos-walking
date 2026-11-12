"""One walk (openday/walks.py): the walk window (it starts on the click, ends when they stop, arrive at the X or
the operator clicks They've stopped, gives up on nobody), then the thesis's gait features, which must give the
thesis code's own numbers."""

import numpy as np
import pytest

import openday
from openday import chain, recordings, walks
from openday.walks import WalkWindow

from helpers import HOP, cols


def test_walk_starts_on_the_click(cfg):
    w = WalkWindow(cfg, 1.0, 10.0, HOP)
    w.add(cols([100.0] * 40 + [1.0] * 200))                        # 2.1 s of motion, the first 1 s before the click
    assert w.t[0] >= 1.0
    assert not w.seen and w.ended and w.reason == "limit"          # only ~1.1 s after the click: no walk


def test_walk_ends_on_quiet(cfg):
    w = WalkWindow(cfg, 0.0, 10.0, HOP)
    w.add(cols([100.0] * 50 + [1.0] * 60))                         # 2.7 s walking, then still
    assert w.ended and w.reason == "stopped" and w.seen
    assert 2.6 + 0.5 <= w.t_end_col <= 2.7 + 0.5 + 0.3             # quiet 0.5 s (+ the smoothing's delay)
    arr = w.arrays()                                               # 100 never reaches the arrival level (80 x e0):
    assert w.t_arrive == arr["t"][arr["active"]].max()             # the arrival is the last active column


def test_walk_hard_limit(cfg):
    max_s = float(cfg["walk"]["max_s"])
    w = WalkWindow(cfg, 0.0, 10.0, HOP)                            # e0 = 10 / active_k: 100 stays below 80 x e0
    w.add(cols([100.0] * int((max_s + 1.0) / HOP)))                # never stops, never peaks
    assert w.ended and w.reason == "limit" and abs(w.t_end_col - max_s) < HOP + 1e-9


def test_no_walk_when_too_short(cfg):
    w = WalkWindow(cfg, 0.0, 10.0, HOP)
    w.add(cols([100.0] * 20 + [1.0] * 200))                        # about 1 s of motion
    assert w.ended and not w.seen


def _arrive_walk(cfg, rise_s=4.0, top=400.0, flat=100.0, after_s=6.0, e0=1.0, start=8.0, linear=False):
    """Energy (in units of e0) climbing from `start` to `top` over rise_s (geometric, as the walker nears the boards:
    the top is well above the walk's median), then flat at `flat` (still above the active threshold: someone
    standing at the X, close to the boards)."""
    k = float(cfg["dsp"]["active_k"])
    n = int(round(rise_s / HOP))
    rise = np.linspace(start, top, n) if linear else np.geomspace(start, top, n)
    e = list(rise) + [flat] * int(after_s / HOP)
    w = WalkWindow(cfg, 0.0, k * e0, HOP, e0=e0)
    w.add(cols(np.array(e) * e0))
    return w, (n - 1) * HOP                                        # the walk, the time of the top of the climb


def test_walk_ends_on_arrival(cfg):
    w, t_top = _arrive_walk(cfg)
    hold = float(cfg["walk"]["arrive_hold_s"])
    assert w.ended and w.reason == "arrived" and w.seen
    assert abs(w.t_arrive - t_top) <= 3 * HOP                      # the peak (the 5-column median lags 2 columns)
    assert hold <= w.t_end_col - w.t_arrive <= hold + HOP + 1e-9   # no new peak for arrive_hold_s
    assert w.peaked and w.peak >= float(cfg["walk"]["arrive_min_x"])


def test_a_longer_hold_only_ends_later(cfg):
    """hold_s (the app's screen window: walk.screen_hold_s) only moves the end: the same arrival, and the same
    columns up to the shorter window's end."""
    k = float(cfg["dsp"]["active_k"])
    e = list(np.geomspace(8.0, 400.0, 75)) + [100.0] * 60
    short, long = (WalkWindow(cfg, 0.0, k, HOP, e0=1.0, hold_s=h) for h in (None, 1.0))
    for w in (short, long):
        w.add(cols(e))
    assert short.arrive_hold == float(cfg["walk"]["arrive_hold_s"]) and long.arrive_hold == 1.0
    assert short.reason == long.reason == "arrived" and short.t_arrive == long.t_arrive
    assert 1.0 <= long.t_end_col - long.t_arrive < 1.0 + HOP
    n = len(short.t)
    for key, v in short.arrays().items():
        assert np.array_equal(v, long.arrays()[key][:n], equal_nan=True), key


def test_arrival_needs_the_level(cfg):
    w, _ = _arrive_walk(cfg, top=60.0, flat=50.0)                  # peaks below arrive_min_x (80) x e0
    assert w.reason == "limit" and not w.peaked
    arr = w.arrays()
    assert w.t_arrive == arr["t"][arr["active"]].max()             # then the arrival is the last active column


def test_arrival_scales_with_e0(cfg):
    w, t_top = _arrive_walk(cfg, e0=7.73)                          # the same walk in a noisier room
    assert w.reason == "arrived" and abs(w.t_arrive - t_top) <= 3 * HOP
    w = WalkWindow(cfg, 0.0, 10.0, HOP)                            # e0 defaults to threshold / active_k
    assert abs(w.e0 - 10.0 / float(cfg["dsp"]["active_k"])) < 1e-12


def test_single_early_spike_does_not_end_the_walk(cfg):
    """A one-column spike far above the arrival level right after the click: the running median drops it, so the
    walk runs on to its real arrival."""
    e = [20.0] * 4 + [5000.0] + list(np.geomspace(8.0, 400.0, 70)) + [100.0] * 100
    w = WalkWindow(cfg, 0.0, float(cfg["dsp"]["active_k"]), HOP, e0=1.0)
    w.add(cols(e))
    assert w.reason == "arrived" and abs(w.t_arrive - 74 * HOP) <= 3 * HOP


def test_early_burst_does_not_end_before_seen(cfg):
    """A short burst far above the arrival level just after the click, then ordinary walking below it: the walk can
    only end once it has been seen (min_active_s active), never during the burst."""
    e = [800.0] * 8 + [30.0] * 200
    w = WalkWindow(cfg, 0.0, float(cfg["dsp"]["active_k"]), HOP, e0=1.0)
    w.add(cols(e))
    assert w.ended and w.seen and w.reason == "arrived"
    assert w.t_end_col >= float(cfg["walk"]["min_active_s"]) - HOP
    assert w.t_peak < 1.0 and w.t_arrive == w.t_peak


def test_bump_mid_walk_needs_the_jump(cfg):
    """A bump above the arrival level in the middle of a walk (a step close to a strong path), then the real climb to
    the X: with the jump test (arrive_jump_x) only the real peak, many times the walk's median, ends the walk."""
    k = float(cfg["dsp"]["active_k"])
    e = [30.0] * 25 + [100.0] * 15 + [30.0] * 20 + list(np.geomspace(30.0, 900.0, 15)) + [100.0] * 100
    t_top = (25 + 15 + 20 + 14) * HOP
    cfg["walk"]["arrive_jump_x"] = 4.0
    w = WalkWindow(cfg, 0.0, k, HOP, e0=1.0)
    w.add(cols(e))
    assert w.reason == "arrived" and abs(w.t_arrive - t_top) <= 3 * HOP
    cfg["walk"]["arrive_jump_x"] = 0.0                             # without it the bump ends the walk mid-way
    w = WalkWindow(cfg, 0.0, k, HOP, e0=1.0)
    w.add(cols(e))
    assert w.reason == "arrived" and w.t_arrive < 2.5 and w.t_end_col < 3.5


def test_gentle_climb_is_not_an_arrival_with_the_jump_test(cfg):
    cfg["walk"]["arrive_jump_x"] = 4.0
    w, _ = _arrive_walk(cfg, start=10.0, top=200.0, flat=150.0, linear=True)   # peak only ~2 x the walk's median
    assert w.reason == "limit" and not w.peaked
    arr = w.arrays()
    assert w.t_arrive == arr["t"][arr["active"]].max()


def test_operator_stop(cfg):
    stop_min = float(cfg["walk"]["stop_min_s"])
    w = WalkWindow(cfg, 0.0, 10.0, HOP)
    w.add(cols([100.0] * 20))                                      # about 1 s walking
    assert not w.request_stop(stop_min - 0.2) and w.t_stop is None  # too soon after the click: a double press
    assert w.request_stop(3.0) and not w.ended                     # its columns have not arrived yet
    w.add(cols([100.0] * 200, t0=20 * HOP))
    assert w.ended and w.reason == "operator" and w.seen
    assert 3.0 <= w.t_end_col < 3.0 + HOP
    assert not w.request_stop(4.0)                                 # ended: nothing more to stop


def test_operator_stop_after_its_columns(cfg):
    """A stop whose column is already in ends the walk on that column, and the columns after it are dropped: the
    same walk the offline evaluation gets by applying the stop first."""
    e = [100.0] * 30 + [300.0] * 30
    w = WalkWindow(cfg, 0.0, 10.0, HOP)
    w.add(cols(e))                                                 # 3.2 s in already
    assert w.request_stop(1.7) and w.ended and w.reason == "operator"
    assert w.t_end_col == w.t[-1] and 1.7 <= w.t_end_col < 1.7 + HOP
    off = WalkWindow(cfg, 0.0, 10.0, HOP)
    off.request_stop(1.7)
    off.add(cols(e))
    assert (off.reason, off.t_end_col, off.n_active, off.peak, off.t_peak, off.t_arrive) == \
        (w.reason, w.t_end_col, w.n_active, w.peak, w.t_peak, w.t_arrive)
    for k, v in off.arrays().items():
        assert np.array_equal(v, w.arrays()[k], equal_nan=True), k


def test_operator_stop_before_any_column(cfg):
    w = WalkWindow(cfg, 1.0, 10.0, HOP)
    assert w.request_stop(1.0 + 2.0) and not w.ended
    w.add(cols([100.0] * 120))
    assert w.reason == "operator" and abs(w.t_end_col - 3.0) < HOP


def test_quiet_and_arrival_come_before_the_operator(cfg):
    w = WalkWindow(cfg, 0.0, 10.0, HOP)
    w.request_stop(5.0)
    w.add(cols([100.0] * 50 + [1.0] * 100))                        # stops at 2.7 s, long before the press
    assert w.reason == "stopped" and w.t_end_col < 5.0


def test_give_up_nothing(cfg):
    give_up = float(cfg["walk"]["give_up_s"])
    w = WalkWindow(cfg, 0.0, 10.0, HOP)
    w.add(cols([1.0] * int((float(cfg["walk"]["max_s"]) + 1.0) / HOP)))   # nobody walks
    assert w.ended and w.reason == "nothing" and not w.seen
    assert give_up - 1e-9 <= w.t_end_col < give_up + HOP
    assert w.t_arrive is None


def test_give_up_needs_little_activity(cfg):
    w = WalkWindow(cfg, 0.0, 10.0, HOP)
    w.add(cols([100.0] * 10 + [1.0] * 200))                        # about 0.5 s active: not "nothing"
    assert w.active_s >= float(cfg["walk"]["give_up_active_s"])
    assert w.ended and w.reason == "limit" and not w.seen


# ---- the gait features: the copies of the thesis code against the originals, on a real walk --------------------------

def _sample(cfg):
    """4 s of a real Room B walk (P3, day 1, natural, first S pass), RX2, each packet divided by its mean."""
    fr = recordings.load_rx2("P3", 1, "natural")
    a0 = recordings.passes("P3", 1, "natural", "S")[0]["anchor_times"][0]
    seg = recordings.segment(fr, a0, a0 + 4.0)
    return chain.amplitudes(seg["data"], chain.live_bins(cfg))


@pytest.mark.needs_code
def test_copies_match_the_thesis(cfg):
    openday.use_code()
    import pipeline.identification.engine as E
    import pipeline.identification.features as F
    AN = _sample(cfg)
    assert AN.shape[0] > 500
    for band in ("b08_35", "dop"):
        assert np.array_equal(walks.bandpass(AN, band, axis=0), F.bandpass(AN, band, axis=0))
        assert np.array_equal(walks.reduce_link(AN, band, "pca"), F.reduce_link(AN, band, "pca"))
    rms = walks.reduce_link(AN, "b08_35", "rms")[0]
    assert walks.cadence_estimates(rms) == F.cadence_estimates(rms)
    ests = [F.cadence_estimates(walks.reduce_link(AN, "b08_35", r)[0]) for r in ("rms", "pca")]
    assert walks.strong_cadence(ests) == E.strong_cadence(ests)
    assert walks.resolve_cadence(ests) == F.resolve_cadence(ests)
    bd = walks.bandpass(AN, "dop", axis=0)
    dop = walks.reduce_link(AN, "dop", "pca", n_pc=6, bmat=bd)[1:]
    for c in (None, 1.8):
        mine, theirs = walks.doppler_features_h(dop, cadence_hz=c), E.doppler_features_h(dop, cadence_hz=c)
        assert mine.keys() == theirs.keys() and all(np.allclose(mine[k], theirs[k], equal_nan=True) for k in mine)
    assert np.isclose(walks.torso_hz_h(AN, bmat=bd), E.torso_hz_h(AN, bmat=bd))
    f = np.linspace(4, 70, 34)
    mag = np.abs(np.random.default_rng(0).normal(size=(20, 34)))
    assert np.array_equal(walks.contour_interp(f, mag, 50.0), E.contour_interp(f, mag, 50.0), equal_nan=True)


@pytest.mark.needs_code
def test_single_link_is_live_features_for_one_link(cfg):
    """single_link reproduces engine.live_features' RX2 columns when only RX2's channels vote on the cadence."""
    openday.use_code()
    import pipeline.identification.engine as E
    import pipeline.identification.features as F
    AN = _sample(cfg)
    mine = walks.single_link(AN)
    bg = F.bandpass(AN, "b08_35", axis=0)
    ch = [F.bandpass(np.sqrt(np.mean(bg ** 2, axis=1)), "b08_35"), F.reduce_link(AN, "b08_35", "pca", n_pc=6, bmat=bg)[0]]
    cad = E.strong_cadence([F.cadence_estimates(x) for x in ch])["cadence_hz"]
    bd = F.bandpass(AN, "dop", axis=0)
    dop = E.doppler_features_h(F.reduce_link(AN, "dop", "pca", n_pc=6, bmat=bd)[1:], win_s=0.2, cadence_hz=cad)
    assert np.isclose(mine["cadence_hz"], cad if cad is not None else np.nan, equal_nan=True)
    for k, v in dop.items():
        assert np.isclose(mine[k], v), k
    d = F.bandpass(np.sqrt(np.mean(bd ** 2, axis=1)), "b08_35")
    assert np.isclose(mine["moddepth_dop"], np.std(d) / max(float(AN.mean()), 1e-6))
    assert np.isclose(mine["sub_dispersion"], np.median(np.std(bg, axis=0)) / 1e-9)
    assert np.isclose(mine["torso_hz"], E.torso_hz_h(AN, bmat=bd))


# ---- OpenDay's own four measures (walks.shape_features) ---------------------------------------------------------------

def _synthetic_walk(n=900, seed=0):
    """A synthetic walk's raw window (n x 103, each packet's mean 1): a 1.8 Hz step rhythm modulating a 12 Hz and a
    35 Hz Doppler tone whose strength drifts smoothly across the subcarriers, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None] / walks.FS
    k = np.arange(103)[None, :]
    env = 1.0 + 0.6 * np.sin(2 * np.pi * 1.8 * t)
    mot = 0.04 * env * (np.sin(2 * np.pi * 12 * t + k / 9.0) + 0.5 * np.sin(2 * np.pi * 35 * t + k / 4.0))
    a = 1.0 + 0.3 * np.cos(k / 13.0) + mot + 0.01 * rng.normal(size=(n, 103))
    return a / a.mean(axis=1, keepdims=True)


def test_shape_features_finite_and_scale_free(cfg):
    """The four new measures are finite on a walk, in their ranges, and unchanged when every amplitude is scaled by
    the same factor (a louder or quieter room); the thesis's measures are the same with or without them."""
    groups = chain.subcarriers(chain.live_bins(cfg))[1]
    AN = _synthetic_walk()
    f = walks.single_link(AN, groups)
    assert set(walks.SHAPE) <= set(walks.GAIT_NAMES) and all(np.isfinite(f[n]) for n in walks.SHAPE)
    assert f["moddepth_dop10"] > 0 and -1 <= f["adj4"] <= 1 and 0 <= f["ev21"] <= 1 and 0 < f["pc1_frac"] <= 1
    for c in (0.37, 4.2):
        g = walks.single_link(AN * c, groups)
        for n in walks.SHAPE:
            assert g[n] == pytest.approx(f[n], rel=1e-9), n
    plain = walks.single_link(AN)                                  # without the subcarrier map: adj4 alone missing
    assert np.isnan(plain["adj4"])
    for n in walks.GAIT_NAMES:
        if n != "adj4":
            assert plain[n] == f[n] or (np.isnan(plain[n]) and np.isnan(f[n])), n
    t = np.arange(len(AN)) / walks.FS                              # walk_gait passes the map: adj4 is there
    gw = walks.walk_gait(t, AN, 0.0, float(t[-1]), cfg)
    assert all(gw[n] == pytest.approx(f[n], rel=1e-12) for n in walks.SHAPE)


def test_shape_features_by_hand(cfg):
    """Each new measure recomputed from its definition (walks.shape_features' docstring)."""
    from scipy import signal
    sc, groups = chain.subcarriers(chain.live_bins(cfg))
    AN = _synthetic_walk(seed=1)
    f = walks.single_link(AN, groups)
    sos = signal.butter(4, [10.0 / 75.0, 70.0 / 75.0], btype="bandpass", output="sos")
    b10 = signal.sosfiltfilt(sos, AN, axis=0)
    d = walks.bandpass(np.sqrt((b10 ** 2).mean(axis=1)), "b08_35")
    assert f["moddepth_dop10"] == pytest.approx(np.std(d) / AN.mean(), rel=1e-9)
    bd = walks.bandpass(AN, "dop", axis=0)
    b = bd - bd.mean(axis=0)
    ev = np.sort(np.linalg.eigh(np.cov(b.T))[0])[::-1]
    assert f["pc1_frac"] == pytest.approx(ev[0] / ev.sum(), rel=1e-9)
    assert f["ev21"] == pytest.approx(ev[1] / ev[0], rel=1e-9)
    per = np.array([(bd / AN.mean(axis=0))[:, groups == i].mean(axis=1) for i in range(len(sc))])
    r = [np.corrcoef(per[i], per[i + 4])[0, 1] for i in range(len(sc) - 4)]
    assert f["adj4"] == pytest.approx(np.mean(r), rel=1e-9)


@pytest.mark.needs_code
def test_shape_features_on_a_real_walk(cfg):
    """On 4 s of a real Room B walk: finite, and scale-free."""
    groups = chain.subcarriers(chain.live_bins(cfg))[1]
    AN = np.asarray(_sample(cfg), float)                           # (float32 from the chain: scaled exactly here)
    f, g = walks.single_link(AN, groups), walks.single_link(AN * 2.5, groups)
    for n in walks.SHAPE:
        assert np.isfinite(f[n]) and g[n] == pytest.approx(f[n], rel=1e-9), n
