"""The signal chain (openday/chain.py): a synthetic Doppler tone reads as the right speed, the live bins, the
gain-invariant normalisation, live and offline batches giving the same columns, the receiver's clock; then the 3D
picture's movement bars on their own (the Ridge class): frequency order, one ridge per waterfall column, where the
signal wobbles."""

import copy

import numpy as np
import pytest

from openday import chain, recordings
from openday.chain import Ridge, subcarriers

from helpers import packets


def _run(cfg, variant, lt, tm, data, batch, base=None):
    ch = chain.Chain(cfg, variant, None, base)
    out = [ch.push(lt[i:i + batch], tm[i:i + batch], data[i:i + batch]) for i in range(0, len(lt), batch)]
    return ch, [c for c in out if len(c)]


@pytest.mark.parametrize("variant", ["A", "B"])
@pytest.mark.parametrize("f", [12.0, 25.0, 40.0])
def test_tone_gives_speed(cfg, variant, f):
    """A synthetic Doppler tone at f Hz reads as f x lambda/2 (0.0609 m/s per Hz at F = 1) within 3 %."""
    cfg["radio"]["bistatic_factor"] = 1.0
    lt, tm, data = packets(4.0, 0.0, seed=1)
    ch, cols = _run(cfg, variant, lt, tm, data, 8)
    c = chain.Columns(np.concatenate([x.t for x in cols]), np.concatenate([x.power for x in cols]),
                    np.concatenate([x.gap for x in cols]))
    base = chain.make_baseline(c, cfg, ch.freqs, variant, ch.spec.nfft)
    lt2, tm2, data2 = packets(6.0, f, seed=2, lt0_us=int(lt[-1]) + 6667)
    ch2, cols2 = _run(cfg, variant, np.concatenate([lt, lt2]), np.concatenate([tm, tm2 + tm[-1] - 99.0]),
                      np.concatenate([data, data2]), 8, base)
    t = np.concatenate([x.t for x in cols2])
    f50 = np.concatenate([x.f50 for x in cols2])
    late = t > t[-1] - 3.0                                         # well inside the tone
    v = float(chain.speed_of(np.median(f50[late]), cfg))
    assert abs(v - f * 0.0609) / (f * 0.0609) < 0.03, (v, f * 0.0609)
    assert abs(0.12177 / 2 - 0.0609) < 1e-3


def test_live_bins(cfg):
    b = chain.live_bins(cfg)
    assert len(b) == 103
    assert not {93, 94, 95} & set(b.tolist())
    assert not ({0, 1, 64} | set(range(27, 38)) | set(range(96, 104))) & set(b.tolist())


def test_normalisation_is_gain_invariant(cfg):
    rng = np.random.default_rng(3)
    data = rng.integers(-30, 30, (20, 256)).astype(np.int8)
    a1 = chain.amplitudes(data, chain.live_bins(cfg))
    gains = rng.integers(2, 4, (20, 1)).astype(np.int8)
    a2 = chain.amplitudes(data * gains, chain.live_bins(cfg))       # a per-packet gain factor
    assert np.allclose(a1, a2, atol=1e-5)
    assert np.allclose(a1.mean(axis=1), 1.0)


@pytest.mark.needs_code
def test_batch_size_does_not_change_columns(cfg):
    """Offline (big batches) equals live (small batches): the chain is causal and incremental."""
    fr = recordings.segment(recordings.load_rx2("P4", 1, "natural"), *(lambda a: (a[0] - 3, a[-1] + 1))(
        recordings.passes("P4", 1, "natural", "S")[0]["anchor_times"]))
    outs = []
    for batch in (8, 75, len(fr)):
        _ch, cols = _run(cfg, "A", fr["local_timestamp"], fr["t_mono"], fr["data"], batch)
        outs.append((np.concatenate([c.t for c in cols]), np.concatenate([c.power for c in cols])))
    for t, p in outs[1:]:
        assert np.array_equal(t, outs[0][0])
        assert np.allclose(p, outs[0][1], rtol=1e-9, atol=1e-12)


def test_device_clock_unwraps_and_restarts(cfg):
    c = chain.DeviceClock(cfg)
    mod = int(cfg["clock"]["modulus_us"])
    lt = np.array([mod - 20_000, mod - 13_000, mod - 6_000, 1_000, 8_000])
    t, _tm, restarted = c.push(lt, np.arange(5) * 0.007 + 50)
    assert not restarted and np.all(np.diff(t) > 0) and c.wraps == 1
    _t, _tm, restarted = c.push(np.array([500_000_000, 500_007_000]), np.array([51.0, 51.007]))
    _t, _tm, restarted = c.push(np.array([12_000, 19_000]), np.array([51.1, 51.107]))   # board rebooted
    assert restarted and c.wraps == 0


# ---- the 3D picture's movement bars (Ridge on its own) ---------------------------------------------------------------

def test_the_ridge_runs_in_frequency_order(cfg):
    """The 103 live bins are two estimates (legacy LTF, HT-LTF) of subcarriers -26..-1 and +1..+28, each half in
    FFT order: a ridge puts them in frequency order, one point per subcarrier."""
    bins = chain.live_bins(cfg)
    sc, at = subcarriers(bins)
    assert list(sc) == list(range(-26, 0)) + list(range(1, 29)) and len(sc) == 54
    assert (bins < 64).sum() == 51 and (bins >= 64).sum() == 52    # legacy LTF, HT-LTF
    assert (np.bincount(at) == 2).sum() == 49                      # 49 subcarriers measured by both halves
    r = Ridge(cfg)
    assert r.points == 54 and list(r.sc) == list(sc)


def test_smoothing_stays_centred(cfg):
    """A spike on one subcarrier stays centred on it for any ridge_smooth (else the ridge would sit off its
    numbers): an even one is taken as the next odd one."""
    for k in (3, 4, 5):
        c = copy.deepcopy(cfg)
        c["ui"]["ridge_smooth"] = k
        r = Ridge(c)
        q = np.ones(r.points)
        q[27] = 2.0
        r.add_shape(q)
        z = r.heights()[-1]
        assert r.smooth == k | 1 and len(z) == r.points
        bump = z - z.min()                                         # (a shape: the spike over a flat floor)
        assert np.sum(np.arange(r.points) * bump) / bump.sum() == pytest.approx(27.0)


def _wobble(cfg, n, lo=-20, hi=-10, seed=3, move=0.8, common=0.03):
    """n samples of the live bins at 150 a second: a level of 40 on every bin, a gain that drifts on all of them at
    once (common), noise, and (move > 0) subcarriers lo..hi (both estimates) flickering together at 15 Hz, the
    way a walker makes them."""
    rng = np.random.default_rng(seed)
    sc, at = subcarriers(chain.live_bins(cfg))
    num = sc[at]
    t = np.arange(n)[:, None] / 150.0
    gain = 1.0 + common * np.sin(2 * np.pi * 7.0 * t + rng.uniform(0, 6)) * np.sin(2 * np.pi * 0.7 * t)
    x = 40.0 * gain + rng.normal(0.0, 0.3, (n, len(num)))
    on = (num >= lo) & (num <= hi)
    x[:, on] += move * np.sin(2 * np.pi * 15.0 * t)
    return x


def test_movement_one_ridge_per_hop_once_fitted_whatever_the_batches(cfg):
    """No ridge until the movement is first fitted (pca_window_s, like the waterfall's first column), then one
    per hop; the same samples give the same ridges however the packets are batched."""
    r = Ridge(cfg)
    assert r.pca_n == 300 and r.hop == 8
    x = _wobble(cfg, 1200)
    r.push(x[:299])
    assert len(r.heights()) == 0
    r.push(x[299:300])
    assert len(r.heights()) == 1
    r.push(x[300:307])
    assert len(r.heights()) == 1
    r.push(x[307:308])
    assert len(r.heights()) == 2
    whole = Ridge(cfg)
    whole.push(x)
    assert whole.heights().shape == (whole.n, whole.points) and np.isfinite(whole.heights()).all()
    for size in (1, 7, 8, 13, 75, 301):
        r = Ridge(cfg)
        for i in range(0, len(x), size):
            r.push(x[i:i + size])
        assert np.allclose(r.heights(), whole.heights(), rtol=1e-6, atol=1e-9), size
    r.reset()
    assert len(r.heights()) == 0
    r.push(x[:300])
    assert len(r.heights()) == 1                                   # (a restart waits for its own fit again)


def test_movement_shows_where_the_signal_wobbles(cfg):
    """The subcarriers a walker makes flicker stand tallest, at their own numbers (frequency order, both estimates
    together); a shape's mean is 1; a gain that moves every subcarrier at once (the waterfall's dropped PC1) is not
    movement."""
    sc = list(Ridge(cfg).sc)
    for lo, hi in ((-20, -10), (5, 15)):
        r = Ridge(cfg)
        r.push(_wobble(cfg, 900, lo, hi))
        z = r.heights()[-1]
        on = np.array([lo <= s <= hi for s in sc])
        inner = np.array([lo + 2 <= s <= hi - 2 for s in sc])
        assert z.mean() == pytest.approx(1.0)
        assert z[inner].min() > 2.0 * np.median(z[~on])
        assert sc[int(np.argmax(z))] in range(lo, hi + 1)
    r = Ridge(cfg)
    r.push(_wobble(cfg, 900, move=0.0))                            # only the common gain and noise: no place
    z = r.heights()[-1]                                            # stands out much
    assert z.max() < 2.0


def test_movement_each_ridge_eases_into_its_new_shape(cfg):
    r = Ridge(cfg)
    m = r.points
    a = 1.0 + 0.5 * np.sin(np.linspace(0.0, 6.0, m))
    b = 1.0 + 0.5 * np.cos(np.linspace(0.0, 9.0, m))

    def norm(q):
        q = r._smooth(q)
        return q / q.mean()

    r.add_shape(a)
    r.add_shape(b)
    z = r.heights()
    assert r.ease == 0.5 and z[0] == pytest.approx(norm(a))
    assert z[1] == pytest.approx(norm(a) + 0.5 * (norm(b) - norm(a)))
    c = copy.deepcopy(cfg)
    c["ui"]["ridge_ease"] = 1.0
    r = Ridge(c)
    r.add_shape(a)
    r.add_shape(b)
    assert r.heights()[1] == pytest.approx(norm(b))


def test_movement_a_wobble_counts_as_a_share_of_its_level_and_the_edge_is_capped_first(cfg):
    """The same wobble on a bin at half the level is twice the share; +27 and +28, wobbling twice as much as the rest
    (the band's noisy edge), are capped at +26 before the smoothing, so +26 does not take up their noise."""
    r = Ridge(cfg)
    sc, at = subcarriers(chain.live_bins(cfg))
    num = sc[at]
    y = np.sin(2 * np.pi * 15.0 * np.arange(r.win) / 150.0)[:, None] * np.ones(len(num))   # the same everywhere
    r.add_wobble(y, np.where((num >= 5) & (num <= 15), 20.0, 40.0))
    z, s = r.heights()[-1], list(r.sc)
    inner = [s.index(k) for k in range(7, 14)]
    assert z[inner] == pytest.approx(np.full(7, 2.0 * z[[i for i, k in enumerate(s) if k <= 0]].mean()))
    r = Ridge(cfg)
    y2 = y.copy()
    y2[:, num >= 27] *= 2.0
    r.add_wobble(y2, np.full(len(num), 40.0))
    z = r.heights()[-1]
    assert np.ptp(z) < 1e-9                                        # (+26 as level as the rest)
