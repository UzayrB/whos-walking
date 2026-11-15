"""Who walked (openday/models.py): the nearest-centre rule (m0), the trained model (a saved model deciding the same
after loading, the shipped model's manifest), and the learning walks' pools and the visitor count (no names)."""

import json
import os

import numpy as np
import pytest

import openday
from openday import models
from openday import models as M
from openday.models import Pool, Tally


def test_classifier_separated_clusters(cfg):
    rng = np.random.default_rng(0)
    names = cfg["features"]["names"]
    sd = np.array([cfg["classifier"]["prior_sd"][n] for n in names])
    mu_u, mu_v = np.array([1.0, 100, 2.0, 0.3]), np.array([1.4, 115, 2.6, 0.45])
    pools = {"uzayr": mu_u + rng.normal(0, 1, (5, 4)) * sd * 0.3, "visitor": mu_v + rng.normal(0, 1, (3, 4)) * sd * 0.3}
    for _ in range(20):
        d = models.decide_m0(mu_u + rng.normal(0, 1, 4) * sd * 0.3, pools, names, cfg)
        assert d.label == "uzayr"
        assert d.contrib["speed"] > 0                              # speed pulled towards Uzayr
        assert d.speed_only.label == "uzayr"
        assert models.decide_m0(mu_v + rng.normal(0, 1, 4) * sd * 0.3, pools, names, cfg).label == "visitor"


def test_not_sure_below_tau(cfg):
    names = cfg["features"]["names"]
    pools = {"uzayr": np.array([[1.0, 100, 2.0, 0.3]] * 3), "visitor": np.array([[1.4, 115, 2.6, 0.45]] * 3)}
    mid = np.array([1.2, 107.5, 2.3, 0.375]) + np.array([0.004, 0, 0, 0])
    d = models.decide_m0(mid, pools, names, cfg, tau=0.1)
    assert d.margin < 0.1 and d.not_sure
    d = models.decide_m0(np.array([1.0, 100, 2.0, 0.3]), pools, names, cfg, tau=0.1)
    assert not d.not_sure and d.label == "uzayr"


def test_spread_never_tiny(cfg):
    names = cfg["features"]["names"]
    pools = {"uzayr": np.array([[1.0, 100, 2.0, 0.3]] * 3), "visitor": np.array([[1.4, 115, 2.6, 0.45]] * 3)}
    s = models.fit_spread(pools, np.array([cfg["classifier"]["prior_sd"][n] for n in names]))
    assert np.all(s >= np.array([cfg["classifier"]["prior_sd"][n] for n in names]))


# ---- the trained model -----------------------------------------------------------------------------------------------

def test_saved_model_gives_identical_decisions(tmp_path):
    """A trained model (m1, here made up) saved and loaded again decides every walk exactly as before. A model file
    with an option this app no longer has (m2, PLDA, session centring) is not loaded: the app says so and uses m0."""
    rng = np.random.default_rng(1)
    walks_ = []
    for _ in range(30):
        f = dict(zip(M.ALL, rng.normal(size=len(M.ALL))))
        f.update(speed=1.0 + 0.1 * rng.normal(), moddepth_dop=0.02, sub_dispersion=1e7, moddepth_dop10=0.01)
        walks_.append(f)
    X = np.array([M.vec(f, M.ALL) for f in walks_])
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    m = M.Model("m1", list(M.ALL), mu, sd, tau=0.0836, P=rng.normal(size=(len(M.ALL), len(M.ALL))),
                lam=rng.random(len(M.ALL)))
    p = str(tmp_path / "m1.json")
    m.save(p)
    back = M.Model.load(p)
    pools = {"uzayr": walks_[:3], "visitor": walks_[3:6]}
    for f in walks_[6:]:
        a, b = M.decide(m, f, pools), M.decide(back, f, pools)
        assert (a.label, a.not_sure, a.why) == (b.label, b.not_sure, b.why) and a.margin == b.margin
    cfg = {"classifier": {"kind": "m1", "model_path": p, "model_tau": 0.1}}
    for old in ({"plda": True}, {"session_centre": True}, {"kind": "m2"}):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dict(m.to_json(), **old), f)
        logs: list = []
        assert M.load_model(cfg, logs.append) is None and "using m0" in logs[0], old


@pytest.mark.needs_code
@pytest.mark.needs_model
def test_shipped_model_is_thesis_only_and_loads(cfg_shipped):
    c = cfg_shipped["classifier"]
    if c["kind"] == "m0":
        return
    m = M.Model.load(os.path.join(openday.OPENDAY, c["model_path"]))
    assert m.kind == c["kind"] and len(m.names) == len(m.mu) == len(m.sd)
    man = json.load(open(os.path.join(openday.OPENDAY, "state", "model", "manifest.json")))
    assert set(man["trained_on"]["people"]) == set(openday.PEOPLE)            # codes only, never names
    bad = openday._people().get("excluded", [])                         # recorded but never used
    assert not any(x in f for x in bad for f in man["trained_on"]["files"])
    # (9 Nov retrain: Room A walks of the same six people, all three receivers, join the spread; was RX2 only)
    assert all(f.startswith(("Code/data/room_B/identification/", "Code/data/room_A/"))
               and f.endswith(("_RX1.bin", "_RX2.bin", "_RX3.bin", ".gt.json")) for f in man["trained_on"]["files"])
    assert not any("bench" in f for f in man["trained_on"]["files"])          # no home or bench recording


def test_the_shipped_model_lines(cfg_shipped):
    """As shipped: the trained model in state/model and its close-call line; no How-sure numbers in the public
    config (How sure is words from the margin against that line, sure_words)."""
    c = cfg_shipped["classifier"]
    assert (c["model_path"], c["model_tau"]) == ("state/model/model.json", 0.046)
    assert not {"second_look_sure", "cal_m", "cal_p", "cal2_m", "cal2_p"} & set(c)


def test_a_model_file_with_the_older_how_sure_curves_still_loads(tmp_path):
    """An older model file (with How sure curves, well formed or not) loads, keeps its tau from the config and
    decides exactly as the same file without them; save writes no curve."""
    rng = np.random.default_rng(3)
    walks_ = [dict(zip(M.ALL, rng.normal(size=len(M.ALL))), speed=1.0 + 0.2 * rng.normal()) for _ in range(9)]
    m = M.Model("m1", ["speed", "swing"], np.zeros(2), np.ones(2), tau=0.05, P=np.eye(2), lam=np.ones(2))
    p = str(tmp_path / "model.json")
    m.save(p)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    assert not {"cal_m", "cal_p", "cal2_m", "cal2_p"} & set(d)
    cfg = {"classifier": {"kind": "m1", "model_path": p, "model_tau": 0.1}}
    pools = {"uzayr": walks_[:3], "visitor": walks_[3:6]}
    for curves in ({"cal_m": [0.0, 0.4], "cal_p": [0.55, 0.95], "cal2_m": [0.0, 0.2], "cal2_p": [0.5, 0.7]},
                   {"cal_m": None, "cal_p": None, "cal2_m": None, "cal2_p": None},
                   {"cal2_m": [0.2, 0.0], "cal2_p": ["a", None]}):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dict(d, **curves), f)
        logs: list = []
        lm = M.load_model(cfg, logs.append)
        assert lm is not None and lm.tau == 0.1 and logs == ["model m1 (2 features)"], curves
        for f in walks_[6:]:
            a, b = M.decide(m, f, pools), M.decide(lm, f, pools)
            assert (a.label, a.margin, a.why) == (b.label, b.margin, b.why)


def test_how_sure_is_words_from_the_margin_against_tau():
    """No number: "Close call" under tau (an answer the second look left close, or with it off), "Leaning" from
    tau, "Fairly sure" from 4 tau."""
    tau = 0.046
    assert M.sure_words(0.0, tau) == M.sure_words(tau - 1e-9, tau) == "Close call"
    assert M.sure_words(tau, tau) == M.sure_words(4 * tau - 1e-9, tau) == "Leaning"
    assert M.sure_words(4 * tau, tau) == M.sure_words(1.0, tau) == "Fairly sure"


def test_the_echo_reading_never_reaches_a_decision():
    """The Echo tile's reading rides along in a walk's readings (and its pools'), but no decision reads it."""
    rng = np.random.default_rng(2)
    walks_ = [dict(zip(M.ALL, rng.normal(size=len(M.ALL))), speed=1.0 + 0.2 * rng.normal()) for _ in range(9)]
    m = M.Model("m1", ["speed", "swing"], np.zeros(2), np.ones(2), tau=0.05, P=np.eye(2), lam=np.ones(2))
    loud = [dict(f, echo=float(rng.uniform(5, 35))) for f in walks_]
    for f, g in zip(walks_[6:], loud[6:]):
        a, b = M.decide(m, f, {"uzayr": walks_[:3], "visitor": walks_[3:6]}), \
            M.decide(m, g, {"uzayr": loud[:3], "visitor": loud[3:6]})
        assert (a.label, a.margin, a.dist, a.contrib) == (b.label, b.margin, b.dist, b.contrib)


# ---- the pools and the visitor count ---------------------------------------------------------------------------------

F = {"speed": 1.1, "steps": 100.0, "swing": 2.0, "bounce": 0.3}


def test_visitor_pool_never_written(tmp_path):
    before = set(os.listdir(tmp_path))
    v = Pool(5)
    v.add(F)
    assert set(os.listdir(tmp_path)) == before


def test_tally_has_no_names(tmp_path):
    t = Tally(str(tmp_path / "tally.json"))
    assert t.new_visitor() == 1
    d = json.load(open(tmp_path / "tally.json"))
    assert set(d) == {"date", "right", "wrong", "not_sure", "visitors"} and list(d["visitors"]) == ["1"]


def test_every_measure_has_plain_words_and_the_new_scale_is_log():
    """Every measure a model can use belongs to one plain-word group of the screen's "Told apart by" line, and the new
    movement-strength measure is compared on a log scale like the old one."""
    from openday import walks
    covered = set().union(*(set(m) for _w, m in walks.GAIT_PLAIN))
    assert set(M.ALL) <= covered
    assert "moddepth_dop10" in M.LOG and M.vec({"moddepth_dop10": float(np.e)}, ["moddepth_dop10"])[0] == pytest.approx(1.0)
    assert M.why({"adj4": 1.0, "pc1_frac": 0.5, "speed": 0.1}, "uzayr") == ["signal shape", "speed"]


# ---- the second look: two walks as one -------------------------------------------------------------------------------

def test_two_walks_averaged_as_the_model_sees_them():
    """models.average_walks: each reading the mean of the two walks; one missing (NaN or None) takes the other's;
    the log-scale readings (models.LOG) their geometric mean, as the model compares them."""
    a = {"speed": 1.0, "steps": 100.0, "moddepth_dop": 0.01, "c50_cv": float("nan"), "swing": None}
    b = {"speed": 1.4, "steps": 110.0, "moddepth_dop": 0.04, "c50_cv": 0.3, "swing": 2.0, "bounce": 0.3}
    f = M.average_walks([a, b])
    assert f["speed"] == pytest.approx(1.2) and f["steps"] == pytest.approx(105.0)
    assert f["moddepth_dop"] == pytest.approx(0.02)               # sqrt(0.01 * 0.04)
    assert f["c50_cv"] == pytest.approx(0.3) and f["swing"] == pytest.approx(2.0) and f["bounce"] == 0.3
    assert np.isnan(M.average_walks([{"x": float("nan")}, {"x": float("nan")}])["x"])
    assert M.vec(f, ["moddepth_dop"])[0] == pytest.approx(np.mean([np.log(0.01), np.log(0.04)]))
