"""Who walked (openday/models.py): the nearest-centre rule (m0), the trained model (eval-pair's folds, a saved
model deciding the same after loading, the shipped model's manifest), and the learning walks' pools and the
visitor count (no names)."""

import itertools
import json
import os

import numpy as np
import pytest

import openday
from openday import evaluate, models
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

def _rows(seed=0):
    """Synthetic walks for the six people: every name, day, block and direction, person-specific feature means."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, name in enumerate(openday.PEOPLE):
        centre = rng.normal(size=len(M.ALL))
        for day, block, d in itertools.product((1, 2), ("natural", "slow", "fast"), ("S", "N")):
            for i in range(10 if block == "natural" else 5):
                f = dict(zip(M.ALL, centre + 0.5 * rng.normal(size=len(M.ALL))))
                f["speed"], f["moddepth_dop"], f["sub_dispersion"] = 1.0 + 0.05 * k + 0.03 * rng.normal(), 0.02, 1e7
                rows.append({"name": name, "day": day, "block": block, "dir": d, "pass_id": i, "gt": 1.0, "f": f})
    return rows


def test_folds_never_train_on_a_test_person():
    D = evaluate.Data(_rows())
    for A_, B_ in itertools.combinations(openday.PEOPLE, 2):
        for use_N in (False, True):
            tr = evaluate.train_indices(D, (A_, B_), use_N)
            assert len(tr) and not {A_, B_} & set(D.name[tr])
        fold = evaluate.fit_fold(D, (A_, B_))
        assert A_ not in fold["_train_people"] and B_ not in fold["_train_people"]
        assert len(fold["_train_people"]) == 4


def test_saved_model_gives_identical_decisions(tmp_path):
    D = evaluate.Data(_rows(1))
    names = evaluate.CORE
    X = D.X[:, [M.ALL.index(n) for n in names]]
    rng = np.random.default_rng(2)
    pools = {"uzayr": [D.rows[i]["f"] for i in range(3)], "visitor": [D.rows[i]["f"] for i in range(200, 203)]}
    for m in (evaluate.fit_m1(X, D.name, names), evaluate.fit_m1(X, D.name, names, keep=3),
              evaluate.fit_m2(X, D.name, D.rec, names)):
        m.tau, m.plda = 0.0836, m.kind == "m1"
        p = str(tmp_path / f"{m.kind}.json")
        m.save(p)
        back = M.Model.load(p)
        for i in rng.choice(len(D.rows), 25, replace=False):
            a, b = M.decide(m, D.rows[i]["f"], pools), M.decide(back, D.rows[i]["f"], pools)
            assert (a.label, a.not_sure, a.why) == (b.label, b.not_sure, b.why) and a.margin == b.margin


def test_the_manifest_code_files_exist():
    """eval-pair --write dates these files in the model's manifest: each must exist, or the write would fail."""
    assert evaluate.CODE_FILES
    for f in evaluate.CODE_FILES:
        assert os.path.isfile(os.path.join(openday.OPENDAY, "openday", f)), f


@pytest.mark.needs_code
@pytest.mark.needs_model
def test_shipped_model_is_thesis_only_and_loads(cfg_shipped):
    c = cfg_shipped["classifier"]
    if c["kind"] == "m0":
        return
    m = M.Model.load(os.path.join(openday.OPENDAY, c["model_path"]))
    assert m.kind == c["kind"] and len(m.names) == len(m.mu) == len(m.sd)
    man = json.load(open(os.path.join(openday.OPENDAY, "state", "model", "manifest.json")))
    assert set(man["trained_on"]["people"]) == {openday.recording_name(p) for p in openday.PEOPLE}
    bad = openday._people().get("excluded", [])                         # recorded but never used
    assert not any(x in f for x in bad for f in man["trained_on"]["files"])
    # (9 Nov retrain: Room A walks of the same six people, all three receivers, join the spread; was RX2 only)
    assert all(f.startswith(("Code/data/room_B/identification/", "Code/data/room_A/"))
               and f.endswith(("_RX1.bin", "_RX2.bin", "_RX3.bin", ".gt.json")) for f in man["trained_on"]["files"])
    assert not any("bench" in f for f in man["trained_on"]["files"])          # no home or bench recording


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
    assert M.why({"adj4": 1.0, "pc1_frac": 0.5, "speed": 0.1}, "uzayr") == ["body echo", "speed"]


# ---- the second look: two walks as one, and its How sure ------------------------------------------------------------

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


def test_how_sure_from_both_walks_reads_the_models_own_curve(tmp_path, cfg_shipped):
    """Model.how_sure_both: the two-walk curve in the model file (cal2_m / cal2_p, fitted separately after training,
    laptop only like the model; no How-sure numbers in the public config), capped at 99 %; a model file without one (an older file) reads the
    one-walk curve. The curve survives save / load; load_model drops a malformed one (and says so): lengths that
    differ, nothing in it, a value not finite, margins or shares falling, a share outside 0-1, not numbers, half a
    curve; the model itself still loads."""
    assert not {"second_look_sure", "cal_m", "cal_p", "cal2_m", "cal2_p"} & set(cfg_shipped["classifier"])
    m = M.Model("m1", ["speed"], np.zeros(1), np.ones(1), cal_m=np.array([0.0, 0.4]), cal_p=np.array([0.55, 0.95]))
    assert m.how_sure_both(0.2) == pytest.approx(m.how_sure(0.2)) == pytest.approx(0.75)    # no two-walk curve
    m.cal2_m, m.cal2_p = np.array([0.0, 0.2, 0.5]), np.array([0.5, 0.7, 1.0])               # (synthetic)
    assert m.how_sure_both(0.1) == pytest.approx(0.6) and m.how_sure_both(-1.0) == pytest.approx(0.5)
    assert m.how_sure_both(0.5) == 0.99 and m.how_sure(0.2) == pytest.approx(0.75)          # the one-walk curve stays
    p = str(tmp_path / "model.json")
    m.save(p)
    back = M.Model.load(p)
    assert back.cal2_m.tolist() == m.cal2_m.tolist() and back.cal2_p.tolist() == m.cal2_p.tolist()
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    del d["cal2_m"], d["cal2_p"]                                                             # an older model file
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)
    assert M.Model.load(p).cal2_m is None and M.Model.load(p).how_sure_both(0.2) == pytest.approx(0.75)
    cfg = {"classifier": {"kind": "m1", "model_path": p, "model_tau": 0.1}}
    nan, inf = float("nan"), float("inf")
    for cm, cp, kept in (([0.0, 0.2], [0.5, 0.7], True),                  # well formed
                         ([0.0, 0.2], [0.5], False),                       # lengths differ
                         ([0.2, 0.0], [0.5, 0.7], False),                  # margins falling
                         ([], [], False),                                  # empty
                         ([0.0, nan], [0.5, 0.7], False),                  # a margin not finite (NaN, inf)
                         ([0.0, inf], [0.5, 0.7], False),
                         ([0.0, 0.2], [0.5, None], False),                 # a share missing (null)
                         ([0.0, 0.2], [0.5, 1.2], False),                  # a share above 1 (or below 0)
                         ([0.0, 0.2], [-0.1, 0.7], False),
                         ([0.0, 0.2], [0.7, 0.5], False),                  # shares falling: wider is never less sure
                         ([0.0, 0.2], ["a", "b"], False),                  # not numbers: the model still loads
                         ([[0.0, 0.2]], [[0.5, 0.7]], False),              # not a list of numbers
                         (None, [0.5, 0.7], False)):                       # half a curve
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dict(d, cal2_m=cm, cal2_p=cp), f)
        logs: list = []
        lm = M.load_model(cfg, logs.append)
        assert lm is not None, (cm, cp)                                    # the model itself always loads
        assert (lm.cal2_m is not None) == kept and (lm.cal2_p is not None) == kept, (cm, cp)
        assert lm.how_sure_both(0.1) == pytest.approx(0.6 if kept else 0.65), (cm, cp)
        assert ("not usable" in logs[0]) != kept and ("its own curve" in logs[-1]) == kept, (cm, cp, logs)


@pytest.mark.needs_model
def test_installed_models_two_walk_curve_is_well_formed(cfg_shipped):
    """The installed model's two-walk How sure curve, when it has one: as many knots as values, margins rising,
    values rising from at least 50 % (PAV), never above 100 %; load_model keeps it."""
    m = M.load_model(cfg_shipped, lambda s: None)
    if m is None or m.cal2_m is None:
        pytest.skip("the installed model has no two-walk curve (How sure from both walks reads the one-walk curve)")
    cm, cp = np.asarray(m.cal2_m), np.asarray(m.cal2_p)
    assert len(cm) == len(cp) >= 2 and np.all(np.diff(cm) > 0) and np.all(np.diff(cp) >= 0)
    assert 0.5 <= cp[0] and cp[-1] <= 1.0 and m.how_sure_both(1.0) <= 0.99
