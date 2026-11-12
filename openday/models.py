"""Who walked: the nearest-centre rule (m0), the trained models (m1, m2) and the learning walks.

This file, in order:
  1. The nearest-centre rule (m0): decide_m0, one walk's readouts against the two pools, with a margin and
     "not sure".
  2. The trained models: the features they use, Model (saved as JSON) and load_model. evaluate.py fits them
     on the thesis people.
  3. Deciding one walk with a model (decide), two walks averaged for the second look (average_walks; its How
     sure is Model.how_sure_both), and the plain words for what separated it (why).
  4. The pools (each person's learning walks, in memory only) and the day's tally.

Two-person decisions for people the model never saw in training (the speaker-verification idea).

Every model learns, from the thesis people, how walks of one person differ from walks of two different people,
and then compares a new walk with the two enrolled people of the day:

  m0  nearest centroid on the four readouts (speed, steps, swing, bounce): decide_m0. It needs no training, and
      the app falls back to it when no trained model loads.
  m1  learned whitening: the within-person covariance of the training people (shrunk towards its diagonal)
      whitens the features; the between-person covariance is then diagonalised in that space, and `keep`
      directions with the largest between/within ratio are kept (0 keeps all). Nearest centroid there. With
      `plda` (calibration C2), each centroid is shrunk towards the training mean by its number of walks and the
      distance allows for the centroid's remaining uncertainty: a light PLDA.
  m2  pairwise same/different: L2 logistic regression on |za - zb| and za * zb (standardised features) over
      pairs of training walks. Same-person pairs are taken only ACROSS recordings, so "same recording" cannot be
      learned as "same person" (different people are always different recordings). On the day: the mean
      P(same) of the walk against each person's enrolment walks; the higher wins.

Calibration C1 (`session_centre`) centres the standardisation on the session's own enrolment walks (both
people, no labels). Only thesis data is ever fitted or saved here; a visitor's walks are only ever passed in.
"""

from __future__ import annotations

import itertools
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

import openday
from openday import walks

# --- 1. The nearest-centre rule (m0) ---

# Two-person nearest centroid in standardised feature space, with a margin and "not sure".

CLASSES = ("uzayr", "visitor")


@dataclass
class Decision:
    label: str                       # "uzayr" or "visitor": the nearer centroid
    margin: float                    # (d_far - d_near) / (d_far + d_near), 0..1
    not_sure: bool                   # margin below tau
    dist: dict[str, float]
    contrib: dict[str, float]        # per feature: > 0 pulls towards Uzayr, < 0 towards the visitor
    speed_only: Optional["Decision"] = field(default=None, repr=False)
    why: list = field(default_factory=list)   # plain words for what separated this walk (why)


def fit_spread(pools: dict[str, np.ndarray], prior_sd: np.ndarray) -> np.ndarray:
    """Per feature: max(pooled within-person SD of the current pools, prior SD)."""
    ss, dof = np.zeros(len(prior_sd)), 0
    for x in pools.values():
        if len(x) > 1:
            ss += ((x - x.mean(axis=0)) ** 2).sum(axis=0)
            dof += len(x) - 1
    pooled = np.sqrt(ss / dof) if dof > 0 else np.zeros(len(prior_sd))
    return np.maximum(pooled, prior_sd)


def decide_m0(x: np.ndarray, pools: dict[str, np.ndarray], names: Sequence[str], cfg: dict,
              tau: Optional[float] = None, with_speed_only: bool = True) -> Optional[Decision]:
    """Classify one walk's feature vector `x` (ordered as `names`) against the Uzayr and visitor pools.

    NaN features (e.g. no step-rate peak) are left out of both distances. Returns None if a pool is empty.
    """
    if any(len(pools.get(c, ())) == 0 for c in CLASSES):
        return None
    c = cfg["classifier"]
    tau = float(c["tau"] if tau is None else tau)
    prior = np.array([float(c["prior_sd"][n]) for n in names])
    P = {k: np.asarray(v, float) for k, v in pools.items()}
    P = {k: np.where(np.isnan(v), np.nanmean(v, axis=0), v) for k, v in P.items()}
    s = fit_spread(P, prior)
    mu = {k: np.nanmean(P[k], axis=0) for k in CLASSES}
    ok = np.isfinite(x) & np.isfinite(mu["uzayr"]) & np.isfinite(mu["visitor"])
    z = {k: np.where(ok, (x - mu[k]) / s, 0.0) for k in CLASSES}
    d = {k: float(np.sqrt((z[k] ** 2).sum())) for k in CLASSES}
    near, far = sorted(CLASSES, key=lambda k: d[k])
    tot = d[near] + d[far]
    margin = (d[far] - d[near]) / tot if tot > 0 else 0.0
    contrib = {n: float(z["visitor"][i] ** 2 - z["uzayr"][i] ** 2) for i, n in enumerate(names)}
    out = Decision(near, float(margin), bool(margin < tau), d, contrib)
    if with_speed_only and "speed" in names and len(names) > 1:
        i = list(names).index("speed")
        out.speed_only = decide_m0(x[i:i + 1], {k: v[:, i:i + 1] for k, v in P.items()}, ["speed"], cfg, tau, False)
    return out


# --- 2. The trained models ---

READOUTS = ["speed", "steps", "swing", "bounce"]
ALL = READOUTS + walks.GAIT_NAMES
LOG = ("moddepth_dop", "sub_dispersion", "moddepth_dop10")   # scale-like and heavy-tailed: compared on a log scale


def vec(f: dict, names: Sequence[str]) -> np.ndarray:
    v = np.array([float(f.get(n, np.nan)) if f.get(n) is not None else np.nan for n in names], float)
    for i, n in enumerate(names):
        if n in LOG:
            v[i] = np.log(v[i]) if v[i] > 0 else np.nan
    return v


@dataclass
class Model:
    kind: str                                      # "m0", "m1" or "m2"
    names: list
    mu: np.ndarray                                 # training mean and SD (standardisation)
    sd: np.ndarray
    tau: float = 0.08
    prior_sd: dict = field(default_factory=dict)   # m0 only
    P: Optional[np.ndarray] = None                 # m1: (r, d) projection from standardised features
    lam: Optional[np.ndarray] = None               # m1: between-person variance per direction (within = 1)
    w: Optional[np.ndarray] = None                 # m2: logistic weights on [|za - zb|, za * zb]
    b: float = 0.0
    plda: bool = False                             # C2's centroid shrinkage (m1)
    session_centre: bool = False                   # C1
    meta: dict = field(default_factory=dict)
    cal_m: Optional[np.ndarray] = None             # "How sure": margin knots ...
    cal_p: Optional[np.ndarray] = None             # ... and how often decisions that clear were right (leave-two-out)
    cal2_m: Optional[np.ndarray] = None            # the same for an answer from both walks (the second look): the
    cal2_p: Optional[np.ndarray] = None            # two walks' margin -> how often such answers were right

    def how_sure(self, margin: float) -> Optional[float]:
        """The share of leave-two-out decisions with this margin that were right (rising with the margin)."""
        return _read_curve(self.cal_m, self.cal_p, margin)

    def how_sure_both(self, margin: float) -> Optional[float]:
        """How sure for an answer from both walks (classifier.second_look): the share of leave-two-out close calls
        walked once more whose answer from both walks, with this margin, was right (cal2_m / cal2_p: not part of
        training, fitted separately afterwards by training_home/fit_cal2.py and added to the model file, laptop
        only; a newly trained model has none until it is refitted). Without that curve: the one-walk curve
        (how_sure)."""
        if self.cal2_m is None or not len(self.cal2_m):
            return self.how_sure(margin)
        return _read_curve(self.cal2_m, self.cal2_p, margin)

    # -- save / load (JSON only, thesis data only) ------------------------------------------------------
    def to_json(self) -> dict:
        arr = lambda a: None if a is None else np.asarray(a, float).tolist()  # noqa: E731
        return {"kind": self.kind, "names": list(self.names), "mu": arr(self.mu), "sd": arr(self.sd),
                "tau": float(self.tau), "prior_sd": dict(self.prior_sd), "P": arr(self.P), "lam": arr(self.lam),
                "w": arr(self.w), "b": float(self.b), "plda": bool(self.plda),
                "session_centre": bool(self.session_centre), "meta": self.meta, "cal_m": arr(self.cal_m),
                "cal_p": arr(self.cal_p), "cal2_m": arr(self.cal2_m), "cal2_p": arr(self.cal2_p)}

    @staticmethod
    def from_json(d: dict) -> "Model":
        arr = lambda a: None if a is None else np.asarray(a, float)  # noqa: E731

        def extra(a):                                      # the two-walk curve (optional, fitted separately): one
            try:                                           # that is not numbers never stops the model loading
                return arr(a)                              # (load_model then drops it, logged)
            except (TypeError, ValueError):
                return np.array([np.nan])
        return Model(d["kind"], list(d["names"]), arr(d["mu"]), arr(d["sd"]), float(d["tau"]),
                     dict(d.get("prior_sd", {})), arr(d.get("P")), arr(d.get("lam")), arr(d.get("w")),
                     float(d.get("b", 0.0)), bool(d.get("plda", False)), bool(d.get("session_centre", False)),
                     dict(d.get("meta", {})), arr(d.get("cal_m")), arr(d.get("cal_p")), extra(d.get("cal2_m")),
                     extra(d.get("cal2_p")))

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "Model":
        with open(path, encoding="utf-8") as f:
            return Model.from_json(json.load(f))

    # -- standardised features --------------------------------------------------------------------------
    def z(self, X: np.ndarray, centre: Optional[np.ndarray] = None) -> np.ndarray:
        """(X - mean) / SD with missing values at the mean (z = 0). `centre` replaces the training mean (C1)."""
        mu = self.mu if centre is None else np.where(np.isfinite(centre), centre, self.mu)
        Z = (np.atleast_2d(X) - mu) / self.sd
        return np.where(np.isfinite(Z), Z, 0.0)


def _read_curve(cm: Optional[np.ndarray], cp: Optional[np.ndarray], margin: float) -> Optional[float]:
    """A How-sure curve read at this margin, capped at 99 % (never "100 %"); None without a curve."""
    if cm is None or not len(cm):
        return None
    return min(float(np.interp(margin, cm, cp)), 0.99)


def _curve_ok(cm: Optional[np.ndarray], cp: Optional[np.ndarray]) -> bool:
    """A How-sure curve that can be read: margin knots and shares, as many of each (at least one), all finite, the
    knots never falling, the shares between 0 and 1 and never falling (a wider margin is never less sure)."""
    if cm is None or cp is None or np.ndim(cm) != 1 or np.ndim(cp) != 1 or len(cm) != len(cp) or not len(cm):
        return False
    return bool(np.all(np.isfinite(cm)) and np.all(np.isfinite(cp)) and np.all(np.diff(cm) >= 0)
                and np.all((cp >= 0.0) & (cp <= 1.0)) and np.all(np.diff(cp) >= 0))


def model_file(cfg: dict) -> str:
    """classifier.model_path, inside OpenDay/ unless it is an absolute path."""
    p = cfg["classifier"]["model_path"]
    return p if os.path.isabs(p) else os.path.join(openday.OPENDAY, p)


def load_model(cfg: dict, log=print) -> Optional[Model]:
    """The trained model the config names (classifier.kind, model_path), its tau set to classifier.model_tau. None
    for kind m0, or when the file is missing or unreadable (logged; the nearest-centre rule m0 is used then). Its
    two-walk How sure curve (cal2_m / cal2_p, the second look's, fitted separately after training) is kept only when
    it is well formed (_curve_ok); else it is dropped (logged) and an answer from both walks reads the one-walk
    curve."""
    c = cfg["classifier"]
    if c.get("kind", "m0") == "m0":
        return None
    p = model_file(cfg)
    try:
        m = Model.load(p)
    except (OSError, ValueError, KeyError) as exc:
        log(f"model {p} not loaded ({exc}): using m0")
        return None
    m.tau = float(c.get("model_tau", m.tau))
    if (m.cal2_m is not None or m.cal2_p is not None) and not _curve_ok(m.cal2_m, m.cal2_p):
        log("model: its two-walk How sure curve (cal2_m / cal2_p) is not usable: the one-walk curve is read instead")
        m.cal2_m = m.cal2_p = None
    log(f"model {m.kind} ({len(m.names)} features" + ("; How sure from both walks: its own curve)"
                                                       if m.cal2_m is not None else ")"))
    return m


# --- 3. Deciding one walk with a model ---


def _pair_feats(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    return np.concatenate([np.abs(za - zb), za * zb], axis=-1)


def _session_centre(m: Model, enrol: dict) -> Optional[np.ndarray]:
    if not m.session_centre:
        return None
    allx = np.concatenate([np.atleast_2d(v) for v in enrol.values()])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a feature missing in every walk stays missing
        return np.nanmean(allx, axis=0)


def scores(m: Model, x: np.ndarray, enrol: dict) -> tuple[dict, float]:
    """Per person, a score where lower is nearer (m0, m1: distance; m2: 1 - mean P(same)), and the margin."""
    if m.kind == "m0":
        prior = {"classifier": {"tau": m.tau, "prior_sd": m.prior_sd}}
        d = decide_m0(np.asarray(x, float), {k: np.asarray(v, float) for k, v in enrol.items()}, m.names, prior,
                      with_speed_only=False)
        return d.dist, d.margin
    centre = _session_centre(m, enrol)
    zx = m.z(x, centre)[0]
    out = {}
    if m.kind == "m1":
        yx = m.P @ zx
        for k, E in enrol.items():
            Y = m.z(E, centre) @ m.P.T
            n = len(Y)
            if m.plda:
                s = n * m.lam / (n * m.lam + 1.0)
                c, v = s * Y.mean(axis=0), m.lam / (n * m.lam + 1.0)
                out[k] = float(np.sqrt((((yx - c) ** 2) / (1.0 + v)).sum()))
            else:
                out[k] = float(np.sqrt(((yx - Y.mean(axis=0)) ** 2).sum()))
        a, b = sorted(out.values())
        return out, (b - a) / (a + b) if a + b > 0 else 0.0
    if m.kind == "m2":
        for k, E in enrol.items():
            ZE = m.z(E, centre)
            p = 1.0 / (1.0 + np.exp(-(_pair_feats(np.broadcast_to(zx, ZE.shape), ZE) @ m.w + m.b)))
            out[k] = float(1.0 - p.mean())
        a, b = sorted(out.values())
        return out, float(b - a)
    raise ValueError(m.kind)


def decide(m: Model, f: dict, pools: dict) -> Optional[Decision]:
    """One walk's feature dict `f` against pools {"uzayr": [feature dicts], "visitor": [...]}: a Decision with
    the margin, not sure below tau, the per-readout pulls the tiles show, and the plain words for the line."""
    if any(len(pools.get(k, ())) == 0 for k in CLASSES):
        return None
    x = vec(f, m.names)
    enrol = {k: np.array([vec(g, m.names) for g in pools[k]]) for k in CLASSES}
    dist, margin = scores(m, x, enrol)
    near = min(CLASSES, key=lambda k: dist[k])
    pulls = _pulls(m, x, enrol)
    d = Decision(near, float(margin), bool(margin < m.tau), dist, pulls)
    d.why = why(pulls, near)
    xs = vec(f, ["speed"])
    es = {k: np.array([vec(g, ["speed"]) for g in pools[k]]) for k in CLASSES}
    sp = Model("m0", ["speed"], np.zeros(1), np.ones(1), m.tau, {"speed": m.prior_sd.get("speed", 0.06)})
    if np.isfinite(xs).all():
        dd, mm = scores(sp, xs, es)
        d.speed_only = Decision(min(CLASSES, key=lambda k: dd[k]), float(mm), bool(mm < m.tau), dd, {})
    return d


def average_walks(fs: Sequence[dict]) -> dict:
    """Several walks' feature dicts as one, feature by feature: the second look (classifier.second_look) decides on
    the average of the close call and the walk after it. A reading missing (NaN) in one walk takes the other's.
    The readings the models compare on a log scale (LOG) are averaged on that scale, as the model sees them."""
    out: dict = {}
    for k in dict.fromkeys(k for f in fs for k in f):
        vals = [f.get(k) for f in fs]
        try:
            v = np.array([np.nan if x is None else float(x) for x in vals], float)
        except (TypeError, ValueError):                    # (not a number: the newest walk's)
            out[k] = vals[-1]
            continue
        ok = np.isfinite(v) & ((v > 0) if k in LOG else True)
        if not ok.any():
            out[k] = float("nan")
        elif k in LOG:
            out[k] = float(np.exp(np.mean(np.log(v[ok]))))
        else:
            out[k] = float(np.mean(v[ok]))
    return out


def _pulls(m: Model, x: np.ndarray, enrol: dict) -> dict:
    """Per feature, in standardised units: > 0 pulls towards Uzayr, < 0 towards the visitor (squared distances)."""
    zx = m.z(x)[0]
    cu, cv = m.z(enrol["uzayr"]).mean(axis=0), m.z(enrol["visitor"]).mean(axis=0)
    ok = np.isfinite(x)
    c = np.where(ok, (zx - cv) ** 2 - (zx - cu) ** 2, 0.0)
    return {n: float(c[i]) for i, n in enumerate(m.names)}


def why(pulls: dict, label: str, top: int = 3) -> list[str]:
    """The plain-word groups that pulled most towards the decision (walks.GAIT_PLAIN), strongest first."""
    sign = 1.0 if label == "uzayr" else -1.0
    g = []
    for words, members in walks.GAIT_PLAIN:
        v = [pulls[n] for n in members if n in pulls]
        if v:
            g.append((sign * float(np.mean(v)), words))
    g.sort(reverse=True)
    return [w for s, w in g[:top] if s > 0.05]


# --- 4. The pools and the day's tally ---

# Walk pools and the day's tally.
#
# The pools (Uzayr's learning walks, the visitor's) live in memory only: they are never written, and the visitor's
# name is not stored here at all (the app holds it). The tally (state/tally.json) holds counts only.

_ids = itertools.count(1)


def _atomic_write(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


class Pool:
    """A rolling pool of walks (the newest `cap` kept), in memory only."""

    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.walks: list[dict] = []

    def add(self, feats: dict[str, float]) -> int:
        w = {"id": next(_ids), "t": time.time(), "f": dict(feats)}
        self.walks = (self.walks + [w])[-self.cap:]
        return w["id"]

    def remove(self, wid: int) -> bool:
        n = len(self.walks)
        self.walks = [w for w in self.walks if w["id"] != wid]
        return len(self.walks) < n

    def clear(self) -> None:
        self.walks = []

    def __len__(self) -> int:
        return len(self.walks)

    def matrix(self, names) -> np.ndarray:
        return np.array([[w["f"].get(n, np.nan) for n in names] for w in self.walks], float).reshape(-1, len(names))

    def means(self, names) -> dict[str, float]:
        m = self.matrix(names)
        return {n: float(np.nanmean(m[:, i])) if len(m) and np.isfinite(m[:, i]).any() else float("nan")
                for i, n in enumerate(names)}


class Tally:
    """The day's players, each under an anonymous index, and their right / wrong / not sure counts (the app counts
    no answer, so these stay 0). No names."""

    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self.doc = {"date": time.strftime("%Y-%m-%d"), "right": 0, "wrong": 0, "not_sure": 0, "visitors": {}}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("date") == self.doc["date"]:
                self.doc = d

    def new_visitor(self) -> int:
        idx = len(self.doc["visitors"]) + 1
        self.doc["visitors"][str(idx)] = {"right": 0, "wrong": 0, "not_sure": 0}
        self.save()
        return idx

    def save(self) -> None:
        if self.path:
            _atomic_write(self.path, self.doc)
