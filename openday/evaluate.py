"""Two helpers for fitting a trained model: fit_standard (each measure's mean and spread) and calibrate (the "How
sure" curve). The app never calls them; the retraining tools on the laptop (training_home/, not in this repo) do.

Scoring the home-test (bench) recordings is in bench.py.
"""

from __future__ import annotations

import numpy as np


def fit_standard(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu, np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)


def calibrate(margins: np.ndarray, correct: np.ndarray, bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Margin -> share right: equal-count bins, then pooled-adjacent-violators so it never falls as the margin
    grows. Returns (bin centre margins, share right)."""
    o = np.argsort(margins)
    m, c = np.asarray(margins, float)[o], np.asarray(correct, float)[o]
    parts = np.array_split(np.arange(len(m)), bins)
    x = [float(np.median(m[p])) for p in parts if len(p)]
    y = [float(c[p].mean()) for p in parts if len(p)]
    w = [len(p) for p in parts if len(p)]
    blocks = [[yi, wi, [i]] for i, (yi, wi) in enumerate(zip(y, w))]
    k = 0
    while k < len(blocks) - 1:                         # pool adjacent violators
        if blocks[k][0] > blocks[k + 1][0]:
            a, b = blocks[k], blocks.pop(k + 1)
            a[0] = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1])
            a[1] += b[1]
            a[2] += b[2]
            k = max(k - 1, 0)
        else:
            k += 1
    fit = np.empty(len(y))
    for val, _w, idx in blocks:
        fit[idx] = val
    return np.array(x), fit
