"""Metrics (protocol section 5).

Calibration is measured top-label for every system: the confidence attached to the predicted
label against whether that label is correct. This is the only form available for all systems
(LLMs return one verbalized confidence), so it is used uniformly.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def macro_f1(gold: list, pred: list, labels: list) -> float:
    f = []
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def ece_equal_mass(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins, independent of row order.

    Items are ordered by confidence; a block of tied confidences that straddles a bin boundary is
    split fractionally (each tied item contributes its share of weight to each bin it overlaps),
    so the result does not depend on the arbitrary order of tied items.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    levels = np.unique(conf)
    edges = np.linspace(0, n, bins + 1)
    w = np.zeros(bins); sc = np.zeros(bins); sy = np.zeros(bins)
    start = 0.0
    for lv in levels:
        m = conf == lv
        cnt = m.sum()
        ybar = correct[m].mean()
        lo, hi = start, start + cnt
        for b in range(bins):
            ov = max(0.0, min(hi, edges[b + 1]) - max(lo, edges[b]))
            if ov > 0:
                w[b] += ov; sc[b] += ov * lv; sy[b] += ov * ybar
        start = hi
    nz = w > 0
    return float(np.sum(np.abs(sc[nz] / w[nz] - sy[nz] / w[nz]) * w[nz]) / n)


def brier_top(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((conf - correct) ** 2))


def risk_at_coverage(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Expected error rate among the most confident `coverage` share, ties broken uniformly at random.

    Items are accepted in descending confidence. A tied block straddling the cut contributes
    its mean error rate proportionally to the fraction of it that is accepted.
    """
    n = len(conf)
    k = coverage * n
    if k <= 0:
        return math.nan
    err = 1 - correct
    levels = sorted(set(conf.tolist()), reverse=True)
    taken, errors = 0.0, 0.0
    for lv in levels:
        m = conf == lv
        cnt = m.sum()
        take = min(cnt, k - taken)
        errors += take * err[m].mean()
        taken += take
        if taken >= k - 1e-9:
            break
    return errors / taken


def aurc(conf: np.ndarray, correct: np.ndarray, steps: int = 100) -> float:
    covs = np.linspace(1 / steps, 1, steps)
    return float(np.mean([risk_at_coverage(conf, correct, c) for c in covs]))


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only": b, "b_only": c, "p": float(p)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda x: x[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def bootstrap_ci(fn, *arrays, n: int = 10000, seed: int = 20260927, clusters: np.ndarray | None = None,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over items, or over clusters if given (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    stats = []
    if clusters is None:
        for _ in range(n):
            idx = rng.integers(0, N, N)
            stats.append(fn(*[a[idx] for a in arrays]))
    else:
        uniq = np.unique(clusters)
        members = {u: np.where(clusters == u)[0] for u in uniq}
        for _ in range(n):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([members[u] for u in pick])
            stats.append(fn(*[a[idx] for a in arrays]))
    stats = np.array(stats, dtype=float)
    stats = stats[~np.isnan(stats)]
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))
