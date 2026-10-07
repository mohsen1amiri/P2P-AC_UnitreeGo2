"""Statistics used in the paper.

* ``N80`` (Sec. VI-C): environment steps until the evaluated success rate first reaches 80 %. A seed that never
  reaches it within the budget is *censored*; it is reported separately (never silently dropped).
* Aggregation over seeds: mean, sample standard deviation and a percentile-bootstrap 95 % confidence interval.
* Success rates of a fixed controller (planner, hardware): Wilson score interval.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def n_to_threshold(t: np.ndarray, success: np.ndarray, threshold: float = 0.8) -> Optional[float]:
    """First evaluation step at which the success rate reaches ``threshold`` (None if never)."""
    t = np.asarray(t, dtype=float)
    s = np.asarray(success, dtype=float)
    order = np.argsort(t)
    t, s = t[order], s[order]
    hit = np.nonzero(s >= threshold - 1e-12)[0]
    return float(t[hit[0]]) if hit.size else None


def bootstrap_ci(values: Sequence[float], level: float = 0.95, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if v.size == 0:
        return math.nan, math.nan
    if v.size == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def summarize_seeds(values: Sequence[Optional[float]]) -> dict:
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    lo, hi = bootstrap_ci(v)
    return {
        "mean": float(v.mean()) if v.size else math.nan,
        "std": float(v.std(ddof=1)) if v.size > 1 else (0.0 if v.size == 1 else math.nan),
        "ci_lo": lo, "ci_hi": hi, "n": int(v.size),
    }


def summarize_n80(values: Sequence[Optional[float]], budget: float) -> dict:
    """``values`` per seed (None = censored)."""
    reached = [v for v in values if v is not None]
    s = summarize_seeds(reached)
    censored = len(values) - len(reached)
    lb = [v if v is not None else budget for v in values]
    s.update({"n_seeds": len(values), "n_reached": len(reached), "n_censored": censored,
              "mean_censored_as_budget": float(np.mean(lb)) if lb else math.nan})
    return s


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def welch_test(a: Sequence[float], b: Sequence[float]) -> dict:
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return {"t": math.nan, "p": math.nan}
    r = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"t": float(r.statistic), "p": float(r.pvalue), "mannwhitney_u": float(u.statistic), "mannwhitney_p": float(u.pvalue)}


def interpolate_curve(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of a per-seed learning curve onto a common grid (NaN beyond the last point)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    out = np.interp(grid, t, y)
    out[grid > t[-1] + 1e-9] = np.nan
    return out
