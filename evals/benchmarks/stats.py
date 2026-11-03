"""The one percentile/summary implementation used by every benchmark summary and comparison.

Percentiles use the nearest-rank method: the value at 1-based rank ceil(q * n)
of the sorted sample (q in (0, 1]). It never interpolates, so it always returns
an observed value; with n = 18, p95 is the maximum. It is used identically for
every variant, so comparisons are like-for-like.
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable

PERCENTILE_METHOD = "nearest-rank: sorted value at 1-based rank ceil(q*n); no interpolation"


def nearest_rank(values: Iterable[float], q: float) -> float | None:
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def summarize(values: Iterable[float | None]) -> dict:
    """n, p50, p95, min, max, mean and coefficient of variation of the finite values."""
    data = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not data:
        return {"n": 0, "p50": None, "p95": None, "min": None, "max": None, "mean": None, "cv": None}
    mean = statistics.fmean(data)
    cv = statistics.stdev(data) / mean if len(data) > 1 and mean else None
    return {"n": len(data), "p50": nearest_rank(data, 0.5), "p95": nearest_rank(data, 0.95), "min": min(data),
            "max": max(data), "mean": round(mean, 1), "cv": None if cv is None else round(cv, 4)}


def median(values: Iterable[float | None]) -> float | None:
    return nearest_rank([v for v in values if v is not None], 0.5)


def improvement_pct(baseline: float | None, variant: float | None) -> float | None:
    """Lower-is-better (latency, memory): 100 * (baseline - variant) / baseline; None if unavailable."""
    if baseline is None or variant is None or baseline <= 0:
        return None
    return round(100 * (baseline - variant) / baseline, 1)


def increase_pct(baseline: float | None, variant: float | None) -> float | None:
    """Higher-is-better (throughput): 100 * (variant - baseline) / baseline; None if unavailable."""
    if baseline is None or variant is None or baseline <= 0:
        return None
    return round(100 * (variant - baseline) / baseline, 1)
