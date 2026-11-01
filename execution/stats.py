"""
Statistics for the empirical evaluation (owner: Anaswara).

Reviewer guardrail: medians with 95% confidence intervals and Wilcoxon
signed-rank paired tests; report losses, not only wins.

* ``bootstrap_ci``      percentile bootstrap CI of a statistic (default median)
* ``wilcoxon_paired``   paired two-sided Wilcoxon signed-rank test + win/loss/tie
* ``summarize_outcomes`` per-strategy metric suite used by E1/E2:
    median latency (+CI), P90/P99 latency over all runs of the chosen plans,
    budget violation rate, feasibility rate, speedup vs Spark default,
    Oracle regret
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


def median(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def geometric_mean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and v > 0], dtype=float)
    return float(np.exp(np.mean(np.log(arr)))) if arr.size else float("nan")


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.median,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    stats = np.apply_along_axis(stat, 1, arr[idx])
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """
    Two-sided Wilcoxon signed-rank test on paired samples a vs b.

    ``median_diff`` = median(a - b); negative means ``a`` is lower (faster /
    cheaper). ``wins`` counts pairs where a < b.
    """

    pairs = [
        (x, y) for x, y in zip(a, b)
        if x is not None and y is not None and not (math.isnan(x) or math.isnan(y))
    ]
    diffs = np.asarray([x - y for x, y in pairs], dtype=float)
    scale = np.asarray([max(abs(x), abs(y), 1e-12) for x, y in pairs], dtype=float)
    # treat relative differences below 1e-9 as ties (floating-point noise)
    nonzero = np.abs(diffs) > 1e-9 * scale
    result: Dict[str, Any] = {
        "n_pairs": int(len(pairs)),
        "n_nonzero": int(nonzero.sum()),
        "wins": int((nonzero & (diffs < 0)).sum()),
        "losses": int((nonzero & (diffs > 0)).sum()),
        "ties": int((~nonzero).sum()),
        "median_diff": float(np.median(diffs)) if diffs.size else float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
    }
    if result["n_nonzero"] == 0:
        result["p_value"] = 1.0
        return result
    try:
        from scipy.stats import wilcoxon

        test = wilcoxon(diffs[nonzero], zero_method="wilcox", alternative="two-sided")
        result["statistic"] = float(test.statistic)
        result["p_value"] = float(test.pvalue)
    except Exception as exc:  # pragma: no cover - scipy missing
        result["error"] = str(exc)
    return result


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""

    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = min(1.0, (m - rank) * p_values[i])
        running = max(running, value)
        adjusted[i] = running
    return adjusted


def summarize_outcomes(
    outcomes: Sequence[Dict[str, Any]],
    group_keys: Sequence[str] = ("strategy",),
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Aggregate per-query outcome rows (one row per query x strategy) into the
    full metric suite, grouped by ``group_keys``.
    """

    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in outcomes:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)

    summary: List[Dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        selected = [r for r in rows if r.get("plan_id")]
        lat = [r["latency_s"] for r in selected]
        cost = [r["cost"] for r in selected]
        run_lat = [x for r in selected for x in (r.get("run_latencies_s") or [])]
        lo, hi = bootstrap_ci(lat, level=level, n_resamples=n_resamples, seed=seed)
        clo, chi = bootstrap_ci(cost, level=level, n_resamples=n_resamples, seed=seed)
        regrets = [r["oracle_regret"] for r in selected if r.get("oracle_regret") is not None]
        speedups = [r["speedup_vs_spark_default"] for r in selected
                    if r.get("speedup_vs_spark_default") is not None]
        entry = dict(zip(group_keys, key))
        entry.update({
            "n_queries": len(rows),
            "n_selected": len(selected),
            "median_latency_s": median(lat),
            "latency_ci_low": lo,
            "latency_ci_high": hi,
            "p90_latency_s": percentile(run_lat, 90),
            "p99_latency_s": percentile(run_lat, 99),
            "median_cost": median(cost),
            "cost_ci_low": clo,
            "cost_ci_high": chi,
            "budget_violation_rate": (
                float(np.mean([r["budget_violated"] for r in selected])) if selected else float("nan")
            ),
            "run_violation_rate": (
                float(np.mean([r["run_violation_rate"] for r in selected])) if selected else float("nan")
            ),
            "feasibility_rate": (
                float(np.mean([not r["infeasible_fallback"] for r in rows])) if rows else float("nan")
            ),
            "median_oracle_regret": median(regrets),
            "mean_oracle_regret": float(np.mean(regrets)) if regrets else float("nan"),
            "geomean_speedup_vs_spark_default": geometric_mean(speedups),
            "oracle_hit_rate": (
                float(np.mean([r["matches_oracle"] for r in selected])) if selected else float("nan")
            ),
        })
        summary.append(entry)
    return summary


def paired_tests(
    outcomes: Sequence[Dict[str, Any]],
    reference: str,
    metric: str = "latency_s",
    pair_keys: Sequence[str] = ("query_id",),
    group_keys: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Wilcoxon test of ``reference`` vs every other strategy, Holm-adjusted."""

    by_group: Dict[tuple, Dict[str, Dict[tuple, float]]] = {}
    for row in outcomes:
        if not row.get("plan_id"):
            continue
        g = tuple(row[k] for k in group_keys)
        by_group.setdefault(g, {}).setdefault(row["strategy"], {})[
            tuple(row[k] for k in pair_keys)
        ] = row[metric]

    tests: List[Dict[str, Any]] = []
    for g, strategies in by_group.items():
        ref = strategies.get(reference, {})
        block: List[Dict[str, Any]] = []
        for other, values in sorted(strategies.items()):
            if other == reference:
                continue
            keys = sorted(set(ref) & set(values))
            res = wilcoxon_paired([ref[k] for k in keys], [values[k] for k in keys])
            entry = dict(zip(group_keys, g))
            entry.update({"reference": reference, "baseline": other, "metric": metric})
            entry.update(res)
            block.append(entry)
        adjusted = holm_correction([b["p_value"] if not math.isnan(b["p_value"]) else 1.0 for b in block])
        for b, p in zip(block, adjusted):
            b["p_value_holm"] = p
        tests.extend(block)
    return tests


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"
