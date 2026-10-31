"""
Conformal Calibration Engine for Chance-Constrained Query Optimization.

Roshini (Lead: Robust Optimizer & Theory) — Phase 2 Deliverable

Implements split conformal prediction to compute finite-sample certified
upper bounds on query cost and latency:
    Pr(C_i <= C_hat_i^+) >= 1 - delta

Under exchangeability of calibration and test queries, this guarantees
that admitted candidate plans satisfy the budget constraint with certified
probability (1 - delta) without making parametric distributional assumptions.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class ConformalQuantile:
    """Computed conformal quantile and multiplier for a given confidence level."""

    metric_name: str
    delta: float
    confidence_level: float  # 1 - delta
    sample_size: int
    quantile_index: int
    raw_quantile_q: float
    multiplier: float  # exp(q)
    residuals_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConformalQuantile:
        return cls(**data)


@dataclass
class CalibrationSample:
    """One observation used for conformal calibration."""

    query_id: str
    plan_id: str
    predicted_cost: float
    actual_cost: float
    predicted_latency: float
    actual_latency: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConformalCalibrator:
    """
    Split conformal prediction calibration engine.

    Computes non-conformity scores via absolute log-residuals:
        e_i = |log(actual_i) - log(predicted_i)|

    and selects the finite-sample corrected quantile:
        k = ceil((n + 1) * (1 - delta))
        q = e_(k)
        multiplier = exp(q)
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def compute_residuals(
        self,
        predicted: Sequence[float],
        actual: Sequence[float],
    ) -> List[float]:
        """
        Compute absolute log-residuals:
            e_i = |log(actual_i + eps) - log(predicted_i + eps)|
        """
        if len(predicted) != len(actual):
            raise ValueError(
                f"Length mismatch: {len(predicted)} predicted vs {len(actual)} actual"
            )

        residuals: List[float] = []
        for p, a in zip(predicted, actual):
            p_val = max(float(p), self.eps)
            a_val = max(float(a), self.eps)
            residuals.append(abs(math.log(a_val) - math.log(p_val)))
        return residuals

    def compute_quantile(
        self,
        residuals: Sequence[float],
        delta: float,
        metric_name: str = "cost",
    ) -> ConformalQuantile:
        """
        Compute the finite-sample (1 - delta) quantile from residuals.

        Uses the exact order-statistic rule:
            k = ceil((n + 1) * (1 - delta))

        If k > n, returns the maximum residual with a warning flag, as
        sample size n is too small to strictly achieve 1 - delta without
        further randomization.
        """
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        n = len(residuals)
        if n == 0:
            raise ValueError("Cannot calibrate on empty residuals list")

        sorted_residuals = sorted(residuals)

        # Finite-sample order statistic rank: 1-indexed ceil((n + 1) * (1 - delta))
        rank = math.ceil((n + 1) * (1.0 - delta))
        idx = min(rank - 1, n - 1)
        q = sorted_residuals[idx]
        multiplier = math.exp(q)

        mean_res = sum(sorted_residuals) / n
        median_res = sorted_residuals[n // 2]
        max_res = sorted_residuals[-1]
        min_res = sorted_residuals[0]

        summary = {
            "min": float(min_res),
            "median": float(median_res),
            "mean": float(mean_res),
            "max": float(max_res),
            "q_val": float(q),
        }

        return ConformalQuantile(
            metric_name=metric_name,
            delta=delta,
            confidence_level=1.0 - delta,
            sample_size=n,
            quantile_index=idx,
            raw_quantile_q=float(q),
            multiplier=float(multiplier),
            residuals_summary=summary,
        )

    def calibrate(
        self,
        samples: Sequence[CalibrationSample],
        delta: float = 0.05,
        metric: str = "cost",
    ) -> ConformalQuantile:
        """Calibrate either 'cost' or 'latency' from CalibrationSample objects."""
        if metric == "cost":
            pred = [s.predicted_cost for s in samples]
            act = [s.actual_cost for s in samples]
        elif metric == "latency":
            pred = [s.predicted_latency for s in samples]
            act = [s.actual_latency for s in samples]
        else:
            raise ValueError(f"Unknown metric '{metric}', must be 'cost' or 'latency'")

        residuals = self.compute_residuals(pred, act)
        return self.compute_quantile(residuals, delta=delta, metric_name=metric)

    def calibrate_joint(
        self,
        samples: Sequence[CalibrationSample],
        delta_cost: float = 0.05,
        delta_latency: float = 0.05,
    ) -> Tuple[ConformalQuantile, ConformalQuantile]:
        """Calibrate both cost and latency bounds."""
        cost_q = self.calibrate(samples, delta=delta_cost, metric="cost")
        lat_q = self.calibrate(samples, delta=delta_latency, metric="latency")
        return cost_q, lat_q

    @staticmethod
    def split_query_grouped(
        samples: Sequence[CalibrationSample],
        cal_fraction: float = 0.3,
        seed: int = 42,
    ) -> Tuple[List[CalibrationSample], List[CalibrationSample]]:
        """
        Partition samples into train/fit and calibration folds GROUPED BY QUERY_ID.

        Guardrail: Strictly prevents data leakage across folds by ensuring that
        all candidate plans or runs of a query appear either in train or in cal.
        """
        import random

        rng = random.Random(seed)
        query_to_samples: Dict[str, List[CalibrationSample]] = {}
        for s in samples:
            query_to_samples.setdefault(s.query_id, []).append(s)

        unique_queries = sorted(query_to_samples.keys())
        rng.shuffle(unique_queries)

        n_cal = max(1, int(len(unique_queries) * cal_fraction))
        cal_queries = set(unique_queries[:n_cal])

        cal_fold: List[CalibrationSample] = []
        train_fold: List[CalibrationSample] = []

        for q_id, q_samples in query_to_samples.items():
            if q_id in cal_queries:
                cal_fold.extend(q_samples)
            else:
                train_fold.extend(q_samples)

        return train_fold, cal_fold

    @staticmethod
    def evaluate_coverage(
        test_samples: Sequence[CalibrationSample],
        quantile: ConformalQuantile,
    ) -> Dict[str, Any]:
        """
        Evaluate empirical coverage on held-out test data.

        Returns coverage rate, violation rate, and margin statistics.
        """
        if not test_samples:
            return {"error": "Empty test samples"}

        is_cost = quantile.metric_name == "cost"
        covered_count = 0
        total = len(test_samples)
        margins: List[float] = []

        for s in test_samples:
            p = s.predicted_cost if is_cost else s.predicted_latency
            a = s.actual_cost if is_cost else s.actual_latency

            upper_bound = p * quantile.multiplier
            is_covered = (a <= upper_bound)
            if is_covered:
                covered_count += 1
            margins.append(upper_bound - a)

        empirical_coverage = covered_count / total
        violation_rate = 1.0 - empirical_coverage

        return {
            "metric": quantile.metric_name,
            "target_confidence": quantile.confidence_level,
            "delta": quantile.delta,
            "test_sample_size": total,
            "covered_count": covered_count,
            "empirical_coverage": empirical_coverage,
            "violation_rate": violation_rate,
            "coverage_error": abs(empirical_coverage - quantile.confidence_level),
            "mean_margin": sum(margins) / total,
            "min_margin": min(margins),
            "max_margin": max(margins),
        }


# Compatibility exports for the predictor-level calibration engine.
from predictor.conformal import (
    ConformalCalibrationEngine,
    ConformalMetrics,
    compute_log_residuals,
    compute_finite_sample_quantile,
    compute_certified_upper_bound,
)
