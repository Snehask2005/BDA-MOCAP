r"""
Chance-Constrained Conformal Calibration Engine.
Part of WP4 and Phase 2 Deliverables for Hridhika (Learned Predictor Lead).

Mathematical Foundation (Split Conformal Prediction):
Given calibration pairs (C_i, \hat{C}_i) for i = 1, ..., n:
1. Nonconformity score (log-space absolute residual):
     e_i = |log(C_i) - log(\hat{C}_i)|
2. Finite-sample quantile at level (1 - \delta):
     k = ceil((n + 1) * (1 - \delta))
     q = sorted(e_1, ..., e_n)[k - 1]
3. Certified cost upper bound:
     \hat{C}^+_i = \hat{C}_i * exp(q)
4. Theoretical guarantee under exchangeability:
     Pr(C_{n+1} <= \hat{C}^+_{n+1}) >= 1 - \delta
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConformalMetrics:
    """Evaluation summary for a specific risk level delta."""

    delta: float
    target_coverage: float
    empirical_coverage: float
    coverage_error: float
    quantile_q: float
    point_violation_rate: float
    conformal_violation_rate: float
    mean_inflation_factor: float


def compute_log_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute absolute log-residuals:
    e_i = |log(y_true_i) - log(y_pred_i)|
    Clips values to avoid log(0).
    """
    y_t = np.maximum(np.asarray(y_true, dtype=np.float64), 1e-6)
    y_p = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-6)
    return np.abs(np.log(y_t) - np.log(y_p))


def compute_finite_sample_quantile(
    residuals: np.ndarray,
    delta: float,
) -> float:
    """
    Finite-sample conformal quantile:
    q = sorted(residuals)[ceil((n + 1) * (1 - delta)) - 1]
    """
    res = np.sort(np.asarray(residuals, dtype=np.float64))
    n = len(res)
    if n == 0:
        return 0.0

    k = math.ceil((n + 1) * (1.0 - delta))
    if k > n:
        # Sample size insufficient for exact finite-sample certification; use maximum residual
        return float(res[-1])
    order_idx = max(0, min(k - 1, n - 1))
    return float(res[order_idx])


def compute_certified_upper_bound(
    y_pred: np.ndarray,
    q: float,
) -> np.ndarray:
    r"""
    Formulate certified upper bound:
    \hat{C}^+_i = \hat{C}_i * exp(q)
    """
    y_p = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-6)
    return y_p * math.exp(q)


class ConformalCalibrationEngine:
    """
    Conformal calibration engine managing quantile estimation across
    risk levels delta in {0.01, 0.05, 0.1, 0.2} for cost and latency.
    """

    def __init__(self, deltas: Optional[List[float]] = None):
        self.deltas = deltas or [0.01, 0.05, 0.1, 0.2]
        self.cost_quantiles: Dict[float, float] = {}
        self.latency_quantiles: Dict[float, float] = {}
        self.cost_residuals: np.ndarray = np.array([])
        self.latency_residuals: np.ndarray = np.array([])
        self.num_calibration_samples: int = 0

    def fit(
        self,
        y_true_cost: np.ndarray,
        y_pred_cost: np.ndarray,
        y_true_latency: Optional[np.ndarray] = None,
        y_pred_latency: Optional[np.ndarray] = None,
    ) -> "ConformalCalibrationEngine":
        """
        Fit conformal quantiles using calibration fold data.
        """
        self.cost_residuals = compute_log_residuals(y_true_cost, y_pred_cost)
        self.num_calibration_samples = len(self.cost_residuals)

        for d in self.deltas:
            self.cost_quantiles[d] = compute_finite_sample_quantile(self.cost_residuals, d)

        if y_true_latency is not None and y_pred_latency is not None:
            self.latency_residuals = compute_log_residuals(y_true_latency, y_pred_latency)
            for d in self.deltas:
                self.latency_quantiles[d] = compute_finite_sample_quantile(self.latency_residuals, d)

        return self

    def predict_cost_bound(self, y_pred_cost: np.ndarray, delta: float = 0.05) -> np.ndarray:
        r"""Return certified cost upper bound \hat{C}^+ for risk level delta."""
        if delta not in self.cost_quantiles:
            q = compute_finite_sample_quantile(self.cost_residuals, delta)
        else:
            q = self.cost_quantiles[delta]
        return compute_certified_upper_bound(y_pred_cost, q)

    def predict_latency_bound(self, y_pred_latency: np.ndarray, delta: float = 0.05) -> np.ndarray:
        r"""Return certified latency upper bound \hat{L}^+ for risk level delta."""
        if not self.latency_quantiles:
            # Fall back to cost quantile if latency calibration was omitted
            q = self.cost_quantiles.get(delta, 0.1)
        else:
            q = self.latency_quantiles.get(delta, compute_finite_sample_quantile(self.latency_residuals, delta))
        return compute_certified_upper_bound(y_pred_latency, q)

    def evaluate_test_set(
        self,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
        budgets: Optional[np.ndarray] = None,
    ) -> Dict[float, ConformalMetrics]:
        """
        Evaluate empirical coverage and violation rates on a strictly held-out test set.
        Target criterion: empirical frequency within +/- 2% of (1 - delta).
        """
        y_true = np.asarray(y_test_true, dtype=np.float64)
        y_pred = np.asarray(y_test_pred, dtype=np.float64)
        n = len(y_true)

        results: Dict[float, ConformalMetrics] = {}

        for d in self.deltas:
            q = self.cost_quantiles.get(d, compute_finite_sample_quantile(self.cost_residuals, d))
            y_bound = compute_certified_upper_bound(y_pred, q)

            # Empirical coverage: fraction of test plans where true <= upper bound
            covered = np.sum(y_true <= y_bound)
            empirical_cov = float(covered / n) if n > 0 else 1.0
            target_cov = 1.0 - d
            cov_error = empirical_cov - target_cov

            # Budget violation rates under synthetic or provided budgets
            if budgets is None:
                # Relative budget: 1.1x the point prediction or true cost
                test_budgets = y_pred * 1.15
            else:
                test_budgets = np.asarray(budgets, dtype=np.float64)

            # Point prediction admission: admit if y_pred <= B
            point_admitted = y_pred <= test_budgets
            point_violations = np.sum(point_admitted & (y_true > test_budgets))
            point_viol_rate = float(point_violations / np.sum(point_admitted)) if np.sum(point_admitted) > 0 else 0.0

            # Conformal admission: admit if y_bound <= B
            conformal_admitted = y_bound <= test_budgets
            conformal_violations = np.sum(conformal_admitted & (y_true > test_budgets))
            conformal_viol_rate = float(conformal_violations / np.sum(conformal_admitted)) if np.sum(conformal_admitted) > 0 else 0.0

            inflation = float(math.exp(q))

            results[d] = ConformalMetrics(
                delta=d,
                target_coverage=target_cov,
                empirical_coverage=empirical_cov,
                coverage_error=cov_error,
                quantile_q=q,
                point_violation_rate=point_viol_rate,
                conformal_violation_rate=conformal_viol_rate,
                mean_inflation_factor=inflation,
            )

        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "deltas": self.deltas,
            "cost_quantiles": {str(k): float(v) for k, v in self.cost_quantiles.items()},
            "latency_quantiles": {str(k): float(v) for k, v in self.latency_quantiles.items()},
            "cost_residuals": self.cost_residuals.tolist(),
            "latency_residuals": self.latency_residuals.tolist(),
            "num_calibration_samples": self.num_calibration_samples,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConformalCalibrationEngine":
        with open(path, "r") as f:
            data = json.load(f)
        engine = cls(deltas=data["deltas"])
        engine.cost_quantiles = {float(k): float(v) for k, v in data["cost_quantiles"].items()}
        engine.latency_quantiles = {float(k): float(v) for k, v in data["latency_quantiles"].items()}
        engine.cost_residuals = np.array(data["cost_residuals"])
        engine.latency_residuals = np.array(data["latency_residuals"])
        engine.num_calibration_samples = data.get("num_calibration_samples", len(engine.cost_residuals))
        return engine
