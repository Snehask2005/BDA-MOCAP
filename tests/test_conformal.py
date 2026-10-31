"""Unit tests for Conformal Calibration Engine and Chance-Constrained Bounds."""
import numpy as np
import pytest
from predictor.conformal import (
    ConformalCalibrationEngine,
    compute_log_residuals,
    compute_finite_sample_quantile,
    compute_certified_upper_bound,
)
from selector.bounds import (
    filter_chance_constrained_feasible,
    compute_plan_confidence_interval,
)


def test_compute_log_residuals():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 10.0, 60.0])
    # |log(10) - log(10)| = 0
    # |log(20) - log(10)| = log(2) ~ 0.6931
    # |log(30) - log(60)| = log(2) ~ 0.6931
    res = compute_log_residuals(y_true, y_pred)
    assert res[0] == 0.0
    assert abs(res[1] - np.log(2)) < 1e-4
    assert abs(res[2] - np.log(2)) < 1e-4


def test_finite_sample_quantile_rule():
    # n = 99 calibration residuals: 0.01, 0.02, ..., 0.99
    res = np.linspace(0.01, 0.99, 99)
    delta = 0.10  # 90% confidence
    # k = ceil((99 + 1) * (1 - 0.10)) = ceil(100 * 0.90) = 90
    q = compute_finite_sample_quantile(res, delta)
    assert abs(q - 0.90) < 1e-2


def test_certified_upper_bound():
    y_pred = np.array([10.0, 20.0])
    q = np.log(1.5)  # 50% inflation factor
    bound = compute_certified_upper_bound(y_pred, q)
    assert abs(bound[0] - 15.0) < 1e-4
    assert abs(bound[1] - 30.0) < 1e-4


def test_conformal_engine_coverage():
    rng = np.random.RandomState(42)
    # 200 calibration plans
    y_true_cal = rng.uniform(1.0, 100.0, size=200)
    noise_cal = rng.normal(0.0, 0.1, size=200)
    y_pred_cal = y_true_cal * np.exp(noise_cal)

    engine = ConformalCalibrationEngine(deltas=[0.05, 0.10])
    engine.fit(y_true_cal, y_pred_cal)

    assert 0.05 in engine.cost_quantiles
    assert 0.10 in engine.cost_quantiles
    assert engine.cost_quantiles[0.05] > engine.cost_quantiles[0.10]

    # Test set of 500 plans
    y_true_test = rng.uniform(1.0, 100.0, size=500)
    noise_test = rng.normal(0.0, 0.1, size=500)
    y_pred_test = y_true_test * np.exp(noise_test)

    eval_results = engine.evaluate_test_set(y_true_test, y_pred_test)

    # Coverage should satisfy finite-sample guarantee: empirical coverage >= 1 - delta - epsilon
    m95 = eval_results[0.05]
    assert m95.empirical_coverage >= 0.95 - 0.02
    assert abs(m95.empirical_coverage - 0.95) <= 0.05

    m90 = eval_results[0.10]
    assert m90.empirical_coverage >= 0.90 - 0.02
    assert abs(m90.empirical_coverage - 0.90) <= 0.05


def test_filter_chance_constrained_feasible():
    class DummyPlan:
        def __init__(self, plan_id, query_id, strategy):
            self.plan_id = plan_id
            self.query_id = query_id
            self.strategy = strategy

    candidates = [
        DummyPlan("P01", "Q01", "default"),
        DummyPlan("P02", "Q01", "broadcast"),
        DummyPlan("P03", "Q01", "merge"),
    ]

    cost_estimates = np.array([8.0, 12.0, 15.0])
    cost_bounds = np.array([9.5, 14.0, 18.0])
    lat_estimates = np.array([2.0, 1.0, 3.0])
    lat_bounds = np.array([2.5, 1.2, 3.8])

    budget = 10.0
    deadline = 3.0

    result = filter_chance_constrained_feasible(
        candidates=candidates,
        cost_estimates=cost_estimates,
        cost_upper_bounds=cost_bounds,
        latency_estimates=lat_estimates,
        latency_upper_bounds=lat_bounds,
        budget=budget,
        deadline=deadline,
    )

    # P01: cost_bound 9.5 <= 10.0 and lat_bound 2.5 <= 3.0 -> FEASIBLE
    # P02: cost_bound 14.0 > 10.0 -> INFEASIBLE (over budget)
    # P03: cost_bound 18.0 > 10.0 and lat_bound 3.8 > 3.0 -> INFEASIBLE
    assert len(result.feasible_plans) == 1
    assert result.feasible_plans[0].plan_id == "P01"
    assert len(result.infeasible_plans) == 2
