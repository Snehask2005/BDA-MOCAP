"""Unit tests for ML models, grouped query cross-validation, and rank metrics."""
import numpy as np
import pytest
from predictor.models import (
    AnalyticalBaselineModel,
    LogRidgePredictor,
    LightGBMPredictor,
    LambdaRankPredictor,
    QueryGroupedKFold,
    spearman_rank_correlation,
    kendall_tau,
    top1_hit_rate,
    distinct_values_count,
)


def test_analytical_baseline_value_collapse():
    # 5 plans of the same query with identical scan bytes but varying joins/shuffles
    X = np.array([
        [1e8, 1e7, 1e7, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1e8, 2e7, 2e7, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1e8, 5e7, 5e7, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1e8, 8e7, 8e7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1e8, 9e7, 9e7, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    baseline = AnalyticalBaselineModel()
    preds = baseline.predict(X)
    # Proves legacy value collapse (all 5 evaluate to same cost)
    assert len(np.unique(preds)) == 1


def test_log_ridge_predictor():
    rng = np.random.RandomState(42)
    X = rng.uniform(10, 1000, size=(50, 17))
    # Synthetic ground truth
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 1.2 * X[:, 4] + rng.normal(0, 1, size=50)
    y = np.maximum(y, 1.0)

    model = LogRidgePredictor(alpha=0.1)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 50
    assert np.all(preds > 0)
    rho = spearman_rank_correlation(y, preds)
    assert rho > 0.85


def test_lightgbm_predictor():
    rng = np.random.RandomState(42)
    X = rng.uniform(1, 100, size=(60, 17))
    y = 2.0 * np.log1p(X[:, 0]) + 0.5 * (X[:, 1] ** 0.5) + rng.normal(0, 0.1, size=60)
    y = np.maximum(y, 0.1)

    model = LightGBMPredictor(n_estimators=40, learning_rate=0.08, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 60
    assert np.all(preds > 0)
    rho = spearman_rank_correlation(y, preds)
    assert rho > 0.80


def test_query_grouped_kfold_zero_leakage():
    # 20 plans spanning 4 distinct queries
    queries = ["Q01"] * 5 + ["Q02"] * 5 + ["Q03"] * 5 + ["Q04"] * 5
    cv = QueryGroupedKFold(n_splits=2, shuffle=True, seed=42)

    for train_idx, test_idx in cv.split(queries):
        train_q = set([queries[i] for i in train_idx])
        test_q = set([queries[i] for i in test_idx])
        # Crucial guardrail: zero intersection between train and test queries!
        assert len(train_q.intersection(test_q)) == 0


def test_rank_correlation_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

    assert spearman_rank_correlation(y_true, y_pred) == pytest.approx(1.0)
    assert kendall_tau(y_true, y_pred) == pytest.approx(1.0)

    # Inverted ranking
    y_inv = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert spearman_rank_correlation(y_true, y_inv) == pytest.approx(-1.0)
    assert kendall_tau(y_true, y_inv) == pytest.approx(-1.0)


def test_top1_hit_rate():
    queries = ["Q01", "Q01", "Q01", "Q02", "Q02", "Q02"]
    # Q01 best is plan 0; Q02 best is plan 4
    y_true = np.array([1.0, 2.0, 3.0, 5.0, 2.0, 4.0])
    y_pred = np.array([1.2, 1.9, 2.8, 4.8, 2.1, 3.9])

    rate = top1_hit_rate(queries, y_true, y_pred)
    assert rate == 1.0
