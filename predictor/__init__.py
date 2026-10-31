"""
MOCAP Learned Cost Predictor & Conformal Calibration Module.
Owned by Hridhika (Lead Learned Predictor).

Key Components:
- cbo_stats: Spark CBO statistics configuration, EXPLAIN COST parsing, and validation.
- features: Catalyst physical plan feature extractor (scans, shuffles, exchanges, join types).
- models: Log-Ridge, LightGBM, and LambdaRank pairwise ranking models + Grouped Query CV.
- conformal: Finite-sample split conformal calibration engine for certified cost upper bounds.
"""

from predictor.cbo_stats import (
    CBOCostStats,
    parse_explain_cost,
    enable_cbo,
    enforce_hint_testing_mode,
    analyze_table,
    verify_cbo_active,
)
from predictor.features import (
    PlanFeatures,
    extract_plan_features,
    extract_features_from_plan_text,
)
from predictor.models import (
    LogRidgePredictor,
    LightGBMPredictor,
    LambdaRankPredictor,
    AnalyticalBaselineModel,
    QueryGroupedKFold,
    spearman_rank_correlation,
    kendall_tau,
    top1_hit_rate,
)
from predictor.conformal import (
    ConformalCalibrationEngine,
    ConformalMetrics,
    compute_log_residuals,
    compute_finite_sample_quantile,
    compute_certified_upper_bound,
)

__all__ = [
    "CBOCostStats",
    "parse_explain_cost",
    "enable_cbo",
    "enforce_hint_testing_mode",
    "analyze_table",
    "verify_cbo_active",
    "PlanFeatures",
    "extract_plan_features",
    "extract_features_from_plan_text",
    "LogRidgePredictor",
    "LightGBMPredictor",
    "LambdaRankPredictor",
    "AnalyticalBaselineModel",
    "QueryGroupedKFold",
    "spearman_rank_correlation",
    "kendall_tau",
    "top1_hit_rate",
    "ConformalCalibrationEngine",
    "ConformalMetrics",
    "compute_log_residuals",
    "compute_finite_sample_quantile",
    "compute_certified_upper_bound",
]
