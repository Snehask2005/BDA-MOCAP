"""
Re-export of Conformal Calibration Engine for the Selector/Optimizer module.
Allows Stage 3 Robust Optimizer to access conformal quantiles directly.
"""
from predictor.conformal import (
    ConformalCalibrationEngine,
    ConformalMetrics,
    compute_log_residuals,
    compute_finite_sample_quantile,
    compute_certified_upper_bound,
)

__all__ = [
    "ConformalCalibrationEngine",
    "ConformalMetrics",
    "compute_log_residuals",
    "compute_finite_sample_quantile",
    "compute_certified_upper_bound",
]
