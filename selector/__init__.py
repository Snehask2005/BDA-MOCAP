"""
MOCAP Robust Optimizer & Chance-Constrained Selector Module.
Owned by Roshini (Optimizer & Theory Lead), in collaboration with Hridhika (ML/Conformal).
"""

from selector.bounds import (
    BoundFilterResult,
    filter_chance_constrained_feasible,
    compute_plan_confidence_interval,
)
from selector.conformal import ConformalCalibrationEngine

__all__ = [
    "BoundFilterResult",
    "filter_chance_constrained_feasible",
    "compute_plan_confidence_interval",
    "ConformalCalibrationEngine",
]
