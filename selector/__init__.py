"""
MOCAP Selector Package — Robust Chance-Constrained Query Optimization.

Roshini (Lead: Robust Optimizer & Theory) — Phases 2 & 3 Deliverable

Public API:
-----------
Conformal Calibration & Bounds:
    ConformalCalibrator         — Split conformal prediction residual engine
    ConformalQuantile           — Conformal quantile and multiplier descriptor
    CalibrationSample           — Observation record for calibration
    PlanBounds                  — Certified upper/lower bounds per candidate
    ConformalBoundsComputer     — Factory computing PlanBounds from point predictions
    IntervalFilter              — Chance-constrained admissibility filter (C_hat^+ <= B)
    IntervalFilterSummary       — Diagnostics of the interval filtering pass
    FilterRejectionReason       — Rejection classification enum

Pareto Frontier:
    ParetoExtractor             — High-performance O(N log N) 2D Pareto extractor
    ParetoPlan                  — Non-dominated candidate with confidence bounds
    ParetoFrontierResult        — Complete Pareto frontier output

Decision Policies:
    DecisionEngine              — Decision evaluator and candidate ranker
    DecisionPolicyType          — Supported policies (Fastest under budget, Scalarized, etc.)
    DecisionOutcome             — Winning plan selection and rankings
    DecisionScore               — Granular score breakdown per candidate

Infeasible Fallbacks:
    InfeasibleEngine            — Three-mode fallback engine (zero crashes)
    InfeasibleMode              — Fallback mode enum (Min Cost, Min Budget, Refusal)
    InfeasibleResolution        — Strongly-typed fallback resolution
    InfeasibleDiagnostics       — Comprehensive audit of infeasibility causes

Unified Optimizer:
    RobustOptimizer             — End-to-end Stage 3 chance-constrained planner
    RobustSelectionResult       — Full selection and telemetry output
"""

from selector.bounds import (
    ConformalBoundsComputer,
    FilterRejectionReason,
    IntervalFilter,
    IntervalFilterSummary,
    PlanBounds,
    PlanFilterResult,
)
from selector.conformal import (
    CalibrationSample,
    ConformalCalibrator,
    ConformalQuantile,
)
from selector.decision import (
    DecisionEngine,
    DecisionOutcome,
    DecisionPolicyType,
    DecisionScore,
)
from selector.infeasible import (
    CandidateInfeasibilityDetail,
    InfeasibleDiagnostics,
    InfeasibleEngine,
    InfeasibleMode,
    InfeasibleResolution,
)
from selector.pareto import (
    ParetoExtractor,
    ParetoFrontierResult,
    ParetoPlan,
)
from selector.robust_optimizer import (
    RobustOptimizer,
    RobustSelectionResult,
)

__all__ = [
    # Calibration & Bounds
    "ConformalCalibrator",
    "ConformalQuantile",
    "CalibrationSample",
    "PlanBounds",
    "PlanFilterResult",
    "ConformalBoundsComputer",
    "IntervalFilter",
    "IntervalFilterSummary",
    "FilterRejectionReason",
    # Pareto
    "ParetoExtractor",
    "ParetoPlan",
    "ParetoFrontierResult",
    # Decision
    "DecisionEngine",
    "DecisionPolicyType",
    "DecisionOutcome",
    "DecisionScore",
    # Infeasible
    "InfeasibleEngine",
    "InfeasibleMode",
    "InfeasibleResolution",
    "InfeasibleDiagnostics",
    "CandidateInfeasibilityDetail",
    # Unified Optimizer
    "RobustOptimizer",
    "RobustSelectionResult",
]


# Backward compatibility exports for the original Phase 2 API.
from selector.bounds import (
    ConfidenceInterval,
    EvaluatedPlan,
    BoundFilterResult,
    compute_plan_confidence_interval,
    filter_chance_constrained_feasible,
)
from selector.conformal import (
    ConformalCalibrationEngine,
    ConformalMetrics,
    compute_log_residuals,
    compute_finite_sample_quantile,
    compute_certified_upper_bound,
)

__all__.extend([
    "ConfidenceInterval",
    "EvaluatedPlan",
    "BoundFilterResult",
    "compute_plan_confidence_interval",
    "filter_chance_constrained_feasible",
    "ConformalCalibrationEngine",
    "ConformalMetrics",
    "compute_log_residuals",
    "compute_finite_sample_quantile",
    "compute_certified_upper_bound",
])
