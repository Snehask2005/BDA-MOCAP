"""
MOCAP optimizer package.

Roshini (Member 3) — Core MOCAP Optimizer

Public API
----------
MOCAPSelector      — unified three-stage selection (constraints → Pareto → Lagrangian)
ConstraintsFilter  — hard budget / deadline filtering with infeasibility reasons
ParetoSelector     — Pareto-dominance filtering with tie handling
LagrangianRanker   — Lagrangian multi-objective ranking with score breakdown
SparkDefaultBaseline         — minimum-cost Spark-default comparison policy
GreedyBudgetPruningBaseline  — budget-filtered minimum-latency baseline
AblationConfig               — ablation experiment configuration dataclass
AblationVariant              — ablation variant enum
ALL_ABLATIONS                — pre-built ablation configuration registry
"""

from mocap.optimizer.ablations import (
    AblationConfig,
    AblationVariant,
    ALL_ABLATIONS,
    get_ablation,
    list_ablations,
)
from mocap.optimizer.baselines import (
    BaselineResult,
    GreedyBudgetPruningBaseline,
    SparkDefaultBaseline,
)
from mocap.optimizer.constraints import ConstraintsFilter, InfeasibilityRecord
from mocap.optimizer.lagrangian import LagrangianRanker, ScoredPlan
from mocap.optimizer.pareto import ParetoSelector
from mocap.optimizer.selector import MOCAPSelector, SelectionResult

__all__ = [
    # Core optimizer
    "MOCAPSelector",
    "SelectionResult",
    # Stages
    "ConstraintsFilter",
    "InfeasibilityRecord",
    "ParetoSelector",
    "LagrangianRanker",
    "ScoredPlan",
    # Baselines
    "SparkDefaultBaseline",
    "GreedyBudgetPruningBaseline",
    "BaselineResult",
    # Ablations
    "AblationConfig",
    "AblationVariant",
    "ALL_ABLATIONS",
    "get_ablation",
    "list_ablations",
]
