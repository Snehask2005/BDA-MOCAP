r"""
Chance-Constrained Plan Bounds & Feasibility Filter.
Part of Phase 2 & Phase 3 Integration (Hridhika & Roshini).

Filters candidate plans via certified conformal upper bounds:
  Admit plan p_i  <===>  \hat{C}^+_i <= B  and  \hat{L}^+_i <= D
Guarantees Pr(C_i <= B) >= 1 - \delta under exchangeability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConfidenceInterval:
    """Certified interval [lower, upper] for cost or latency."""
    point_estimate: float
    upper_bound: float
    lower_bound: float
    delta: float
    quantile_q: float


@dataclass
class EvaluatedPlan:
    """Candidate plan bundled with certified conformal bounds."""
    plan_id: str
    query_id: str
    strategy: str
    estimated_cost: float
    certified_cost_bound: float
    estimated_latency: float
    certified_latency_bound: float
    is_cost_feasible: bool = True
    is_latency_feasible: bool = True
    cost_headroom: float = 0.0
    latency_headroom: float = 0.0
    raw_plan: Any = None


@dataclass
class BoundFilterResult:
    """Partitioned results of chance-constrained filtering."""
    feasible_plans: List[EvaluatedPlan]
    infeasible_plans: List[EvaluatedPlan]
    cost_budget: float
    latency_deadline: Optional[float]
    delta: float
    all_evaluated: List[EvaluatedPlan] = field(default_factory=list)

    @property
    def has_feasible_plan(self) -> bool:
        return len(self.feasible_plans) > 0


def compute_plan_confidence_interval(
    point_estimate: float,
    quantile_q: float,
    delta: float = 0.05,
) -> ConfidenceInterval:
    """Compute symmetric/log-space confidence interval for an estimate."""
    factor = float(np.exp(quantile_q))
    upper = point_estimate * factor
    lower = point_estimate / factor
    return ConfidenceInterval(
        point_estimate=point_estimate,
        upper_bound=upper,
        lower_bound=lower,
        delta=delta,
        quantile_q=quantile_q,
    )


def filter_chance_constrained_feasible(
    candidates: List[Any],
    cost_estimates: np.ndarray,
    cost_upper_bounds: np.ndarray,
    latency_estimates: np.ndarray,
    latency_upper_bounds: np.ndarray,
    budget: float,
    deadline: Optional[float] = None,
    delta: float = 0.05,
) -> BoundFilterResult:
    r"""
    Filter candidate plans using conformal bounds \hat{C}^+ <= B and \hat{L}^+ <= D.
    Replaces the mathematically inert Lagrangian penalty term.
    """
    feasible: List[EvaluatedPlan] = []
    infeasible: List[EvaluatedPlan] = []
    all_evaluated: List[EvaluatedPlan] = []

    for i, cand in enumerate(candidates):
        plan_id = getattr(cand, "plan_id", f"P{i+1:02d}")
        query_id = getattr(cand, "query_id", "Q01")
        strategy = getattr(cand, "strategy", "default")

        c_est = float(cost_estimates[i])
        c_bound = float(cost_upper_bounds[i])
        l_est = float(latency_estimates[i])
        l_bound = float(latency_upper_bounds[i])

        cost_feas = c_bound <= budget
        lat_feas = (deadline is None) or (l_bound <= deadline)

        evaluated = EvaluatedPlan(
            plan_id=plan_id,
            query_id=query_id,
            strategy=strategy,
            estimated_cost=c_est,
            certified_cost_bound=c_bound,
            estimated_latency=l_est,
            certified_latency_bound=l_bound,
            is_cost_feasible=cost_feas,
            is_latency_feasible=lat_feas,
            cost_headroom=budget - c_bound,
            latency_headroom=(deadline - l_bound) if deadline is not None else float("inf"),
            raw_plan=cand,
        )

        all_evaluated.append(evaluated)
        if cost_feas and lat_feas:
            feasible.append(evaluated)
        else:
            infeasible.append(evaluated)

    return BoundFilterResult(
        feasible_plans=feasible,
        infeasible_plans=infeasible,
        cost_budget=budget,
        latency_deadline=deadline,
        delta=delta,
        all_evaluated=all_evaluated,
    )
