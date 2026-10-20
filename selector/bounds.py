"""
Conformal Bounds and Chance-Constrained Interval Filter.

Roshini (Lead: Robust Optimizer & Theory) — Phase 2 Deliverable

Maps point predictions (C_hat, L_hat) into certified conformal upper bounds:
    C_hat^+ = C_hat * exp(q_C)
    L_hat^+ = L_hat * exp(q_L)

And enforces the chance-constrained budget admissibility test:
    Admit candidate plan p_i  <=>  C_hat_i^+ <= B  (and L_hat_i^+ <= D)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from selector.conformal import ConformalQuantile


class FilterRejectionReason(str, Enum):
    """Reason why a candidate plan was rejected by the interval filter."""

    PASSED = "PASSED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    BOTH_EXCEEDED = "BOTH_EXCEEDED"
    MISSING_METRICS = "MISSING_METRICS"
    INVALID_VALUES = "INVALID_VALUES"


@dataclass
class PlanBounds:
    """Certified confidence bounds for a single candidate plan."""

    plan_id: str
    query_id: str
    point_cost: float
    point_latency: float
    conformal_cost_upper: float  # C_hat^+
    conformal_latency_upper: float  # L_hat^+
    conformal_cost_lower: float  # C_hat^-
    conformal_latency_lower: float  # L_hat^-
    delta_cost: float
    delta_latency: float
    cost_multiplier: float
    latency_multiplier: float
    raw_plan: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_cost_feasible(self, budget: float, use_conformal: bool = True) -> bool:
        test_cost = self.conformal_cost_upper if use_conformal else self.point_cost
        return test_cost <= budget

    def is_latency_feasible(
        self, deadline: Optional[float], use_conformal: bool = True
    ) -> bool:
        if deadline is None:
            return True
        test_lat = self.conformal_latency_upper if use_conformal else self.point_latency
        return test_lat <= deadline

    def is_feasible(
        self, budget: float, deadline: Optional[float] = None, use_conformal: bool = True
    ) -> bool:
        return self.is_cost_feasible(budget, use_conformal) and self.is_latency_feasible(
            deadline, use_conformal
        )

    def cost_margin(self, budget: float, use_conformal: bool = True) -> float:
        """Surplus/deficit: positive means under budget (slack), negative means overrun."""
        test_cost = self.conformal_cost_upper if use_conformal else self.point_cost
        return budget - test_cost

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_plan", None)
        return d


@dataclass
class PlanFilterResult:
    """Outcome of filtering a candidate through the interval filter."""

    plan: PlanBounds
    admitted: bool
    rejection_reason: FilterRejectionReason
    budget_slack: float  # budget - cost_test (positive = surplus, negative = deficit)
    deadline_slack: Optional[float] = None  # deadline - lat_test


@dataclass
class IntervalFilterSummary:
    """Aggregate diagnostics of the interval filtering pass."""

    total_candidates: int
    admitted_count: int
    rejected_count: int
    budget: float
    deadline: Optional[float]
    use_conformal: bool
    min_cost_upper: float
    min_point_cost: float
    min_latency_upper: float
    min_point_latency: float
    admitted_plans: List[PlanBounds] = field(default_factory=list)
    rejection_records: List[PlanFilterResult] = field(default_factory=list)


class ConformalBoundsComputer:
    """Converts point predictions into certified confidence intervals."""

    def __init__(
        self,
        cost_quantile: ConformalQuantile,
        latency_quantile: Optional[ConformalQuantile] = None,
    ) -> None:
        self.cost_quantile = cost_quantile
        self.latency_quantile = latency_quantile or ConformalQuantile(
            metric_name="latency",
            delta=cost_quantile.delta,
            confidence_level=cost_quantile.confidence_level,
            sample_size=cost_quantile.sample_size,
            quantile_index=cost_quantile.quantile_index,
            raw_quantile_q=cost_quantile.raw_quantile_q,
            multiplier=cost_quantile.multiplier,
        )

    def compute_bounds(
        self,
        plan_id: str,
        query_id: str,
        point_cost: float,
        point_latency: float,
        raw_plan: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanBounds:
        c_p = max(0.0, float(point_cost))
        l_p = max(0.0, float(point_latency))

        c_upper = c_p * self.cost_quantile.multiplier
        l_upper = l_p * self.latency_quantile.multiplier

        c_lower = c_p / self.cost_quantile.multiplier if self.cost_quantile.multiplier > 0 else 0.0
        l_lower = l_p / self.latency_quantile.multiplier if self.latency_quantile.multiplier > 0 else 0.0

        return PlanBounds(
            plan_id=plan_id,
            query_id=query_id,
            point_cost=c_p,
            point_latency=l_p,
            conformal_cost_upper=c_upper,
            conformal_latency_upper=l_upper,
            conformal_cost_lower=c_lower,
            conformal_latency_lower=l_lower,
            delta_cost=self.cost_quantile.delta,
            delta_latency=self.latency_quantile.delta,
            cost_multiplier=self.cost_quantile.multiplier,
            latency_multiplier=self.latency_quantile.multiplier,
            raw_plan=raw_plan,
            metadata=metadata or {},
        )


class IntervalFilter:
    """
    Chance-constrained admissibility filter.

    Admits plan p_i iff C_hat_i^+ <= B and L_hat_i^+ <= D.
    Provides diagnostic reasons and slack margins for all rejected candidates.
    """

    def __init__(self, use_conformal: bool = True) -> None:
        self.use_conformal = use_conformal

    def filter_plans(
        self,
        plans: Sequence[PlanBounds],
        budget: float,
        deadline: Optional[float] = None,
    ) -> IntervalFilterSummary:
        if not plans:
            return IntervalFilterSummary(
                total_candidates=0,
                admitted_count=0,
                rejected_count=0,
                budget=budget,
                deadline=deadline,
                use_conformal=self.use_conformal,
                min_cost_upper=float("inf"),
                min_point_cost=float("inf"),
                min_latency_upper=float("inf"),
                min_point_latency=float("inf"),
            )

        admitted: List[PlanBounds] = []
        rejections: List[PlanFilterResult] = []

        for p in plans:
            cost_val = p.conformal_cost_upper if self.use_conformal else p.point_cost
            lat_val = p.conformal_latency_upper if self.use_conformal else p.point_latency

            cost_ok = (cost_val <= budget)
            lat_ok = (deadline is None) or (lat_val <= deadline)

            b_slack = budget - cost_val
            d_slack = (deadline - lat_val) if deadline is not None else None

            if cost_ok and lat_ok:
                admitted.append(p)
                rejections.append(
                    PlanFilterResult(
                        plan=p,
                        admitted=True,
                        rejection_reason=FilterRejectionReason.PASSED,
                        budget_slack=b_slack,
                        deadline_slack=d_slack,
                    )
                )
            else:
                if (not cost_ok) and (not lat_ok):
                    reason = FilterRejectionReason.BOTH_EXCEEDED
                elif not cost_ok:
                    reason = FilterRejectionReason.BUDGET_EXCEEDED
                else:
                    reason = FilterRejectionReason.DEADLINE_EXCEEDED

                rejections.append(
                    PlanFilterResult(
                        plan=p,
                        admitted=False,
                        rejection_reason=reason,
                        budget_slack=b_slack,
                        deadline_slack=d_slack,
                    )
                )

        min_c_upper = min(p.conformal_cost_upper for p in plans)
        min_p_cost = min(p.point_cost for p in plans)
        min_l_upper = min(p.conformal_latency_upper for p in plans)
        min_p_lat = min(p.point_latency for p in plans)

        return IntervalFilterSummary(
            total_candidates=len(plans),
            admitted_count=len(admitted),
            rejected_count=len(plans) - len(admitted),
            budget=budget,
            deadline=deadline,
            use_conformal=self.use_conformal,
            min_cost_upper=min_c_upper,
            min_point_cost=min_p_cost,
            min_latency_upper=min_l_upper,
            min_point_latency=min_p_lat,
            admitted_plans=admitted,
            rejection_records=rejections,
        )
