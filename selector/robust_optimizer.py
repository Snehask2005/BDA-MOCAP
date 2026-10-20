"""
Robust Optimizer — Stage 3 Unified Chance-Constrained Query Planner.

Roshini (Lead: Robust Optimizer & Theory) — Core Phase 2 & 3 Deliverable

Integrates:
    Stage 3a: Conformal Bounds & Interval Filter (C_hat^+ <= B, L_hat^+ <= D)
    Stage 3b: 2D Pareto Frontier Extraction (O(N log N) sort-and-scan)
    Stage 3c: Non-Inert Decision Policy (Fastest Under Budget / Scalarized)
    Fallback: Three-Mode Infeasible Engine (Zero crashes, rich diagnostics)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from selector.bounds import (
    ConformalBoundsComputer,
    FilterRejectionReason,
    IntervalFilter,
    IntervalFilterSummary,
    PlanBounds,
)
from selector.conformal import ConformalCalibrator, ConformalQuantile
from selector.decision import (
    DecisionEngine,
    DecisionOutcome,
    DecisionPolicyType,
    DecisionScore,
)
from selector.infeasible import (
    InfeasibleDiagnostics,
    InfeasibleEngine,
    InfeasibleMode,
    InfeasibleResolution,
)


@dataclass
class RobustSelectionResult:
    """Complete diagnostic output of the robust chance-constrained optimizer."""

    query_id: str
    budget: float
    deadline: Optional[float]
    confidence_delta: float
    target_confidence: float  # 1 - delta

    # Counts across stages
    candidate_count: int
    feasible_count: int
    pareto_count: int

    # Stage runtimes in milliseconds
    bounds_time_ms: float
    filter_time_ms: float
    pareto_time_ms: float
    decision_time_ms: float
    total_planning_time_ms: float

    # Outcome
    is_feasible: bool
    selected_plan: Optional[PlanBounds]
    decision_outcome: Optional[DecisionOutcome]
    infeasible_resolution: Optional[InfeasibleResolution]

    # Stage payloads
    filter_summary: Optional[IntervalFilterSummary] = None
    pareto_plans: List[PlanBounds] = field(default_factory=list)
    pareto_metadata: Dict[str, Any] = field(default_factory=dict)
    selection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "budget": self.budget,
            "deadline": self.deadline,
            "confidence_delta": self.confidence_delta,
            "candidate_count": self.candidate_count,
            "feasible_count": self.feasible_count,
            "pareto_count": self.pareto_count,
            "total_planning_time_ms": self.total_planning_time_ms,
            "is_feasible": self.is_feasible,
            "selected_plan_id": self.selected_plan.plan_id if self.selected_plan else None,
            "selection_reason": self.selection_reason,
            "pareto_plan_ids": [p.plan_id for p in self.pareto_plans],
        }


class RobustOptimizer:
    """
    Unified Chance-Constrained Robust Optimizer.
    """

    def __init__(
        self,
        cost_multiplier: float = 1.15,
        latency_multiplier: float = 1.10,
        delta: float = 0.05,
        decision_policy: DecisionPolicyType = DecisionPolicyType.FASTEST_UNDER_BUDGET,
        infeasible_mode: InfeasibleMode = InfeasibleMode.MIN_COST_OVER_BUDGET,
        scalarization_alpha: float = 0.5,
        use_conformal: bool = True,
        cost_quantile: Optional[ConformalQuantile] = None,
        latency_quantile: Optional[ConformalQuantile] = None,
    ) -> None:
        self.delta = delta
        self.confidence_level = 1.0 - delta
        self.decision_policy = decision_policy
        self.infeasible_mode = infeasible_mode
        self.use_conformal = use_conformal

        # Initialize conformal quantiles
        if cost_quantile is not None:
            self.cost_quantile = cost_quantile
        else:
            self.cost_quantile = ConformalQuantile(
                metric_name="cost",
                delta=delta,
                confidence_level=1.0 - delta,
                sample_size=100,
                quantile_index=95,
                raw_quantile_q=math.log(max(cost_multiplier, 1.0)),
                multiplier=cost_multiplier,
            )

        if latency_quantile is not None:
            self.latency_quantile = latency_quantile
        else:
            self.latency_quantile = ConformalQuantile(
                metric_name="latency",
                delta=delta,
                confidence_level=1.0 - delta,
                sample_size=100,
                quantile_index=95,
                raw_quantile_q=math.log(max(latency_multiplier, 1.0)),
                multiplier=latency_multiplier,
            )

        self.bounds_computer = ConformalBoundsComputer(
            cost_quantile=self.cost_quantile,
            latency_quantile=self.latency_quantile,
        )
        self.interval_filter = IntervalFilter(use_conformal=use_conformal)
        self.decision_engine = DecisionEngine(
            policy=decision_policy,
            alpha=scalarization_alpha,
            beta=1.0 - scalarization_alpha,
        )
        self.infeasible_engine = InfeasibleEngine(mode=infeasible_mode)

    def select(
        self,
        candidate_items: Sequence[Tuple[str, float, float, Any]],
        budget: float,
        deadline: Optional[float] = None,
        query_id: str = "QUERY_1",
    ) -> RobustSelectionResult:
        """
        Main selection pipeline.

        Args:
            candidate_items: Sequence of tuples (plan_id, point_cost, point_latency, raw_plan_obj)
            budget: Execution cost budget
            deadline: Optional latency deadline
            query_id: Unique query identifier
        """
        import math
        start_total = time.perf_counter()

        # -------------------------------------------------------------
        # Stage 3a (Part 1): Compute Conformal Bounds
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        bounded_plans: List[PlanBounds] = []
        for item in candidate_items:
            p_id, p_cost, p_lat, raw_p = item
            b = self.bounds_computer.compute_bounds(
                plan_id=p_id,
                query_id=query_id,
                point_cost=p_cost,
                point_latency=p_lat,
                raw_plan=raw_p,
            )
            bounded_plans.append(b)
        t_bounds = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Stage 3a (Part 2): Interval Filtering
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        filter_summary = self.interval_filter.filter_plans(
            plans=bounded_plans,
            budget=budget,
            deadline=deadline,
        )
        t_filter = (time.perf_counter() - t0) * 1000.0

        # Handle empty candidates or infeasible scenarios
        if not filter_summary.admitted_plans:
            infeasible_res = self.infeasible_engine.resolve(
                all_candidates=bounded_plans,
                budget=budget,
                deadline=deadline,
                query_id=query_id,
            )
            total_time = (time.perf_counter() - start_total) * 1000.0

            return RobustSelectionResult(
                query_id=query_id,
                budget=budget,
                deadline=deadline,
                confidence_delta=self.delta,
                target_confidence=self.confidence_level,
                candidate_count=len(bounded_plans),
                feasible_count=0,
                pareto_count=0,
                bounds_time_ms=t_bounds,
                filter_time_ms=t_filter,
                pareto_time_ms=0.0,
                decision_time_ms=0.0,
                total_planning_time_ms=total_time,
                is_feasible=False,
                selected_plan=infeasible_res.fallback_plan,
                decision_outcome=None,
                infeasible_resolution=infeasible_res,
                filter_summary=filter_summary,
                pareto_plans=[],
                pareto_metadata={},
                selection_reason=infeasible_res.message,
            )

        # -------------------------------------------------------------
        # Stage 3b: 2D Pareto Frontier Extraction
        # -------------------------------------------------------------
        from selector.pareto import ParetoExtractor

        t0 = time.perf_counter()
        pareto_extractor = ParetoExtractor(
            use_conformal_cost=self.use_conformal,
            use_conformal_latency=False,
        )
        pareto_res = pareto_extractor.extract_frontier(filter_summary.admitted_plans)
        t_pareto = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Stage 3c: Decision Policy Selection
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        decision_outcome = self.decision_engine.select(
            frontier=pareto_res.frontier,
            budget=budget,
            deadline=deadline,
        )
        t_decision = (time.perf_counter() - t0) * 1000.0

        total_time = (time.perf_counter() - start_total) * 1000.0

        return RobustSelectionResult(
            query_id=query_id,
            budget=budget,
            deadline=deadline,
            confidence_delta=self.delta,
            target_confidence=self.confidence_level,
            candidate_count=len(bounded_plans),
            feasible_count=filter_summary.admitted_count,
            pareto_count=pareto_res.frontier_count,
            bounds_time_ms=t_bounds,
            filter_time_ms=t_filter,
            pareto_time_ms=t_pareto,
            decision_time_ms=t_decision,
            total_planning_time_ms=total_time,
            is_feasible=True,
            selected_plan=decision_outcome.selected_plan,
            decision_outcome=decision_outcome,
            infeasible_resolution=None,
            filter_summary=filter_summary,
            pareto_plans=[p.plan for p in pareto_res.frontier],
            pareto_metadata=pareto_res.to_dict(),
            selection_reason=decision_outcome.selection_reason,
        )


import math  # Ensure import is present in module
