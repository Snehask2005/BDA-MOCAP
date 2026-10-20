"""
Three-Mode Infeasible Fallback Engine.

Roshini (Lead: Robust Optimizer & Theory) — Phase 3 Deliverable

When no candidate plan satisfies the chance-constrained budget:
    C_hat_i^+ <= B  (or deadline L_hat_i^+ <= D)
the optimizer invokes this engine.

Guarantees:
    1. Zero unhandled runtime exceptions, null pointers, or crashes.
    2. Informative diagnostic records specifying exact cost deficits.
    3. Three distinct, caller-selectable operational policies:
        Mode 1: MIN_COST_OVER_BUDGET (Best-effort minimal budget overrun)
        Mode 2: MIN_BUDGET_REPORTING (Calculates minimum required budget)
        Mode 3: FORMAL_REFUSAL       (Strict rejection with diagnostic payload)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from selector.bounds import FilterRejectionReason, PlanBounds, PlanFilterResult


class InfeasibleMode(str, Enum):
    """The three operating modes when all candidates violate constraints."""

    MIN_COST_OVER_BUDGET = "MIN_COST_OVER_BUDGET"
    MIN_BUDGET_REPORTING = "MIN_BUDGET_REPORTING"
    FORMAL_REFUSAL = "FORMAL_REFUSAL"


@dataclass
class CandidateInfeasibilityDetail:
    """Detailed reason why a specific candidate plan was infeasible."""

    plan_id: str
    point_cost: float
    point_latency: float
    conformal_cost_upper: float
    conformal_latency_upper: float
    budget_overrun: float  # conformal_cost_upper - budget
    budget_deficit_percent: float
    deadline_overrun: Optional[float]
    primary_failure: FilterRejectionReason


@dataclass
class InfeasibleDiagnostics:
    """Complete diagnostic summary of the constraint violation across all candidates."""

    query_id: str
    target_budget: float
    target_deadline: Optional[float]
    total_candidates: int
    mode_invoked: InfeasibleMode

    # Bound extremes
    min_conformal_cost: float
    max_conformal_cost: float
    min_point_cost: float
    min_conformal_latency: float
    min_point_latency: float

    # Required budgets
    min_budget_required: float  # Minimum budget to admit at least ONE plan
    min_budget_for_fastest: float  # Budget required to admit the fastest plan
    recommended_budget_with_slack: float  # 5% buffer over min_budget_required
    deficit_to_admit_any: float  # min_conformal_cost - target_budget

    # Per-candidate audit
    candidate_details: List[CandidateInfeasibilityDetail] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode_invoked"] = self.mode_invoked.value
        return d


@dataclass
class InfeasibleResolution:
    """The structured return from the InfeasibleEngine."""

    mode: InfeasibleMode
    is_refusal: bool
    fallback_plan: Optional[PlanBounds]
    overrun_amount: float
    diagnostics: InfeasibleDiagnostics
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "is_refusal": self.is_refusal,
            "fallback_plan_id": self.fallback_plan.plan_id if self.fallback_plan else None,
            "overrun_amount": self.overrun_amount,
            "message": self.message,
            "diagnostics": self.diagnostics.to_dict(),
        }


class InfeasibleEngine:
    """
    Handles infeasible planning scenarios across the three defined modes.
    """

    def __init__(
        self,
        mode: InfeasibleMode = InfeasibleMode.MIN_COST_OVER_BUDGET,
        buffer_slack_percent: float = 0.05,
    ) -> None:
        self.mode = mode
        self.buffer_slack_percent = buffer_slack_percent

    def analyze(
        self,
        all_candidates: Sequence[PlanBounds],
        budget: float,
        deadline: Optional[float] = None,
        query_id: str = "UNKNOWN",
    ) -> InfeasibleDiagnostics:
        """Analyze candidates to produce rich diagnostic metrics."""
        if not all_candidates:
            return InfeasibleDiagnostics(
                query_id=query_id,
                target_budget=budget,
                target_deadline=deadline,
                total_candidates=0,
                mode_invoked=self.mode,
                min_conformal_cost=float("inf"),
                max_conformal_cost=float("inf"),
                min_point_cost=float("inf"),
                min_conformal_latency=float("inf"),
                min_point_latency=float("inf"),
                min_budget_required=float("inf"),
                min_budget_for_fastest=float("inf"),
                recommended_budget_with_slack=float("inf"),
                deficit_to_admit_any=float("inf"),
                candidate_details=[],
            )

        details: List[CandidateInfeasibilityDetail] = []
        for p in all_candidates:
            c_up = p.conformal_cost_upper
            l_up = p.conformal_latency_upper
            b_overrun = max(0.0, c_up - budget)
            deficit_pct = (b_overrun / budget * 100.0) if budget > 0 else 0.0
            d_overrun = max(0.0, l_up - deadline) if deadline is not None else None

            if b_overrun > 0 and (d_overrun is not None and d_overrun > 0):
                prim = FilterRejectionReason.BOTH_EXCEEDED
            elif b_overrun > 0:
                prim = FilterRejectionReason.BUDGET_EXCEEDED
            elif d_overrun is not None and d_overrun > 0:
                prim = FilterRejectionReason.DEADLINE_EXCEEDED
            else:
                prim = FilterRejectionReason.PASSED

            details.append(
                CandidateInfeasibilityDetail(
                    plan_id=p.plan_id,
                    point_cost=p.point_cost,
                    point_latency=p.point_latency,
                    conformal_cost_upper=c_up,
                    conformal_latency_upper=l_up,
                    budget_overrun=b_overrun,
                    budget_deficit_percent=deficit_pct,
                    deadline_overrun=d_overrun,
                    primary_failure=prim,
                )
            )

        min_c_up = min(p.conformal_cost_upper for p in all_candidates)
        max_c_up = max(p.conformal_cost_upper for p in all_candidates)
        min_p_cost = min(p.point_cost for p in all_candidates)
        min_l_up = min(p.conformal_latency_upper for p in all_candidates)
        min_p_lat = min(p.point_latency for p in all_candidates)

        fastest_plan = min(all_candidates, key=lambda p: p.point_latency)
        budget_for_fastest = fastest_plan.conformal_cost_upper

        min_req = min_c_up
        rec_budget = min_req * (1.0 + self.buffer_slack_percent)
        deficit = max(0.0, min_req - budget)

        return InfeasibleDiagnostics(
            query_id=query_id,
            target_budget=budget,
            target_deadline=deadline,
            total_candidates=len(all_candidates),
            mode_invoked=self.mode,
            min_conformal_cost=min_c_up,
            max_conformal_cost=max_c_up,
            min_point_cost=min_p_cost,
            min_conformal_latency=min_l_up,
            min_point_latency=min_p_lat,
            min_budget_required=min_req,
            min_budget_for_fastest=budget_for_fastest,
            recommended_budget_with_slack=rec_budget,
            deficit_to_admit_any=deficit,
            candidate_details=details,
        )

    def resolve(
        self,
        all_candidates: Sequence[PlanBounds],
        budget: float,
        deadline: Optional[float] = None,
        query_id: str = "UNKNOWN",
        mode_override: Optional[InfeasibleMode] = None,
    ) -> InfeasibleResolution:
        active_mode = mode_override or self.mode
        diag = self.analyze(all_candidates, budget, deadline, query_id)
        diag.mode_invoked = active_mode

        if not all_candidates:
            return InfeasibleResolution(
                mode=active_mode,
                is_refusal=True,
                fallback_plan=None,
                overrun_amount=0.0,
                diagnostics=diag,
                message="Infeasible: Zero candidates provided to optimizer.",
            )

        # -----------------------------------------------------------------
        # Mode 1: MIN_COST_OVER_BUDGET
        # -----------------------------------------------------------------
        if active_mode == InfeasibleMode.MIN_COST_OVER_BUDGET:
            # Select candidate with minimal conformal cost upper bound
            best_plan = min(
                all_candidates,
                key=lambda p: (p.conformal_cost_upper, p.point_latency, p.plan_id),
            )
            overrun = max(0.0, best_plan.conformal_cost_upper - budget)
            msg = (
                f"Infeasible Fallback [MIN_COST_OVER_BUDGET]: Selected plan '{best_plan.plan_id}' "
                f"with minimum certified cost C_hat^+={best_plan.conformal_cost_upper:.6f} "
                f"(overrun of {overrun:.6f} over budget {budget:.6f})."
            )
            return InfeasibleResolution(
                mode=active_mode,
                is_refusal=False,
                fallback_plan=best_plan,
                overrun_amount=overrun,
                diagnostics=diag,
                message=msg,
            )

        # -----------------------------------------------------------------
        # Mode 2: MIN_BUDGET_REPORTING
        # -----------------------------------------------------------------
        elif active_mode == InfeasibleMode.MIN_BUDGET_REPORTING:
            msg = (
                f"Infeasible Fallback [MIN_BUDGET_REPORTING]: No plan satisfies budget B={budget:.6f}. "
                f"Minimum budget required to admit any plan is {diag.min_budget_required:.6f} "
                f"(deficit of {diag.deficit_to_admit_any:.6f}). "
                f"Recommended budget with safety buffer: {diag.recommended_budget_with_slack:.6f}. "
                f"Budget required for fastest plan: {diag.min_budget_for_fastest:.6f}."
            )
            return InfeasibleResolution(
                mode=active_mode,
                is_refusal=True,
                fallback_plan=None,
                overrun_amount=diag.deficit_to_admit_any,
                diagnostics=diag,
                message=msg,
            )

        # -----------------------------------------------------------------
        # Mode 3: FORMAL_REFUSAL
        # -----------------------------------------------------------------
        elif active_mode == InfeasibleMode.FORMAL_REFUSAL:
            msg = (
                f"Infeasible Formal Refusal: Query '{query_id}' rejected. "
                f"Zero of {len(all_candidates)} plans satisfy chance-constrained budget B={budget:.6f}. "
                f"Cheapest certified plan requires {diag.min_conformal_cost:.6f} "
                f"(deficit: {diag.deficit_to_admit_any / budget * 100.0:.2f}%)."
            )
            return InfeasibleResolution(
                mode=active_mode,
                is_refusal=True,
                fallback_plan=None,
                overrun_amount=diag.deficit_to_admit_any,
                diagnostics=diag,
                message=msg,
            )

        else:
            raise ValueError(f"Unknown InfeasibleMode: {active_mode}")
