"""
Hard-constraint filtering for the MOCAP optimizer.

Roshini (Member 3) — Core MOCAP Optimizer

Filters candidate plans before Pareto/Lagrangian ranking.
Only plans satisfying:
    estimated_cost  <= budget    (always enforced)
    estimated_latency <= deadline (enforced when deadline is provided)
are passed to the multi-objective ranking stage.

An infeasibility reason is attached to every rejected plan so that
experiment logs can distinguish *why* a plan was rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class InfeasibilityRecord:
    """Records why a candidate plan was rejected."""

    plan_id: str
    estimated_cost: float
    estimated_latency: Optional[float]
    budget: float
    deadline: Optional[float]
    reason: str  # "budget_exceeded" | "deadline_exceeded" | "both_exceeded"


class ConstraintsFilter:
    """
    Enforces hard constraints before Pareto / Lagrangian ranking.

    Rule 1 (always applied):
        estimated_cost <= budget

    Rule 2 (applied when deadline is not None):
        estimated_latency <= deadline

    Plans failing either rule are excluded from the feasible set and
    recorded in ``infeasibility_log`` for diagnostics and paper tables.
    """

    def __init__(
        self,
        budget: float,
        deadline: Optional[float] = None,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be strictly positive")
        if deadline is not None and deadline <= 0:
            raise ValueError("deadline must be strictly positive when provided")

        self.budget = budget
        self.deadline = deadline

        # Populated by filter_feasible_plans for diagnostics.
        self.infeasibility_log: List[InfeasibilityRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_feasible_plans(
        self,
        candidate_plans,
        plan_metrics_map: Dict,
    ) -> List[Tuple]:
        """
        Return ``(plan, metrics)`` pairs whose estimated cost and latency
        satisfy the configured hard constraints.

        Also populates ``self.infeasibility_log`` with a record for every
        rejected plan, including the specific violation reason.

        Parameters
        ----------
        candidate_plans:
            Iterable of CandidatePlan objects.
        plan_metrics_map:
            Mapping from plan_id -> PlanMetrics.

        Returns
        -------
        List of ``(plan, metrics)`` tuples that are feasible.
        """

        self.infeasibility_log.clear()

        feasible: List[Tuple] = []

        for plan in candidate_plans:
            metrics = plan_metrics_map.get(plan.plan_id)

            if metrics is None:
                # No prediction available — conservatively reject.
                self.infeasibility_log.append(
                    InfeasibilityRecord(
                        plan_id=plan.plan_id,
                        estimated_cost=float("nan"),
                        estimated_latency=None,
                        budget=self.budget,
                        deadline=self.deadline,
                        reason="no_metrics",
                    )
                )
                continue

            cost = float(metrics.estimated_cost)
            latency = (
                float(metrics.estimated_latency)
                if metrics.estimated_latency is not None
                else None
            )

            cost_ok = cost <= self.budget
            latency_ok = (
                True
                if (self.deadline is None or latency is None)
                else latency <= self.deadline
            )

            if cost_ok and latency_ok:
                feasible.append((plan, metrics))
            else:
                # Determine specific rejection reason.
                if not cost_ok and not latency_ok:
                    reason = "both_exceeded"
                elif not cost_ok:
                    reason = "budget_exceeded"
                else:
                    reason = "deadline_exceeded"

                self.infeasibility_log.append(
                    InfeasibilityRecord(
                        plan_id=plan.plan_id,
                        estimated_cost=cost,
                        estimated_latency=latency,
                        budget=self.budget,
                        deadline=self.deadline,
                        reason=reason,
                    )
                )

        return feasible

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def infeasibility_summary(self) -> Dict:
        """
        Return a summary dict suitable for experiment logs.

        Keys:
            budget_exceeded_count   — plans rejected solely by cost
            deadline_exceeded_count — plans rejected solely by deadline
            both_exceeded_count     — plans rejected by both constraints
            no_metrics_count        — plans with no prediction available
            total_rejected          — total rejected plans
        """
        counts: Dict[str, int] = {
            "budget_exceeded_count": 0,
            "deadline_exceeded_count": 0,
            "both_exceeded_count": 0,
            "no_metrics_count": 0,
        }

        for record in self.infeasibility_log:
            key = f"{record.reason}_count"
            counts[key] = counts.get(key, 0) + 1

        counts["total_rejected"] = len(self.infeasibility_log)

        return counts

    def is_infeasible(self) -> bool:
        """Return True when zero feasible plans exist."""
        return len(self.infeasibility_log) > 0

    def failure_reason(
        self,
        candidate_count: int,
        feasible_count: int,
    ) -> Optional[str]:
        """
        Return a human-readable failure string when no feasible plans exist,
        or None when at least one plan is feasible.

        Intended for ``MOCAPResult.failure_reason``.
        """
        if feasible_count > 0:
            return None

        summary = self.infeasibility_summary()

        if candidate_count == 0:
            return "No candidate plans were generated."

        if summary["no_metrics_count"] == candidate_count:
            return (
                "No candidate could be evaluated: predictions are missing."
            )

        if summary["budget_exceeded_count"] == candidate_count:
            return (
                f"No candidate satisfies the budget "
                f"(B = {self.budget:.6g}). "
                f"All {candidate_count} candidates exceed the cost limit."
            )

        if summary["deadline_exceeded_count"] == candidate_count:
            return (
                f"No candidate satisfies the deadline "
                f"(D = {self.deadline:.6g} s). "
                f"All {candidate_count} candidates exceed the latency limit."
            )

        return (
            f"No candidate satisfies the constraints "
            f"(B = {self.budget:.6g}"
            + (f", D = {self.deadline:.6g} s" if self.deadline else "")
            + f"). All {candidate_count} candidates are infeasible."
        )