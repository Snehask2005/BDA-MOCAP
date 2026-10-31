"""
Baseline selection policies for MOCAP comparison.

Roshini (Member 3) — Core MOCAP Optimizer

Two baselines are implemented:

1.  SparkDefaultBaseline
    Simulates Spark's default Catalyst choice: pick the candidate with
    the LOWEST estimated cost among ALL candidates (no budget filtering).
    This approximates what Spark would choose without budget awareness.

2.  GreedyBudgetPruningBaseline
    Apply the hard budget constraint, then greedily select the plan with
    the minimum predicted latency among feasible candidates.  This matches
    a common cost-aware heuristic: "cheapest valid plan that is also
    fastest".

Both baselines operate on the same candidate set and predicted metrics
as MOCAP, ensuring a fair comparison (no candidate generation advantage).

Neither baseline uses Pareto filtering or Lagrangian ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class BaselineResult:
    """Output of one baseline selection."""

    baseline_name: str
    query_id: str

    selected_plan_id: Optional[str]
    selected_strategy: Optional[str]
    predicted_cost: Optional[float]
    predicted_latency: Optional[float]

    candidate_count: int
    feasible_count: int

    budget: Optional[float]
    failure_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline_name,
            "query_id": self.query_id,
            "selected_plan": self.selected_plan_id or "",
            "selected_strategy": self.selected_strategy or "",
            "predicted_cost": self.predicted_cost,
            "predicted_latency": self.predicted_latency,
            "candidate_count": self.candidate_count,
            "feasible_count": self.feasible_count,
            "budget": self.budget,
            "failure_reason": self.failure_reason or "",
        }


class SparkDefaultBaseline:
    """
    Simulates Spark's default Catalyst plan selection.

    Strategy: pick the candidate with the **lowest predicted cost** among
    all generated candidates (no hard budget filtering).

    Rationale: Catalyst optimises for resource efficiency without explicit
    budget awareness.  This baseline demonstrates what MOCAP adds over the
    Spark default.
    """

    NAME = "spark_default"

    def select(
        self,
        query_id: str,
        candidates,
        metrics: Dict,
        budget: Optional[float] = None,
    ) -> BaselineResult:
        """Select the minimum-cost candidate regardless of budget."""

        if not candidates:
            return BaselineResult(
                baseline_name=self.NAME,
                query_id=query_id,
                selected_plan_id=None,
                selected_strategy=None,
                predicted_cost=None,
                predicted_latency=None,
                candidate_count=0,
                feasible_count=0,
                budget=budget,
                failure_reason="No candidates available.",
            )

        # Collect all candidates with available metrics.
        evaluated = [
            (plan, metrics[plan.plan_id])
            for plan in candidates
            if plan.plan_id in metrics
        ]

        if not evaluated:
            return BaselineResult(
                baseline_name=self.NAME,
                query_id=query_id,
                selected_plan_id=None,
                selected_strategy=None,
                predicted_cost=None,
                predicted_latency=None,
                candidate_count=len(candidates),
                feasible_count=0,
                budget=budget,
                failure_reason="No metrics available for any candidate.",
            )

        best_plan, best_metrics = min(
            evaluated,
            key=lambda item: float(item[1].estimated_cost or float("inf")),
        )

        return BaselineResult(
            baseline_name=self.NAME,
            query_id=query_id,
            selected_plan_id=best_plan.plan_id,
            selected_strategy=(
                getattr(best_plan, "actual_join_strategy", None)
                or getattr(best_plan, "strategy", None)
            ),
            predicted_cost=float(best_metrics.estimated_cost or 0.0),
            predicted_latency=float(best_metrics.estimated_latency or 0.0),
            candidate_count=len(candidates),
            feasible_count=len(evaluated),
            budget=budget,
            failure_reason=None,
        )


class GreedyBudgetPruningBaseline:
    """
    Budget-aware greedy baseline.

    Strategy:
        1. Filter candidates to those with estimated_cost <= budget.
        2. Among feasible candidates, pick the one with the **lowest
           predicted latency**.

    This represents the simplest budget-constrained policy and serves as
    a lower-bound comparison for MOCAP's multi-objective optimisation.
    """

    NAME = "greedy_budget_pruning"

    def __init__(
        self,
        budget: float,
        deadline: Optional[float] = None,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be strictly positive")
        self.budget = budget
        self.deadline = deadline

    def select(
        self,
        query_id: str,
        candidates,
        metrics: Dict,
    ) -> BaselineResult:
        """Apply hard budget filter then pick minimum-latency candidate."""

        if not candidates:
            return BaselineResult(
                baseline_name=self.NAME,
                query_id=query_id,
                selected_plan_id=None,
                selected_strategy=None,
                predicted_cost=None,
                predicted_latency=None,
                candidate_count=0,
                feasible_count=0,
                budget=self.budget,
                failure_reason="No candidates available.",
            )

        feasible: List[Tuple] = []

        for plan in candidates:
            m = metrics.get(plan.plan_id)
            if m is None:
                continue
            cost = float(m.estimated_cost or 0.0)
            latency = float(m.estimated_latency or 0.0)
            cost_ok = cost <= self.budget
            lat_ok = (
                True
                if self.deadline is None
                else latency <= self.deadline
            )
            if cost_ok and lat_ok:
                feasible.append((plan, m))

        if not feasible:
            return BaselineResult(
                baseline_name=self.NAME,
                query_id=query_id,
                selected_plan_id=None,
                selected_strategy=None,
                predicted_cost=None,
                predicted_latency=None,
                candidate_count=len(candidates),
                feasible_count=0,
                budget=self.budget,
                failure_reason=(
                    f"No candidate satisfies the budget "
                    f"(B = {self.budget:.6g})."
                ),
            )

        best_plan, best_metrics = min(
            feasible,
            key=lambda item: float(item[1].estimated_latency or float("inf")),
        )

        return BaselineResult(
            baseline_name=self.NAME,
            query_id=query_id,
            selected_plan_id=best_plan.plan_id,
            selected_strategy=(
                getattr(best_plan, "actual_join_strategy", None)
                or getattr(best_plan, "strategy", None)
            ),
            predicted_cost=float(best_metrics.estimated_cost or 0.0),
            predicted_latency=float(best_metrics.estimated_latency or 0.0),
            candidate_count=len(candidates),
            feasible_count=len(feasible),
            budget=self.budget,
            failure_reason=None,
        )
