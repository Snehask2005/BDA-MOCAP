"""
Replanner hook for AdaptiveController.

Builds the `replanner(SelectedPlan) -> Optional[SelectedPlan]` callback from
the candidate set and predicted metrics of a finished MOCAP planning run
(MOCAPResult.candidates / MOCAPResult.metrics). It does NOT call the
optimizer or execute anything: it only picks among candidates that were
already generated and predicted.

Policy: among candidates not yet tried whose predicted cost is lower than
the cancelled plan's expected cost, return the cheapest (ties broken by
lower predicted latency). Returns None when no cheaper candidate exists,
which makes AdaptiveController fall back to degraded (AQP) mode.

Selection is relative to the configured model and candidate set; it is not
claimed to be globally optimal.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from mocap.interfaces import SelectedPlan


def make_cost_replanner(
    candidates: Iterable,
    metrics: Dict[str, object],
) -> Callable[[SelectedPlan], Optional[SelectedPlan]]:
    """
    candidates: iterable of CandidatePlan (plan_id, query_id, sql,
                strategy, actual_join_strategy).
    metrics:    {plan_id: PlanMetrics} (estimated_cost, estimated_latency).
    """
    candidates = list(candidates)
    tried: set[str] = set()

    def replan(current: SelectedPlan) -> Optional[SelectedPlan]:
        tried.add(current.plan_id)

        options = []
        for cand in candidates:
            if cand.plan_id in tried:
                continue

            m = metrics.get(cand.plan_id)
            if m is None:
                continue

            if m.estimated_cost >= current.expected_cost:
                continue

            options.append((m.estimated_cost, m.estimated_latency, cand, m))

        if not options:
            return None

        options.sort(key=lambda item: (item[0], item[1]))
        _cost, _latency, cand, m = options[0]

        return SelectedPlan(
            plan_id=cand.plan_id,
            query_id=current.query_id,
            selected_strategy=(cand.actual_join_strategy or cand.strategy),
            expected_cost=m.estimated_cost,
            expected_latency=m.estimated_latency,
            selection_reason=(
                f"adaptive replan: projected overrun of {current.plan_id}; "
                "cheapest untried candidate with lower predicted cost"
            ),
            physical_plan_sql=cand.sql,
            budget=current.budget,
            deadline=current.deadline,
            accuracy_tolerance=current.accuracy_tolerance,
        )

    return replan
