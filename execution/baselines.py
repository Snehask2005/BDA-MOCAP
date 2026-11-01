"""
Selection strategies compared in E1/E2/E5 (owner: Anaswara).

All strategies choose from the SAME measured candidate pool of a query, so
the comparison isolates the selection rule (plan Section 7, fair baselines).

    mocap               Roshini's RobustOptimizer: conformal upper-bound filter
                        (C^+ <= B), Pareto frontier, fastest-under-budget
    mocap_point         same pipeline but filtering on point predictions
                        (ablation: what the conformal bound adds)
    greedy_cheapest     lowest predicted cost, ignores latency and budget
    greedy_fastest      lowest predicted latency among plans whose predicted
                        cost <= B (falls back to cheapest predicted)
    random              uniformly random candidate (seeded per query)
    oracle              lowest MEASURED latency among plans whose measured
                        cost <= B (falls back to cheapest measured)
    spark_default_*     Spark's own planner on the original SQL, with CBO and
                        AQE on/off -- measured separately, not chosen from the
                        pool (see harness.measure_query)

Predicted-only strategies never look at measured values; only ``oracle``
does, which is what makes it an upper bound rather than a baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence


class Candidate(Protocol):
    plan_id: str
    strategy: str
    pred_cost: Optional[float]
    pred_latency: Optional[float]
    actual_cost: Optional[float]
    actual_latency: Optional[float]


@dataclass
class Choice:
    """A strategy's decision for one query."""

    strategy: str
    plan_id: Optional[str]
    infeasible_fallback: bool = False
    reason: str = ""
    planning_ms: float = 0.0
    details: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}


def _usable(cands: Sequence[Candidate], predicted: bool) -> List[Candidate]:
    if predicted:
        return [
            c for c in cands
            if c.pred_cost is not None and c.pred_latency is not None
            and c.actual_latency is not None
        ]
    return [c for c in cands if c.actual_cost is not None and c.actual_latency is not None]


def greedy_cheapest(cands: Sequence[Candidate], budget: float) -> Choice:
    pool = _usable(cands, predicted=True)
    if not pool:
        return Choice("greedy_cheapest", None, reason="no usable candidates")
    best = min(pool, key=lambda c: (c.pred_cost, c.pred_latency, c.plan_id))
    return Choice("greedy_cheapest", best.plan_id, reason="min predicted cost")


def greedy_fastest(cands: Sequence[Candidate], budget: float) -> Choice:
    pool = _usable(cands, predicted=True)
    if not pool:
        return Choice("greedy_fastest", None, reason="no usable candidates")
    feasible = [c for c in pool if c.pred_cost <= budget]
    if feasible:
        best = min(feasible, key=lambda c: (c.pred_latency, c.pred_cost, c.plan_id))
        return Choice("greedy_fastest", best.plan_id, reason="min predicted latency, predicted cost <= B")
    best = min(pool, key=lambda c: (c.pred_cost, c.plan_id))
    return Choice("greedy_fastest", best.plan_id, infeasible_fallback=True,
                  reason="no plan predicted under budget; cheapest predicted")


def random_choice(cands: Sequence[Candidate], budget: float, seed: int, query_id: str) -> Choice:
    pool = _usable(cands, predicted=False)
    if not pool:
        return Choice("random", None, reason="no usable candidates")
    rng = random.Random(f"{seed}:{query_id}")
    pick = rng.choice(sorted(pool, key=lambda c: c.plan_id))
    return Choice("random", pick.plan_id, reason=f"seeded random (seed={seed})")


def oracle(cands: Sequence[Candidate], budget: float) -> Choice:
    pool = _usable(cands, predicted=False)
    if not pool:
        return Choice("oracle", None, reason="no measured candidates")
    feasible = [c for c in pool if c.actual_cost <= budget]
    if feasible:
        best = min(feasible, key=lambda c: (c.actual_latency, c.actual_cost, c.plan_id))
        return Choice("oracle", best.plan_id, reason="min measured latency, measured cost <= B")
    best = min(pool, key=lambda c: (c.actual_cost, c.plan_id))
    return Choice("oracle", best.plan_id, infeasible_fallback=True,
                  reason="nothing measured under budget; cheapest measured")


def mocap_select(
    cands: Sequence[Candidate],
    budget: float,
    query_id: str,
    cost_quantile,
    latency_quantile,
    use_conformal: bool = True,
    name: Optional[str] = None,
    policy=None,
) -> Choice:
    """Select with Roshini's RobustOptimizer (selector package)."""

    from selector import DecisionPolicyType, InfeasibleMode, RobustOptimizer

    label = name or ("mocap" if use_conformal else "mocap_point")
    pool = _usable(cands, predicted=True)
    if not pool:
        return Choice(label, None, reason="no usable candidates")
    optimizer = RobustOptimizer(
        delta=cost_quantile.delta,
        decision_policy=policy or DecisionPolicyType.FASTEST_UNDER_BUDGET,
        infeasible_mode=InfeasibleMode.MIN_COST_OVER_BUDGET,
        use_conformal=use_conformal,
        cost_quantile=cost_quantile,
        latency_quantile=latency_quantile,
    )
    items = [(c.plan_id, float(c.pred_cost), float(c.pred_latency), None) for c in pool]
    result = optimizer.select(items, budget=budget, deadline=None, query_id=query_id)
    plan_id = result.selected_plan.plan_id if result.selected_plan is not None else None
    return Choice(
        label,
        plan_id,
        infeasible_fallback=not result.is_feasible,
        reason=result.selection_reason,
        planning_ms=result.total_planning_time_ms,
        details={
            "feasible_count": result.feasible_count,
            "pareto_count": result.pareto_count,
            "cost_multiplier": cost_quantile.multiplier,
        },
    )


PREDICTED_STRATEGIES = ("mocap", "mocap_point", "greedy_cheapest", "greedy_fastest")
POOL_STRATEGIES = PREDICTED_STRATEGIES + ("random", "oracle")
