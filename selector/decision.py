"""
Decision Policies for Pareto Frontier Selection.

Roshini (Lead: Robust Optimizer & Theory) — Phase 3 Deliverable

Deprecates the mathematically inert Lagrangian penalty term:
    lambda * max(0, C_hat - B)
which identically collapsed to zero under hard-filtered feasibility.

Replaces it with two well-founded operational decision policies:
    Policy A: Fastest Under Budget (minimizes latency subject to certified budget)
    Policy B: Normalized Multi-Objective Scalarization (Pareto trade-off weighting)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from selector.bounds import PlanBounds
from selector.pareto import ParetoPlan


class DecisionPolicyType(str, Enum):
    """Available decision policies for selecting from the Pareto frontier."""

    FASTEST_UNDER_BUDGET = "FASTEST_UNDER_BUDGET"
    NORMALIZED_SCALARIZATION = "NORMALIZED_SCALARIZATION"
    CHEAPEST_FEASIBLE = "CHEAPEST_FEASIBLE"
    LEGACY_LAGRANGIAN_ABLATION = "LEGACY_LAGRANGIAN_ABLATION"


@dataclass
class DecisionScore:
    """Detailed score breakdown for a candidate evaluated under a decision policy."""

    plan: PlanBounds
    policy: DecisionPolicyType
    final_score: float
    rank: int
    normalized_latency: float = 0.0
    normalized_cost: float = 0.0
    latency_weight: float = 1.0
    cost_weight: float = 0.0
    lagrangian_penalty: float = 0.0  # Kept strictly for ablation verification
    selection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("plan", None)
        d["plan_id"] = self.plan.plan_id
        return d


@dataclass
class DecisionOutcome:
    """Final decision output selecting the optimal execution plan."""

    selected_plan: PlanBounds
    policy: DecisionPolicyType
    score_details: DecisionScore
    ranked_candidates: List[DecisionScore]
    frontier_size: int
    selection_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_plan_id": self.selected_plan.plan_id,
            "policy": self.policy.value,
            "score": self.score_details.final_score,
            "selection_reason": self.selection_reason,
            "frontier_size": self.frontier_size,
            "ranked_plan_ids": [s.plan.plan_id for s in self.ranked_candidates],
        }


class DecisionEngine:
    """
    Evaluates and ranks Pareto-optimal candidates according to the chosen policy.
    """

    def __init__(
        self,
        policy: DecisionPolicyType = DecisionPolicyType.FASTEST_UNDER_BUDGET,
        alpha: float = 0.5,  # Latency weight in scalarization
        beta: float = 0.5,   # Cost weight in scalarization
        eps: float = 1e-9,
    ) -> None:
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def select(
        self,
        frontier: Sequence[ParetoPlan],
        budget: float,
        deadline: Optional[float] = None,
    ) -> DecisionOutcome:
        if not frontier:
            raise ValueError("Cannot select from an empty Pareto frontier")

        plans = [p.plan for p in frontier]

        if self.policy == DecisionPolicyType.FASTEST_UNDER_BUDGET:
            return self._select_fastest(plans, budget, deadline)
        elif self.policy == DecisionPolicyType.NORMALIZED_SCALARIZATION:
            return self._select_scalarized(plans, budget, deadline)
        elif self.policy == DecisionPolicyType.CHEAPEST_FEASIBLE:
            return self._select_cheapest(plans, budget, deadline)
        elif self.policy == DecisionPolicyType.LEGACY_LAGRANGIAN_ABLATION:
            return self._select_legacy_lagrangian(plans, budget, deadline)
        else:
            raise ValueError(f"Unknown decision policy: {self.policy}")

    def _select_fastest(
        self,
        plans: List[PlanBounds],
        budget: float,
        deadline: Optional[float],
    ) -> DecisionOutcome:
        """
        Policy A: Fastest plan under budget.

        Since certified budget feasibility (C_hat^+ <= B) is already guaranteed
        at level 1 - delta, the primary objective is minimizing query runtime.
        Ties are broken by lower conformal cost upper bound.
        """
        sorted_plans = sorted(
            plans,
            key=lambda p: (p.point_latency, p.conformal_cost_upper, p.plan_id),
        )

        ranked: List[DecisionScore] = []
        for rank_idx, p in enumerate(sorted_plans, start=1):
            score = DecisionScore(
                plan=p,
                policy=DecisionPolicyType.FASTEST_UNDER_BUDGET,
                final_score=p.point_latency,
                rank=rank_idx,
                normalized_latency=p.point_latency,
                normalized_cost=p.conformal_cost_upper,
                latency_weight=1.0,
                cost_weight=0.0,
                lagrangian_penalty=0.0,
                selection_reason=(
                    f"Selected as fastest feasible plan under certified budget B={budget} "
                    f"(estimated latency={p.point_latency:.4f}s, cost upper bound={p.conformal_cost_upper:.6f})"
                ),
            )
            ranked.append(score)

        winner = ranked[0]
        return DecisionOutcome(
            selected_plan=winner.plan,
            policy=DecisionPolicyType.FASTEST_UNDER_BUDGET,
            score_details=winner,
            ranked_candidates=ranked,
            frontier_size=len(plans),
            selection_reason=winner.selection_reason,
            metadata={"budget": budget, "deadline": deadline},
        )

    def _select_scalarized(
        self,
        plans: List[PlanBounds],
        budget: float,
        deadline: Optional[float],
    ) -> DecisionOutcome:
        """
        Policy B: Multi-objective normalized cost/latency scalarization.

        Normalizes latency and cost across the current Pareto frontier:
            L_norm = (L - L_min) / (L_max - L_min + eps)
            C_norm = (C^+ - C^+_min) / (C^+_max - C^+_min + eps)
            Score = alpha * L_norm + (1 - alpha) * C_norm
        """
        l_min = min(p.point_latency for p in plans)
        l_max = max(p.point_latency for p in plans)
        c_min = min(p.conformal_cost_upper for p in plans)
        c_max = max(p.conformal_cost_upper for p in plans)

        l_span = max(l_max - l_min, self.eps)
        c_span = max(c_max - c_min, self.eps)

        total_weight = self.alpha + self.beta
        w_lat = (self.alpha / total_weight) if total_weight > 0 else 0.5
        w_cost = (self.beta / total_weight) if total_weight > 0 else 0.5

        scored: List[Tuple[float, PlanBounds, float, float]] = []
        for p in plans:
            norm_l = (p.point_latency - l_min) / l_span
            norm_c = (p.conformal_cost_upper - c_min) / c_span
            composite_score = w_lat * norm_l + w_cost * norm_c
            scored.append((composite_score, p, norm_l, norm_c))

        scored.sort(key=lambda item: (item[0], item[1].point_latency, item[1].plan_id))

        ranked: List[DecisionScore] = []
        for rank_idx, (s_val, p, nl, nc) in enumerate(scored, start=1):
            score = DecisionScore(
                plan=p,
                policy=DecisionPolicyType.NORMALIZED_SCALARIZATION,
                final_score=s_val,
                rank=rank_idx,
                normalized_latency=nl,
                normalized_cost=nc,
                latency_weight=w_lat,
                cost_weight=w_cost,
                lagrangian_penalty=0.0,
                selection_reason=(
                    f"Selected by normalized scalarization (score={s_val:.4f}, "
                    f"latency_norm={nl:.3f}, cost_norm={nc:.3f}, alpha={w_lat:.2f})"
                ),
            )
            ranked.append(score)

        winner = ranked[0]
        return DecisionOutcome(
            selected_plan=winner.plan,
            policy=DecisionPolicyType.NORMALIZED_SCALARIZATION,
            score_details=winner,
            ranked_candidates=ranked,
            frontier_size=len(plans),
            selection_reason=winner.selection_reason,
            metadata={"alpha": w_lat, "beta": w_cost, "budget": budget},
        )

    def _select_cheapest(
        self,
        plans: List[PlanBounds],
        budget: float,
        deadline: Optional[float],
    ) -> DecisionOutcome:
        """Min-cost feasible plan."""
        sorted_plans = sorted(
            plans,
            key=lambda p: (p.conformal_cost_upper, p.point_latency, p.plan_id),
        )
        ranked = [
            DecisionScore(
                plan=p,
                policy=DecisionPolicyType.CHEAPEST_FEASIBLE,
                final_score=p.conformal_cost_upper,
                rank=i,
                selection_reason=f"Selected as cheapest feasible plan (C_hat^+={p.conformal_cost_upper:.6f})",
            )
            for i, p in enumerate(sorted_plans, start=1)
        ]
        winner = ranked[0]
        return DecisionOutcome(
            selected_plan=winner.plan,
            policy=DecisionPolicyType.CHEAPEST_FEASIBLE,
            score_details=winner,
            ranked_candidates=ranked,
            frontier_size=len(plans),
            selection_reason=winner.selection_reason,
        )

    def _select_legacy_lagrangian(
        self,
        plans: List[PlanBounds],
        budget: float,
        deadline: Optional[float],
        lambda_mult: float = 1.0,
    ) -> DecisionOutcome:
        """
        Legacy Lagrangian formulation for ablation study:
            Score = alpha * norm_lat + beta * norm_cost + lambda * max(0, C_hat - B)

        Confirms audit finding: since C_hat <= B for all surviving plans,
        max(0, C_hat - B) is identically 0.0, showing lambda is inert.
        """
        l_min = min(p.point_latency for p in plans)
        l_max = max(p.point_latency for p in plans)
        c_min = min(p.point_cost for p in plans)
        c_max = max(p.point_cost for p in plans)
        l_span = max(l_max - l_min, self.eps)
        c_span = max(c_max - c_min, self.eps)

        scored: List[Tuple[float, PlanBounds, float, float, float]] = []
        for p in plans:
            nl = (p.point_latency - l_min) / l_span
            nc = (p.point_cost - c_min) / c_span
            penalty = lambda_mult * max(0.0, p.point_cost - budget)
            score_val = 0.5 * nl + 0.5 * nc + penalty
            scored.append((score_val, p, nl, nc, penalty))

        scored.sort(key=lambda x: (x[0], x[1].point_latency))
        ranked = [
            DecisionScore(
                plan=p,
                policy=DecisionPolicyType.LEGACY_LAGRANGIAN_ABLATION,
                final_score=s,
                rank=i,
                normalized_latency=nl,
                normalized_cost=nc,
                lagrangian_penalty=pen,
                selection_reason=f"Legacy Lagrangian score={s:.4f} (penalty={pen:.4f})",
            )
            for i, (s, p, nl, nc, pen) in enumerate(scored, start=1)
        ]
        winner = ranked[0]
        return DecisionOutcome(
            selected_plan=winner.plan,
            policy=DecisionPolicyType.LEGACY_LAGRANGIAN_ABLATION,
            score_details=winner,
            ranked_candidates=ranked,
            frontier_size=len(plans),
            selection_reason=winner.selection_reason,
            metadata={"lambda": lambda_mult, "penalty": winner.lagrangian_penalty},
        )
