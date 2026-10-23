"""
2D Pareto Frontier Extraction for Multi-Objective Candidate Plans.

Roshini (Lead: Robust Optimizer & Theory) — Phase 3 Deliverable

Extracts non-dominated candidate query execution plans balancing:
    Objective 1: Execution Latency (minimize)
    Objective 2: Cloud Cost (minimize conformal upper bound C_hat^+)

Achieves strict O(N log N) time complexity via sort-and-scan, guaranteeing
planning latency well under the required < 50ms threshold.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from selector.bounds import PlanBounds


@dataclass
class ParetoPlan:
    """A non-dominated candidate plan enriched with confidence bounds."""

    plan: PlanBounds
    pareto_rank: int = 1
    is_equivalent_group_leader: bool = True
    equivalent_plans: List[PlanBounds] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return self.plan.conformal_cost_upper

    @property
    def latency(self) -> float:
        return self.plan.point_latency


@dataclass
class ParetoFrontierResult:
    """Complete result of Pareto frontier extraction."""

    frontier: List[ParetoPlan]
    dominated_plans: List[PlanBounds]
    total_candidates: int
    frontier_count: int
    dominated_count: int
    extraction_time_ms: float
    min_cost: float
    max_cost: float
    min_latency: float
    max_latency: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "frontier_count": self.frontier_count,
            "dominated_count": self.dominated_count,
            "extraction_time_ms": self.extraction_time_ms,
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "frontier_plan_ids": [p.plan.plan_id for p in self.frontier],
        }


class ParetoExtractor:
    """
    High-performance 2D Pareto frontier extractor.

    Given a set of feasible candidate plans with cost and latency estimates,
    finds all non-dominated plans where:
        A dominates B  <=>  (Cost(A) <= Cost(B) and Lat(A) <= Lat(B))
                            and (Cost(A) < Cost(B) or Lat(A) < Lat(B))

    Sorts by Cost ascending (breaking ties with Latency ascending), then scans
    maintaining the minimum latency seen so far.
    """

    def __init__(
        self,
        use_conformal_cost: bool = True,
        use_conformal_latency: bool = False,
        tolerance: float = 1e-9,
    ) -> None:
        self.use_conformal_cost = use_conformal_cost
        self.use_conformal_latency = use_conformal_latency
        self.tolerance = tolerance

    def _get_cost(self, p: PlanBounds) -> float:
        return p.conformal_cost_upper if self.use_conformal_cost else p.point_cost

    def _get_latency(self, p: PlanBounds) -> float:
        return p.conformal_latency_upper if self.use_conformal_latency else p.point_latency

    def extract_frontier(
        self,
        plans: Sequence[PlanBounds],
    ) -> ParetoFrontierResult:
        start_t = time.perf_counter()

        if not plans:
            return ParetoFrontierResult(
                frontier=[],
                dominated_plans=[],
                total_candidates=0,
                frontier_count=0,
                dominated_count=0,
                extraction_time_ms=0.0,
                min_cost=0.0,
                max_cost=0.0,
                min_latency=0.0,
                max_latency=0.0,
            )

        # Pair each plan with its evaluated (cost, latency)
        evaluated: List[Tuple[float, float, PlanBounds]] = [
            (self._get_cost(p), self._get_latency(p), p) for p in plans
        ]

        # Sort primarily by cost ascending, secondarily by latency ascending
        evaluated.sort(key=lambda item: (item[0], item[1]))

        frontier_plans: List[ParetoPlan] = []
        dominated_plans: List[PlanBounds] = []

        # Min latency observed on the frontier
        min_latency_so_far = float("inf")

        # Group duplicate (cost, latency) values within tolerance
        current_group: List[Tuple[float, float, PlanBounds]] = []

        for item in evaluated:
            if not current_group:
                current_group.append(item)
                continue

            last_cost, last_lat, _ = current_group[-1]
            cost, lat, _ = item

            if (
                abs(cost - last_cost) <= self.tolerance
                and abs(lat - last_lat) <= self.tolerance
            ):
                current_group.append(item)
            else:
                # Process finished group
                leader_cost, leader_lat, leader_plan = current_group[0]
                if leader_lat < min_latency_so_far - self.tolerance:
                    pareto_p = ParetoPlan(
                        plan=leader_plan,
                        equivalent_plans=[g[2] for g in current_group[1:]],
                    )
                    frontier_plans.append(pareto_p)
                    min_latency_so_far = leader_lat
                else:
                    for _, _, p in current_group:
                        dominated_plans.append(p)

                current_group = [item]

        # Process final group
        if current_group:
            leader_cost, leader_lat, leader_plan = current_group[0]
            if leader_lat < min_latency_so_far - self.tolerance:
                pareto_p = ParetoPlan(
                    plan=leader_plan,
                    equivalent_plans=[g[2] for g in current_group[1:]],
                )
                frontier_plans.append(pareto_p)
                min_latency_so_far = leader_lat
            else:
                for _, _, p in current_group:
                    dominated_plans.append(p)

        end_t = time.perf_counter()
        elapsed_ms = (end_t - start_t) * 1000.0

        costs = [self._get_cost(p.plan) for p in frontier_plans]
        lats = [self._get_latency(p.plan) for p in frontier_plans]

        min_c = min(costs) if costs else 0.0
        max_c = max(costs) if costs else 0.0
        min_l = min(lats) if lats else 0.0
        max_l = max(lats) if lats else 0.0

        return ParetoFrontierResult(
            frontier=frontier_plans,
            dominated_plans=dominated_plans,
            total_candidates=len(plans),
            frontier_count=len(frontier_plans),
            dominated_count=len(dominated_plans),
            extraction_time_ms=elapsed_ms,
            min_cost=min_c,
            max_cost=max_c,
            min_latency=min_l,
            max_latency=max_l,
        )
