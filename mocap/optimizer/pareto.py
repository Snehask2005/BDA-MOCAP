"""
Pareto-dominance filtering for the MOCAP optimizer.

Roshini (Member 3) — Core MOCAP Optimizer

A plan p dominates q (in the cost-latency space) if:
    p is no worse than q in *every* objective, AND
    p is strictly better than q in *at least one* objective.

Plans that are not dominated by any other plan form the Pareto front.
These plans represent the best available trade-offs and are the only
candidates forwarded to Lagrangian ranking.

Tie / equivalence handling:
    Two plans are *equivalent* if they have identical predicted cost and
    identical predicted latency. In this case neither dominates the other,
    and both remain on the Pareto front. The caller (LagrangianRanker) is
    responsible for breaking such ties.

Numerical tolerance:
    A small relative tolerance (1e-9) guards against floating-point noise
    causing spurious dominance decisions.
"""

from __future__ import annotations

from typing import List, Tuple

_EPSILON = 1e-9  # relative tolerance for dominance comparison


def _dominates(
    cost_a: float,
    lat_a: float,
    cost_b: float,
    lat_b: float,
) -> bool:
    """
    Return True if plan A dominates plan B.

    Dominance: A <= B in all objectives AND A < B in at least one.

    A relative tolerance guards against floating-point artefacts.
    """
    tol_cost = _EPSILON * max(abs(cost_a), abs(cost_b), 1e-30)
    tol_lat = _EPSILON * max(abs(lat_a), abs(lat_b), 1e-30)

    a_no_worse_cost = cost_a <= cost_b + tol_cost
    a_no_worse_lat = lat_a <= lat_b + tol_lat

    a_strictly_better = (
        cost_a < cost_b - tol_cost
        or lat_a < lat_b - tol_lat
    )

    return a_no_worse_cost and a_no_worse_lat and a_strictly_better


class ParetoSelector:
    """
    Pareto-optimal front computation for the MOCAP two-objective space
    (predicted cost, predicted latency).

    Usage (instance):
        selector = ParetoSelector()
        front = selector.compute_front(feasible_plans)
        print(selector.dominated_plans)

    Usage (class-level, backward-compatible):
        front = ParetoSelector.get_pareto_optimal(feasible_plans)

    Attributes
    ----------
    dominated_plans : list
        Populated after ``compute_front``; contains (plan, metrics) pairs
        that were dominated. Useful for paper tables and diagnostics.
    equivalent_groups : list
        Groups of plans that are mutually equivalent (identical cost and
        latency). Each group is a list of (plan, metrics) pairs.
    """

    def __init__(self) -> None:
        self.dominated_plans: List[Tuple] = []
        self.equivalent_groups: List[List[Tuple]] = []

    # ------------------------------------------------------------------
    # Primary instance method
    # ------------------------------------------------------------------

    def compute_front(
        self,
        feasible_plans: List[Tuple],
    ) -> List[Tuple]:
        """
        Return the Pareto-optimal subset of ``feasible_plans``.

        Parameters
        ----------
        feasible_plans : list of (CandidatePlan, PlanMetrics)
            Feasible candidates from the constraints filter.

        Returns
        -------
        list of (CandidatePlan, PlanMetrics)
            Plans not dominated by any other feasible plan.
        """
        self.dominated_plans.clear()
        self.equivalent_groups.clear()

        if not feasible_plans:
            return []

        # Single plan is trivially Pareto-optimal.
        if len(feasible_plans) == 1:
            return list(feasible_plans)

        pareto_front: List[Tuple] = []

        for i, (p_curr, m_curr) in enumerate(feasible_plans):
            cost_curr = float(m_curr.estimated_cost or 0.0)
            lat_curr = float(m_curr.estimated_latency or 0.0)

            is_dominated = False

            for j, (p_other, m_other) in enumerate(feasible_plans):
                if i == j:
                    continue

                cost_other = float(m_other.estimated_cost or 0.0)
                lat_other = float(m_other.estimated_latency or 0.0)

                if _dominates(cost_other, lat_other, cost_curr, lat_curr):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append((p_curr, m_curr))
            else:
                self.dominated_plans.append((p_curr, m_curr))

        # Identify equivalent groups on the Pareto front (ties in both
        # objectives).  These are reported for diagnostics but all remain
        # on the front so LagrangianRanker can break the tie via score.
        self._find_equivalent_groups(pareto_front)

        return pareto_front

    # ------------------------------------------------------------------
    # Backward-compatible class-level API
    # ------------------------------------------------------------------

    @classmethod
    def get_pareto_optimal(
        cls,
        feasible_plans: List[Tuple],
    ) -> List[Tuple]:
        """
        Class-level convenience method (backward-compatible).

        Existing callers (pipeline.py, experiments/*.py) using:
            ``ParetoSelector.get_pareto_optimal(feasible)``
        continue to work without modification.
        """
        return cls().compute_front(feasible_plans)

    @classmethod
    def static_pareto(
        cls,
        feasible_plans: List[Tuple],
    ) -> List[Tuple]:
        """Alternative class-level convenience alias."""
        return cls().compute_front(feasible_plans)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_equivalent_groups(self, pareto_front: List[Tuple]) -> None:
        visited = set()
        for i, (p_i, m_i) in enumerate(pareto_front):
            if p_i.plan_id in visited:
                continue
            group = [(p_i, m_i)]
            cost_i = float(m_i.estimated_cost or 0.0)
            lat_i = float(m_i.estimated_latency or 0.0)
            for j, (p_j, m_j) in enumerate(pareto_front):
                if i == j or p_j.plan_id in visited:
                    continue
                cost_j = float(m_j.estimated_cost or 0.0)
                lat_j = float(m_j.estimated_latency or 0.0)
                tol_c = _EPSILON * max(abs(cost_i), abs(cost_j), 1e-30)
                tol_l = _EPSILON * max(abs(lat_i), abs(lat_j), 1e-30)
                if (
                    abs(cost_i - cost_j) <= tol_c
                    and abs(lat_i - lat_j) <= tol_l
                ):
                    group.append((p_j, m_j))
                    visited.add(p_j.plan_id)
            if len(group) > 1:
                self.equivalent_groups.append(group)
            visited.add(p_i.plan_id)

    def dominance_summary(self) -> dict:
        """Return a summary dict for experiment logs."""
        return {
            "pareto_size": None,         # set by caller after filtering
            "dominated_count": len(self.dominated_plans),
            "equivalent_groups": len(self.equivalent_groups),
        }