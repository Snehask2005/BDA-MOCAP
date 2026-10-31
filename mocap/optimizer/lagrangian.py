"""
Lagrangian multi-objective ranking for the MOCAP optimizer.

Roshini (Member 3) — Core MOCAP Optimizer

The Lagrangian objective combines normalized cost and latency using
configurable weights (alpha, beta) and an optional penalty for any
residual constraint violation:

    J(p) = alpha  * L_hat(p)
          + beta   * C_hat(p)
          + gamma  * A_hat(p)     [accuracy term — optional]
          + delta  * V_hat(p)     [variance term — optional]
          + lambda * max(0, C(p) - B)   [violation penalty]

where:
    L_hat  = min-max normalised predicted latency   in [0, 1]
    C_hat  = min-max normalised predicted cost      in [0, 1]
    A_hat  = normalized accuracy proxy              in [0, 1]
    V_hat  = normalized variance / uncertainty      in [0, 1]
    B      = hard budget
    lambda = Lagrange multiplier (violation penalty coefficient)

Hard constraints are applied BEFORE this ranker; therefore the violation
term is normally zero.  lambda is exposed for ablation experiments.

Plans are sorted by ascending J (lower score = better).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ScoredPlan:
    """A ranked candidate with its Lagrangian score and component breakdown."""

    plan_id: str
    query_id: str
    strategy: str
    score: float

    normalized_cost: float
    normalized_latency: float
    normalized_accuracy: float
    normalized_variance: float
    violation_penalty: float

    estimated_cost: float
    estimated_latency: float

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "query_id": self.query_id,
            "strategy": self.strategy,
            "score": self.score,
            "normalized_cost": self.normalized_cost,
            "normalized_latency": self.normalized_latency,
            "normalized_accuracy": self.normalized_accuracy,
            "normalized_variance": self.normalized_variance,
            "violation_penalty": self.violation_penalty,
            "estimated_cost": self.estimated_cost,
            "estimated_latency": self.estimated_latency,
        }


class LagrangianRanker:
    """
    Rank feasible Pareto-optimal candidates using a Lagrangian score.

    Parameters
    ----------
    alpha : float
        Weight for normalized predicted latency.
    beta : float
        Weight for normalized predicted cost.
    gamma : float
        Weight for accuracy component (optional; default 0).
    delta : float
        Weight for variance / uncertainty component (optional; default 0).
    lambda_mult : float
        Violation penalty multiplier. Non-zero values are intended for
        ablation experiments where hard filtering is disabled.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.0,
        delta: float = 0.0,
        lambda_mult: float = 0.0,
    ) -> None:
        if alpha < 0 or beta < 0 or gamma < 0 or delta < 0 or lambda_mult < 0:
            raise ValueError("All Lagrangian weights must be non-negative.")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_mult = lambda_mult

    # ------------------------------------------------------------------
    # Public configuration description (for paper / pseudocode)
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """Return a description dict for experiment logs and pseudocode."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "lambda_mult": self.lambda_mult,
            "objective": (
                "J = alpha*L_hat + beta*C_hat "
                "+ gamma*A_hat + delta*V_hat "
                "+ lambda*max(0, C - B)"
            ),
        }

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """
        Min-max normalize a single value to [0, 1].

        When min == max (all candidates have the same value for this
        objective), the normalized value is 0 so the objective does not
        influence ranking.
        """
        if maximum <= minimum:
            return 0.0

        return (value - minimum) / (maximum - minimum)

    # ------------------------------------------------------------------
    # Per-plan score
    # ------------------------------------------------------------------

    def compute_score(
        self,
        plan,
        metrics,
        budget: float,
        cost_range: Optional[Tuple[float, float]] = None,
        latency_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute the Lagrangian objective J for one plan.

        Returns
        -------
        (score, norm_cost, norm_latency, norm_accuracy, norm_variance)
        """
        cost = float(metrics.estimated_cost or 0.0)
        latency = float(metrics.estimated_latency or 0.0)

        # Accuracy and variance are optional extension fields.
        accuracy = float(getattr(metrics, "accuracy_proxy", 0.0) or 0.0)
        variance = float(getattr(metrics, "variance", 0.0) or 0.0)

        # Use identity range when range is degenerate.
        if cost_range is None:
            cost_range = (cost, cost)
        if latency_range is None:
            latency_range = (latency, latency)

        norm_cost = self._normalize(cost, *cost_range)
        norm_latency = self._normalize(latency, *latency_range)
        norm_accuracy = accuracy       # assumed pre-normalised [0, 1]
        norm_variance = variance       # assumed pre-normalised [0, 1]

        violation = max(0.0, cost - budget)
        violation_penalty = self.lambda_mult * violation

        score = (
            self.alpha * norm_latency
            + self.beta * norm_cost
            + self.gamma * norm_accuracy
            + self.delta * norm_variance
            + violation_penalty
        )

        return (
            score,
            norm_cost,
            norm_latency,
            norm_accuracy,
            norm_variance,
        )

    # ------------------------------------------------------------------
    # Full ranking
    # ------------------------------------------------------------------

    def rank_candidates(
        self,
        feasible_plans: List[Tuple],
        budget: float,
    ) -> List[Tuple]:
        """
        Rank all plans in ``feasible_plans`` by ascending Lagrangian score.

        Parameters
        ----------
        feasible_plans : list of (CandidatePlan, PlanMetrics)
        budget : float
            Hard budget used for violation penalty computation.

        Returns
        -------
        list of (CandidatePlan, PlanMetrics, score)
            Sorted ascending by score (lowest = best).
        """
        if not feasible_plans:
            return []

        costs = [
            float(m.estimated_cost or 0.0)
            for _, m in feasible_plans
        ]
        latencies = [
            float(m.estimated_latency or 0.0)
            for _, m in feasible_plans
        ]

        cost_range = (min(costs), max(costs))
        latency_range = (min(latencies), max(latencies))

        scored = []

        for plan, metrics in feasible_plans:
            (
                score,
                norm_cost,
                norm_latency,
                norm_accuracy,
                norm_variance,
            ) = self.compute_score(
                plan,
                metrics,
                budget,
                cost_range=cost_range,
                latency_range=latency_range,
            )

            scored.append((plan, metrics, score))

        # Ascending sort: lowest J = best plan.
        scored.sort(key=lambda item: item[2])

        return scored

    # ------------------------------------------------------------------
    # Rich ranking with ScoredPlan records
    # ------------------------------------------------------------------

    def rank_with_details(
        self,
        feasible_plans: List[Tuple],
        budget: float,
    ) -> List[ScoredPlan]:
        """
        Return a list of ``ScoredPlan`` objects with full score breakdown.

        Useful for experiment tables and paper results.
        """
        if not feasible_plans:
            return []

        costs = [float(m.estimated_cost or 0.0) for _, m in feasible_plans]
        latencies = [float(m.estimated_latency or 0.0) for _, m in feasible_plans]

        cost_range = (min(costs), max(costs))
        latency_range = (min(latencies), max(latencies))

        results = []

        for plan, metrics in feasible_plans:
            (
                score,
                norm_cost,
                norm_latency,
                norm_accuracy,
                norm_variance,
            ) = self.compute_score(
                plan,
                metrics,
                budget,
                cost_range=cost_range,
                latency_range=latency_range,
            )

            violation = max(0.0, float(metrics.estimated_cost or 0.0) - budget)

            results.append(
                ScoredPlan(
                    plan_id=plan.plan_id,
                    query_id=plan.query_id,
                    strategy=(
                        getattr(plan, "actual_join_strategy", None)
                        or getattr(plan, "strategy", "unknown")
                    ),
                    score=score,
                    normalized_cost=norm_cost,
                    normalized_latency=norm_latency,
                    normalized_accuracy=norm_accuracy,
                    normalized_variance=norm_variance,
                    violation_penalty=self.lambda_mult * violation,
                    estimated_cost=float(metrics.estimated_cost or 0.0),
                    estimated_latency=float(metrics.estimated_latency or 0.0),
                )
            )

        results.sort(key=lambda sp: sp.score)

        return results