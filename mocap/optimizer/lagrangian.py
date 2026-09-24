"""
Lagrangian multi-objective ranking for MOCAP.

Hard constraints are applied before this ranker. Therefore the ranker
operates only on feasible plans.

Cost and latency are min-max normalized across the candidate set so that
the two objectives are comparable despite having different units.
"""

from __future__ import annotations


class LagrangianRanker:
    """
    Rank feasible plans using:

        J = alpha * L_hat
          + beta  * C_hat
          + gamma * A_hat
          + delta * V_hat
          + lambda * violation

    Since hard feasibility filtering happens first, the violation term is
    normally zero.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.0,
        delta: float = 0.0,
        lambda_mult: float = 0.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_mult = lambda_mult

    @staticmethod
    def _normalize(value: float, minimum: float, maximum: float) -> float:
        if maximum <= minimum:
            return 0.0
        return (value - minimum) / (maximum - minimum)

    def compute_score(
        self,
        plan,
        metrics,
        budget,
        cost_range=None,
        latency_range=None,
    ):
        cost = float(metrics.estimated_cost or 0.0)
        latency = float(metrics.estimated_latency or 0.0)

        if cost_range is None:
            cost_range = (cost, cost)

        if latency_range is None:
            latency_range = (latency, latency)

        normalized_cost = self._normalize(
            cost,
            cost_range[0],
            cost_range[1],
        )

        normalized_latency = self._normalize(
            latency,
            latency_range[0],
            latency_range[1],
        )

        # Accuracy/variance are currently optional extensions.
        accuracy = float(getattr(metrics, "accuracy_tolerance", 0.0) or 0.0)
        variance = float(getattr(metrics, "variance", 0.0) or 0.0)

        score = (
            self.alpha * normalized_latency
            + self.beta * normalized_cost
            + self.gamma * accuracy
            + self.delta * variance
        )

        violation = max(0.0, cost - budget)

        score += self.lambda_mult * violation

        return score

    def rank_candidates(self, feasible_plans, budget):
        if not feasible_plans:
            return []

        costs = [
            float(metrics.estimated_cost or 0.0)
            for _, metrics in feasible_plans
        ]

        latencies = [
            float(metrics.estimated_latency or 0.0)
            for _, metrics in feasible_plans
        ]

        cost_range = (min(costs), max(costs))
        latency_range = (min(latencies), max(latencies))

        scored = []

        for plan, metrics in feasible_plans:
            score = self.compute_score(
                plan,
                metrics,
                budget,
                cost_range=cost_range,
                latency_range=latency_range,
            )

            scored.append(
                (plan, metrics, score)
            )

        scored.sort(key=lambda item: item[2])

        return scored