"""
Unified MOCAP optimizer entry point.

Roshini (Member 3) — Core MOCAP Optimizer

The MOCAPSelector encapsulates the three-stage selection pipeline:

    Stage 1 — Hard constraint filtering (budget + deadline)
               [ConstraintsFilter]

    Stage 2 — Pareto-dominance filtering
               [ParetoSelector]

    Stage 3 — Lagrangian ranking
               [LagrangianRanker]

It returns a SelectionResult that bundles the chosen plan, diagnostic
metadata, and all intermediate stage outputs for experiment logging.

An ablation configuration (AblationConfig) can be injected to skip or
replace individual stages for ablation experiments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mocap.interfaces import PlanMetrics, SelectedPlan
from mocap.optimizer.ablations import AblationConfig
from mocap.optimizer.constraints import ConstraintsFilter
from mocap.optimizer.lagrangian import LagrangianRanker, ScoredPlan
from mocap.optimizer.pareto import ParetoSelector


@dataclass
class SelectionResult:
    """
    Complete output of one MOCAP optimizer selection run.

    Suitable for experiment logs and paper tables.
    """

    query_id: str
    budget: float
    deadline: Optional[float]

    # Counts at each stage
    candidate_count: int
    feasible_count: int
    pareto_count: int

    # Stage timing (seconds)
    constraint_time_s: float
    pareto_time_s: float
    ranking_time_s: float
    total_time_s: float

    # Selected plan (None if no feasible candidate exists)
    selected_plan: Optional[SelectedPlan]
    best_score: Optional[float]
    ranked_details: List[ScoredPlan] = field(default_factory=list)

    # Intermediate outputs
    infeasibility_summary: Dict = field(default_factory=dict)
    pareto_dominated_count: int = 0
    pareto_equivalent_groups: int = 0

    # Failure diagnostics
    failure_reason: Optional[str] = None
    ablation_variant: Optional[str] = None

    def to_dict(self) -> dict:
        """Flatten to a dict suitable for CSV/JSON logging."""
        return {
            "query_id": self.query_id,
            "budget": self.budget,
            "deadline": self.deadline,
            "candidate_count": self.candidate_count,
            "feasible_count": self.feasible_count,
            "pareto_count": self.pareto_count,
            "constraint_time_s": self.constraint_time_s,
            "pareto_time_s": self.pareto_time_s,
            "ranking_time_s": self.ranking_time_s,
            "total_time_s": self.total_time_s,
            "selected_plan_id": (
                self.selected_plan.plan_id
                if self.selected_plan
                else None
            ),
            "selected_strategy": (
                self.selected_plan.selected_strategy
                if self.selected_plan
                else None
            ),
            "expected_cost": (
                self.selected_plan.expected_cost
                if self.selected_plan
                else None
            ),
            "expected_latency": (
                self.selected_plan.expected_latency
                if self.selected_plan
                else None
            ),
            "best_score": self.best_score,
            "pareto_dominated_count": self.pareto_dominated_count,
            "pareto_equivalent_groups": self.pareto_equivalent_groups,
            "failure_reason": self.failure_reason,
            "ablation_variant": self.ablation_variant,
            **{
                f"infeasible_{k}": v
                for k, v in self.infeasibility_summary.items()
            },
        }


class MOCAPSelector:
    """
    Three-stage MOCAP plan selector.

    Parameters
    ----------
    alpha, beta, gamma, delta, lambda_mult : float
        Lagrangian objective weights.  See LagrangianRanker for semantics.
    ablation : AblationConfig, optional
        When provided, stages can be skipped or reconfigured for ablation.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.0,
        delta: float = 0.0,
        lambda_mult: float = 0.0,
        ablation: Optional[AblationConfig] = None,
    ) -> None:
        if ablation is not None:
            # Ablation overrides individual weights.
            alpha = ablation.alpha
            beta = ablation.beta
            gamma = ablation.gamma
            delta = ablation.delta
            lambda_mult = ablation.lambda_mult

        self.ablation = ablation

        self._ranker = LagrangianRanker(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            lambda_mult=lambda_mult,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        query_id: str,
        candidates,
        metrics: Dict[str, PlanMetrics],
        budget: float,
        deadline: Optional[float] = None,
        accuracy_tolerance: Optional[float] = None,
    ) -> SelectionResult:
        """
        Run the three-stage selection pipeline.

        Parameters
        ----------
        query_id : str
        candidates : list of CandidatePlan
        metrics : dict mapping plan_id -> PlanMetrics
        budget : float
            Hard budget constraint.
        deadline : float, optional
            Hard deadline constraint (seconds).
        accuracy_tolerance : float, optional
            Passed through to SelectedPlan for the execution layer.

        Returns
        -------
        SelectionResult
        """
        t_start = time.perf_counter()

        # Apply ablation override for accuracy tolerance.
        if (
            self.ablation is not None
            and self.ablation.force_accuracy_tolerance is not None
        ):
            accuracy_tolerance = self.ablation.force_accuracy_tolerance

        # ---------------------------------------------------------------
        # Stage 1: Hard constraint filtering
        # ---------------------------------------------------------------
        t0 = time.perf_counter()

        constraint_filter = ConstraintsFilter(
            budget=budget,
            deadline=deadline,
        )

        feasible = constraint_filter.filter_feasible_plans(
            candidates,
            metrics,
        )

        t_constraint = time.perf_counter() - t0

        if not feasible:
            t_total = time.perf_counter() - t_start
            return SelectionResult(
                query_id=query_id,
                budget=budget,
                deadline=deadline,
                candidate_count=len(candidates) if candidates else 0,
                feasible_count=0,
                pareto_count=0,
                constraint_time_s=t_constraint,
                pareto_time_s=0.0,
                ranking_time_s=0.0,
                total_time_s=t_total,
                selected_plan=None,
                best_score=None,
                infeasibility_summary=constraint_filter.infeasibility_summary(),
                failure_reason=constraint_filter.failure_reason(
                    len(candidates) if candidates else 0, 0
                ),
                ablation_variant=(
                    self.ablation.variant.value if self.ablation else None
                ),
            )

        # ---------------------------------------------------------------
        # Stage 2: Pareto filtering (may be ablated)
        # ---------------------------------------------------------------
        t0 = time.perf_counter()

        use_pareto = (
            self.ablation is None or self.ablation.use_pareto
        )

        pareto_selector = ParetoSelector()

        if use_pareto:
            pareto_front = pareto_selector.compute_front(feasible)
        else:
            pareto_front = list(feasible)

        t_pareto = time.perf_counter() - t0

        if not pareto_front:
            t_total = time.perf_counter() - t_start
            return SelectionResult(
                query_id=query_id,
                budget=budget,
                deadline=deadline,
                candidate_count=len(candidates) if candidates else 0,
                feasible_count=len(feasible),
                pareto_count=0,
                constraint_time_s=t_constraint,
                pareto_time_s=t_pareto,
                ranking_time_s=0.0,
                total_time_s=t_total,
                selected_plan=None,
                best_score=None,
                infeasibility_summary=constraint_filter.infeasibility_summary(),
                pareto_dominated_count=len(
                    pareto_selector.dominated_plans
                ),
                failure_reason="Pareto frontier is empty after filtering.",
                ablation_variant=(
                    self.ablation.variant.value if self.ablation else None
                ),
            )

        # ---------------------------------------------------------------
        # Stage 3: Lagrangian ranking (may be ablated)
        # ---------------------------------------------------------------
        t0 = time.perf_counter()

        use_lagrangian = (
            self.ablation is None or self.ablation.use_lagrangian
        )

        if use_lagrangian:
            ranked = self._ranker.rank_candidates(pareto_front, budget)
            ranked_details = self._ranker.rank_with_details(
                pareto_front, budget
            )
        else:
            # No-Lagrangian ablation: select by minimum cost directly.
            ranked = sorted(
                [(p, m, float(m.estimated_cost or 0.0)) for p, m in pareto_front],
                key=lambda x: x[2],
            )
            ranked_details = []

        t_ranking = time.perf_counter() - t0

        if not ranked:
            t_total = time.perf_counter() - t_start
            return SelectionResult(
                query_id=query_id,
                budget=budget,
                deadline=deadline,
                candidate_count=len(candidates) if candidates else 0,
                feasible_count=len(feasible),
                pareto_count=len(pareto_front),
                constraint_time_s=t_constraint,
                pareto_time_s=t_pareto,
                ranking_time_s=t_ranking,
                total_time_s=t_total,
                selected_plan=None,
                best_score=None,
                infeasibility_summary=constraint_filter.infeasibility_summary(),
                pareto_dominated_count=len(pareto_selector.dominated_plans),
                failure_reason="Ranking produced no scored candidates.",
                ablation_variant=(
                    self.ablation.variant.value if self.ablation else None
                ),
            )

        best_plan, best_metrics, best_score = ranked[0]

        # ---------------------------------------------------------------
        # Construct SelectedPlan (contract for Anaswara's executor)
        # ---------------------------------------------------------------
        selected = SelectedPlan(
            plan_id=best_plan.plan_id,
            query_id=query_id,
            selected_strategy=(
                getattr(best_plan, "actual_join_strategy", None)
                or getattr(best_plan, "strategy", "unknown")
            ),
            expected_cost=float(best_metrics.estimated_cost or 0.0),
            expected_latency=float(best_metrics.estimated_latency or 0.0),
            selection_reason=(
                self._selection_reason(
                    use_pareto=use_pareto,
                    use_lagrangian=use_lagrangian,
                    pareto_size=len(pareto_front),
                    feasible_size=len(feasible),
                    ablation=(
                        self.ablation.variant.value if self.ablation else None
                    ),
                )
            ),
            physical_plan_sql=getattr(best_plan, "sql", None),
            budget=budget,
            deadline=deadline,
            accuracy_tolerance=accuracy_tolerance,
        )

        t_total = time.perf_counter() - t_start

        return SelectionResult(
            query_id=query_id,
            budget=budget,
            deadline=deadline,
            candidate_count=len(candidates) if candidates else 0,
            feasible_count=len(feasible),
            pareto_count=len(pareto_front),
            constraint_time_s=t_constraint,
            pareto_time_s=t_pareto,
            ranking_time_s=t_ranking,
            total_time_s=t_total,
            selected_plan=selected,
            best_score=best_score,
            ranked_details=ranked_details,
            infeasibility_summary=constraint_filter.infeasibility_summary(),
            pareto_dominated_count=len(pareto_selector.dominated_plans),
            pareto_equivalent_groups=len(pareto_selector.equivalent_groups),
            failure_reason=None,
            ablation_variant=(
                self.ablation.variant.value if self.ablation else None
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _selection_reason(
        use_pareto: bool,
        use_lagrangian: bool,
        pareto_size: int,
        feasible_size: int,
        ablation: Optional[str],
    ) -> str:
        parts = [
            f"Selected by MOCAP optimizer: "
            f"{feasible_size} feasible → "
            f"{pareto_size} Pareto-optimal"
        ]
        if not use_pareto:
            parts.append("[Pareto disabled]")
        if not use_lagrangian:
            parts.append("[Lagrangian replaced by min-cost]")
        if ablation:
            parts.append(f"[ablation={ablation}]")
        return "; ".join(parts) + "."
