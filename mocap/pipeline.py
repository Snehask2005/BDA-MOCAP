"""
MOCAP end-to-end planning pipeline.

Planning path:

    QueryRequest
        -> Candidate generation
        -> Pre-execution prediction
        -> Hard constraint filtering
        -> Pareto filtering
        -> Lagrangian ranking
        -> SelectedPlan

Execution is deliberately separate from planning. Runtime observations
are collected after the selected plan executes and are used for calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates
from mocap.plans.representation import CandidatePlan

from mocap.cost.predictor import (
    PredictorConfig,
    predict_candidates,
)
from mocap.cost.learned import CalibrationModel
from mocap.cost.analytical import PlanMetrics

from mocap.optimizer.constraints import ConstraintsFilter
from mocap.optimizer.pareto import ParetoSelector
from mocap.optimizer.lagrangian import LagrangianRanker

from mocap.interfaces import SelectedPlan

from mocap.cost.statistics import extract_plan_statistics


@dataclass
class MOCAPResult:
    """Complete output of one MOCAP planning run."""

    query: QueryRequest
    candidates: List[CandidatePlan]
    metrics: Dict[str, PlanMetrics]

    feasible: List[tuple]
    pareto_front: List[tuple]

    selected_plan: Optional[SelectedPlan]
    selected_score: Optional[float] = None

    failure_reason: Optional[str] = None


class MOCAPPipeline:
    """
    End-to-end MOCAP planner.

    Candidate plans are evaluated using a pre-execution predictor.
    Only the selected plan should subsequently be executed by the
    execution layer.
    """

    def __init__(
        self,
        spark: SparkSession,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.0,
        delta: float = 0.0,
        lambda_mult: float = 0.0,
        num_cores: int = 1,
        predictor_config: Optional[PredictorConfig] = None,
        calibration_model: Optional[CalibrationModel] = None,
    ):
        self.spark = spark
        self.num_cores = num_cores
        self.predictor_config = predictor_config or PredictorConfig()
        self.calibration_model = calibration_model

        self.ranker = LagrangianRanker(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            lambda_mult=lambda_mult,
        )

    def generate_candidates(
        self,
        query: QueryRequest,
    ) -> List[CandidatePlan]:
        """Generate physical-plan candidates using Spark/Catalyst."""

        return generate_join_candidates(
            query,
            self.spark,
        )

    def estimate_candidates(
        self,
        candidates: List[CandidatePlan],
    ) -> Dict[str, PlanMetrics]:
        """
        Predict candidate metrics BEFORE execution.

        Each candidate is analyzed independently so that candidate-specific
        Catalyst statistics, including intermediate join sizes, influence
        its prediction.

        No candidate query is executed here.
        """

        metrics: Dict[str, PlanMetrics] = {}

        for candidate in candidates:
            # Construct and optimize this candidate's Spark plan.
            # This does not execute the query.
            dataframe = self.spark.sql(
                candidate.sql
            )

            # Extract statistics specific to this candidate.
            statistics = extract_plan_statistics(
                dataframe
            )

            # Predict cost/latency for this candidate using its own
            # Catalyst statistics.
            candidate_metrics = predict_candidates(
                [candidate],
                config=self.predictor_config,
                statistics=statistics,
                calibration_model=self.calibration_model,
            )

            metrics.update(candidate_metrics)

        return metrics

    def select_plan(
        self,
        query: QueryRequest,
        candidates: List[CandidatePlan],
        metrics: Dict[str, PlanMetrics],
    ):
        """Apply hard constraints, Pareto filtering and ranking."""

        # ---------------------------------------------------------
        # 1. Hard budget/deadline filtering
        # ---------------------------------------------------------

        constraint_filter = ConstraintsFilter(
            budget=query.budget,
            deadline=query.deadline,
        )

        feasible = constraint_filter.filter_feasible_plans(
            candidates,
            metrics,
        )

        if not feasible:
            return (
                None,
                feasible,
                [],
                None,
                "No candidate satisfies the constraints.",
            )

        # ---------------------------------------------------------
        # 2. Pareto filtering
        # ---------------------------------------------------------

        pareto_front = ParetoSelector.get_pareto_optimal(
            feasible
        )

        if not pareto_front:
            return (
                None,
                feasible,
                pareto_front,
                None,
                "Pareto frontier is empty.",
            )

        # ---------------------------------------------------------
        # 3. Lagrangian ranking
        # ---------------------------------------------------------

        ranked = self.ranker.rank_candidates(
            pareto_front,
            query.budget,
        )

        if not ranked:
            return (
                None,
                feasible,
                pareto_front,
                None,
                "No candidate could be ranked.",
            )

        best_plan, best_metrics, best_score = ranked[0]

        # ---------------------------------------------------------
        # 4. Construct SelectedPlan
        # ---------------------------------------------------------

        selected = SelectedPlan(
            plan_id=best_plan.plan_id,
            query_id=query.query_id,
            selected_strategy=(
                best_plan.actual_join_strategy
                or best_plan.strategy
            ),
            expected_cost=best_metrics.estimated_cost,
            expected_latency=best_metrics.estimated_latency,
            selection_reason=(
                "Selected by MOCAP: feasible under budget/deadline, "
                "Pareto-optimal, and lowest normalized Lagrangian score."
            ),
            physical_plan_sql=best_plan.sql,
            budget=query.budget,
            deadline=query.deadline,
            accuracy_tolerance=query.accuracy_tolerance,
        )

        return (
            selected,
            feasible,
            pareto_front,
            best_score,
            None,
        )

    def run(
        self,
        query: QueryRequest,
    ) -> MOCAPResult:
        """
        Generate, predict, constrain and select a physical plan.

        This method performs planning only. It does not execute the
        selected query.
        """

        # ---------------------------------------------------------
        # 1. Candidate generation
        # ---------------------------------------------------------

        candidates = self.generate_candidates(
            query
        )

        if not candidates:
            return MOCAPResult(
                query=query,
                candidates=[],
                metrics={},
                feasible=[],
                pareto_front=[],
                selected_plan=None,
                failure_reason=(
                    "No candidate plans were generated."
                ),
            )

        # ---------------------------------------------------------
        # 2. Candidate-specific prediction
        # ---------------------------------------------------------

        # Each candidate is independently analyzed using its own
        # Catalyst statistics. This constructs/optimizes plans but
        # does not execute them.
        metrics = self.estimate_candidates(
            candidates
        )

        # ---------------------------------------------------------
        # 3. Constraint filtering + Pareto + ranking
        # ---------------------------------------------------------

        (
            selected,
            feasible,
            pareto_front,
            score,
            failure_reason,
        ) = self.select_plan(
            query=query,
            candidates=candidates,
            metrics=metrics,
        )

        # ---------------------------------------------------------
        # 4. Return complete planning result
        # ---------------------------------------------------------

        return MOCAPResult(
            query=query,
            candidates=candidates,
            metrics=metrics,
            feasible=feasible,
            pareto_front=pareto_front,
            selected_plan=selected,
            selected_score=score,
            failure_reason=failure_reason,
        )


def run_mocap(
    query: QueryRequest,
    spark: SparkSession,
    **kwargs,
) -> MOCAPResult:
    """
    Convenience function for running MOCAP.
    """

    pipeline = MOCAPPipeline(
        spark=spark,
        **kwargs,
    )

    return pipeline.run(
        query
    )