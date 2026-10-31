"""
Validate pre-execution predictions against post-execution telemetry.

This is the "Validate analytical predictions against Spark execution
telemetry" checklist item: for each candidate, it produces both a
prediction (mocap.cost.predictor, no execution) and ground-truth
telemetry (mocap.cost.estimator, real execution), then reports how far
apart they were.

Kept deliberately free of any calibration-fitting logic -- this module
only *measures* accuracy; mocap.calibration.* is what learns from it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from pyspark.sql import SparkSession

from mocap.cost.estimator import observe_execution_telemetry
from mocap.cost.pricing import PricingConfig
from mocap.cost.statistics import extract_plan_statistics
from mocap.cost.predictor import PredictorConfig, predict_plan_metrics
from mocap.cost.learned import CalibrationModel
from mocap.interfaces import ExecutionTelemetry, PlanMetrics
from mocap.plans.representation import CandidatePlan


@dataclass(frozen=True)
class PredictionValidationResult:
    """One candidate's prediction paired with what actually happened."""

    plan_id: str
    query_id: str
    strategy: str

    predicted_cost: float
    actual_cost: float
    cost_absolute_error: float
    cost_relative_error: Optional[float]  # None if actual_cost == 0

    predicted_latency: float
    actual_latency: float
    latency_absolute_error: float
    latency_relative_error: Optional[float]

    prediction: PlanMetrics
    telemetry: ExecutionTelemetry


def _relative_error(predicted: float, actual: float) -> Optional[float]:
    if actual == 0:
        return None
    return abs(predicted - actual) / abs(actual)


def compare_prediction_to_telemetry(
    prediction: PlanMetrics,
    telemetry: ExecutionTelemetry,
    strategy: str,
) -> PredictionValidationResult:
    """Pair up one prediction and one telemetry observation and score the gap."""
    if prediction.plan_id != telemetry.plan_id:
        raise ValueError("prediction and telemetry refer to different plans")

    return PredictionValidationResult(
        plan_id=prediction.plan_id,
        query_id=prediction.query_id,
        strategy=strategy,
        predicted_cost=prediction.estimated_cost,
        actual_cost=telemetry.actual_cost,
        cost_absolute_error=abs(prediction.estimated_cost - telemetry.actual_cost),
        cost_relative_error=_relative_error(prediction.estimated_cost, telemetry.actual_cost),
        predicted_latency=prediction.estimated_latency,
        actual_latency=telemetry.actual_latency,
        latency_absolute_error=abs(prediction.estimated_latency - telemetry.actual_latency),
        latency_relative_error=_relative_error(prediction.estimated_latency, telemetry.actual_latency),
        prediction=prediction,
        telemetry=telemetry,
    )


def validate_candidate(
    candidate: CandidatePlan,
    spark: SparkSession,
    predictor_config: Optional[PredictorConfig] = None,
    pricing_config: Optional[PricingConfig] = None,
    calibration_model: Optional[CalibrationModel] = None,
    num_cores: int = 1,
) -> PredictionValidationResult:
    """
    Predict, then execute, then compare -- for exactly one candidate.

    Statistics are extracted from the DataFrame's *optimized* (not yet
    executed) plan, so the prediction genuinely does not see execution
    results before it's made.
    """
    df = spark.sql(candidate.sql)

    statistics = extract_plan_statistics(df)

    prediction = predict_plan_metrics(
        candidate,
        config=predictor_config,
        pricing_config=pricing_config,
        statistics=statistics,
        calibration_model=calibration_model,
    )

    start = time.time()
    df.collect()
    elapsed = time.time() - start

    telemetry = observe_execution_telemetry(
        candidate=candidate,
        dataframe=df,
        elapsed=elapsed,
        num_cores=num_cores,
        config=pricing_config,
    )

    strategy = candidate.actual_join_strategy or candidate.strategy

    return compare_prediction_to_telemetry(prediction, telemetry, strategy)


def validate_candidates(
    candidates: List[CandidatePlan],
    spark: SparkSession,
    predictor_config: Optional[PredictorConfig] = None,
    pricing_config: Optional[PricingConfig] = None,
    calibration_model: Optional[CalibrationModel] = None,
    num_cores: int = 1,
) -> List[PredictionValidationResult]:
    """Validate every candidate in `candidates`, one Spark execution each."""
    return [
        validate_candidate(
            candidate,
            spark,
            predictor_config=predictor_config,
            pricing_config=pricing_config,
            calibration_model=calibration_model,
            num_cores=num_cores,
        )
        for candidate in candidates
    ]


def summarize_validation(results: List[PredictionValidationResult]) -> dict:
    """
    Aggregate error statistics across many validation results -- the
    headline numbers for the "prediction accuracy" paper result
    (MAE/RMSE/correlation for cost and latency).
    """
    if not results:
        raise ValueError("results must not be empty")

    n = len(results)

    cost_errors = [r.cost_absolute_error for r in results]
    latency_errors = [r.latency_absolute_error for r in results]

    def _mae(errors: List[float]) -> float:
        return sum(errors) / len(errors)

    def _rmse(errors: List[float]) -> float:
        return (sum(e ** 2 for e in errors) / len(errors)) ** 0.5

    return {
        "n_candidates": n,
        "cost_mae": _mae(cost_errors),
        "cost_rmse": _rmse(cost_errors),
        "latency_mae": _mae(latency_errors),
        "latency_rmse": _rmse(latency_errors),
    }
