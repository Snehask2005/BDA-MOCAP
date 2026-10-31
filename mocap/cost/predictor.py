"""
Pre-execution cost and latency predictor for MOCAP.

The predictor uses structural features extracted from a Spark physical plan
together with Catalyst statistics describing the workload size.

It does not execute candidate queries.

The default predictor is a transparent structural baseline. An optional
learned calibration model can correct the predicted monetary cost using
independent execution telemetry collected by the calibration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mocap.cost.analytical import estimate_plan_metrics
from mocap.cost.learned import CalibrationModel, calibrate
from mocap.cost.pricing import PricingConfig, load_pricing_config
from mocap.cost.statistics_model import PlanStatistics
from mocap.interfaces import PlanMetrics
from mocap.plans.representation import CandidatePlan


@dataclass(frozen=True)
class PredictorConfig:
    """
    Transparent structural prediction parameters.

    These values define the initial planning prior. They are NOT presented
    as empirically learned coefficients.

    Monetary conversion is handled separately by PricingConfig.
    """

    base_cpu_seconds: float = 0.25
    base_scan_bytes: float = 64.0 * 1024 * 1024

    cpu_per_join: float = 0.35
    cpu_per_exchange: float = 0.20
    cpu_per_broadcast_exchange: float = 0.08
    cpu_per_sort: float = 0.15
    cpu_per_aggregate: float = 0.12
    cpu_per_filter: float = 0.03
    cpu_per_scan: float = 0.05
    cpu_per_plan_depth: float = 0.02

    latency_per_cpu_second: float = 1.0
    latency_per_exchange: float = 0.15
    latency_per_sort: float = 0.10
    latency_per_aggregate: float = 0.05

    broadcast_factor: float = 0.75
    shuffle_hash_factor: float = 1.10
    sort_merge_factor: float = 1.00
    nested_loop_factor: float = 1.50

    # Statistics-aware scaling.
    #
    # The predictor treats base_scan_bytes as the workload size at which
    # structural coefficients represent one unit of data-processing work.
    # The floor prevents tiny inputs from eliminating fixed data-processing
    # work, while the ceiling prevents pathological statistics from exploding
    # the prediction.
    min_workload_scale: float = 0.25
    max_workload_scale: float = 1000.0

    # Candidate-specific intermediate workload.
    #
    # This captures the size of the largest reliable join input/intermediate
    # observed by Catalyst. It is deliberately converted into a bounded
    # scale so pathological statistics cannot dominate prediction.
    intermediate_workload_weight: float = 0.20


def _strategy_factor(
    plan: CandidatePlan,
    config: PredictorConfig,
) -> float:
    strategy = (
        plan.actual_join_strategy
        or plan.strategy
        or ""
    ).lower()

    if "broadcast" in strategy:
        return config.broadcast_factor

    if "shuffle" in strategy:
        return config.shuffle_hash_factor

    if "sortmerge" in strategy or "sort_merge" in strategy:
        return config.sort_merge_factor

    if "nestedloop" in strategy or "nested_loop" in strategy:
        return config.nested_loop_factor

    return 1.0


def _workload_scale(
    statistics: Optional[PlanStatistics],
    config: PredictorConfig,
) -> float:
    """
    Convert Catalyst input statistics into a bounded workload scale.

    A workload equal to base_scan_bytes has scale 1.0.

    Missing/unknown statistics preserve the previous structural baseline
    through a scale of 1.0.
    """

    if statistics is None:
        return 1.0

    input_bytes = statistics.input_bytes

    if input_bytes is None or input_bytes <= 0:
        return 1.0

    raw_scale = input_bytes / config.base_scan_bytes

    return min(
        max(raw_scale, config.min_workload_scale),
        config.max_workload_scale,
    )

def _intermediate_workload_scale(
    statistics: Optional[PlanStatistics],
    config: PredictorConfig,
) -> float:
    """
    Convert intermediate join statistics into a bounded workload scale.

    Missing/unknown statistics contribute no additional intermediate-work
    signal.

    The scale is normalized against base_scan_bytes and capped to avoid
    pathological Catalyst estimates dominating the predictor.
    """

    if statistics is None:
        return 0.0

    intermediate_bytes = statistics.intermediate_bytes

    if intermediate_bytes is None or intermediate_bytes <= 0:
        return 0.0

    raw_scale = (
        intermediate_bytes
        / config.base_scan_bytes
    )

    return min(
        max(raw_scale, 0.0),
        config.max_workload_scale,
    )


def _estimate_cpu_seconds(
    plan: CandidatePlan,
    config: PredictorConfig,
    workload_scale: float = 1.0,
    intermediate_scale: float = 0.0,
) -> float:
    """
    Estimate aggregate CPU-seconds.

    Fixed/base overhead remains independent of workload size. Data-dependent
    operator work scales with Catalyst's estimated input size.
    """

    fixed_cpu = config.base_cpu_seconds

    data_dependent_cpu = (
        plan.num_joins * config.cpu_per_join
        + plan.num_exchanges * config.cpu_per_exchange
        + plan.num_broadcast_exchanges
        * config.cpu_per_broadcast_exchange
        + plan.num_sorts * config.cpu_per_sort
        + plan.num_aggregates * config.cpu_per_aggregate
        + plan.num_filters * config.cpu_per_filter
        + plan.num_scans * config.cpu_per_scan
        + plan.plan_depth * config.cpu_per_plan_depth
    )

    intermediate_work = (
        data_dependent_cpu
        * intermediate_scale
        * config.intermediate_workload_weight
    )

    return max(
        (
            fixed_cpu
            + data_dependent_cpu * workload_scale
            + intermediate_work
        )
        * _strategy_factor(plan, config),
        0.0,
    )


def _estimate_scan_bytes(
    plan: CandidatePlan,
    config: PredictorConfig,
    statistics: Optional[PlanStatistics] = None,
) -> float:
    if (
        statistics is not None
        and statistics.input_bytes > 0
    ):
        return statistics.input_bytes

    return config.base_scan_bytes * max(1, plan.num_scans)


def _estimate_shuffle_bytes(
    plan: CandidatePlan,
    config: PredictorConfig,
    statistics: Optional[PlanStatistics] = None,
) -> float:
    """
    Estimate shuffle volume from the workload size.

    Regular exchanges represent full-scale shuffle work while broadcast
    exchanges receive a smaller relative shuffle footprint.
    """

    regular_exchanges = max(
        0,
        plan.num_exchanges - plan.num_broadcast_exchanges,
    )

    broadcast_exchanges = plan.num_broadcast_exchanges

    if statistics is not None and statistics.input_bytes > 0:
        input_bytes = statistics.input_bytes
    else:
        input_bytes = config.base_scan_bytes * max(1, plan.num_scans)

    return input_bytes * (
        regular_exchanges
        + 0.25 * broadcast_exchanges
    )


def _estimate_latency(
    plan: CandidatePlan,
    cpu_seconds: float,
    config: PredictorConfig,
) -> float:
    latency = (
        cpu_seconds * config.latency_per_cpu_second
        + plan.num_exchanges * config.latency_per_exchange
        + plan.num_sorts * config.latency_per_sort
        + plan.num_aggregates * config.latency_per_aggregate
    )

    return max(latency, 0.001)


def _estimate_resources(
    plan: CandidatePlan,
    config: PredictorConfig,
    statistics: Optional[PlanStatistics] = None,
) -> tuple[float, float, float, float]:
    workload_scale = _workload_scale(
        statistics,
        config,
    )

    intermediate_scale = _intermediate_workload_scale(
        statistics,
        config,
    )

    cpu_seconds = _estimate_cpu_seconds(
        plan,
        config,
        workload_scale,
        intermediate_scale,
    )

    bytes_scanned = _estimate_scan_bytes(
        plan,
        config,
        statistics,
    )

    bytes_shuffled = _estimate_shuffle_bytes(
        plan,
        config,
        statistics,
    )

    latency = _estimate_latency(
        plan,
        cpu_seconds,
        config,
    )

    return (
        cpu_seconds,
        bytes_scanned,
        bytes_shuffled,
        latency,
    )


def _apply_calibration(
    metrics: PlanMetrics,
    calibration_model: Optional[CalibrationModel],
) -> PlanMetrics:
    """
    Apply learned cost calibration while preserving the predicted
    resource components.

    The learned model is allowed to correct monetary cost only.
    Latency and resource predictions remain unchanged.
    """

    calibrated_cost = calibrate(
        metrics,
        model=calibration_model,
    )

    calibrated_cost = max(0.0, calibrated_cost)

    return PlanMetrics(
        plan_id=metrics.plan_id,
        query_id=metrics.query_id,
        estimated_cost=calibrated_cost,
        estimated_latency=metrics.estimated_latency,
        cpu_component=metrics.cpu_component,
        io_component=metrics.io_component,
        shuffle_component=metrics.shuffle_component,
        cpu_seconds=metrics.cpu_seconds,
        bytes_scanned=metrics.bytes_scanned,
        bytes_shuffled=metrics.bytes_shuffled,
        actual_cost=metrics.actual_cost,
        actual_latency=metrics.actual_latency,
    )


def predict_plan_metrics(
    plan: CandidatePlan,
    config: Optional[PredictorConfig] = None,
    pricing_config: Optional[PricingConfig] = None,
    statistics: Optional[PlanStatistics] = None,
    calibration_model: Optional[CalibrationModel] = None,
) -> PlanMetrics:
    """
    Produce pre-execution predictions for one candidate plan.

    No Spark action or candidate query execution occurs here.

    Catalyst statistics influence data-dependent CPU and shuffle estimates.
    """

    predictor_config = (
        config
        if config is not None
        else PredictorConfig()
    )

    (
        cpu_seconds,
        bytes_scanned,
        bytes_shuffled,
        latency,
    ) = _estimate_resources(
        plan,
        predictor_config,
        statistics,
    )

    pricing = (
        pricing_config
        if pricing_config is not None
        else load_pricing_config()
    )

    metrics = estimate_plan_metrics(
        plan_id=plan.plan_id,
        query_id=plan.query_id,
        cpu_seconds=cpu_seconds,
        bytes_scanned=bytes_scanned,
        bytes_shuffled=bytes_shuffled,
        wall_clock_seconds=latency,
        config=pricing,
    )

    metrics = _apply_calibration(
        metrics,
        calibration_model,
    )

    return metrics


def predict_candidates(
    candidates: list[CandidatePlan],
    config: Optional[PredictorConfig] = None,
    pricing_config: Optional[PricingConfig] = None,
    statistics: Optional[PlanStatistics] = None,
    calibration_model: Optional[CalibrationModel] = None,
) -> dict[str, PlanMetrics]:
    """
    Predict every candidate without executing any candidate query.
    """

    return {
        candidate.plan_id: predict_plan_metrics(
            candidate,
            config=config,
            pricing_config=pricing_config,
            statistics=statistics,
            calibration_model=calibration_model,
        )
        for candidate in candidates
    }
