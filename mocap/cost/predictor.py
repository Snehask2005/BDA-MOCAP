"""
Pre-execution cost and latency predictor for MOCAP.

This module predicts resource usage from candidate physical-plan features.
It does NOT execute the candidate.

The initial predictor is deliberately transparent and deterministic.
Its predictions can later be calibrated using ExecutionTelemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mocap.cost.analytical import estimate_plan_metrics
from mocap.cost.pricing import PricingConfig, load_pricing_config
from mocap.interfaces import PlanMetrics
from mocap.plans.representation import CandidatePlan


@dataclass(frozen=True)
class PredictorConfig:
    """
    Configuration for the initial analytical predictor.

    These are intentionally conservative heuristic coefficients.
    They are not claimed to represent a cloud provider's billing model.
    Calibration will learn corrections from observed executions.
    """

    base_cpu_seconds: float = 0.50
    base_scan_bytes: float = 64.0 * 1024 * 1024

    cpu_per_join: float = 0.35
    cpu_per_exchange: float = 0.20
    cpu_per_sort: float = 0.25
    cpu_per_broadcast_exchange: float = 0.10

    latency_per_cpu_second: float = 1.0
    latency_per_exchange: float = 0.15
    latency_per_sort: float = 0.20

    broadcast_cpu_factor: float = 0.75
    shuffle_cpu_factor: float = 1.10
    sort_merge_cpu_factor: float = 1.00


def _join_strategy_factor(plan: CandidatePlan, config: PredictorConfig) -> float:
    """Return a deterministic relative CPU factor for the join strategy."""
    strategy = (
        plan.actual_join_strategy
        or plan.strategy
        or ""
    ).lower()

    if "broadcast" in strategy:
        return config.broadcast_cpu_factor

    if "shuffle" in strategy:
        return config.shuffle_cpu_factor

    if "sort" in strategy or "merge" in strategy:
        return config.sort_merge_cpu_factor

    return 1.0


def _estimate_resources(
    plan: CandidatePlan,
    config: PredictorConfig,
) -> tuple[float, float, float, float]:
    """
    Estimate CPU-seconds, scanned bytes, shuffled bytes and latency.

    This is a plan-feature model, not an execution measurement.
    """
    joins = max(0, plan.num_joins)
    exchanges = max(0, plan.num_exchanges)
    broadcasts = max(0, plan.num_broadcast_exchanges)
    sorts = max(0, plan.num_sorts)

    cpu_seconds = (
        config.base_cpu_seconds
        + joins * config.cpu_per_join
        + exchanges * config.cpu_per_exchange
        + broadcasts * config.cpu_per_broadcast_exchange
        + sorts * config.cpu_per_sort
    )

    strategy_factor = _join_strategy_factor(plan, config)
    cpu_seconds *= strategy_factor

    # Shuffle-heavy plans are expected to move substantially more data.
    # The estimate is a relative planning signal until calibrated.
    bytes_scanned = config.base_scan_bytes * max(1, joins + 1)

    bytes_shuffled = (
        config.base_scan_bytes
        * (
            exchanges
            + broadcasts * 0.25
        )
    )

    latency = (
        cpu_seconds * config.latency_per_cpu_second
        + exchanges * config.latency_per_exchange
        + sorts * config.latency_per_sort
    )

    return (
        cpu_seconds,
        bytes_scanned,
        bytes_shuffled,
        max(latency, 0.001),
    )


def predict_plan_metrics(
    plan: CandidatePlan,
    config: Optional[PredictorConfig] = None,
    pricing_config: Optional[PricingConfig] = None,
) -> PlanMetrics:
    """
    Produce pre-execution PlanMetrics for a candidate.
    """
    predictor_config = config or PredictorConfig()

    (
        cpu_seconds,
        bytes_scanned,
        bytes_shuffled,
        latency,
    ) = _estimate_resources(
        plan,
        predictor_config,
    )

    return estimate_plan_metrics(
        plan_id=plan.plan_id,
        query_id=plan.query_id,
        cpu_seconds=cpu_seconds,
        bytes_scanned=bytes_scanned,
        bytes_shuffled=bytes_shuffled,
        wall_clock_seconds=latency,
        config=pricing_config or load_pricing_config(),
    )


def predict_candidates(
    candidates: list[CandidatePlan],
    config: Optional[PredictorConfig] = None,
    pricing_config: Optional[PricingConfig] = None,
) -> dict[str, PlanMetrics]:
    """Predict metrics for every candidate without executing them."""
    return {
        candidate.plan_id: predict_plan_metrics(
            candidate,
            config=config,
            pricing_config=pricing_config,
        )
        for candidate in candidates
    }