"""
Analytical cost and latency model.

This module converts resource estimates into predicted monetary cost and
latency. It does NOT execute a query.

Runtime observations are represented separately by ExecutionTelemetry.
"""

from __future__ import annotations

from typing import Optional

from mocap.interfaces import PlanMetrics
from mocap.cost.pricing import (
    PricingConfig,
    cpu_cost,
    io_cost,
    shuffle_cost,
    load_pricing_config,
)


def estimate_plan_metrics(
    plan_id: str,
    query_id: str,
    cpu_seconds: float,
    bytes_scanned: float,
    bytes_shuffled: float,
    wall_clock_seconds: float,
    config: Optional[PricingConfig] = None,
) -> PlanMetrics:
    """
    Convert predicted resource usage into predicted cost/latency.

    `cpu_seconds` is aggregate CPU-seconds.
    `wall_clock_seconds` is predicted elapsed time.
    """
    if cpu_seconds < 0:
        raise ValueError("cpu_seconds must be non-negative")

    if bytes_scanned < 0:
        raise ValueError("bytes_scanned must be non-negative")

    if bytes_shuffled < 0:
        raise ValueError("bytes_shuffled must be non-negative")

    if wall_clock_seconds < 0:
        raise ValueError("wall_clock_seconds must be non-negative")

    cfg = config or load_pricing_config()

    c_cpu = cpu_cost(cpu_seconds, cfg)
    c_io = io_cost(bytes_scanned, cfg)
    c_shuffle = shuffle_cost(bytes_shuffled, cfg)

    return PlanMetrics(
        plan_id=plan_id,
        query_id=query_id,
        estimated_cost=c_cpu + c_io + c_shuffle,
        estimated_latency=wall_clock_seconds,
        cpu_component=c_cpu,
        io_component=c_io,
        shuffle_component=c_shuffle,
        cpu_seconds=cpu_seconds,
        bytes_scanned=bytes_scanned,
        bytes_shuffled=bytes_shuffled,
    )


def attach_actuals(
    metrics: PlanMetrics,
    actual_cost: float,
    actual_latency: float,
) -> PlanMetrics:
    """
    Backward-compatible helper for older callers.

    New calibration code should store observations in ExecutionTelemetry
    instead of attaching actuals directly to prediction objects.
    """
    if actual_cost < 0:
        raise ValueError("actual_cost must be non-negative")

    if actual_latency < 0:
        raise ValueError("actual_latency must be non-negative")

    metrics.actual_cost = actual_cost
    metrics.actual_latency = actual_latency
    return metrics