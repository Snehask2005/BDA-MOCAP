"""
Analytical (formula-based) cost and latency model.

Defines PlanMetrics — the shared object this module hands off to
the optimizer (Student 3) per the team's integration contract:
"Hridhika -> Student 3: PlanMetrics with estimated cost/latency and
CPU/I/O/shuffle components."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mocap.cost.pricing import (
    PricingConfig,
    cpu_cost,
    io_cost,
    shuffle_cost,
    load_pricing_config,
)


@dataclass
class PlanMetrics:
    """Shared object handed from the cost module to the optimizer."""

    plan_id: str
    query_id: str

    estimated_cost: float
    estimated_latency: float

    cpu_component: float
    io_component: float
    shuffle_component: float

    # Raw resource counters the estimate was built from.
    cpu_seconds: float
    bytes_scanned: float
    bytes_shuffled: float

    actual_cost: Optional[float] = None
    actual_latency: Optional[float] = None


def estimate_plan_metrics(
    plan_id: str,
    query_id: str,
    cpu_seconds: float,
    bytes_scanned: float,
    bytes_shuffled: float,
    wall_clock_seconds: float,
    num_cores: int = 1,
    config: Optional[PricingConfig] = None,
) -> PlanMetrics:
    """
    Build a PlanMetrics object from raw telemetry counters. Cost is
    the sum of CPU + I/O + shuffle dollar cost; latency is taken
    directly as the measured wall-clock time.
    """
    cfg = config or load_pricing_config()

    c_cpu = cpu_cost(cpu_seconds, num_cores, cfg)
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


def attach_actuals(metrics: PlanMetrics, actual_cost: float, actual_latency: float) -> PlanMetrics:
    """Record ground-truth cost/latency once the plan has executed."""
    metrics.actual_cost = actual_cost
    metrics.actual_latency = actual_latency
    return metrics
