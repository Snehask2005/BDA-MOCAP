"""
Spark execution telemetry collector.

This module executes a candidate plan and collects observed runtime
measurements. It deliberately does NOT present those measurements as
pre-execution predictions.

The resulting ExecutionTelemetry objects can be used as labeled data
for calibration.
"""

from __future__ import annotations

import time
from typing import Optional

from pyspark.sql import SparkSession

from mocap.cost.pricing import PricingConfig, load_pricing_config
from mocap.interfaces import ExecutionTelemetry
from mocap.plans.representation import CandidatePlan


_SCAN_BYTES_KEYS = (
    "size of files read",
)

_SHUFFLE_BYTES_KEYS = (
    "shuffle bytes written",
    "remote bytes read",
    "local bytes read",
)


def _walk_sql_metrics(java_plan) -> dict:
    """Recursively collect Spark SQL metric values."""
    values: dict = {}

    metrics_map = java_plan.metrics()
    it = metrics_map.iterator()

    while it.hasNext():
        entry = it.next()
        name = str(entry._1()).lower()

        try:
            value = entry._2().value()
        except Exception:
            continue

        values[name] = values.get(name, 0) + value

    children_it = java_plan.children().iterator()

    while children_it.hasNext():
        child = children_it.next()

        for name, value in _walk_sql_metrics(child).items():
            values[name] = values.get(name, 0) + value

    return values


def _bucket_bytes(metric_values: dict) -> tuple[float, float]:
    """Extract scan and shuffle byte counters from Spark metrics."""
    bytes_scanned = 0.0
    bytes_shuffled = 0.0

    for name, value in metric_values.items():
        if any(key in name for key in _SCAN_BYTES_KEYS):
            bytes_scanned += value

        elif any(key in name for key in _SHUFFLE_BYTES_KEYS):
            bytes_shuffled += value

    return bytes_scanned, bytes_shuffled


def _calculate_observed_cost(
    cpu_seconds: float,
    bytes_scanned: float,
    bytes_shuffled: float,
    config: PricingConfig,
) -> float:
    """Calculate monetary cost from observed resource consumption."""
    from mocap.cost.pricing import total_cost

    return total_cost(
        cpu_seconds=cpu_seconds,
        bytes_scanned=bytes_scanned,
        bytes_shuffled=bytes_shuffled,
        config=config,
    )


def collect_execution_telemetry(
    candidate: CandidatePlan,
    spark: SparkSession,
    num_cores: int = 1,
    config: Optional[PricingConfig] = None,
) -> ExecutionTelemetry:
    """
    Execute a candidate and collect ground-truth runtime telemetry.

    This is an observation step, not a prediction step.
    """
    if num_cores <= 0:
        raise ValueError("num_cores must be positive")

    cfg = config or load_pricing_config()

    df = spark.sql(candidate.sql)

    start = time.time()
    df.collect()
    elapsed = time.time() - start

    java_plan = df._jdf.queryExecution().executedPlan()

    metric_values = _walk_sql_metrics(java_plan)

    bytes_scanned, bytes_shuffled = _bucket_bytes(metric_values)

    # Approximation until real executor CPU telemetry is available.
    #
    # elapsed wall-clock seconds × active cores is interpreted as
    # aggregate CPU-seconds.
    cpu_seconds = elapsed * num_cores

    actual_cost = _calculate_observed_cost(
        cpu_seconds=cpu_seconds,
        bytes_scanned=bytes_scanned,
        bytes_shuffled=bytes_shuffled,
        config=cfg,
    )

    return ExecutionTelemetry(
        plan_id=candidate.plan_id,
        query_id=candidate.query_id,
        actual_cost=actual_cost,
        actual_latency=elapsed,
        cpu_seconds=cpu_seconds,
        bytes_scanned=bytes_scanned,
        bytes_shuffled=bytes_shuffled,
        runtime_metrics=metric_values,
    )


# Backward-compatible alias for existing code.
estimate_from_execution = collect_execution_telemetry