"""
Live cost estimator: runs a CandidatePlan on Spark, extracts
Spark's built-in SQL execution metrics from the executed physical
plan, and returns a PlanMetrics via the analytical model.

Telemetry strategy
-------------------
Rather than a custom SparkListener (implementing the full
SparkListenerInterface is fragile and changes across Spark
versions), this walks the already-populated SQLMetric objects on
the executed physical plan after an action has run. That's a
stable, public API — it's exactly what the Spark UI's SQL tab uses.

Known limitation (documented, not hidden): Spark's SQL metrics
expose I/O and shuffle byte counters but not executor CPU time.
CPU seconds is therefore approximated from wall-clock time for now.
If you need real per-executor CPU time, that requires a proper
SparkListener attached via py4j's callback server — worth doing
later, but verify the exact SparkListenerInterface method set for
pyspark==4.2.0 first, since it differs across major versions.
"""
from __future__ import annotations

import time
from typing import Optional

from pyspark.sql import SparkSession

from mocap.cost.analytical import PlanMetrics, estimate_plan_metrics, attach_actuals
from mocap.cost.pricing import PricingConfig, load_pricing_config
from mocap.plans.representation import CandidatePlan

# Substrings (lowercased) of Spark's built-in SQL metric names, used
# to bucket each metric into "bytes scanned" vs "bytes shuffled".
# Extend these if a candidate's plan uses operators not covered yet
# (check with print(metric_values) inside _walk_sql_metrics).
_SCAN_BYTES_KEYS = ("size of files read",)
_SHUFFLE_BYTES_KEYS = ("shuffle bytes written", "remote bytes read", "local bytes read")


def _walk_sql_metrics(java_plan) -> dict:
    """Recursively collect every named SQLMetric value from a Java SparkPlan node and its children."""
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
    bytes_scanned = 0.0
    bytes_shuffled = 0.0
    for name, value in metric_values.items():
        if any(key in name for key in _SCAN_BYTES_KEYS):
            bytes_scanned += value
        elif any(key in name for key in _SHUFFLE_BYTES_KEYS):
            bytes_shuffled += value
    return bytes_scanned, bytes_shuffled


def estimate_from_execution(
    candidate: CandidatePlan,
    spark: SparkSession,
    num_cores: int = 1,
    config: Optional[PricingConfig] = None,
) -> PlanMetrics:
    """
    Execute `candidate.sql`, collect telemetry from the executed
    plan's SQL metrics, and return a PlanMetrics with both the
    estimate and the actual (measured) cost/latency attached — this
    prototype measures by executing rather than predicting
    pre-execution.
    """
    cfg = config or load_pricing_config()

    df = spark.sql(candidate.sql)
    start = time.time()
    df.collect()
    elapsed = time.time() - start

    java_plan = df._jdf.queryExecution().executedPlan()
    metric_values = _walk_sql_metrics(java_plan)
    bytes_scanned, bytes_shuffled = _bucket_bytes(metric_values)

    # See module docstring: approximated until a verified SparkListener is wired in.
    cpu_seconds = elapsed * num_cores

    metrics = estimate_plan_metrics(
        plan_id=candidate.plan_id,
        query_id=candidate.query_id,
        cpu_seconds=cpu_seconds,
        bytes_scanned=bytes_scanned,
        bytes_shuffled=bytes_shuffled,
        wall_clock_seconds=elapsed,
        num_cores=num_cores,
        config=cfg,
    )
    attach_actuals(metrics, actual_cost=metrics.estimated_cost, actual_latency=elapsed)
    return metrics
