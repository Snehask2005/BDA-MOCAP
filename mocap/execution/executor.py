"""
Student 4 - Day 1-2 deliverable: Spark execution wrapper.

Wraps spark.sql() execution of a SelectedPlan, times it, pulls whatever
runtime metrics Spark exposes cheaply, and writes a standardized
JSON-lines execution log (query_id, plan_id, timing, runtime metrics --
Section 8 / Section 16 of the plan).

Written to degrade gracefully with no live Spark cluster attached, so it
can be developed and unit-tested locally before the team's shared
Spark/TPC-H environment is up (Sync 1, end of Day 2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mocap.interfaces import ExecutionResult, SelectedPlan

try:
    from pyspark.sql import SparkSession, DataFrame
except ImportError:  # pragma: no cover - lets you develop without pyspark installed
    SparkSession = None
    DataFrame = None


DEFAULT_LOG_PATH = Path("experiments/results/execution_log.jsonl")


@dataclass
class ExecutionLogEntry:
    query_id: str
    plan_id: str
    started_at: float
    ended_at: float
    duration_s: float
    metrics: Dict[str, Any]
    mode: str


class SparkExecutor:
    """Executes a SelectedPlan on Spark and returns a standardized ExecutionResult."""

    def __init__(
        self,
        spark: Optional["SparkSession"] = None,
        log_path: Path = DEFAULT_LOG_PATH,
        cost_per_second: float = 0.0002,  # placeholder $/s -- replace with Hridhika's pricing.py
    ):
        self.spark = spark
        self.log_path = log_path
        self.cost_per_second = cost_per_second
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- public API -------------------------------------------------------

    def execute(
        self,
        plan: SelectedPlan,
        mode: str = "strict",
        sample_fraction: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Run the plan's SQL, collect runtime metrics, log the run, and return
        a standardized ExecutionResult. `sample_fraction`, if given, applies
        row-level sampling before the triggering action -- used by
        adaptive.py's Degraded Mode.
        """
        if self.spark is None or plan.physical_plan_sql is None:
            return self._simulate(plan, mode, sample_fraction)

        start = time.time()
        df: "DataFrame" = self.spark.sql(plan.physical_plan_sql)
        if sample_fraction is not None:
            df = df.sample(withReplacement=False, fraction=sample_fraction)
        row_count = df.count()  # forces execution; swap for .write / .collect as needed
        end = time.time()

        metrics = self._collect_task_metrics()
        metrics["wall_time_s"] = end - start
        metrics["row_count"] = row_count
        if sample_fraction is not None:
            metrics["sample_fraction"] = sample_fraction

        actual_latency = end - start
        actual_cost = self._estimate_actual_cost(actual_latency)

        self._write_log(plan, start, end, metrics, mode)

        return ExecutionResult(
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            actual_cost=actual_cost,
            actual_latency=actual_latency,
            runtime_metrics=metrics,
            result_summary=f"{row_count} rows",
            execution_mode=mode,
        )

    # ---- helpers ------------------------------------------------------------

    def _collect_task_metrics(self) -> Dict[str, Any]:
        """
        Cheap metrics via statusTracker(). For real per-stage CPU/IO/shuffle
        breakdowns, register a SparkListener (see monitor.py's polling
        approach, or extend this with a proper listener) and merge its
        output in here before handing off to Hridhika's cost model.
        """
        metrics: Dict[str, Any] = {}
        if self.spark is not None:
            tracker = self.spark.sparkContext.statusTracker()
            metrics["active_jobs_at_capture"] = len(tracker.getActiveJobIds())
        return metrics

    def _estimate_actual_cost(self, latency: float) -> float:
        # Placeholder cost function; replace with Hridhika's pricing.py once shared (Sync 2).
        return round(latency * self.cost_per_second, 6)

    def _simulate(
        self, plan: SelectedPlan, mode: str, sample_fraction: Optional[float]
    ) -> ExecutionResult:
        """No SparkSession attached yet -> deterministic stub for local dev/tests."""
        start = time.time()
        time.sleep(0.01)
        end = time.time()
        metrics: Dict[str, Any] = {"wall_time_s": end - start, "simulated": True}
        if sample_fraction is not None:
            metrics["sample_fraction"] = sample_fraction
        self._write_log(plan, start, end, metrics, mode)
        return ExecutionResult(
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            actual_cost=plan.expected_cost * (sample_fraction or 1.0),
            actual_latency=end - start,
            runtime_metrics=metrics,
            result_summary="simulated run (no SparkSession attached)",
            execution_mode=mode,
        )

    def _write_log(self, plan: SelectedPlan, start, end, metrics, mode) -> None:
        entry = ExecutionLogEntry(
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            started_at=start,
            ended_at=end,
            duration_s=end - start,
            metrics=metrics,
            mode=mode,
        )
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry.__dict__) + "\n")
