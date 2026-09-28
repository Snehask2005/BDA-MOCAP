"""
Spark execution wrapper for MOCAP.

Executes a SelectedPlan, observes standardized execution telemetry,
derives model-based actual cost using PricingConfig, and writes a
JSON-lines execution log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mocap.cost.estimator import observe_execution_telemetry
from mocap.cost.pricing import PricingConfig, load_pricing_config
from mocap.calibration.service import CalibrationService
from mocap.interfaces import ExecutionResult, PlanMetrics, SelectedPlan

try:
    from pyspark.sql import DataFrame, SparkSession
except ImportError:  # pragma: no cover
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
    """Executes a SelectedPlan on Spark."""

    def __init__(
        self,
        spark: Optional["SparkSession"] = None,
        log_path: Path = DEFAULT_LOG_PATH,
        pricing: Optional[PricingConfig] = None,
        calibration_service: Optional[CalibrationService] = None,
        calibration_dataset_path: Optional[str] = None,
    ):
        self.spark = spark
        self.log_path = log_path
        self.pricing = pricing or load_pricing_config()
        self.calibration_service = calibration_service
        self.calibration_dataset_path = calibration_dataset_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        plan: SelectedPlan,
        mode: str = "strict",
        sample_fraction: Optional[float] = None,
        prediction: Optional[PlanMetrics] = None,
    ) -> ExecutionResult:
        """
        Execute a selected plan and return standardized execution results.

        With Spark attached, the query is executed exactly once. Telemetry
        is then observed from the already-executed DataFrame.

        Without Spark, a deterministic simulation is returned.
        """
        if self.spark is None or plan.physical_plan_sql is None:
            return self._simulate(plan, mode, sample_fraction)

        if sample_fraction is not None and not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")

        start = time.time()

        df: "DataFrame" = self.spark.sql(plan.physical_plan_sql)

        if sample_fraction is not None:
            df = df.sample(
                withReplacement=False,
                fraction=sample_fraction,
            )

        # Trigger the actual Spark execution exactly once.
        row_count = df.count()

        end = time.time()
        elapsed = end - start

        candidate = _candidate_from_selected_plan(plan, df)

        # Observe the execution that has already happened.
        # This does NOT trigger another Spark action.
        telemetry = observe_execution_telemetry(
            candidate=candidate,
            dataframe=df,
            elapsed=elapsed,
            config=self.pricing,
        )

        metrics = dict(telemetry.runtime_metrics)
        metrics["wall_time_s"] = elapsed
        metrics["row_count"] = row_count

        if sample_fraction is not None:
            metrics["sample_fraction"] = sample_fraction

        # Calibration is deliberately optional. Execution produces the
        # ground-truth telemetry; training remains a separate operation.
        if (
            prediction is not None
            and self.calibration_service is not None
        ):
            if self.calibration_dataset_path:
                self.calibration_service.record_to_dataset(
                    prediction=prediction,
                    telemetry=telemetry,
                    csv_path=self.calibration_dataset_path,
                )
            else:
                self.calibration_service.record(
                    prediction=prediction,
                    telemetry=telemetry,
                )

        self._write_log(plan, start, end, metrics, mode)

        return ExecutionResult(
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            actual_cost=telemetry.actual_cost,
            actual_latency=telemetry.actual_latency,
            runtime_metrics=metrics,
            result_summary=f"{row_count} rows",
            execution_mode=mode,
        )

    def _simulate(
        self,
        plan: SelectedPlan,
        mode: str,
        sample_fraction: Optional[float],
    ) -> ExecutionResult:
        """Deterministic simulation when no SparkSession is attached."""
        start = time.time()
        time.sleep(0.01)
        end = time.time()

        metrics: Dict[str, Any] = {
            "wall_time_s": end - start,
            "simulated": True,
        }

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

    def _write_log(
        self,
        plan: SelectedPlan,
        start: float,
        end: float,
        metrics: Dict[str, Any],
        mode: str,
    ) -> None:
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


def _candidate_from_selected_plan(
    plan: SelectedPlan,
    dataframe: "DataFrame",
):
    """
    Build the CandidatePlan required by the shared telemetry estimator.

    Structural features are extracted from the executed physical plan.
    """
    from mocap.plans.representation import (
        CandidatePlan,
        extract_physical_plan_features,
    )

    physical_plan = (
        dataframe._jdf.queryExecution().executedPlan().toString()
    )

    features = extract_physical_plan_features(physical_plan)

    return CandidatePlan(
        query_id=plan.query_id,
        plan_id=plan.plan_id,
        strategy=plan.selected_strategy,
        sql=plan.physical_plan_sql,
        physical_plan=physical_plan,
        fingerprint=plan.plan_id,
        actual_join_strategy=features.actual_join_strategy,
        num_joins=features.num_joins,
        num_broadcast_joins=features.num_broadcast_joins,
        num_shuffle_hash_joins=features.num_shuffle_hash_joins,
        num_sort_merge_joins=features.num_sort_merge_joins,
        num_nested_loop_joins=features.num_nested_loop_joins,
        num_exchanges=features.num_exchanges,
        num_broadcast_exchanges=features.num_broadcast_exchanges,
        num_sorts=features.num_sorts,
        num_aggregates=features.num_aggregates,
        num_filters=features.num_filters,
        num_scans=features.num_scans,
        plan_depth=features.plan_depth,
    )
