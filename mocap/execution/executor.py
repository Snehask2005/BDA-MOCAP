"""
Spark execution wrapper.

Wraps spark.sql() execution of a SelectedPlan, times it, pulls SQL-metric
telemetry (bytes scanned / shuffled) from the executed physical plan, and
writes a standardized JSON-lines execution log.

Degrades gracefully with no live Spark cluster attached (spark=None), so
it stays unit-testable without a JVM. Set allow_simulation=False for real
experiment runs so a missing SparkSession fails loudly instead of quietly
reporting fake numbers.

Execution guarantees
--------------------
* A plan is executed exactly once. The only fallback (``.count()``) is used
  when the "noop" sink itself is unavailable -- never after a real failure
  or a cancellation.
* A cancelled run is logged with ``actual_cost=None`` so it can never be
  mistaken for a completed observation by the calibration pipeline. The
  partial (wasted) cost is reported separately via ``ExecutorCancelled``
  and ``extra_metrics["partial_cost_estimate"]``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from mocap.interfaces import ExecutionResult, ExecutionTelemetry, SelectedPlan

if TYPE_CHECKING:  # pragma: no cover
    from mocap.calibration.service import CalibrationService

try:
    from pyspark.sql import SparkSession, DataFrame
except ImportError:  # pragma: no cover - lets you develop without pyspark installed
    SparkSession = None
    DataFrame = None

try:
    from mocap.cost.pricing import total_cost as _pricing_total_cost
    from mocap.cost.pricing import load_pricing_config as _load_pricing_config
except ImportError:  # pragma: no cover - keep the executor usable if cost module moves/renames
    _pricing_total_cost = None
    _load_pricing_config = None


DEFAULT_LOG_PATH = Path("experiments/results/execution_log.jsonl")

# Substrings (lowercased) of Spark's built-in SQL metric names, used to
# bucket each metric into "bytes scanned" vs "bytes shuffled". Verify
# against `print(metric_values)` if a candidate uses an operator not
# covered here (this list matches mocap/cost/estimator.py's).
_SCAN_BYTES_KEYS = ("size of files read",)
_SHUFFLE_BYTES_KEYS = ("shuffle bytes written", "remote bytes read", "local bytes read")


@dataclass
class ExecutionLogEntry:
    run_id: str
    logical_run_id: str
    role: str                      # "plan" | "fallback" | "ground_truth"
    query_id: str
    plan_id: str
    mode: str                      # "strict" | "adaptive" | "degraded"
    status: str                    # "ok" | "failed" | "cancelled"
    simulated: bool
    started_at: float
    ended_at: float
    duration_s: float
    budget: Optional[float]
    deadline: Optional[float]
    accuracy_tolerance: Optional[float]
    sample_fraction: Optional[float]
    expected_cost: Optional[float]
    expected_latency: Optional[float]
    actual_cost: Optional[float]
    actual_latency: Optional[float]
    cpu_seconds: Optional[float]
    bytes_scanned: Optional[float]
    bytes_shuffled: Optional[float]
    split: Optional[str]           # "calibration" | "eval" | None
    error: Optional[str] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


class ExecutorCancelled(Exception):
    """
    Raised when a job group tagged by this executor was cancelled mid-run.

    elapsed_s / partial_cost describe the work done before cancellation
    (cost is a wall-clock * cores approximation; I/O and shuffle bytes are
    unknown for a cancelled run).
    """

    def __init__(
        self,
        message: str = "job group cancelled",
        elapsed_s: Optional[float] = None,
        partial_cost: Optional[float] = None,
    ):
        super().__init__(message)
        self.elapsed_s = elapsed_s
        self.partial_cost = partial_cost


class SparkExecutor:
    """Executes a SelectedPlan on Spark and returns a standardized ExecutionResult."""

    def __init__(
        self,
        spark: Optional["SparkSession"] = None,
        log_path: Path = DEFAULT_LOG_PATH,
        num_cores: Optional[int] = None,
        allow_simulation: bool = True,
        pricing: Optional[Any] = None,
        calibration_service: Optional["CalibrationService"] = None,
        calibration_dataset_path: Optional[str] = None,
    ):
        self.calibration_service = calibration_service
        self.calibration_dataset_path = calibration_dataset_path
        self.spark = spark
        self.log_path = Path(log_path)
        self.allow_simulation = allow_simulation
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Shared PricingConfig (same object RuntimeMonitor / AdaptiveController use).
        if pricing is not None:
            self.pricing = pricing
        elif _load_pricing_config is not None:
            self.pricing = _load_pricing_config()
        else:
            self.pricing = None

        if num_cores is not None:
            self.num_cores = num_cores
        elif spark is not None:
            try:
                self.num_cores = max(1, spark.sparkContext.defaultParallelism)
            except Exception:
                self.num_cores = 1
        else:
            self.num_cores = 1

    # ---- public API -------------------------------------------------------

    def execute(
        self,
        plan: SelectedPlan,
        mode: str = "strict",
        sample_fraction: Optional[float] = None,
        job_group: Optional[str] = None,
        role: str = "plan",
        split: Optional[str] = None,
        logical_run_id: Optional[str] = None,
        seed: Optional[int] = None,
        prediction: Optional[Any] = None,
    ) -> ExecutionResult:
        """
        Run the plan's SQL, collect telemetry, log the run, and return a
        standardized ExecutionResult.

        job_group: if given and a real SparkSession is attached, tags this
        thread's Spark actions with sc.setJobGroup(job_group, ...,
        interruptOnCancel=True) so a RuntimeMonitor elsewhere can cancel
        this specific run with sc.cancelJobGroup(job_group). Must be called
        from the thread that will trigger the Spark action.

        seed: seed for row-level sampling (degraded mode) so AQP runs are
        reproducible.

        Raises ExecutorCancelled if the job group was cancelled.
        """
        run_id = uuid.uuid4().hex[:12]
        logical_run_id = logical_run_id or run_id

        if self.spark is None or plan.physical_plan_sql is None:
            if not self.allow_simulation:
                raise RuntimeError(
                    "SparkExecutor.execute() called with no SparkSession attached "
                    "and allow_simulation=False. Real experiment runs must not use "
                    "simulated results -- pass a live SparkSession or set "
                    "allow_simulation=True for local/unit-test development."
                )
            return self._simulate(
                plan, mode, sample_fraction, run_id, logical_run_id, role, split
            )

        # A Spark job group is sticky: once set on a thread, it stays
        # attached to every subsequent action until explicitly changed.
        # Always set a fresh tag -- the caller's job_group if cancellation
        # tracking is wanted, otherwise a throwaway per-call tag -- so a
        # run never silently inherits a previous (possibly already-
        # cancelled) group from an earlier call on the same executor.
        effective_group = job_group or f"mocap-untracked-{run_id}"
        self.spark.sparkContext.setJobGroup(
            effective_group, f"{plan.plan_id}:{mode}", interruptOnCancel=(job_group is not None)
        )

        start = time.time()
        status = "ok"
        error_msg: Optional[str] = None
        metrics: Dict[str, Any] = {}
        bytes_scanned = bytes_shuffled = 0.0
        cpu_seconds = 0.0
        row_count: Optional[int] = None

        try:
            df: "DataFrame" = self.spark.sql(plan.physical_plan_sql)
            if sample_fraction is not None:
                df = df.sample(
                    withReplacement=False, fraction=sample_fraction, seed=seed
                )

            # Force full execution without letting column pruning on a
            # trailing .count() understate the real scan/shuffle cost.
            # .count() is used ONLY if the "noop" sink itself is missing.
            # Any other failure (including cancellation) must propagate:
            # re-running the query after a real failure would execute the
            # plan twice and double-count its cost.
            try:
                df.write.format("noop").mode("overwrite").save()
            except Exception as write_exc:
                if not self._looks_like_missing_sink(str(write_exc)):
                    raise
                row_count = df.count()

            end = time.time()
            elapsed = end - start

            try:
                java_plan = df._jdf.queryExecution().executedPlan()
                metric_values = self._walk_sql_metrics(java_plan)
                bytes_scanned, bytes_shuffled = self._bucket_bytes(metric_values)
            except Exception:
                # A plan lacking the expected Java SQL-metrics API (or a
                # non-Spark DataFrame-like object in tests) shouldn't sink
                # an otherwise-successful run; fall back to zero bytes
                # rather than crashing.
                bytes_scanned, bytes_shuffled = 0.0, 0.0

            # Spark's SQL metrics expose I/O and shuffle bytes but not
            # executor CPU time directly; approximated from wall-clock
            # time as a documented limitation (see estimator.py).
            cpu_seconds = elapsed * self.num_cores

        except Exception as exc:
            end = time.time()
            elapsed = end - start
            msg = str(exc)

            if self._looks_like_cancellation(msg):
                partial_cost = self._compute_cost(elapsed * self.num_cores, 0.0, 0.0)
                # actual_cost=None: a cancelled run is NOT a calibration observation.
                self._write_log(
                    run_id, logical_run_id, role, plan, mode, "cancelled", False,
                    start, end, plan.budget, plan.deadline, plan.accuracy_tolerance,
                    sample_fraction, None, elapsed, elapsed * self.num_cores,
                    None, None, split, msg,
                    {"wall_time_s": elapsed, "partial_cost_estimate": partial_cost},
                )
                raise ExecutorCancelled(
                    msg or "job group cancelled",
                    elapsed_s=elapsed,
                    partial_cost=partial_cost,
                ) from exc

            self._write_log(
                run_id, logical_run_id, role, plan, mode, "failed", False,
                start, end, plan.budget, plan.deadline, plan.accuracy_tolerance,
                sample_fraction, None, None, None, None, None, split, msg, {},
            )
            raise

        metrics["wall_time_s"] = end - start
        metrics["bytes_scanned"] = bytes_scanned
        metrics["bytes_shuffled"] = bytes_shuffled
        if row_count is not None:
            metrics["row_count"] = row_count
        if sample_fraction is not None:
            metrics["sample_fraction"] = sample_fraction
            if seed is not None:
                metrics["sample_seed"] = seed

        actual_latency = end - start
        actual_cost = self._compute_cost(cpu_seconds, bytes_scanned, bytes_shuffled)

        self._write_log(
            run_id, logical_run_id, role, plan, mode, status, False,
            start, end, plan.budget, plan.deadline, plan.accuracy_tolerance,
            sample_fraction, actual_cost, actual_latency, cpu_seconds,
            bytes_scanned, bytes_shuffled, split, error_msg, metrics,
        )

        result = ExecutionResult(
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            actual_cost=actual_cost,
            actual_latency=actual_latency,
            runtime_metrics=metrics,
            result_summary=(f"{row_count} rows" if row_count is not None else "noop sink"),
            execution_mode=mode,
        )

        # Only completed real runs are calibration observations.
        if prediction is not None and self.calibration_service is not None:
            telemetry = ExecutionTelemetry(
                plan_id=plan.plan_id,
                query_id=plan.query_id,
                actual_cost=actual_cost,
                actual_latency=actual_latency,
                cpu_seconds=cpu_seconds,
                bytes_scanned=bytes_scanned,
                bytes_shuffled=bytes_shuffled,
                runtime_metrics=dict(metrics),
            )
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

        return result

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _looks_like_cancellation(message: str) -> bool:
        # Confirmed against a live pyspark==4.2.0 local session
        # (sc.cancelJobGroup): the wrapped Py4JJavaError's message
        # contains the structured error code "SPARK_JOB_CANCELLED" and
        # "SQLSTATE: XXKDA". Match on those first -- they're specific to
        # an actual cancellation and won't false-positive on an unrelated
        # failure. Fall back to the loose "cancelled" substring for older
        # Spark versions that may not use this error-code format.
        lowered = message.lower()
        if "spark_job_cancelled" in lowered or "sqlstate: xxkda" in lowered:
            return True
        return "cancelled" in lowered or "canceled" in lowered

    @staticmethod
    def _looks_like_missing_sink(message: str) -> bool:
        """True only when the 'noop' data source itself could not be found."""
        lowered = message.lower()
        return (
            "data_source_not_found" in lowered
            or "failed to find data source" in lowered
            or "failed to find the data source" in lowered
        )

    def _compute_cost(self, cpu_seconds: float, bytes_scanned: float, bytes_shuffled: float) -> float:
        """
        Monetary cost via the shared pricing module.

        `cpu_seconds` is already AGGREGATE CPU-seconds (wall-clock * cores);
        pricing.total_cost must not be given the core count again.
        """
        if _pricing_total_cost is not None and self.pricing is not None:
            return _pricing_total_cost(
                cpu_seconds,
                bytes_scanned,
                bytes_shuffled,
                config=self.pricing,
            )
        # Fallback only if mocap.cost.pricing isn't importable.
        return round(cpu_seconds * 0.0000133, 6)

    def _walk_sql_metrics(self, java_plan) -> dict:
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
            for name, value in self._walk_sql_metrics(child).items():
                values[name] = values.get(name, 0) + value
        return values

    @staticmethod
    def _bucket_bytes(metric_values: dict) -> tuple[float, float]:
        bytes_scanned = 0.0
        bytes_shuffled = 0.0
        for name, value in metric_values.items():
            if any(key in name for key in _SCAN_BYTES_KEYS):
                bytes_scanned += value
            elif any(key in name for key in _SHUFFLE_BYTES_KEYS):
                bytes_shuffled += value
        return bytes_scanned, bytes_shuffled

    def _simulate(
        self,
        plan: SelectedPlan,
        mode: str,
        sample_fraction: Optional[float],
        run_id: str,
        logical_run_id: str,
        role: str,
        split: Optional[str],
    ) -> ExecutionResult:
        """No SparkSession attached -> deterministic stub for local dev/tests."""
        start = time.time()
        time.sleep(0.01)
        end = time.time()
        metrics: Dict[str, Any] = {"wall_time_s": end - start, "simulated": True}
        if sample_fraction is not None:
            metrics["sample_fraction"] = sample_fraction

        actual_cost = plan.expected_cost * (sample_fraction or 1.0)
        actual_latency = end - start

        self._write_log(
            run_id, logical_run_id, role, plan, mode, "ok", True,
            start, end, plan.budget, plan.deadline, plan.accuracy_tolerance,
            sample_fraction, actual_cost, actual_latency, None, None, None,
            split, None, metrics,
        )
        return ExecutionResult(
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            actual_cost=actual_cost,
            actual_latency=actual_latency,
            runtime_metrics=metrics,
            result_summary="simulated run (no SparkSession attached)",
            execution_mode=mode,
        )

    def _write_log(
        self, run_id, logical_run_id, role, plan, mode, status, simulated,
        start, end, budget, deadline, accuracy_tolerance, sample_fraction,
        actual_cost, actual_latency, cpu_seconds, bytes_scanned, bytes_shuffled,
        split, error, extra_metrics,
    ) -> None:
        entry = ExecutionLogEntry(
            run_id=run_id,
            logical_run_id=logical_run_id,
            role=role,
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            mode=mode,
            status=status,
            simulated=simulated,
            started_at=start,
            ended_at=end,
            duration_s=end - start,
            budget=budget,
            deadline=deadline,
            accuracy_tolerance=accuracy_tolerance,
            sample_fraction=sample_fraction,
            expected_cost=plan.expected_cost,
            expected_latency=plan.expected_latency,
            actual_cost=actual_cost,
            actual_latency=actual_latency,
            cpu_seconds=cpu_seconds,
            bytes_scanned=bytes_scanned,
            bytes_shuffled=bytes_shuffled,
            split=split,
            error=error,
            extra_metrics=extra_metrics,
        )
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry.__dict__, default=str) + "\n")
        except Exception as log_exc:  # pragma: no cover - logging must never crash a run
            print(f"WARNING: failed to write execution log entry: {log_exc}")
