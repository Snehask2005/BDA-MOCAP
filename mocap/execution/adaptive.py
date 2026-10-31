"""
Student 4 - Adaptive Mode + Degraded Mode / AQP fallback.

Execution policies:

  Strict    - run the selected plan exactly as chosen; no intervention.
  Adaptive  - a RuntimeMonitor projects cost while the plan runs. If the
              projection exceeds budget * overrun_tolerance, the running
              Spark job group is CANCELLED (sc.cancelJobGroup) and the
              controller either replans (callback into the optimizer) or
              degrades to sampling. The cancelled plan is not re-run.
  Degraded  - run an Approximate Query Processing (AQP) fallback using
              uniform row-level sampling and (optionally) measure the
              accuracy error against the exact result.

What runtime intervention is actually possible (for the paper):
  * Cancellation is job-group level: the in-flight Spark job is
    interrupted and its partial work is lost; the cost already spent is
    reported as `cancelled_partial_cost`. This is cancel-and-restart, NOT
    mid-query plan switching.
  * The projection is CPU-only and based on task progress of stages
    submitted so far (see monitor.py), so it can be biased low early on.
  * AQP sampling here is applied to the query OUTPUT (df.sample), so it
    does not reduce join/scan work; measured cost saving may be small.
    Input-side sampling (TABLESAMPLE) would be required for real savings.
"""

from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from mocap.interfaces import ExecutionResult, SelectedPlan
from mocap.execution.executor import ExecutorCancelled, SparkExecutor
from mocap.execution.monitor import ProgressSample, RuntimeMonitor


class ExecutionMode(str, Enum):
    STRICT = "strict"
    ADAPTIVE = "adaptive"
    DEGRADED = "degraded"


@dataclass
class AdaptiveConfig:
    sample_fraction: float = 0.1     # first AQP fallback: uniform sampling
    overrun_tolerance: float = 1.0   # projected_cost > budget * tolerance triggers action
    max_replans: int = 1

    cancel_on_overrun: bool = True   # False = observe-only (record, never cancel)
    poll_interval_s: float = 1.0
    min_elapsed_s: float = 1.0       # ignore projections before this much runtime
    min_progress: float = 0.05       # ...or before this task-progress fraction

    sample_seed: int = 42            # reproducible AQP sampling
    measure_accuracy: bool = False   # evaluation-only: runs extra count queries


class AdaptiveController:
    """
    Orchestrates executor.py + monitor.py to implement Adaptive and
    Degraded execution policies on top of a base SparkExecutor.
    """

    def __init__(
        self,
        executor: SparkExecutor,
        spark=None,
        config: Optional[AdaptiveConfig] = None,
        replanner: Optional[Callable[[SelectedPlan], Optional[SelectedPlan]]] = None,
    ):
        """
        `replanner` is the hook into the optimizer: given the current
        SelectedPlan, it should return a cheaper SelectedPlan (or None if
        it can't find one). If unset, Adaptive Mode degrades to sampling.
        """
        self.executor = executor
        self.spark = spark
        self.config = config or AdaptiveConfig()
        self.replanner = replanner

    def run(self, plan: SelectedPlan) -> ExecutionResult:
        """Entry point: runs Strict if there's no budget, Adaptive otherwise."""
        if plan.budget is None:
            return self.executor.execute(plan, mode=ExecutionMode.STRICT.value)

        return self._run_monitored(plan, self.config.max_replans)

    # ---- Adaptive branch ------------------------------------------------------

    def _run_monitored(
        self,
        plan: SelectedPlan,
        replans_left: int,
        carried_cost: float = 0.0,
    ) -> ExecutionResult:
        """
        Execute `plan` once under monitoring. On projected overrun the job
        group is cancelled and the overrun is handled WITHOUT re-running
        the cancelled plan.
        """
        job_group = f"mocap-{plan.plan_id}-{uuid.uuid4().hex[:8]}"
        events: list = []

        def _on_overrun(sample: ProgressSample, projected_cost: float) -> None:
            events.append((sample, projected_cost))
            if self.config.cancel_on_overrun and self.spark is not None:
                self.spark.sparkContext.cancelJobGroup(job_group)

        monitor = RuntimeMonitor(
            spark=self.spark,
            budget=plan.budget * self.config.overrun_tolerance,
            poll_interval_s=self.config.poll_interval_s,
            pricing=getattr(self.executor, "pricing", None),
            on_projected_overrun=_on_overrun,
            num_cores=getattr(self.executor, "num_cores", 1) or 1,
            job_group=job_group,
            min_elapsed_s=self.config.min_elapsed_s,
            min_progress=self.config.min_progress,
        )

        cancelled: Optional[ExecutorCancelled] = None
        result: Optional[ExecutionResult] = None

        # Job-group tagging only matters with a live SparkSession; without
        # one there is nothing to track or cancel, so call execute() exactly
        # as a plain executor expects.
        exec_kwargs = {"mode": ExecutionMode.ADAPTIVE.value}
        if self.spark is not None:
            exec_kwargs["job_group"] = job_group

        monitor.start()
        try:
            result = self.executor.execute(plan, **exec_kwargs)
        except ExecutorCancelled as exc:
            if not events:
                raise  # cancelled by someone else, not by our monitor
            cancelled = exc
        finally:
            monitor.stop()

        if cancelled is None:
            # Finished (possibly even though an overrun was projected --
            # e.g. cancel_on_overrun=False, or the job completed first).
            # Never re-execute a plan that already completed.
            if events:
                result.runtime_metrics["projected_overrun_observed"] = True
                result.runtime_metrics["projected_cost_at_overrun"] = events[-1][1]
            self._add_cost_info(result, carried_cost)
            return result

        carried = carried_cost + (cancelled.partial_cost or 0.0)
        return self._handle_overrun(plan, events[-1], replans_left, carried)

    def _handle_overrun(
        self,
        plan: SelectedPlan,
        event,
        replans_left: Optional[int] = None,
        carried_cost: float = 0.0,
    ) -> ExecutionResult:
        _sample, projected_cost = event
        if replans_left is None:
            replans_left = self.config.max_replans

        if self.replanner is not None and replans_left > 0:
            cheaper_plan = self.replanner(plan)
            if cheaper_plan is not None:
                cheaper_plan.budget = plan.budget
                if cheaper_plan.deadline is None:
                    cheaper_plan.deadline = plan.deadline
                if cheaper_plan.accuracy_tolerance is None:
                    cheaper_plan.accuracy_tolerance = plan.accuracy_tolerance

                result = self._run_monitored(
                    cheaper_plan, replans_left - 1, carried_cost
                )
                result.runtime_metrics.setdefault("adaptive_action", "replanned")
                result.runtime_metrics.setdefault("replanned_from", plan.plan_id)
                result.runtime_metrics.setdefault(
                    "projected_cost_at_overrun", projected_cost
                )
                return result

        # No replanner wired up, or it couldn't find a cheaper plan -> degrade.
        result = self.run_degraded(plan)
        result.runtime_metrics["adaptive_action"] = "degraded"
        result.runtime_metrics["overrun_plan_id"] = plan.plan_id
        result.runtime_metrics["projected_cost_at_overrun"] = projected_cost
        self._add_cost_info(result, carried_cost)
        return result

    def _executor_accepts(self, name: str) -> bool:
        """True if executor.execute() takes keyword `name` (or **kwargs)."""
        try:
            params = inspect.signature(self.executor.execute).parameters
        except (TypeError, ValueError):
            return False
        return name in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    @staticmethod
    def _add_cost_info(result: ExecutionResult, carried_cost: float) -> None:
        """Report work wasted on cancelled attempts alongside the final cost."""
        if carried_cost > 0:
            result.runtime_metrics["cancelled_partial_cost"] = carried_cost
            result.runtime_metrics["total_cost_including_cancelled"] = (
                result.actual_cost + carried_cost
            )

    # ---- Degraded / AQP branch -------------------------------------------------

    def run_degraded(self, plan: SelectedPlan) -> ExecutionResult:
        """
        First AQP fallback: uniform row-level sampling.
        Executes the same query at `sample_fraction` (seeded, so
        reproducible) and reports the accuracy error alongside the cost.
        """
        degraded_plan = self._scale_plan_for_sampling(plan, self.config.sample_fraction)
        exec_kwargs = {
            "mode": ExecutionMode.DEGRADED.value,
            "sample_fraction": self.config.sample_fraction,
        }
        if self._executor_accepts("seed"):
            exec_kwargs["seed"] = self.config.sample_seed

        approx_result = self.executor.execute(degraded_plan, **exec_kwargs)
        approx_result.runtime_metrics.setdefault("sample_fraction", self.config.sample_fraction)
        approx_result.runtime_metrics["sampling_stage"] = "post_query_output"
        approx_result.runtime_metrics["accuracy_error"] = self._estimate_accuracy_error(
            plan, approx_result
        )
        return approx_result

    def _scale_plan_for_sampling(self, plan: SelectedPlan, fraction: float) -> SelectedPlan:
        return SelectedPlan(
            plan_id=f"{plan.plan_id}_degraded",
            query_id=plan.query_id,
            selected_strategy=f"{plan.selected_strategy}+sampling",
            expected_cost=plan.expected_cost * fraction,
            expected_latency=plan.expected_latency * fraction,
            selection_reason="degraded mode: budget overrun, uniform sampling fallback",
            physical_plan_sql=plan.physical_plan_sql,
            budget=plan.budget,
            deadline=plan.deadline,
            accuracy_tolerance=plan.accuracy_tolerance,
        )

    def _estimate_accuracy_error(
        self, plan: SelectedPlan, approx_result: ExecutionResult
    ) -> Optional[float]:
        """
        Percent relative error of the row-count estimate:

            estimate = sampled_rows / fraction
            error_%  = |estimate - exact_rows| / exact_rows * 100

        EVALUATION-ONLY: runs an exact COUNT(*) and a seeded sampled
        count, whose cost is NOT included in the reported run cost.
        Disabled unless AdaptiveConfig.measure_accuracy is True.
        Returns None when disabled or when it cannot be computed.
        """
        if (
            not self.config.measure_accuracy
            or self.spark is None
            or plan.physical_plan_sql is None
        ):
            return None

        fraction = approx_result.runtime_metrics.get("sample_fraction")
        if not fraction:
            return None

        try:
            self.spark.sparkContext.setJobGroup(
                f"mocap-aqp-eval-{uuid.uuid4().hex[:8]}",
                "aqp accuracy evaluation (not part of reported cost)",
            )
            sql = plan.physical_plan_sql

            exact = self.spark.sql(sql).count()
            sampled = (
                self.spark.sql(sql)
                .sample(
                    withReplacement=False,
                    fraction=fraction,
                    seed=self.config.sample_seed,
                )
                .count()
            )
        except Exception as exc:
            approx_result.runtime_metrics["accuracy_error_failure"] = str(exc)[:200]
            return None

        approx_result.runtime_metrics["exact_row_count"] = exact
        approx_result.runtime_metrics["sampled_row_count"] = sampled
        approx_result.runtime_metrics["accuracy_error_basis"] = "row_count"

        if exact <= 0:
            return None

        estimate = sampled / fraction
        return abs(estimate - exact) / exact * 100.0
