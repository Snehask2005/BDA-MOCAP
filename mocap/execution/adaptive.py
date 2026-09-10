"""
Student 4 - Day 5-8 deliverable: Adaptive Mode + Degraded Mode / AQP fallback.

Implements the three execution policies referenced in the presentation --
Strict, Adaptive, Degraded (Section 8):

  Strict    - run the selected plan exactly as chosen; no intervention.
  Adaptive  - if RuntimeMonitor detects a projected budget overrun, either
              trigger replanning (a callback into Student 3's selector) or
              fall back to a cheaper execution strategy.
  Degraded  - if no exact plan is/becomes feasible within budget, run an
              Approximate Query Processing (AQP) fallback using sampling
              and report accuracy error alongside the cost saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from mocap.interfaces import ExecutionResult, SelectedPlan
from mocap.execution.executor import SparkExecutor
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
        `replanner` is the hook into Student 3's optimizer: given the
        current SelectedPlan, it should return a cheaper SelectedPlan (or
        None if it can't find one). Leave it unset during early development
        -- Adaptive Mode will fall back straight to sampling instead.
        """
        self.executor = executor
        self.spark = spark
        self.config = config or AdaptiveConfig()
        self.replanner = replanner

    def run(self, plan: SelectedPlan) -> ExecutionResult:
        """Entry point: runs Strict if there's no budget, Adaptive otherwise."""
        if plan.budget is None:
            return self.executor.execute(plan, mode=ExecutionMode.STRICT.value)

        overrun_events = []

        def _on_overrun(sample: ProgressSample, projected_cost: float) -> None:
            overrun_events.append((sample, projected_cost))

        monitor = RuntimeMonitor(
            spark=self.spark,
            budget=plan.budget * self.config.overrun_tolerance,
            on_projected_overrun=_on_overrun,
        )
        monitor.start()
        try:
            result = self.executor.execute(plan, mode=ExecutionMode.ADAPTIVE.value)
        finally:
            monitor.stop()

        if overrun_events:
            return self._handle_overrun(plan, overrun_events[-1])
        return result

    # ---- Adaptive branch ------------------------------------------------------

    def _handle_overrun(self, plan: SelectedPlan, event) -> ExecutionResult:
        _sample, _projected_cost = event
        if self.replanner is not None and self.config.max_replans > 0:
            cheaper_plan = self.replanner(plan)
            if cheaper_plan is not None:
                cheaper_plan.budget = plan.budget
                return self.executor.execute(cheaper_plan, mode=ExecutionMode.ADAPTIVE.value)
        # No replanner wired up yet, or it couldn't find a cheaper plan -> degrade.
        return self.run_degraded(plan)

    # ---- Degraded / AQP branch -------------------------------------------------

    def run_degraded(self, plan: SelectedPlan) -> ExecutionResult:
        """
        First AQP fallback: uniform row-level sampling (Day 7-8 deliverable).
        Executes the same query at `sample_fraction` and reports the
        accuracy error alongside the cost saving.
        """
        degraded_plan = self._scale_plan_for_sampling(plan, self.config.sample_fraction)
        approx_result = self.executor.execute(
            degraded_plan,
            mode=ExecutionMode.DEGRADED.value,
            sample_fraction=self.config.sample_fraction,
        )
        approx_result.runtime_metrics.setdefault("sample_fraction", self.config.sample_fraction)
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
        )

    def _estimate_accuracy_error(
        self, plan: SelectedPlan, approx_result: ExecutionResult
    ) -> Optional[float]:
        """
        Placeholder: returns None until a ground-truth full result (or a
        cached one from a prior experiment run) is available to diff
        against. evaluation/metrics.py should own the real error metric
        (e.g. relative error on an aggregate value) -- wire it in here.
        """
        return None
