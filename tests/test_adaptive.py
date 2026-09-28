from mocap.execution.adaptive import (
    AdaptiveConfig,
    AdaptiveController,
    ExecutionMode,
)
from mocap.execution.executor import SparkExecutor
from mocap.interfaces import ExecutionResult, SelectedPlan


def make_plan(budget=1.0, plan_id="p1"):
    return SelectedPlan(
        plan_id=plan_id,
        query_id="q1",
        selected_strategy="sort_merge_join",
        expected_cost=0.5,
        expected_latency=2.0,
        selection_reason="test plan",
        physical_plan_sql="SELECT 1",
        budget=budget,
    )


class RecordingExecutor(SparkExecutor):
    def __init__(self):
        super().__init__(spark=None)
        self.calls = []

    def execute(self, plan, mode="strict", sample_fraction=None):
        self.calls.append(
            {
                "plan_id": plan.plan_id,
                "mode": mode,
                "sample_fraction": sample_fraction,
            }
        )

        return ExecutionResult(
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            actual_cost=plan.expected_cost,
            actual_latency=plan.expected_latency,
            runtime_metrics={},
            result_summary="test",
            execution_mode=mode,
        )


def test_strict_mode_without_budget():
    executor = RecordingExecutor()
    controller = AdaptiveController(
        executor=executor,
        spark=None,
    )

    result = controller.run(make_plan(budget=None))

    assert result.execution_mode == ExecutionMode.STRICT.value
    assert len(executor.calls) == 1
    assert executor.calls[0]["mode"] == ExecutionMode.STRICT.value


def test_degraded_mode_uses_sampling():
    executor = RecordingExecutor()

    controller = AdaptiveController(
        executor=executor,
        spark=None,
        config=AdaptiveConfig(sample_fraction=0.2),
    )

    result = controller.run_degraded(make_plan())

    assert result.execution_mode == ExecutionMode.DEGRADED.value
    assert executor.calls[-1]["mode"] == ExecutionMode.DEGRADED.value
    assert executor.calls[-1]["sample_fraction"] == 0.2
    assert result.runtime_metrics["sample_fraction"] == 0.2
    assert "accuracy_error" in result.runtime_metrics


def test_replanner_is_used_when_overrun_occurs():
    executor = RecordingExecutor()

    original = make_plan(
        budget=1.0,
        plan_id="original",
    )

    cheaper = make_plan(
        budget=1.0,
        plan_id="cheaper",
    )
    cheaper.expected_cost = 0.2

    def replanner(plan):
        assert plan.plan_id == "original"
        return cheaper

    controller = AdaptiveController(
        executor=executor,
        spark=None,
        config=AdaptiveConfig(max_replans=1),
        replanner=replanner,
    )

    result = controller._handle_overrun(
        original,
        event=(None, 2.0),
    )

    assert result.plan_id == "cheaper"
    assert result.execution_mode == ExecutionMode.ADAPTIVE.value
    assert len(executor.calls) == 1
    assert executor.calls[0]["plan_id"] == "cheaper"


def test_overrun_without_replanner_degrades():
    executor = RecordingExecutor()

    controller = AdaptiveController(
        executor=executor,
        spark=None,
        config=AdaptiveConfig(sample_fraction=0.25),
        replanner=None,
    )

    plan = make_plan(budget=1.0)

    result = controller._handle_overrun(
        plan,
        event=(None, 2.0),
    )

    assert result.execution_mode == ExecutionMode.DEGRADED.value
    assert executor.calls[-1]["sample_fraction"] == 0.25


def test_replanner_none_result_falls_back_to_degraded():
    executor = RecordingExecutor()

    controller = AdaptiveController(
        executor=executor,
        spark=None,
        config=AdaptiveConfig(sample_fraction=0.25),
        replanner=lambda plan: None,
    )

    plan = make_plan(budget=1.0)

    result = controller._handle_overrun(
        plan,
        event=(None, 2.0),
    )

    assert result.execution_mode == ExecutionMode.DEGRADED.value
    assert executor.calls[-1]["sample_fraction"] == 0.25
