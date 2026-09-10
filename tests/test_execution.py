"""
Quick smoke tests for Student 4's module. These do NOT require a live
Spark cluster -- executor.py/monitor.py degrade to simulated behavior
when no SparkSession is attached, so this file is safe to run from Day 1,
before the team's shared Spark environment (Sync 1) is ready.

Run with: pytest tests/test_execution.py -v
"""

from mocap.interfaces import SelectedPlan
from mocap.execution.executor import SparkExecutor
from mocap.execution.adaptive import AdaptiveController, AdaptiveConfig


def make_plan(budget=1.0) -> SelectedPlan:
    return SelectedPlan(
        plan_id="p1",
        query_id="q1",
        selected_strategy="sort_merge_join",
        expected_cost=0.5,
        expected_latency=2.0,
        selection_reason="lowest Lagrangian cost under budget",
        physical_plan_sql="SELECT * FROM lineitem LIMIT 10",
        budget=budget,
    )


def test_executor_simulated_run():
    executor = SparkExecutor(spark=None)
    result = executor.execute(make_plan())
    assert result.query_id == "q1"
    assert result.plan_id == "p1"
    assert result.actual_latency >= 0


def test_adaptive_controller_strict_path_without_budget():
    executor = SparkExecutor(spark=None)
    controller = AdaptiveController(executor=executor, spark=None)
    plan = make_plan(budget=None)
    result = controller.run(plan)
    assert result.execution_mode == "strict"


def test_degraded_mode_produces_sampled_result():
    executor = SparkExecutor(spark=None)
    controller = AdaptiveController(
        executor=executor, spark=None, config=AdaptiveConfig(sample_fraction=0.2)
    )
    plan = make_plan(budget=0.0001)  # unreachable budget -> force degraded path directly
    result = controller.run_degraded(plan)
    assert result.execution_mode == "degraded"
    assert result.runtime_metrics["sample_fraction"] == 0.2
    assert "accuracy_error" in result.runtime_metrics
