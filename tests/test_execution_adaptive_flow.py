"""
JVM-free tests for the execution / monitoring / adaptive flow.

Uses fakes for Spark and the executor so they run anywhere.
"""

import threading
from types import SimpleNamespace

import pytest

from mocap.cost.pricing import PricingConfig
from mocap.execution.adaptive import AdaptiveConfig, AdaptiveController
from mocap.execution.executor import ExecutorCancelled, SparkExecutor
from mocap.execution.monitor import ProgressSample, RuntimeMonitor
from mocap.interfaces import ExecutionResult, SelectedPlan

# 3600 $/vcore-hour => cost == cpu_seconds, easy to reason about.
PRICING = PricingConfig(
    currency="USD",
    cost_per_vcore_hour=3600.0,
    cost_per_gb_scanned=0.0,
    cost_per_gb_shuffled=0.0,
    cost_per_gb_hour_memory=0.0,
    penalty_per_second=0.0,
)


def make_plan(plan_id="P1", budget=0.001):
    return SelectedPlan(
        plan_id=plan_id,
        query_id="Q1",
        selected_strategy="SortMergeJoin",
        expected_cost=0.0005,
        expected_latency=1.0,
        selection_reason="test",
        physical_plan_sql="SELECT 1",
        budget=budget,
    )


class FakeTracker:
    def getJobIdsForGroup(self, group):
        return [1]

    def getActiveJobIds(self):
        return [1]

    def getJobInfo(self, jid):
        return SimpleNamespace(stageIds=[10])

    def getStageInfo(self, sid):
        return SimpleNamespace(numCompletedTasks=1, numTasks=2)


class FakeSparkContext:
    def __init__(self):
        self.cancelled = []
        self.cancel_event = threading.Event()
        self._tracker = FakeTracker()

    def statusTracker(self):
        return self._tracker

    def cancelJobGroup(self, group):
        self.cancelled.append(group)
        self.cancel_event.set()


class FakeSpark:
    def __init__(self):
        self.sparkContext = FakeSparkContext()


class FakeExecutor:
    """Blocks plan 'P1' (when tracked) until cancelled, like a long Spark job."""

    pricing = PRICING
    num_cores = 1

    def __init__(self, spark, block_p1=True):
        self.spark = spark
        self.block_p1 = block_p1
        self.calls = []

    def execute(self, plan, mode="strict", job_group=None, **kwargs):
        self.calls.append((plan.plan_id, mode, job_group))
        if self.block_p1 and plan.plan_id == "P1" and job_group is not None:
            if self.spark.sparkContext.cancel_event.wait(timeout=5):
                raise ExecutorCancelled("cancelled", elapsed_s=0.1, partial_cost=0.5)
        return ExecutionResult(
            query_id=plan.query_id,
            plan_id=plan.plan_id,
            actual_cost=1.0,
            actual_latency=0.1,
            execution_mode=mode,
        )


def fast_config(**kw):
    return AdaptiveConfig(
        poll_interval_s=0.01, min_elapsed_s=0.0, min_progress=0.0, **kw
    )


def test_overrun_cancels_and_degrades_without_rerunning_original():
    spark = FakeSpark()
    ex = FakeExecutor(spark)
    result = AdaptiveController(ex, spark=spark, config=fast_config()).run(make_plan())

    assert len(spark.sparkContext.cancelled) == 1
    # original ran once (cancelled), then ONLY the degraded plan ran
    assert [c[0] for c in ex.calls] == ["P1", "P1_degraded"]
    assert result.runtime_metrics["adaptive_action"] == "degraded"
    assert result.runtime_metrics["cancelled_partial_cost"] == 0.5
    assert result.runtime_metrics["total_cost_including_cancelled"] == pytest.approx(1.5)


def test_overrun_replans_to_cheaper_plan():
    spark = FakeSpark()
    ex = FakeExecutor(spark)
    cheaper = make_plan("P2", budget=None)
    ctl = AdaptiveController(
        ex, spark=spark, config=fast_config(), replanner=lambda p: cheaper
    )
    result = ctl.run(make_plan())

    assert [c[0] for c in ex.calls] == ["P1", "P2"]
    assert result.plan_id == "P2"
    assert result.runtime_metrics["adaptive_action"] == "replanned"
    assert result.runtime_metrics["replanned_from"] == "P1"
    assert cheaper.budget == 0.001  # budget propagated


def test_observe_only_mode_never_cancels_or_reruns():
    spark = FakeSpark()
    ex = FakeExecutor(spark, block_p1=False)
    ctl = AdaptiveController(
        ex, spark=spark, config=fast_config(cancel_on_overrun=False)
    )
    result = ctl.run(make_plan())

    assert spark.sparkContext.cancelled == []
    assert len(ex.calls) == 1
    assert result.plan_id == "P1"


def test_no_budget_runs_strict_once():
    spark = FakeSpark()
    ex = FakeExecutor(spark)
    result = AdaptiveController(ex, spark=spark).run(make_plan(budget=None))
    assert len(ex.calls) == 1
    assert ex.calls[0][1] == "strict"
    assert result.plan_id == "P1"


def test_monitor_projection_scales_with_cores():
    sample = ProgressSample(t=0, completed_tasks=1, total_tasks=2, elapsed_s=1.0)
    one = RuntimeMonitor(None, budget=1, pricing=PRICING, num_cores=1)
    four = RuntimeMonitor(None, budget=1, pricing=PRICING, num_cores=4)
    assert one.project_total_cost(sample) == pytest.approx(2.0)
    assert four.project_total_cost(sample) == pytest.approx(8.0)


def test_monitor_fires_callback_only_once():
    spark = FakeSpark()
    fired = []
    mon = RuntimeMonitor(
        spark,
        budget=0.0001,
        poll_interval_s=0.01,
        pricing=PRICING,
        on_projected_overrun=lambda s, p: fired.append(p),
        job_group="g",
    )
    mon.start()
    threading.Event().wait(0.2)
    mon.stop()
    assert len(fired) == 1


def test_executor_cost_uses_pricing_config(tmp_path):
    real = PricingConfig(
        currency="USD",
        cost_per_vcore_hour=0.048,
        cost_per_gb_scanned=0.005,
        cost_per_gb_shuffled=0.010,
        cost_per_gb_hour_memory=0.0,
        penalty_per_second=0.0,
    )
    ex = SparkExecutor(spark=None, log_path=tmp_path / "log.jsonl", pricing=real)
    gib = 1024 ** 3
    # 3600 cpu-s = 0.048, 1 GiB scanned = 0.005, 1 GiB shuffled = 0.010
    assert ex._compute_cost(3600, gib, gib) == pytest.approx(0.063)


def test_cancellation_detection():
    assert SparkExecutor._looks_like_cancellation("... SPARK_JOB_CANCELLED ...")
    assert not SparkExecutor._looks_like_cancellation("table not found")
    assert SparkExecutor._looks_like_missing_sink("[DATA_SOURCE_NOT_FOUND] noop")
    assert not SparkExecutor._looks_like_missing_sink("OutOfMemoryError")
