"""Unit tests for Spark CBO statistics and plan verification."""
import pytest
from predictor.cbo_stats import (
    CBOCostStats,
    parse_explain_cost,
    verify_plan_matches_hint,
    enable_cbo,
    enforce_hint_testing_mode,
)


SPARK_EXPLAIN_SAMPLE = """
== Optimized Logical Plan ==
Aggregate [c_mktsegment#10], [c_mktsegment#10, count(1) AS order_count#12L], Statistics(sizeInBytes=120.0 KiB, rowCount=1.00E+3)
+- Join Inner, (c_custkey#0 = o_custkey#5), Statistics(sizeInBytes=4.8 MiB, rowCount=4.50E+4)
   :- Filter isnotnull(c_custkey#0), Statistics(sizeInBytes=1.2 MiB, rowCount=2.00E+4)
   :  +- Relation default.customer[c_custkey#0,c_mktsegment#10] parquet, Statistics(sizeInBytes=1.5 MiB, rowCount=2.50E+4)
   +- Filter (isnotnull(o_custkey#5) AND (o_totalprice#7 > 1000.0)), Statistics(sizeInBytes=3.6 MiB, rowCount=3.00E+4)
      +- Relation default.orders[o_orderkey#4,o_custkey#5,o_totalprice#7] parquet, Statistics(sizeInBytes=5.0 MiB, rowCount=5.00E+4)
"""


def test_parse_explain_cost():
    stats = parse_explain_cost(SPARK_EXPLAIN_SAMPLE)
    assert len(stats) >= 4

    # Check join stats
    join_stats = [s for s in stats if "Join" in s.node_name or "Aggregate" in s.node_name]
    assert len(join_stats) > 0

    first = stats[0]
    assert first.has_stats is True
    assert first.is_valid is True
    assert first.size_in_bytes is not None
    assert first.size_in_bytes > 100000  # 120 KiB
    assert first.row_count == 1000.0


def test_degenerate_stats_detection():
    degenerate_text = "Relation test_table parquet, Statistics(sizeInBytes=0, rowCount=0)"
    stats = parse_explain_cost(degenerate_text)
    assert len(stats) == 1
    assert stats[0].is_degenerate() is True


def test_verify_plan_matches_hint():
    plan_bhj = "AdaptiveSparkPlan\n+- BroadcastHashJoinExec [c_custkey#0], [o_custkey#5], Inner, BuildRight"
    assert verify_plan_matches_hint(plan_bhj, "broadcast") is True
    assert verify_plan_matches_hint(plan_bhj, "merge") is False

    plan_smj = "AdaptiveSparkPlan\n+- SortMergeJoinExec [c_custkey#0], [o_custkey#5], Inner"
    assert verify_plan_matches_hint(plan_smj, "merge") is True
    assert verify_plan_matches_hint(plan_smj, "broadcast") is False

    plan_shj = "AdaptiveSparkPlan\n+- ShuffledHashJoinExec [c_custkey#0], [o_custkey#5], Inner, BuildRight"
    assert verify_plan_matches_hint(plan_shj, "shuffle_hash") is True


def test_enforce_hint_testing_mode():
    class DummySpark:
        def __init__(self):
            self.conf = {}
        def set(self, k, v):
            self.conf[k] = v

    class DummySession:
        def __init__(self):
            self.conf = DummySpark()

    session = DummySession()
    cfg = enforce_hint_testing_mode(session)
    assert cfg["spark.sql.cbo.joinReorder.enabled"] == "false"
    assert cfg["spark.sql.cbo.enabled"] == "true"
