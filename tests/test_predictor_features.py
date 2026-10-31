"""Unit tests for Catalyst plan feature extractor."""
import numpy as np
import pytest
from predictor.features import (
    PlanFeatures,
    extract_features_from_plan_text,
    extract_plan_features,
)


SAMPLE_PHYSICAL_PLAN = """
AdaptiveSparkPlan isFinalPlan=true
+- == Final Plan ==
   HashAggregate(keys=[c_mktsegment#10], functions=[count(1)])
   +- Exchange hashpartitioning(c_mktsegment#10, 200), ENSURE_REQUIREMENTS, [plan_id=101]
      +- HashAggregate(keys=[c_mktsegment#10], functions=[partial_count(1)])
         +- SortMergeJoin [c_custkey#0], [o_custkey#5], Inner
            :- Sort [c_custkey#0 ASC NULLS FIRST], false, 0
            :  +- Exchange hashpartitioning(c_custkey#0, 200), ENSURE_REQUIREMENTS, [plan_id=102]
            :     +- Filter isnotnull(c_custkey#0)
            :        +- FileScan parquet [c_custkey#0,c_mktsegment#10] Batches: 10, Data size: 50.0 MiB
            +- Sort [o_custkey#5 ASC NULLS FIRST], false, 0
               +- Exchange hashpartitioning(o_custkey#5, 200), ENSURE_REQUIREMENTS, [plan_id=103]
                  +- Filter (isnotnull(o_custkey#5) AND (o_totalprice#7 > 1000.0))
                     +- FileScan parquet [o_orderkey#4,o_custkey#5,o_totalprice#7] Batches: 30, Data size: 120.0 MiB
"""


def test_extract_features_from_plan_text():
    features = extract_features_from_plan_text(
        query_id="Q01",
        plan_id="P01",
        plan_text=SAMPLE_PHYSICAL_PLAN,
        strategy="merge",
    )

    assert features.query_id == "Q01"
    assert features.plan_id == "P01"
    assert features.strategy == "merge"

    # Operators detection
    assert features.smj_count == 1
    assert features.bhj_count == 0
    assert features.exchange_count >= 3
    assert features.sort_count >= 2
    assert features.filter_count >= 2
    assert features.aggregate_count >= 2

    # Scanned data size
    assert features.scan_bytes > 150.0 * 1024 * 1024  # > 150 MiB

    # Structural metrics
    assert features.tree_depth >= 4
    assert features.num_nodes >= 8


def test_feature_vector_conversion():
    features = PlanFeatures(
        query_id="Q02",
        plan_id="P01",
        scan_bytes=1000000.0,
        shuffle_bytes=500000.0,
        max_intermediate_size=800000.0,
        broadcast_join_sizes=0.0,
        exchange_count=2,
        broadcast_exchange_count=0,
        bhj_count=0,
        smj_count=1,
        shj_count=0,
        bnlj_count=0,
        filter_count=1,
        project_count=1,
        aggregate_count=1,
        sort_count=1,
        tree_depth=5,
        num_nodes=12,
        estimated_rows=10000.0,
        strategy="merge",
    )

    vec = features.to_vector()
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 17
    assert len(PlanFeatures.feature_names()) == 17
    assert vec[0] == 1000000.0
    assert vec[1] == 500000.0


def test_universal_extractor_interface():
    class DummyCandidate:
        def __init__(self):
            self.query_id = "Q03"
            self.plan_id = "P02"
            self.strategy = "broadcast"
            self.physical_plan = "BroadcastHashJoin\n+- FileScan parquet Data size: 10.0 MiB"

    candidate = DummyCandidate()
    feat = extract_plan_features(candidate)
    assert feat.query_id == "Q03"
    assert feat.bhj_count == 1
    assert feat.strategy == "broadcast"
