from mocap.plans.representation import (
    extract_physical_plan_features,
)


def test_broadcast_hash_join_features() -> None:
    physical_plan = """
    BroadcastHashJoin [id], [id], Inner
    :- BroadcastExchange
    :  +- Scan ExistingRDD
    +- Scan ExistingRDD
    """

    features = extract_physical_plan_features(
        physical_plan
    )

    assert features["actual_join_strategy"] == (
        "BroadcastHashJoin"
    )
    assert features["num_joins"] == 1
    assert features["num_exchanges"] == 1
    assert features["num_broadcast_exchanges"] == 1


def test_shuffle_hash_join_features() -> None:
    physical_plan = """
    ShuffledHashJoin [id], [id], Inner
    :- Exchange hashpartitioning(id)
    +- Exchange hashpartitioning(id)
    """

    features = extract_physical_plan_features(
        physical_plan
    )

    assert features["actual_join_strategy"] == (
        "ShuffledHashJoin"
    )
    assert features["num_joins"] == 1
    assert features["num_exchanges"] == 2


def test_sort_merge_join_features() -> None:
    physical_plan = """
    SortMergeJoin [id], [id], Inner
    :- Sort [id ASC]
    :  +- Exchange hashpartitioning(id)
    +- Sort [id ASC]
       +- Exchange hashpartitioning(id)
    """

    features = extract_physical_plan_features(
        physical_plan
    )

    assert features["actual_join_strategy"] == (
        "SortMergeJoin"
    )
    assert features["num_joins"] == 1
    assert features["num_exchanges"] == 2
    assert features["num_sorts"] == 2


def test_non_join_plan() -> None:
    physical_plan = """
    Project [id, name]
    +- Scan ExistingRDD
    """

    features = extract_physical_plan_features(
        physical_plan
    )

    assert features["actual_join_strategy"] is None
    assert features["num_joins"] == 0
    assert features["num_exchanges"] == 0
    assert features["num_broadcast_exchanges"] == 0