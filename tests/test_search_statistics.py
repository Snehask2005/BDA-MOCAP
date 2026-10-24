import pytest

from mocap.plans.search_statistics import (
    JoinSearchStatistics,
    build_cardinality_scorer,
)


def test_cardinality_scorer_prefers_selective_join_path():
    aliases = ("f", "a", "b", "c")
    adjacency = {
        "f": {"a", "b", "c"},
        "a": {"f"},
        "b": {"f"},
        "c": {"f"},
    }
    stats = JoinSearchStatistics(
        relation_rows={"f": 1_000_000, "a": 100, "b": 100, "c": 100},
        join_selectivities={
            ("f", "a"): 0.0001,
            ("f", "b"): 0.1,
            ("f", "c"): 0.1,
        },
    )

    scorer = build_cardinality_scorer(aliases, adjacency, stats)
    assert scorer is not None

    selective_first = scorer(("a", "f", "b", "c"))
    less_selective_first = scorer(("b", "f", "a", "c"))

    assert selective_first < less_selective_first


def test_incomplete_statistics_use_fallback_signal():
    scorer = build_cardinality_scorer(
        ("a", "b"),
        {"a": {"b"}, "b": {"a"}},
        JoinSearchStatistics(
            relation_rows={"a": 100},
            join_selectivities={},
        ),
    )

    assert scorer is None


def test_rejects_invalid_selectivity():
    with pytest.raises(ValueError, match="Invalid join selectivity"):
        build_cardinality_scorer(
            ("a", "b"),
            {"a": {"b"}, "b": {"a"}},
            JoinSearchStatistics(
                relation_rows={"a": 10, "b": 20},
                join_selectivities={("a", "b"): 1.5},
            ),
        )


def test_rejects_invalid_row_count():
    with pytest.raises(ValueError, match="Invalid estimated row count"):
        build_cardinality_scorer(
            ("a", "b"),
            {"a": {"b"}, "b": {"a"}},
            JoinSearchStatistics(
                relation_rows={"a": -1, "b": 20},
                join_selectivities={("a", "b"): 0.1},
            ),
        )


def test_accepts_reversed_pair_key():
    scorer = build_cardinality_scorer(
        ("a", "b"),
        {"a": {"b"}, "b": {"a"}},
        JoinSearchStatistics(
            relation_rows={"a": 100, "b": 200},
            join_selectivities={("b", "a"): 0.01},
        ),
    )

    assert scorer is not None
    assert scorer(("a", "b")) >= 0
