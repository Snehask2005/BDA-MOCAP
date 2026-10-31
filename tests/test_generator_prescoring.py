import sqlglot
import pytest

from mocap.plans.generator import (
    _generate_all_sql_variants,
    _pre_score_sql_variants,
    generate_join_candidates,
)
from mocap.query.parser import QueryRequest


def make_six_table_query():
    return QueryRequest(
        query_id="PRESCORE_6",
        sql="""
            SELECT a.id
            FROM table_a a
            JOIN table_b b ON a.id = b.a_id
            JOIN table_c c ON b.id = c.b_id
            JOIN table_d d ON c.id = d.c_id
            JOIN table_e e ON d.id = e.d_id
            JOIN table_f f ON e.id = f.e_id
        """,
        budget=100.0,
    )


def make_variants():
    sizes = {
        "a": 1000.0,
        "b": 100.0,
        "c": 20.0,
        "d": 5.0,
        "e": 10.0,
        "f": 500.0,
    }
    variants = _generate_all_sql_variants(
        make_six_table_query(),
        search_mode="beam",
        beam_width=10,
        table_size_estimates=sizes,
    )
    return variants, sizes


def order_for_sql(sql):
    tree = sqlglot.parse_one(sql)
    return (
        tree.args["from_"].this.alias_or_name,
        *(join.this.alias_or_name for join in tree.args.get("joins") or []),
    )


def test_prescoring_preserves_base_strategies_and_spreads_orders():
    variants, sizes = make_variants()

    selected = _pre_score_sql_variants(
        variants,
        aliases=("a", "b", "c", "d", "e", "f"),
        table_size_estimates=sizes,
        candidate_analysis_limit=12,
    )

    assert len(selected) <= 12
    assert [
        strategy
        for strategy, _ in selected
        if not strategy.startswith("join_order_")
    ] == ["default", "broadcast", "merge", "shuffle_hash"]

    selected_orders = {}
    for strategy, sql in selected:
        if strategy.startswith("join_order_"):
            selected_orders.setdefault(order_for_sql(sql), []).append(strategy)

    assert len(selected_orders) == len(selected) - 4
    assert all(
        len(strategies) == 1
        and strategies[0].endswith("_broadcast")
        for strategies in selected_orders.values()
    )

    # Prefer distinct starting relations before taking a second order
    # from a starting relation.
    assert len({order[0] for order in selected_orders}) >= min(
        5, len(selected_orders)
    )


def test_limit20_covers_all_available_order_broadcast_variants():
    variants, sizes = make_variants()

    selected = _pre_score_sql_variants(
        variants,
        aliases=("a", "b", "c", "d", "e", "f"),
        table_size_estimates=sizes,
        candidate_analysis_limit=20,
    )

    assert len(selected) <= 20

    all_orders = {
        order_for_sql(sql)
        for strategy, sql in variants
        if strategy.startswith("join_order_")
    }
    selected_orders = {}

    for strategy, sql in selected:
        if strategy.startswith("join_order_"):
            selected_orders.setdefault(order_for_sql(sql), []).append(strategy)

    assert set(selected_orders) == all_orders
    assert all(
        any(strategy.endswith("_broadcast") for strategy in strategies)
        for strategies in selected_orders.values()
    )


def test_prescoring_is_deterministic():
    variants, sizes = make_variants()
    kwargs = {
        "aliases": ("a", "b", "c", "d", "e", "f"),
        "table_size_estimates": sizes,
        "candidate_analysis_limit": 12,
    }

    assert _pre_score_sql_variants(variants, **kwargs) == (
        _pre_score_sql_variants(variants, **kwargs)
    )


@pytest.mark.parametrize("limit", [0, 1, -1, True, 1.5])
def test_invalid_candidate_analysis_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="candidate_analysis_limit"):
        generate_join_candidates(
            make_six_table_query(),
            spark=None,
            search_mode="beam",
            candidate_analysis_limit=limit,
        )


def test_candidate_analysis_limit_requires_beam_search():
    with pytest.raises(ValueError, match="requires search_mode='beam'"):
        generate_join_candidates(
            make_six_table_query(),
            spark=None,
            search_mode="exhaustive",
            candidate_analysis_limit=12,
        )


def test_prescoring_returns_original_variants_when_already_within_limit():
    variants = [
        ("default", "SELECT a.id FROM table_a a"),
        ("broadcast", "SELECT /*+ BROADCAST(a) */ a.id FROM table_a a"),
    ]

    selected = _pre_score_sql_variants(
        variants,
        aliases=("a",),
        table_size_estimates={"a": 100.0},
        candidate_analysis_limit=20,
    )

    assert selected is variants
