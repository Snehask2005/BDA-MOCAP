import sqlglot
import pytest

from mocap.plans.generator import (
    _generate_all_sql_variants,
    _generate_join_order_variants,
)
from mocap.query.parser import QueryRequest


def make_six_table_query():
    sql = """
    SELECT a.id
    FROM table_a a
    JOIN table_b b ON a.id = b.a_id
    JOIN table_c c ON b.id = c.b_id
    JOIN table_d d ON c.id = d.c_id
    JOIN table_e e ON d.id = e.d_id
    JOIN table_f f ON e.id = f.e_id
    """
    return QueryRequest(
        query_id="BEAM_6",
        sql=sql,
        budget=100.0,
    )


def test_exhaustive_mode_keeps_existing_six_table_limit():
    query = make_six_table_query()

    assert _generate_join_order_variants(query) == []


def test_beam_mode_generates_bounded_six_table_orders():
    query = make_six_table_query()
    sizes = {
        "a": 1000.0,
        "b": 100.0,
        "c": 20.0,
        "d": 5.0,
        "e": 10.0,
        "f": 500.0,
    }

    variants = _generate_join_order_variants(
        query,
        search_mode="beam",
        beam_width=10,
        table_size_estimates=sizes,
    )

    assert variants
    assert len(variants) <= 10
    assert len({label for label, _ in variants}) == len(variants)

    expected_aliases = {"a", "b", "c", "d", "e", "f"}

    for label, generated_sql in variants:
        order = tuple(label.removeprefix("join_order_").split("_"))
        assert set(order) == expected_aliases
        assert len(order) == 6

        parsed = sqlglot.parse_one(generated_sql)
        from_clause = parsed.args["from_"]
        joins = parsed.args.get("joins") or []
        actual_aliases = {from_clause.this.alias_or_name}
        actual_aliases.update(
            join.this.alias_or_name for join in joins
        )

        assert actual_aliases == expected_aliases
        assert len(joins) == 5


def test_beam_mode_expands_into_strategy_variants():
    query = make_six_table_query()

    variants = _generate_all_sql_variants(
        query,
        search_mode="beam",
        beam_width=5,
        table_size_estimates={
            "a": 1000.0,
            "b": 100.0,
            "c": 20.0,
            "d": 5.0,
            "e": 10.0,
            "f": 500.0,
        },
    )

    labels = [label for label, _ in variants]

    assert any(label.startswith("join_order_") for label in labels)
    assert any(label.endswith("_broadcast") for label in labels)
    assert any(label.endswith("_merge") for label in labels)
    assert any(label.endswith("_shuffle_hash") for label in labels)

    # Four base variants plus at most four SQL variants per beam order.
    assert len(variants) <= 4 + 4 * 5


def test_unknown_search_mode_is_rejected():
    query = make_six_table_query()

    with pytest.raises(ValueError, match="search_mode"):
        _generate_join_order_variants(
            query,
            search_mode="unknown",
        )


def test_beam_mode_rejects_invalid_width():
    query = make_six_table_query()

    with pytest.raises(ValueError, match="beam_width"):
        _generate_join_order_variants(
            query,
            search_mode="beam",
            beam_width=0,
        )


def test_reordering_skips_multi_relation_join_predicates():
    query = QueryRequest(
        query_id="BEAM_MULTIRELATION_PREDICATE",
        sql="""
            SELECT a.id
            FROM table_a a
            JOIN table_b b ON a.id = b.a_id
            JOIN table_c c
              ON a.id = c.a_id AND b.id = c.b_id
        """,
        budget=100.0,
    )

    assert _generate_join_order_variants(
        query,
        search_mode="exhaustive",
    ) == []

    assert _generate_join_order_variants(
        query,
        search_mode="beam",
        beam_width=10,
    ) == []
