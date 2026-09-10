from mocap.plans.generator import _generate_sql_variants
from mocap.query.parser import QueryRequest


JOIN_SQL = """
SELECT
    c.c_mktsegment,
    COUNT(*) AS order_count
FROM customer c
JOIN orders o
    ON c.c_custkey = o.o_custkey
WHERE o.o_totalprice > 1000
GROUP BY c.c_mktsegment
"""


SINGLE_TABLE_SQL = """
SELECT *
FROM customer c
"""


def make_query(sql: str) -> QueryRequest:
    return QueryRequest(
        query_id="Q_TEST",
        sql=sql,
        budget=100.0,
    )


def test_join_query_generates_multiple_strategies() -> None:
    query = make_query(JOIN_SQL)

    variants = _generate_sql_variants(query)

    strategies = [strategy for strategy, _ in variants]

    assert strategies == [
        "default",
        "broadcast",
        "merge",
        "shuffle_hash",
    ]


def test_join_aliases_are_detected_automatically() -> None:
    query = make_query(JOIN_SQL)

    variants = _generate_sql_variants(query)

    sql_by_strategy = dict(variants)

    assert "BROADCAST(c)" in sql_by_strategy["broadcast"]
    assert "MERGE(c, o)" in sql_by_strategy["merge"]
    assert "SHUFFLE_HASH(c, o)" in sql_by_strategy["shuffle_hash"]


def test_generator_does_not_hard_code_customer_aliases() -> None:
    sql = """
    SELECT *
    FROM supplier s
    JOIN nation n
        ON s.n_nationkey = n.n_nationkey
    """

    query = make_query(sql)

    variants = _generate_sql_variants(query)

    sql_by_strategy = dict(variants)

    assert "BROADCAST(s)" in sql_by_strategy["broadcast"]
    assert "MERGE(s, n)" in sql_by_strategy["merge"]
    assert "SHUFFLE_HASH(s, n)" in sql_by_strategy["shuffle_hash"]


def test_non_join_query_only_has_default_candidate() -> None:
    query = make_query(SINGLE_TABLE_SQL)

    variants = _generate_sql_variants(query)

    assert len(variants) == 1
    assert variants[0][0] == "default"
    assert variants[0][1] == SINGLE_TABLE_SQL


def test_original_sql_is_preserved() -> None:
    query = make_query(JOIN_SQL)

    variants = _generate_sql_variants(query)

    default_strategy, default_sql = variants[0]

    assert default_strategy == "default"
    assert default_sql == JOIN_SQL