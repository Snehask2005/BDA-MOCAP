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

def test_generate_join_candidates_uses_real_spark_plans():
    from pyspark.sql import SparkSession

    from mocap.plans.generator import generate_join_candidates

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPCandidateGenerationTest")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        customers = [
            (i, f"customer_{i}")
            for i in range(20)
        ]

        orders = [
            (i, i % 20, float(i + 1))
            for i in range(100)
        ]

        spark.createDataFrame(
            customers,
            ["id", "name"],
        ).createOrReplaceTempView("customer")

        spark.createDataFrame(
            orders,
            ["id", "customer_id", "amount"],
        ).createOrReplaceTempView("orders")

        query = make_query(
            """
            SELECT
                c.id,
                COUNT(o.id) AS order_count
            FROM customer c
            JOIN orders o
                ON c.id = o.customer_id
            GROUP BY c.id
            """
        )

        candidates = generate_join_candidates(
            query,
            spark,
        )

        assert len(candidates) >= 2

        # Every candidate must have a real physical plan and
        # a fingerprint derived from that plan.
        for candidate in candidates:
            assert candidate.physical_plan
            assert candidate.fingerprint
            assert candidate.num_joins >= 1
            assert candidate.actual_join_strategy is not None

        # Candidate strategies must reflect the physical plans,
        # not merely the requested SQL hint.
        actual_strategies = {
            candidate.actual_join_strategy
            for candidate in candidates
        }

        assert len(actual_strategies) >= 2

        # Fingerprints are used for physical-plan deduplication.
        fingerprints = [
            candidate.fingerprint
            for candidate in candidates
        ]

        assert len(fingerprints) == len(set(fingerprints))

    finally:
        spark.stop()


def test_three_table_query_generates_join_order_variants() -> None:
    from mocap.plans.generator import _generate_join_order_variants

    sql = """
    SELECT *
    FROM customer c
    JOIN orders o
        ON c.id = o.customer_id
    JOIN nation n
        ON c.nation_id = n.id
    """

    query = make_query(sql)

    variants = _generate_join_order_variants(query)

    labels = [label for label, _ in variants]

    assert len(variants) > 0

    assert "join_order_o_c_n" in labels
    assert "join_order_c_n_o" in labels
    assert "join_order_n_c_o" in labels


def test_join_order_variants_preserve_join_predicates() -> None:
    from mocap.plans.generator import _generate_join_order_variants

    sql = """
    SELECT *
    FROM customer c
    JOIN orders o
        ON c.id = o.customer_id
    JOIN nation n
        ON c.nation_id = n.id
    """

    query = make_query(sql)

    variants = _generate_join_order_variants(query)

    assert len(variants) > 0

    for label, variant_sql in variants:
        assert "c.id = o.customer_id" in variant_sql
        assert "c.nation_id = n.id" in variant_sql


def test_non_inner_joins_are_not_reordered() -> None:
    from mocap.plans.generator import _generate_join_order_variants

    sql = """
    SELECT *
    FROM customer c
    LEFT JOIN orders o
        ON c.id = o.customer_id
    JOIN nation n
        ON c.nation_id = n.id
    """

    query = make_query(sql)

    variants = _generate_join_order_variants(query)

    assert variants == []

def test_three_table_query_generates_strategy_variants() -> None:
    from mocap.plans.generator import _generate_all_sql_variants

    sql = """
    SELECT *
    FROM customer c
    JOIN orders o
        ON c.id = o.customer_id
    JOIN nation n
        ON c.nation_id = n.id
    """

    query = make_query(sql)

    variants = _generate_all_sql_variants(query)

    labels = {label for label, _ in variants}

    assert "join_order_o_c_n" in labels
    assert "join_order_n_c_o" in labels
    assert "join_order_c_n_o" in labels

    assert "join_order_o_c_n_broadcast" in labels
    assert "join_order_o_c_n_merge" in labels
    assert "join_order_o_c_n_shuffle_hash" in labels

    assert "join_order_n_c_o_broadcast" in labels
    assert "join_order_n_c_o_merge" in labels
    assert "join_order_n_c_o_shuffle_hash" in labels
