from mocap.query.features import extract_query_features


SQL = """
SELECT
    c.c_mktsegment,
    COUNT(*) AS order_count
FROM customer c
JOIN orders o
    ON c.c_custkey = o.o_custkey
WHERE o.o_totalprice > 1000
GROUP BY c.c_mktsegment;
"""


def test_query_features() -> None:
    features = extract_query_features("Q01", SQL)

    assert features.num_tables == 2
    assert features.num_joins == 1
    assert features.num_filters == 1
    assert features.has_group_by is True
    assert features.has_aggregation is True