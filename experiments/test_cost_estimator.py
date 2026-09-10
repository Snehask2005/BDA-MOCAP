"""
Manual demo/inspection script for the cost estimator module.
Mirrors the style of test_query_structure.py / test_sql_parser.py:
a main() you run directly, not a pytest suite.

Seeds tiny in-memory customer/orders tables so this runs standalone
without needing the full TPC-H dataset.

NOTE: assumes QueryRequest(query_id=..., sql=...) — that's the
signature implied by its usage in generator.py. If parser.py's
actual QueryRequest requires more fields (budget, deadline,
accuracy tolerance per the shared-interface doc), add them here.
"""
from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates
from mocap.cost.estimator import estimate_from_execution
from mocap.cost.learned import calibrate

SQL = """
SELECT
    c.c_mktsegment,
    COUNT(*) AS order_count
FROM customer c
JOIN orders o
    ON c.c_custkey = o.o_custkey
WHERE o.o_totalprice > 1000
GROUP BY c.c_mktsegment
"""


def _seed_tables(spark: SparkSession) -> None:
    spark.createDataFrame(
        [(1, "BUILDING"), (2, "AUTOMOBILE"), (3, "BUILDING")],
        ["c_custkey", "c_mktsegment"],
    ).createOrReplaceTempView("customer")

    spark.createDataFrame(
        [(1, 1, 1500.0), (2, 2, 500.0), (3, 3, 2000.0), (4, 1, 3000.0)],
        ["o_orderkey", "o_custkey", "o_totalprice"],
    ).createOrReplaceTempView("orders")


def main() -> None:
    spark = SparkSession.builder.appName("mocap-cost-demo").master("local[2]").getOrCreate()
    _seed_tables(spark)

    query = QueryRequest(query_id="Q01", sql=SQL)
    candidates = generate_join_candidates(query, spark)

    print("=" * 70)
    print("COST ESTIMATOR DEMO")
    print("=" * 70)

    for candidate in candidates:
        metrics = estimate_from_execution(candidate, spark)
        calibrated_cost = calibrate(metrics)

        print(f"\nplan_id={metrics.plan_id}  strategy={candidate.strategy}")
        print(f"  analytical cost = {metrics.estimated_cost:.6f}")
        print(f"  calibrated cost = {calibrated_cost:.6f}")
        print(f"  latency (s)     = {metrics.estimated_latency:.3f}")
        print(
            f"  components: cpu={metrics.cpu_component:.6f} "
            f"io={metrics.io_component:.6f} shuffle={metrics.shuffle_component:.6f}"
        )

    spark.stop()


if __name__ == "__main__":
    main()
