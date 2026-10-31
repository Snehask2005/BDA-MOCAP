from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("MOCAP-Candidate-Generation-Test")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    try:
        # Small test datasets.
        customers = [
            (1, "A"),
            (2, "B"),
            (3, "C"),
            (4, "D"),
        ]

        orders = [
            (1, 100.0),
            (1, 250.0),
            (2, 75.0),
            (3, 500.0),
            (4, 120.0),
        ]

        customer_df = spark.createDataFrame(
            customers,
            ["c_custkey", "c_name"],
        )

        orders_df = spark.createDataFrame(
            orders,
            ["o_custkey", "o_totalprice"],
        )

        customer_df.createOrReplaceTempView("customer")
        orders_df.createOrReplaceTempView("orders")

        sql = """
        SELECT
            c.c_name,
            SUM(o.o_totalprice) AS total_spent
        FROM customer c
        JOIN orders o
            ON c.c_custkey = o.o_custkey
        GROUP BY c.c_name
        """

        query = QueryRequest(
            query_id="Q_SPARK_01",
            sql=sql,
            budget=100.0,
        )

        candidates = generate_join_candidates(
            query,
            spark,
        )

        print()
        print("=" * 70)
        print("MOCAP CANDIDATE PLAN GENERATION")
        print("=" * 70)

        print(f"Query ID: {query.query_id}")
        print(f"Candidates generated: {len(candidates)}")
        print()

        for candidate in candidates:
            print("-" * 70)
            print(f"Plan ID:       {candidate.plan_id}")
            print(f"Strategy:      {candidate.strategy}")
            print(f"Fingerprint:   {candidate.fingerprint}")
            print()
            print("Physical Plan:")
            print(candidate.physical_plan)

        print("=" * 70)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
