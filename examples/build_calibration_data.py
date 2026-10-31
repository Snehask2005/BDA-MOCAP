from pathlib import Path

from pyspark.sql import SparkSession

from mocap.pipeline import MOCAPPipeline
from mocap.query.parser import QueryRequest
from mocap.calibration.dataset import append_records


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPCalibration")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    customers = [(i, f"customer_{i}") for i in range(1000)]
    orders = [
        (i, i % 1000, float(i % 100 + 1))
        for i in range(10000)
    ]

    spark.createDataFrame(
        customers, ["id", "name"]
    ).createOrReplaceTempView("customer")

    spark.createDataFrame(
        orders, ["id", "customer_id", "amount"]
    ).createOrReplaceTempView("orders")

    queries = [
        """
        SELECT COUNT(*) AS total_orders
        FROM orders
        """,
        """
        SELECT customer_id, COUNT(*) AS order_count
        FROM orders
        GROUP BY customer_id
        """,
        """
        SELECT customer_id, SUM(amount) AS total_amount
        FROM orders
        GROUP BY customer_id
        ORDER BY total_amount DESC
        LIMIT 20
        """,
        """
        SELECT
            c.id,
            COUNT(o.id) AS order_count,
            SUM(o.amount) AS total_amount
        FROM customer c
        JOIN orders o
            ON c.id = o.customer_id
        GROUP BY c.id
        ORDER BY total_amount DESC
        LIMIT 20
        """,
    ]

    output = Path("experiments/results/calibration_dataset.csv")

    if output.exists():
        output.unlink()

    pipeline = MOCAPPipeline(spark)
    total_records = 0

    for i, sql in enumerate(queries, start=1):
        query = QueryRequest(
            query_id=f"CAL_{i:02d}",
            sql=sql,
            budget=1.0,
        )

        print(f"\nRunning calibration query {query.query_id}...")

        result = pipeline.run(query)
        records = list(result.metrics.values())

        if records:
            append_records(records, output)
            total_records += len(records)

        print(f"  Candidates: {len(result.candidates)}")
        print(f"  Metrics:    {len(result.metrics)}")
        print(f"  Feasible:   {len(result.feasible)}")

    spark.stop()

    print("\n" + "=" * 60)
    print("CALIBRATION DATASET COMPLETE")
    print("=" * 60)
    print(f"Records : {total_records}")
    print(f"File    : {output}")


if __name__ == "__main__":
    main()
