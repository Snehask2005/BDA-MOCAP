from pyspark.sql import SparkSession


def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MOCAP-Join-Strategy-Comparison")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def main() -> None:
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    customers = (
        spark.range(0, 10_000)
        .withColumnRenamed("id", "customer_id")
    )

    orders = (
        spark.range(0, 50_000)
        .withColumnRenamed("id", "customer_id")
    )

    customers.createOrReplaceTempView("customer")
    orders.createOrReplaceTempView("orders")

    queries = {
        "default": """
            SELECT c.customer_id
            FROM customer c
            JOIN orders o
              ON c.customer_id = o.customer_id
        """,

        "broadcast": """
            SELECT /*+ BROADCAST(c) */ c.customer_id
            FROM customer c
            JOIN orders o
              ON c.customer_id = o.customer_id
        """,

        "merge": """
            SELECT /*+ MERGE(c, o) */ c.customer_id
            FROM customer c
            JOIN orders o
              ON c.customer_id = o.customer_id
        """,

        "shuffle_hash": """
            SELECT /*+ SHUFFLE_HASH(c, o) */ c.customer_id
            FROM customer c
            JOIN orders o
              ON c.customer_id = o.customer_id
        """,
    }

    for strategy, sql in queries.items():
        print("\n" + "=" * 80)
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 80)

        df = spark.sql(sql)

        df.explain(mode="formatted")

    spark.stop()


if __name__ == "__main__":
    main()