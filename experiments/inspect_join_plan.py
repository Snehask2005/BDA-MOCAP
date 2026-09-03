from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("MOCAP-Join-Plan-Inspection")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    customers = spark.range(
        0,
        10_000,
    ).withColumnRenamed("id", "customer_id")

    orders = spark.range(
        0,
        50_000,
    ).withColumnRenamed("id", "order_id")

    orders = orders.withColumnRenamed(
        "order_id",
        "customer_id",
    )

    joined = customers.join(
        orders,
        on="customer_id",
        how="inner",
    )

    print("=" * 70)
    print("JOIN PHYSICAL PLAN")
    print("=" * 70)

    joined.explain(mode="formatted")

    spark.stop()


if __name__ == "__main__":
    main()