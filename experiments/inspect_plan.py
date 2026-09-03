from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("MOCAP-Plan-Inspection")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 70)
    print("MOCAP - SPARK PLAN INSPECTION")
    print("=" * 70)
    print(f"Spark version: {spark.version}")

    df = spark.range(100_000)

    query_df = df.groupBy().sum("id")

    print("\n--- FORMATTED PHYSICAL PLAN ---\n")
    query_df.explain(mode="formatted")

    print("\n" + "=" * 70)

    spark.stop()


if __name__ == "__main__":
    main()