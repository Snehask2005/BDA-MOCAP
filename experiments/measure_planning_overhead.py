import csv
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates
from mocap.cost.statistics import extract_plan_statistics
from mocap.cost.predictor import predict_candidates, PredictorConfig


REPETITIONS = 5


def measure_once(spark, query, predictor_config):

    # Spark default planning baseline
    start = time.perf_counter()

    default_df = spark.sql(query.sql)
    _ = default_df._jdf.queryExecution().executedPlan().toString()

    spark_planning_time = time.perf_counter() - start

    # MOCAP candidate generation
    start = time.perf_counter()

    candidates = generate_join_candidates(
        query=query,
        spark=spark,
    )

    generation_time = time.perf_counter() - start

    # MOCAP statistics + prediction
    start = time.perf_counter()

    for candidate in candidates:

        dataframe = spark.sql(candidate.sql)

        plan_statistics = extract_plan_statistics(dataframe)

        predict_candidates(
            [candidate],
            config=predictor_config,
            statistics=plan_statistics,
        )

    estimation_time = time.perf_counter() - start

    return {
        "spark_planning_seconds": spark_planning_time,
        "candidate_generation_seconds": generation_time,
        "candidate_estimation_seconds": estimation_time,
        "mocap_planning_seconds": generation_time + estimation_time,
        "num_candidates": len(candidates),
    }


def main():

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Planning-Overhead")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # Same asymmetric dataset as the previous experiment
    spark.range(100).withColumnRenamed(
        "id", "n_id"
    ).createOrReplaceTempView("n")

    spark.range(1000).withColumnRenamed(
        "id", "c_id"
    ).createOrReplaceTempView("c")

    spark.range(10000).withColumnRenamed(
        "id", "o_id"
    ).createOrReplaceTempView("o")

    query = QueryRequest(
        query_id="planning_overhead",
        sql="""
            SELECT *
            FROM n
            JOIN c ON n.n_id = c.c_id
            JOIN o ON c.c_id = o.o_id
        """,
        budget=100.0,
        deadline=100.0,
        accuracy_tolerance=0.0,
    )

    predictor_config = PredictorConfig()

    # Warm up Spark before recording measurements.
    # This is not a measured repetition.
    print("Warming up Spark...")
    measure_once(spark, query, predictor_config)

    rows = []

    print("\nMeasuring planning overhead...\n")

    for repetition in range(1, REPETITIONS + 1):

        result = measure_once(
            spark,
            query,
            predictor_config,
        )

        result["repetition"] = repetition

        rows.append(result)

        print(
            f"Run {repetition}: "
            f"Spark={result['spark_planning_seconds']:.4f}s | "
            f"Generation={result['candidate_generation_seconds']:.4f}s | "
            f"Estimation={result['candidate_estimation_seconds']:.4f}s | "
            f"MOCAP={result['mocap_planning_seconds']:.4f}s | "
            f"Candidates={result['num_candidates']}"
        )

    output_dir = Path("results/planning_overhead")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "planning_overhead.csv"

    with output_file.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 65)
    print("PLANNING OVERHEAD SUMMARY")
    print("=" * 65)

    fields = [
        "spark_planning_seconds",
        "candidate_generation_seconds",
        "candidate_estimation_seconds",
        "mocap_planning_seconds",
    ]

    for field in fields:

        values = [row[field] for row in rows]

        print(
            f"{field:<34}"
            f"mean={statistics.mean(values):.4f}s  "
            f"median={statistics.median(values):.4f}s"
        )

    print(f"\nResults saved to: {output_file}")

    spark.stop()


if __name__ == "__main__":
    main()
