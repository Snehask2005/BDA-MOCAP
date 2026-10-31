import contextlib
import csv
import io
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from pyspark.sql import SparkSession

import mocap.plans.generator as generator
from mocap.query.parser import QueryRequest

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "benchmarks/tpcds/full_sf10/parquet"
E3 = ROOT / "benchmarks/tpcds/e3/parquet"
QUERIES = ROOT / "benchmarks/tpcds/e3/queries"
OUTPUT = ROOT / "results/e3/beam40_cap48/parallel_workers_q09_q10_runs.csv"

TABLE_PATHS = {
    "store_sales": FULL / "store_sales",
    "item": FULL / "item",
    "store": FULL / "store",
    "date_dim": FULL / "date_dim",
    "customer_demographics": FULL / "customer_demographics",
    "customer": E3 / "customer",
    "customer_address": E3 / "customer_address",
    "household_demographics": E3 / "household_demographics",
    "time_dim": E3 / "time_dim",
    "promotion": E3 / "promotion",
}

QUERY_FILES = [
    QUERIES / "q09_9_tables.sql",
    QUERIES / "q10_10_tables.sql",
]
WORKER_COUNTS = [1, 2, 4]
REPETITIONS = 3
SEED = 20261029

def run_one(spark, query_path, workers):
    query = QueryRequest(
        query_id=f"parallel_{query_path.stem}",
        sql=query_path.read_text(encoding="utf-8"),
        budget=100.0,
        deadline=300.0,
    )

    captured = io.StringIO()
    started = time.perf_counter()

    with contextlib.redirect_stdout(captured):
        candidates = generator.generate_join_candidates(
            query,
            spark,
            search_mode="beam",
            beam_width=40,
            candidate_workers=workers,
            candidate_analysis_limit=48,
        )

    elapsed = time.perf_counter() - started
    errors = sum(
        "Skipping strategy" in line
        for line in captured.getvalue().splitlines()
    )

    return {
        "query_file": query_path.name,
        "candidate_workers": workers,
        "planning_seconds": elapsed,
        "unique_physical_plans": len(candidates),
        "analysis_error_count": errors,
    }

def main():
    for path in [*TABLE_PATHS.values(), *QUERY_FILES]:
        if not path.exists():
            raise FileNotFoundError(path)

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Parallel-Plan-Analysis")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    configs = [
        (query_path, workers)
        for query_path in QUERY_FILES
        for workers in WORKER_COUNTS
    ]
    rng = random.Random(SEED)
    rows = []

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)
            print(f"Registered {table}")

        print("\nWarm-up runs (not recorded)")
        for query_path, workers in configs:
            row = run_one(spark, query_path, workers)
            print(
                f"{row['query_file']:18} workers={workers} "
                f"plans={row['unique_physical_plans']} "
                f"time={row['planning_seconds']:.3f}s "
                f"errors={row['analysis_error_count']}"
            )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "query_file",
                "candidate_workers",
                "repetition",
                "planning_seconds",
                "unique_physical_plans",
                "analysis_error_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()

            for repetition in range(1, REPETITIONS + 1):
                order = list(configs)
                rng.shuffle(order)
                print(f"\nMeasured repetition {repetition}/{REPETITIONS}")

                for query_path, workers in order:
                    row = run_one(spark, query_path, workers)
                    row["repetition"] = repetition
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)

                    print(
                        f"{row['query_file']:18} workers={workers} "
                        f"plans={row['unique_physical_plans']} "
                        f"time={row['planning_seconds']:.3f}s "
                        f"errors={row['analysis_error_count']}"
                    )

        groups = defaultdict(list)
        for row in rows:
            groups[(row["query_file"], row["candidate_workers"])].append(row)

        print("\n===== PARALLEL WORKER SUMMARY =====")
        medians = {}

        for query_path in QUERY_FILES:
            name = query_path.name
            print(f"\n{name}")

            for workers in WORKER_COUNTS:
                group = groups[(name, workers)]
                median_time = statistics.median(
                    float(row["planning_seconds"]) for row in group
                )
                errors = sum(
                    int(row["analysis_error_count"]) for row in group
                )
                plan_counts = sorted({
                    int(row["unique_physical_plans"]) for row in group
                })
                medians[(name, workers)] = median_time

                print(
                    f"  workers={workers}: median={median_time:.3f}s "
                    f"plans={plan_counts} errors={errors}"
                )

            baseline = medians[(name, 1)]
            for workers in (2, 4):
                parallel = medians[(name, workers)]
                saved = 100 * (baseline - parallel) / baseline
                print(
                    f"  workers={workers} vs 1: "
                    f"{saved:+.1f}% planning-time reduction"
                )

        print(f"\nSaved runs: {OUTPUT}")
        print("No query execution was performed.")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
