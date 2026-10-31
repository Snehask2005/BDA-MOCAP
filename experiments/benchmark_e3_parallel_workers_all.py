import contextlib
import csv
import io
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.plans.generator import generate_join_candidates
from mocap.query.parser import QueryRequest

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "benchmarks/tpcds/full_sf10/parquet"
E3 = ROOT / "benchmarks/tpcds/e3/parquet"
QUERIES = ROOT / "benchmarks/tpcds/e3/queries"
OUT_DIR = ROOT / "results/e3/beam40_cap48"

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

WORKERS = (1, 2, 4)
REPETITIONS = 3
SEED = 20261029

RUN_FIELDS = [
    "timestamp_utc",
    "query_file",
    "table_count",
    "candidate_workers",
    "repetition",
    "planning_seconds",
    "unique_physical_plans",
    "analysis_error_count",
]


def run_one(spark, path, workers):
    query = QueryRequest(
        query_id=f"parallel_{path.stem}",
        sql=path.read_text(encoding="utf-8"),
        budget=100.0,
        deadline=300.0,
    )

    captured = io.StringIO()
    start = time.perf_counter()

    with contextlib.redirect_stdout(captured):
        candidates = generate_join_candidates(
            query,
            spark,
            search_mode="beam",
            beam_width=40,
            candidate_workers=workers,
            candidate_analysis_limit=48,
        )

    elapsed = time.perf_counter() - start
    errors = sum(
        "Skipping strategy" in line
        for line in captured.getvalue().splitlines()
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_file": path.name,
        "table_count": int(path.name[1:3]),
        "candidate_workers": workers,
        "planning_seconds": elapsed,
        "unique_physical_plans": len(candidates),
        "analysis_error_count": errors,
        "fingerprints": {c.fingerprint for c in candidates},
    }


def main():
    query_paths = sorted(QUERIES.glob("q*.sql"))
    if len(query_paths) != 8:
        raise RuntimeError(f"Expected 8 queries, found {len(query_paths)}")

    for path in [*TABLE_PATHS.values(), *query_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-All-Query-Parallel-Analysis")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    configs = [
        (path, workers)
        for path in query_paths
        for workers in WORKERS
    ]
    rng = random.Random(SEED)
    rows = []
    observed = defaultdict(list)

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)
            print(f"Registered {table}")

        print("\nWarm-ups (not recorded)")
        for path, workers in configs:
            row = run_one(spark, path, workers)
            print(
                f"{row['query_file']:18} workers={workers} "
                f"plans={row['unique_physical_plans']} "
                f"time={row['planning_seconds']:.3f}s "
                f"errors={row['analysis_error_count']}"
            )

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        runs_path = OUT_DIR / "parallel_workers_all_runs.csv"

        with runs_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=RUN_FIELDS, lineterminator="\n"
            )
            writer.writeheader()

            for repetition in range(1, REPETITIONS + 1):
                order = list(configs)
                rng.shuffle(order)
                print(f"\nMeasured repetition {repetition}/{REPETITIONS}")

                for path, workers in order:
                    row = run_one(spark, path, workers)
                    row["repetition"] = repetition
                    observed[(path.name, workers)].append(row["fingerprints"])

                    writer.writerow({
                        key: row[key] for key in RUN_FIELDS
                    })
                    f.flush()
                    rows.append(row)

                    print(
                        f"{path.name:18} workers={workers} "
                        f"plans={row['unique_physical_plans']} "
                        f"time={row['planning_seconds']:.3f}s "
                        f"errors={row['analysis_error_count']}"
                    )

        summary = []
        medians = {}

        print("\n===== ALL-QUERY PARALLEL WORKER SUMMARY =====")
        for path in query_paths:
            baseline = None
            print(f"\n{path.name}")

            for workers in WORKERS:
                group = [
                    row for row in rows
                    if row["query_file"] == path.name
                    and row["candidate_workers"] == workers
                ]
                median_time = statistics.median(
                    row["planning_seconds"] for row in group
                )
                medians[(path.name, workers)] = median_time

                stable = all(
                    fp_set == observed[(path.name, workers)][0]
                    for fp_set in observed[(path.name, workers)][1:]
                )
                plan_counts = sorted({
                    row["unique_physical_plans"] for row in group
                })
                errors = sum(row["analysis_error_count"] for row in group)

                saved = ""
                if workers == 1:
                    baseline = median_time
                else:
                    saved = round(100 * (baseline - median_time) / baseline, 2)

                print(
                    f"  workers={workers}: median={median_time:.3f}s "
                    f"plans={plan_counts} errors={errors} "
                    f"fingerprints_stable={stable} "
                    f"saved_vs_1={saved if saved != '' else 'baseline'}"
                )

                summary.append({
                    "query_file": path.name,
                    "table_count": int(path.name[1:3]),
                    "candidate_workers": workers,
                    "median_planning_seconds": f"{median_time:.6f}",
                    "unique_physical_plan_counts": ";".join(
                        str(value) for value in plan_counts
                    ),
                    "fingerprints_stable": stable,
                    "analysis_errors": errors,
                    "planning_saved_percent_vs_1_worker": saved,
                    "query_execution_performed": False,
                })

        summary_path = OUT_DIR / "parallel_workers_all_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(summary[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary)

        print(f"\nSaved runs: {runs_path}")
        print(f"Saved summary: {summary_path}")
        print("No query execution was performed.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
