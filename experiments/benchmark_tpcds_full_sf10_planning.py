import csv
import random
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates


BASE = Path("benchmarks/tpcds/full_sf10/parquet")
SQL_PATH = Path("benchmarks/tpcds/representative/queries/q5_join_order.sql")
OUTPUT = Path("results/tpcds_full_sf10_planning_summary.csv")
RUNS_OUTPUT = Path("results/tpcds_full_sf10_planning_runs.csv")

TABLES = [
    "store_sales",
    "item",
    "store",
    "date_dim",
    "customer_demographics",
]
WIDTHS = (2, 5, 10, 20)
REPETITIONS = 3
SEED = 20261025


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-TPCDS-Full-SF10-Planning")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        for table in TABLES:
            path = BASE / table
            if not path.exists():
                raise FileNotFoundError(f"Missing Parquet table: {path}")
            spark.read.parquet(str(path)).createOrReplaceTempView(table)
            print(f"Registered table: {table}")

        query = QueryRequest(
            query_id="TPCDS_FULL_SF10_PLANNING",
            sql=SQL_PATH.read_text().strip(),
            budget=100.0,
            deadline=100.0,
            accuracy_tolerance=0.0,
        )

        modes = [("exhaustive", None)] + [
            (f"beam_{width}", width) for width in WIDTHS
        ]
        measured_runs = {mode: [] for mode, _ in modes}
        counts = {}
        rng = random.Random(SEED)

        def run_one(mode, width):
            start = time.perf_counter()
            candidates = generate_join_candidates(
                query=query,
                spark=spark,
                search_mode=mode,
                beam_width=20 if width is None else width,
            )
            elapsed = time.perf_counter() - start

            if not candidates:
                raise RuntimeError(f"No candidates generated for {mode}")

            fingerprints = {c.fingerprint for c in candidates}
            join_orders = {
                c.strategy for c in candidates
                if c.strategy.startswith("join_order_")
            }
            return elapsed, len(candidates), len(fingerprints), len(join_orders)

        print("\nWarm-up: one planning pass per mode")
        for mode, width in modes:
            elapsed, candidate_count, unique_count, join_order_count = (
                run_one("exhaustive" if width is None else "beam", width)
            )
            counts[mode] = {
                "candidate_count": candidate_count,
                "unique_fingerprints": unique_count,
                "join_order_variants": join_order_count,
            }
            print(
                f"{mode:12} candidates={candidate_count:3} "
                f"planning={elapsed:.3f}s (warm-up)"
            )

        print(f"\nMeasured runs: {REPETITIONS} per mode")
        for repetition in range(1, REPETITIONS + 1):
            order = list(modes)
            rng.shuffle(order)

            for mode, width in order:
                elapsed, candidate_count, unique_count, join_order_count = (
                    run_one("exhaustive" if width is None else "beam", width)
                )
                measured_runs[mode].append(elapsed)

                expected = counts[mode]
                if (
                    candidate_count != expected["candidate_count"]
                    or unique_count != expected["unique_fingerprints"]
                ):
                    raise RuntimeError(
                        f"Candidate counts changed across runs for {mode}"
                    )

                print(
                    f"run {repetition}/{REPETITIONS} {mode:12} "
                    f"{elapsed:.3f}s"
                )

        summary_rows = []
        run_rows = []

        for mode, width in modes:
            values = measured_runs[mode]
            summary_rows.append({
                "mode": mode,
                "beam_width": "" if width is None else width,
                **counts[mode],
                "measured_repetitions": len(values),
                "median_planning_seconds": f"{statistics.median(values):.6f}",
                "min_planning_seconds": f"{min(values):.6f}",
                "max_planning_seconds": f"{max(values):.6f}",
                "planning_only": True,
                "query_execution_performed": False,
            })

            for repetition, elapsed in enumerate(values, start=1):
                run_rows.append({
                    "mode": mode,
                    "beam_width": "" if width is None else width,
                    "repetition": repetition,
                    "planning_seconds": f"{elapsed:.6f}",
                    "candidate_count": counts[mode]["candidate_count"],
                })

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        for path, rows in [
            (OUTPUT, summary_rows),
            (RUNS_OUTPUT, run_rows),
        ]:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(rows[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {path}")

        print("\n===== FULL-SF10 PLANNING SUMMARY =====")
        for row in summary_rows:
            print(
                f"{row['mode']:12} "
                f"candidates={row['candidate_count']:3} "
                f"median={float(row['median_planning_seconds']):.3f}s "
                f"range={float(row['min_planning_seconds']):.3f}-"
                f"{float(row['max_planning_seconds']):.3f}s"
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
