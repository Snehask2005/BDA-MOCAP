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
OUT = Path("results/parallel_candidate_analysis_summary.csv")
RUNS_OUT = Path("results/parallel_candidate_analysis_runs.csv")

TABLES = [
    "store_sales",
    "item",
    "store",
    "date_dim",
    "customer_demographics",
]
WORKER_COUNTS = (1, 2, 4)
SEARCH_MODES = (
    ("exhaustive", None),
    ("beam_10", 10),
)
REPETITIONS = 3
SEED = 20261026


def candidate_signature(candidates):
    """Capture order and identity to detect output changes."""
    return [
        (
            candidate.plan_id,
            candidate.strategy,
            candidate.fingerprint,
            candidate.sql,
        )
        for candidate in candidates
    ]


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Parallel-Candidate-Analysis-Benchmark")
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
            print(f"Registered: {table}")

        query = QueryRequest(
            query_id="TPCDS_PARALLEL_ANALYSIS",
            sql=SQL_PATH.read_text().strip(),
            budget=100.0,
            deadline=100.0,
            accuracy_tolerance=0.0,
        )

        configs = [
            (mode_name, width, workers)
            for mode_name, width in SEARCH_MODES
            for workers in WORKER_COUNTS
        ]
        rng = random.Random(SEED)
        baseline_signatures = {}
        measurements = {config: [] for config in configs}
        candidate_counts = {}

        def run_one(mode_name, width, workers):
            actual_mode = "exhaustive" if width is None else "beam"
            start = time.perf_counter()
            candidates = generate_join_candidates(
                query=query,
                spark=spark,
                search_mode=actual_mode,
                beam_width=20 if width is None else width,
                candidate_workers=workers,
            )
            elapsed = time.perf_counter() - start
            signature = candidate_signature(candidates)

            if not candidates:
                raise RuntimeError(
                    f"No candidates generated for "
                    f"{mode_name} with {workers} workers."
                )

            key = mode_name
            if key not in baseline_signatures:
                if workers == 1:
                    baseline_signatures[key] = signature
            elif signature != baseline_signatures[key]:
                raise RuntimeError(
                    f"Candidate output changed for {mode_name} "
                    f"with candidate_workers={workers}; "
                    "parallel output must match sequential output."
                )

            # In case worker configurations are visited out of order,
            # establish a sequential baseline explicitly in the caller.
            candidate_counts[(mode_name, workers)] = len(candidates)
            return elapsed, len(candidates), signature

        print("\nEstablishing sequential baselines...")
        for mode_name, width in SEARCH_MODES:
            elapsed, count, signature = run_one(mode_name, width, 1)
            baseline_signatures[mode_name] = signature
            candidate_counts[(mode_name, 1)] = count
            print(
                f"{mode_name:12} workers=1 "
                f"candidates={count:3} warm-up={elapsed:.3f}s"
            )

        # Warm up all parallel configurations before measuring.
        print("\nWarming up parallel configurations...")
        warmups = [
            (mode_name, width, workers)
            for mode_name, width in SEARCH_MODES
            for workers in WORKER_COUNTS
            if workers != 1
        ]
        rng.shuffle(warmups)

        for mode_name, width, workers in warmups:
            elapsed, count, _ = run_one(mode_name, width, workers)
            print(
                f"{mode_name:12} workers={workers} "
                f"candidates={count:3} warm-up={elapsed:.3f}s"
            )

        print(f"\nMeasured repetitions: {REPETITIONS}")
        for repetition in range(1, REPETITIONS + 1):
            order = configs.copy()
            rng.shuffle(order)

            for mode_name, width, workers in order:
                elapsed, count, _ = run_one(
                    mode_name, width, workers
                )
                measurements[(mode_name, width, workers)].append(elapsed)
                print(
                    f"run {repetition}/{REPETITIONS} "
                    f"{mode_name:12} workers={workers} "
                    f"{elapsed:.3f}s"
                )

        summary_rows = []
        run_rows = []

        for mode_name, width in SEARCH_MODES:
            sequential_median = statistics.median(
                measurements[(mode_name, width, 1)]
            )

            for workers in WORKER_COUNTS:
                values = measurements[(mode_name, width, workers)]
                median_seconds = statistics.median(values)

                summary_rows.append({
                    "search_mode": mode_name,
                    "beam_width": "" if width is None else width,
                    "candidate_workers": workers,
                    "candidate_count": candidate_counts[(mode_name, workers)],
                    "repetitions": len(values),
                    "median_seconds": f"{median_seconds:.6f}",
                    "min_seconds": f"{min(values):.6f}",
                    "max_seconds": f"{max(values):.6f}",
                    "speedup_vs_sequential": (
                        f"{sequential_median / median_seconds:.4f}"
                    ),
                    "reduction_vs_sequential_percent": (
                        f"{(1 - median_seconds / sequential_median) * 100:.2f}"
                    ),
                    "outputs_match_sequential": True,
                })

                for repetition, elapsed in enumerate(values, start=1):
                    run_rows.append({
                        "search_mode": mode_name,
                        "beam_width": "" if width is None else width,
                        "candidate_workers": workers,
                        "repetition": repetition,
                        "planning_seconds": f"{elapsed:.6f}",
                    })

        for path, rows in (
            (OUT, summary_rows),
            (RUNS_OUT, run_rows),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(rows[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {path}")

        print("\n===== PARALLEL ANALYSIS SUMMARY =====")
        for row in summary_rows:
            print(
                f"{row['search_mode']:12} "
                f"workers={row['candidate_workers']} "
                f"candidates={row['candidate_count']:3} "
                f"median={float(row['median_seconds']):.3f}s "
                f"reduction={row['reduction_vs_sequential_percent']}%"
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
