import csv
import math
import random
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession
from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates


BASE = Path("benchmarks/tpcds/full_sf10/parquet")
SQL_PATH = Path("benchmarks/tpcds/representative/queries/q5_join_order.sql")
FIRST_RUN = Path("results/tpcds_full_sf10_execution_quality_runs.csv")
RUNS_OUT = Path("results/tpcds_full_sf10_finalist_repeat_runs.csv")
SUMMARY_OUT = Path("results/tpcds_full_sf10_finalist_repeat_summary.csv")

TABLES = [
    "store_sales", "item", "store",
    "date_dim", "customer_demographics",
]
WIDTHS = (2, 5, 10, 20)
TOP_K = 2
REPETITIONS = 3
SEED = 20261026
ABS_TOL = 1e-8
REL_TOL = 1e-12


def equivalent(reference_rows, candidate_rows):
    if len(reference_rows) != len(candidate_rows):
        return False

    def index_rows(rows):
        result = {}
        for row in rows:
            key = tuple(row[:-1])
            if key in result:
                return None
            result[key] = row[-1]
        return result

    ref = index_rows(reference_rows)
    got = index_rows(candidate_rows)

    if ref is None or got is None or ref.keys() != got.keys():
        return False

    for key, expected in ref.items():
        actual = got[key]
        if expected is None or actual is None:
            if expected is not actual:
                return False
        elif not math.isclose(
            float(expected), float(actual),
            rel_tol=REL_TOL, abs_tol=ABS_TOL,
        ):
            return False

    return True


def is_unhinted(candidate):
    return (
        candidate.strategy == "default"
        or (
            candidate.strategy.startswith("join_order_")
            and not candidate.strategy.endswith(
                ("_broadcast", "_shuffle_hash")
            )
        )
    )


def main():
    first_rows = list(csv.DictReader(FIRST_RUN.open()))
    modes = ["exhaustive"] + [f"beam_{w}" for w in WIDTHS]
    selected_by_mode = {}
    selected_fingerprints = set()

    for mode in modes:
        eligible = [
            row for row in first_rows
            if mode in row["memberships"].split(";")
            and row["execution_seconds"]
            and row["result_equivalent"].lower() == "true"
        ]
        eligible.sort(key=lambda row: float(row["execution_seconds"]))
        selected = eligible[:TOP_K]
        if not selected:
            raise RuntimeError(f"No valid finalists found for {mode}")

        selected_by_mode[mode] = {row["fingerprint"] for row in selected}
        selected_fingerprints.update(selected_by_mode[mode])

        print(
            f"{mode}: shortlisted "
            + ", ".join(
                f"{r['strategy']} ({float(r['execution_seconds']):.3f}s)"
                for r in selected
            )
        )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-TPCDS-Full-SF10-Finalist-Repeats")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        for table in TABLES:
            spark.read.parquet(str(BASE / table)).createOrReplaceTempView(table)

        query = QueryRequest(
            query_id="TPCDS_FULL_SF10_EXEC_QUALITY",
            sql=SQL_PATH.read_text().strip(),
            budget=100.0,
            deadline=100.0,
            accuracy_tolerance=0.0,
        )
        reference_rows = spark.sql(query.sql).collect()

        plans = {}
        for mode, width in (
            [("exhaustive", None)]
            + [(f"beam_{w}", w) for w in WIDTHS]
        ):
            candidates = generate_join_candidates(
                query=query,
                spark=spark,
                search_mode="exhaustive" if width is None else "beam",
                beam_width=20 if width is None else width,
            )
            for candidate in candidates:
                if is_unhinted(candidate):
                    plans.setdefault(candidate.fingerprint, candidate)

        missing = selected_fingerprints - set(plans)
        if missing:
            raise RuntimeError(
                f"Could not regenerate shortlisted plans: {sorted(missing)}"
            )

        selected = sorted(selected_fingerprints)
        samples = {fp: [] for fp in selected}
        rng = random.Random(SEED)

        print(f"\nUnique finalists: {len(selected)}")
        print("Running one warm-up per finalist...")
        warmup_order = selected.copy()
        rng.shuffle(warmup_order)

        for fp in warmup_order:
            rows = spark.sql(plans[fp].sql).collect()
            if not equivalent(reference_rows, rows):
                raise RuntimeError(
                    f"Correctness failed during warm-up: "
                    f"{plans[fp].strategy}"
                )

        run_rows = []
        for repetition in range(1, REPETITIONS + 1):
            order = selected.copy()
            rng.shuffle(order)
            print(f"\nMeasured round {repetition}/{REPETITIONS}")

            for fp in order:
                candidate = plans[fp]
                start = time.perf_counter()
                rows = spark.sql(candidate.sql).collect()
                elapsed = time.perf_counter() - start

                passed = equivalent(reference_rows, rows)
                if not passed:
                    raise RuntimeError(
                        f"Correctness failed during repeat: "
                        f"{candidate.strategy}"
                    )

                samples[fp].append(elapsed)
                membership = [
                    mode for mode in modes
                    if fp in selected_by_mode[mode]
                ]

                run_rows.append({
                    "fingerprint": fp,
                    "strategy": candidate.strategy,
                    "selected_for_modes": ";".join(membership),
                    "repetition": repetition,
                    "execution_seconds": f"{elapsed:.6f}",
                    "result_equivalent": passed,
                })
                print(f"  {candidate.strategy}: {elapsed:.3f}s")

        medians = {
            fp: statistics.median(values)
            for fp, values in samples.items()
        }

        # Repeated median of the exhaustive winner from the first run.
        first_exhaustive = [
            row for row in first_rows
            if "exhaustive" in row["memberships"].split(";")
            and row["execution_seconds"]
            and row["result_equivalent"].lower() == "true"
        ]
        first_exhaustive.sort(
            key=lambda row: float(row["execution_seconds"])
        )
        first_run_winner_fp = first_exhaustive[0]["fingerprint"]

        if first_run_winner_fp not in medians:
            raise RuntimeError(
                "The first-run exhaustive winner was not shortlisted."
            )

        winner_reference_median = medians[first_run_winner_fp]
        summary_rows = []

        for mode in modes:
            fps = selected_by_mode[mode]
            best_fp = min(fps, key=lambda fp: medians[fp])
            best_median = medians[best_fp]

            summary_rows.append({
                "mode": mode,
                "shortlisted_candidates": len(fps),
                "repetitions_per_candidate": REPETITIONS,
                "best_repeated_median_seconds": f"{best_median:.6f}",
                "best_strategy": plans[best_fp].strategy,
                "best_fingerprint": best_fp,
                "first_run_exhaustive_winner_retained": (
                    first_run_winner_fp in selected_by_mode[mode]
                ),
                "loss_vs_repeated_exhaustive_winner_percent": (
                    f"{(best_median / winner_reference_median - 1) * 100:.4f}"
                ),
                "timing_scope": "shortlisted finalists, local full-data sample",
            })

        candidate_rows = []
        for fp in selected:
            values = samples[fp]
            candidate_rows.append({
                "fingerprint": fp,
                "strategy": plans[fp].strategy,
                "selected_for_modes": ";".join(
                    mode for mode in modes
                    if fp in selected_by_mode[mode]
                ),
                "single_run_seconds": next(
                    float(row["execution_seconds"])
                    for row in first_rows
                    if row["fingerprint"] == fp
                ),
                "repeat_median_seconds": f"{statistics.median(values):.6f}",
                "repeat_min_seconds": f"{min(values):.6f}",
                "repeat_max_seconds": f"{max(values):.6f}",
                "repeat_stdev_seconds": (
                    f"{statistics.stdev(values):.6f}"
                    if len(values) > 1 else ""
                ),
                "all_repeats_equivalent": True,
            })

        RUNS_OUT.parent.mkdir(parents=True, exist_ok=True)
        for path, rows in (
            (RUNS_OUT, run_rows),
            (SUMMARY_OUT, summary_rows),
            (Path("results/tpcds_full_sf10_finalist_repeat_candidates.csv"),
             candidate_rows),
        ):
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(rows[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {path}")

        print("\n===== REPEATED FINALIST SUMMARY =====")
        for row in summary_rows:
            print(
                f"{row['mode']:12} "
                f"best_median={row['best_repeated_median_seconds']}s "
                f"loss_vs_initial_exhaustive_winner="
                f"{row['loss_vs_repeated_exhaustive_winner_percent']}%"
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
