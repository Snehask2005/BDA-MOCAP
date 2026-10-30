import csv
import math
import random
import time
from pathlib import Path

from pyspark.sql import SparkSession
from mocap.query.parser import QueryRequest
from mocap.plans.generator import generate_join_candidates


BASE = Path("benchmarks/tpcds/full_sf10/parquet")
SQL_PATH = Path("benchmarks/tpcds/representative/queries/q5_join_order.sql")
OUT = Path("results")
RUNS_FILE = OUT / "tpcds_full_sf10_execution_quality_runs.csv"
SUMMARY_FILE = OUT / "tpcds_full_sf10_execution_quality_summary.csv"

TABLES = [
    "store_sales", "item", "store",
    "date_dim", "customer_demographics",
]
WIDTHS = (2, 5, 10, 20)
SEED = 20261025
ABS_TOL = 1e-8
REL_TOL = 1e-12


def equivalent(reference_rows, candidate_rows):
    if len(reference_rows) != len(candidate_rows):
        return False

    def index_rows(rows):
        indexed = {}
        for row in rows:
            key = tuple(row[:-1])
            if key in indexed:
                return None
            indexed[key] = row[-1]
        return indexed

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
    if candidate.strategy == "default":
        return True
    return (
        candidate.strategy.startswith("join_order_")
        and not candidate.strategy.endswith(
            ("_broadcast", "_shuffle_hash")
        )
    )


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-TPCDS-Full-SF10-Execution-Quality")
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
                raise FileNotFoundError(f"Missing table: {path}")
            spark.read.parquet(str(path)).createOrReplaceTempView(table)
            print(f"Registered: {table}")

        query = QueryRequest(
            query_id="TPCDS_FULL_SF10_EXEC_QUALITY",
            sql=SQL_PATH.read_text().strip(),
            budget=100.0,
            deadline=100.0,
            accuracy_tolerance=0.0,
        )

        print("\nExecuting reference query...")
        reference_rows = spark.sql(query.sql).collect()
        print(f"Reference output rows: {len(reference_rows)}")

        modes = [("exhaustive", None)] + [
            (f"beam_{width}", width) for width in WIDTHS
        ]
        mode_sets = {}
        mode_stats = {}
        all_plans = {}

        print("\nGenerating candidates...")
        for mode, width in modes:
            start = time.perf_counter()
            candidates = generate_join_candidates(
                query=query,
                spark=spark,
                search_mode="exhaustive" if width is None else "beam",
                beam_width=20 if width is None else width,
            )
            planning = time.perf_counter() - start
            selected = [c for c in candidates if is_unhinted(c)]

            mode_sets[mode] = {c.fingerprint for c in selected}
            for candidate in selected:
                all_plans.setdefault(candidate.fingerprint, candidate)

            mode_stats[mode] = {
                "beam_width": "" if width is None else width,
                "total_candidates": len(candidates),
                "unhinted_candidates": len(selected),
                "planning_seconds": planning,
            }
            print(
                f"{mode:12} all={len(candidates):3} "
                f"unhinted={len(selected):3} planning={planning:.3f}s"
            )

        exhaustive_set = mode_sets["exhaustive"]
        for mode, fps in mode_sets.items():
            if not fps.issubset(exhaustive_set):
                print(
                    f"NOTE: {mode} has "
                    f"{len(fps - exhaustive_set)} plan(s) not found "
                    "in the exhaustive fingerprints."
                )

        if not exhaustive_set:
            raise RuntimeError("No exhaustive unhinted candidates found.")

        # Randomize candidate order to reduce systematic order effects.
        order = list(all_plans)
        random.Random(SEED).shuffle(order)
        records = []
        OUT.mkdir(parents=True, exist_ok=True)

        # Write each result immediately so completed runs are preserved.
        with RUNS_FILE.open("w", newline="") as file:
            fields = [
                "fingerprint", "strategy", "memberships",
                "execution_seconds", "result_equivalent",
                "result_rows", "error",
            ]
            writer = csv.DictWriter(
                file, fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()

            print(f"\nExecuting {len(order)} unique unhinted candidates...")
            for index, fp in enumerate(order, start=1):
                candidate = all_plans[fp]
                memberships = ";".join(
                    mode for mode, fps in mode_sets.items() if fp in fps
                )
                elapsed_text = ""
                rows_count = ""
                passed = False
                error = ""

                print(
                    f"\n[{index}/{len(order)}] {candidate.strategy}"
                    f" | {memberships}"
                )
                try:
                    start = time.perf_counter()
                    rows = spark.sql(candidate.sql).collect()
                    elapsed = time.perf_counter() - start
                    elapsed_text = f"{elapsed:.6f}"
                    rows_count = len(rows)
                    passed = equivalent(reference_rows, rows)
                    print(
                        f"  {'PASS' if passed else 'FAIL'} "
                        f"| {elapsed:.3f}s | rows={len(rows)}"
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"  ERROR: {error}")

                record = {
                    "fingerprint": fp,
                    "strategy": candidate.strategy,
                    "memberships": memberships,
                    "execution_seconds": elapsed_text,
                    "result_equivalent": passed,
                    "result_rows": rows_count,
                    "error": error,
                }
                writer.writerow(record)
                file.flush()
                records.append(record)

        successful = {
            row["fingerprint"]: float(row["execution_seconds"])
            for row in records
            if row["execution_seconds"] and row["result_equivalent"]
        }

        if not successful:
            raise RuntimeError("No candidate completed with matching results.")

        exhaustive_success = exhaustive_set.intersection(successful)
        if not exhaustive_success:
            raise RuntimeError("No exhaustive candidate passed validation.")

        exhaustive_best_fp = min(
            exhaustive_success, key=lambda fp: successful[fp]
        )
        exhaustive_best_time = successful[exhaustive_best_fp]
        summary_rows = []

        for mode, fps in mode_sets.items():
            passed_fps = fps.intersection(successful)
            best_fp = (
                min(passed_fps, key=lambda fp: successful[fp])
                if passed_fps else None
            )
            best_time = successful[best_fp] if best_fp else None

            summary_rows.append({
                "mode": mode,
                **mode_stats[mode],
                "successful_equivalent_candidates": len(passed_fps),
                "fastest_observed_strategy": (
                    all_plans[best_fp].strategy if best_fp else ""
                ),
                "fastest_observed_fingerprint": best_fp or "",
                "fastest_observed_seconds": (
                    f"{best_time:.6f}" if best_time is not None else ""
                ),
                "exhaustive_best_fingerprint": exhaustive_best_fp,
                "exhaustive_best_retained": (
                    exhaustive_best_fp in fps
                ),
                "observed_pruning_loss_percent": (
                    f"{(best_time / exhaustive_best_time - 1) * 100:.4f}"
                    if best_time is not None else ""
                ),
                "single_run_only": True,
            })

        with SUMMARY_FILE.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(summary_rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print("\n===== FULL-SF10 EXECUTION-QUALITY SUMMARY =====")
        for row in summary_rows:
            print(
                f"{row['mode']:12} "
                f"valid={row['successful_equivalent_candidates']:3} "
                f"fastest={row['fastest_observed_seconds']:>9}s "
                f"best_retained={row['exhaustive_best_retained']} "
                f"observed_loss={row['observed_pruning_loss_percent']}%"
            )

        print(f"\nDetailed runs: {RUNS_FILE}")
        print(f"Summary:       {SUMMARY_FILE}")
        print("NOTE: One execution per candidate; timing differences are preliminary.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
