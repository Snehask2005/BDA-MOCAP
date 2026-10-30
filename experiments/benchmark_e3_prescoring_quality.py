import argparse
import contextlib
import csv
import io
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession

import mocap.plans.generator as generator
from mocap.query.parser import QueryRequest


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "benchmarks/tpcds/full_sf10/parquet"
E3 = ROOT / "benchmarks/tpcds/e3/parquet"
QUERIES = ROOT / "benchmarks/tpcds/e3/queries"
RESULTS = ROOT / "results/e3"

QUERY_FILE = QUERIES / "q10_10_tables.sql"
SEED = 20261027
TOP_FINALISTS = 3

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

RUN_FIELDS = [
    "timestamp_utc",
    "phase",
    "mode",
    "repetition",
    "fingerprint",
    "strategy",
    "execution_seconds",
    "row_count",
    "total_sales",
    "result_equivalent",
    "error",
]

SUMMARY_FIELDS = [
    "mode",
    "candidate_analysis_limit",
    "unique_physical_plans",
    "screened_finalists",
    "best_finalist_strategy",
    "best_finalist_fingerprint",
    "best_finalist_median_seconds",
    "baseline_best_fingerprint_retained",
    "latency_change_percent_vs_baseline",
    "all_screened_results_equivalent",
    "execution_errors",
    "measured_repetitions_per_finalist",
]


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def execute_candidate(spark, candidate):
    """Execute a candidate to completion and time the end-to-end call."""
    started = time.perf_counter()

    try:
        rows = spark.sql(candidate.sql).collect()
        elapsed = time.perf_counter() - started

        if len(rows) != 1:
            raise ValueError(
                f"Expected one aggregate result row; received {len(rows)}"
            )

        values = {
            str(key).lower(): value
            for key, value in rows[0].asDict().items()
        }

        count = int(values["row_count"])
        sales = (
            None
            if values["total_sales"] is None
            else float(values["total_sales"])
        )

        return elapsed, (count, sales), ""

    except Exception as exc:
        return time.perf_counter() - started, None, repr(exc)[:400]


def results_match(left, right):
    if left is None or right is None:
        return False

    if left[0] != right[0]:
        return False

    if left[1] is None or right[1] is None:
        return left[1] is None and right[1] is None

    return math.isclose(
        left[1],
        right[1],
        rel_tol=1e-9,
        abs_tol=1e-5,
    )


def build_run_row(
    *,
    phase,
    mode,
    repetition,
    candidate,
    elapsed,
    result,
    equivalent,
    error,
):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "mode": mode,
        "repetition": repetition,
        "fingerprint": candidate.fingerprint,
        "strategy": candidate.strategy,
        "execution_seconds": f"{elapsed:.6f}",
        "row_count": "" if result is None else result[0],
        "total_sales": (
            "" if result is None or result[1] is None else result[1]
        ),
        "result_equivalent": "" if equivalent is None else equivalent,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Measured repetitions for each finalist (default: 5).",
    )
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    missing = [
        str(path)
        for path in TABLE_PATHS.values()
        if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing input directories:\n" + "\n".join(missing)
        )

    if not QUERY_FILE.is_file():
        raise FileNotFoundError(QUERY_FILE)

    query = QueryRequest(
        query_id="e3_q10_prescoring_quality",
        sql=QUERY_FILE.read_text(encoding="utf-8"),
        budget=100.0,
        deadline=300.0,
    )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-PreScoring-Quality")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    output_runs = RESULTS / "prescoring_q10_quality_runs.csv"
    output_summary = RESULTS / "prescoring_q10_quality_summary.csv"

    rng = random.Random(SEED)
    all_runs = []
    reference_result = None
    reference_assigned = False

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)
            print(f"Registered {table}")

        print(f"\nSpark version: {spark.version}")
        print("Master: local[2]; AQE=false; CBO=false; join reorder=false")
        print("This benchmark EXECUTES queries; it is not planning-only.")

        modes = [
            ("baseline", None),
            ("prescore_12", 12),
            ("prescore_20", 20),
        ]
        candidates_by_mode = {}

        print("\n===== GENERATING CANDIDATE SETS =====")
        for mode, limit in modes:
            captured = io.StringIO()

            with contextlib.redirect_stdout(captured):
                candidates = generator.generate_join_candidates(
                    query,
                    spark,
                    search_mode="beam",
                    beam_width=10,
                    candidate_workers=1,
                    candidate_analysis_limit=limit,
                )

            skipped = [
                line for line in captured.getvalue().splitlines()
                if "Skipping strategy" in line
            ]
            if skipped:
                raise RuntimeError(
                    f"{mode}: candidate generation reported errors: "
                    + "; ".join(skipped[:3])
                )

            candidates_by_mode[mode] = candidates
            print(
                f"{mode:12} limit={str(limit):>4} "
                f"unique_physical_plans={len(candidates)}"
            )

        baseline = candidates_by_mode["baseline"]
        baseline_by_fp = {
            candidate.fingerprint: candidate for candidate in baseline
        }
        baseline_fingerprints = set(baseline_by_fp)

        plan_sets = {
            mode: {candidate.fingerprint for candidate in candidates}
            for mode, candidates in candidates_by_mode.items()
        }

        for mode, fingerprints in plan_sets.items():
            if not fingerprints.issubset(baseline_fingerprints):
                raise RuntimeError(
                    f"{mode} contains fingerprints absent from the baseline"
                )

        # Execute every unique physical plan in the baseline once.
        screen_candidates = list(baseline)
        rng.shuffle(screen_candidates)
        screen_rows_by_fp = {}

        print("\n===== SCREENING ALL BASELINE PLANS =====")
        screen_output = []

        for index, candidate in enumerate(screen_candidates, start=1):
            elapsed, result, error = execute_candidate(spark, candidate)

            if error:
                equivalent = False
            else:
                if not reference_assigned:
                    reference_result = result
                    reference_assigned = True
                equivalent = results_match(result, reference_result)

            row = build_run_row(
                phase="screen",
                mode="baseline_all_plans",
                repetition=0,
                candidate=candidate,
                elapsed=elapsed,
                result=result,
                equivalent=equivalent,
                error=error,
            )
            screen_output.append(row)
            all_runs.append(row)
            screen_rows_by_fp[candidate.fingerprint] = row

            save_csv(output_runs, all_runs, RUN_FIELDS)

            print(
                f"screen {index}/{len(screen_candidates)} "
                f"{candidate.strategy:35} "
                f"{elapsed:.3f}s "
                f"equivalent={equivalent} "
                f"error={'yes' if error else 'no'}"
            )

        if not reference_assigned:
            raise RuntimeError("No candidate produced a successful result.")

        # Choose up to three fastest screened plans from each set.
        finalists_by_mode = {}
        finalists = set()

        print("\n===== SELECTING FINALISTS =====")
        for mode, _limit in modes:
            valid = [
                fingerprint
                for fingerprint in plan_sets[mode]
                if fingerprint in screen_rows_by_fp
                and screen_rows_by_fp[fingerprint]["result_equivalent"] == "True"
                and not screen_rows_by_fp[fingerprint]["error"]
            ]
            valid.sort(
                key=lambda fingerprint: float(
                    screen_rows_by_fp[fingerprint]["execution_seconds"]
                )
            )
            chosen = valid[:TOP_FINALISTS]
            finalists_by_mode[mode] = chosen
            finalists.update(chosen)

            print(
                f"{mode:12}: {len(chosen)} finalists "
                f"from {len(plan_sets[mode])} plans"
            )

        if not finalists:
            raise RuntimeError("No valid finalists were available.")

        # One unmeasured warm-up per unique finalist.
        print("\n===== FINALIST WARM-UP =====")
        for fingerprint in sorted(finalists):
            candidate = baseline_by_fp[fingerprint]
            elapsed, result, error = execute_candidate(spark, candidate)
            equivalent = (
                not error
                and results_match(result, reference_result)
            )
            print(
                f"{candidate.strategy:35} {elapsed:.3f}s "
                f"equivalent={equivalent} "
                f"error={'yes' if error else 'no'}"
            )

        # Measure finalist execution times repeatedly in randomized order.
        finalist_rows = []

        for repetition in range(1, args.repetitions + 1):
            order = list(finalists)
            rng.shuffle(order)

            print(
                f"\n===== FINALIST REPETITION "
                f"{repetition}/{args.repetitions} ====="
            )

            for fingerprint in order:
                candidate = baseline_by_fp[fingerprint]
                elapsed, result, error = execute_candidate(spark, candidate)
                equivalent = (
                    not error
                    and results_match(result, reference_result)
                )

                row = build_run_row(
                    phase="finalist",
                    mode="shared_finalists",
                    repetition=repetition,
                    candidate=candidate,
                    elapsed=elapsed,
                    result=result,
                    equivalent=equivalent,
                    error=error,
                )
                finalist_rows.append(row)
                all_runs.append(row)

                save_csv(output_runs, all_runs, RUN_FIELDS)

                print(
                    f"{candidate.strategy:35} {elapsed:.3f}s "
                    f"equivalent={equivalent} "
                    f"error={'yes' if error else 'no'}"
                )

        # Calculate a median for each finalist.
        times_by_fp = {}
        for fingerprint in finalists:
            times = [
                float(row["execution_seconds"])
                for row in finalist_rows
                if row["fingerprint"] == fingerprint
                and row["result_equivalent"] == "True"
                and not row["error"]
            ]
            if times:
                times_by_fp[fingerprint] = statistics.median(times)

        best_by_mode = {}
        for mode, chosen in finalists_by_mode.items():
            eligible = [
                fingerprint
                for fingerprint in chosen
                if fingerprint in times_by_fp
            ]
            if not eligible:
                continue
            best_by_mode[mode] = min(
                eligible,
                key=lambda fingerprint: times_by_fp[fingerprint],
            )

        if "baseline" not in best_by_mode:
            raise RuntimeError(
                "No baseline finalist had valid repeated execution measurements."
            )

        baseline_best = best_by_mode["baseline"]
        baseline_median = times_by_fp[baseline_best]

        summary_rows = []
        limits = {
            "baseline": "",
            "prescore_12": 12,
            "prescore_20": 20,
        }

        for mode, _limit in modes:
            chosen = finalists_by_mode[mode]
            best = best_by_mode.get(mode)
            best_median = times_by_fp.get(best) if best else None

            if best_median is None:
                latency_change = ""
            else:
                latency_change = (
                    f"{100.0 * (best_median - baseline_median) / baseline_median:.2f}"
                )

            screened_rows = [
                screen_rows_by_fp[fp]
                for fp in plan_sets[mode]
                if fp in screen_rows_by_fp
            ]
            equivalent = all(
                row["result_equivalent"] == "True" and not row["error"]
                for row in screened_rows
            )

            summary_rows.append({
                "mode": mode,
                "candidate_analysis_limit": limits[mode],
                "unique_physical_plans": len(plan_sets[mode]),
                "screened_finalists": "; ".join(
                    baseline_by_fp[fp].strategy for fp in chosen
                ),
                "best_finalist_strategy": (
                    baseline_by_fp[best].strategy if best else ""
                ),
                "best_finalist_fingerprint": best or "",
                "best_finalist_median_seconds": (
                    f"{best_median:.6f}" if best_median is not None else ""
                ),
                "baseline_best_fingerprint_retained": (
                    baseline_best in plan_sets[mode]
                ),
                "latency_change_percent_vs_baseline": latency_change,
                "all_screened_results_equivalent": equivalent,
                "execution_errors": sum(
                    bool(row["error"])
                    for row in screened_rows
                ) + sum(
                    bool(row["error"])
                    for row in finalist_rows
                    if row["fingerprint"] in plan_sets[mode]
                ),
                "measured_repetitions_per_finalist": args.repetitions,
            })

        save_csv(output_summary, summary_rows, SUMMARY_FIELDS)

        print(f"\nSaved raw execution runs: {output_runs}")
        print(f"Saved summary: {output_summary}")
        print("\n===== E3 PRE-SCORING EXECUTION-QUALITY SUMMARY =====")

        for row in summary_rows:
            print(
                f"{row['mode']:12} "
                f"plans={row['unique_physical_plans']:>2} "
                f"best_median={row['best_finalist_median_seconds']:>9}s "
                f"baseline_best_retained="
                f"{row['baseline_best_fingerprint_retained']} "
                f"latency_change="
                f"{row['latency_change_percent_vs_baseline']}% "
                f"results_equivalent="
                f"{row['all_screened_results_equivalent']}"
            )

        print(
            "\nNOTE: These are finalist-based estimates. The full baseline "
            "set was screened once; only its top three screened plans per "
            "configuration were repeated. This is not yet a full exhaustive "
            "pruning-loss estimate."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
