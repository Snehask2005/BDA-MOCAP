import argparse
import csv
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession

import mocap.plans.generator as generator
from mocap.query.parser import QueryRequest


ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "benchmarks/tpcds/representative/parquet"
QUERIES = ROOT / "benchmarks/tpcds/e3/queries"
RESULTS = ROOT / "results/e3"

QUERY_FILE = QUERIES / "q05_5_tables.sql"
SEED = 20261028
DEFAULT_REPETITIONS = 5

TABLE_PATHS = {
    "store_sales": REP / "store_sales",
    "item": REP / "item",
    "store": REP / "store",
    "date_dim": REP / "date_dim",
    "customer_demographics": REP / "customer_demographics",
}

RUN_FIELDS = [
    "timestamp_utc",
    "phase",
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
    "candidate_plan_count",
    "valid_measured_plan_count",
    "best_strategy",
    "best_fingerprint",
    "best_median_seconds",
    "baseline_best_fingerprint_retained",
    "latency_change_percent_vs_baseline",
    "all_candidate_results_equivalent",
    "execution_errors",
    "repetitions_per_plan",
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


def execute_sql(spark, sql):
    started = time.perf_counter()

    try:
        rows = spark.sql(sql).collect()
        elapsed = time.perf_counter() - started

        if len(rows) != 1:
            raise ValueError(f"Expected one result row; received {len(rows)}")

        values = {
            str(key).lower(): value
            for key, value in rows[0].asDict().items()
        }

        result = (
            int(values["row_count"]),
            None if values["total_sales"] is None
            else float(values["total_sales"]),
        )
        return elapsed, result, ""

    except Exception as exc:
        return time.perf_counter() - started, None, repr(exc)[:400]


def results_match(actual, reference):
    if actual is None or reference is None:
        return False
    if actual[0] != reference[0]:
        return False
    if actual[1] is None or reference[1] is None:
        return actual[1] is None and reference[1] is None

    return abs(actual[1] - reference[1]) <= max(
        1e-5,
        1e-9 * abs(reference[1]),
    )


def make_row(phase, repetition, candidate, elapsed, result, matches, error):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "repetition": repetition,
        "fingerprint": candidate.fingerprint,
        "strategy": candidate.strategy,
        "execution_seconds": f"{elapsed:.6f}",
        "row_count": "" if result is None else result[0],
        "total_sales": (
            "" if result is None or result[1] is None else result[1]
        ),
        "result_equivalent": matches,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Measured executions per unique plan (default: 5).",
    )
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    missing = [
        str(path) for path in TABLE_PATHS.values()
        if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    if not QUERY_FILE.is_file():
        raise FileNotFoundError(QUERY_FILE)

    query = QueryRequest(
        query_id="e3_q05_allplans_quality",
        sql=QUERY_FILE.read_text(encoding="utf-8"),
        budget=100.0,
        deadline=300.0,
    )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Q05-AllPlan-Quality")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    runs_path = RESULTS / "prescoring_q05_allplans_runs.csv"
    summary_path = RESULTS / "prescoring_q05_allplans_summary.csv"

    rng = random.Random(SEED)
    all_runs = []

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)
            print(f"Registered {table}")

        _, reference, reference_error = execute_sql(spark, query.sql)
        if reference_error:
            raise RuntimeError(f"Reference query failed: {reference_error}")

        print(
            f"\nReference: row_count={reference[0]}, "
            f"total_sales={reference[1]}"
        )

        configs = [
            ("baseline", None),
            ("prescore_12", 12),
            ("prescore_20", 20),
        ]

        candidates_by_mode = {}

        print("\n===== GENERATING CANDIDATE SETS =====")
        for mode, limit in configs:
            candidates = generator.generate_join_candidates(
                query,
                spark,
                search_mode="beam",
                beam_width=10,
                candidate_workers=1,
                candidate_analysis_limit=limit,
            )
            candidates_by_mode[mode] = candidates
            print(f"{mode:12}: {len(candidates)} unique plans")

        baseline_by_fp = {
            candidate.fingerprint: candidate
            for candidate in candidates_by_mode["baseline"]
        }

        candidate_sets = {
            mode: {candidate.fingerprint for candidate in candidates}
            for mode, candidates in candidates_by_mode.items()
        }

        baseline_fingerprints = set(baseline_by_fp)

        for mode, fingerprints in candidate_sets.items():
            if not fingerprints.issubset(baseline_fingerprints):
                raise RuntimeError(
                    f"{mode} contains a fingerprint absent from baseline"
                )

        fingerprints = sorted(baseline_fingerprints)
        print(f"\nUnique fingerprints to measure: {len(fingerprints)}")
        print(f"Measured repetitions per plan: {args.repetitions}")
        print("Each plan receives one warm-up, then equal measured repetitions.")

        # Warm every unique physical plan once. Warm-ups are not included
        # in timing summaries.
        print("\n===== PLAN WARM-UPS =====")
        for index, fingerprint in enumerate(fingerprints, start=1):
            candidate = baseline_by_fp[fingerprint]
            elapsed, result, error = execute_sql(spark, candidate.sql)
            matches = not error and results_match(result, reference)

            print(
                f"{index:02}/{len(fingerprints):02} "
                f"{candidate.strategy:38} {elapsed:7.3f}s "
                f"equivalent={matches} error={'yes' if error else 'no'}"
            )

        # Save each repetition incrementally. Plans are shuffled to reduce
        # systematic bias from always running in the same order.
        for repetition in range(1, args.repetitions + 1):
            order = list(fingerprints)
            rng.shuffle(order)

            print(
                f"\n===== MEASURED REPETITION "
                f"{repetition}/{args.repetitions} ====="
            )

            for index, fingerprint in enumerate(order, start=1):
                candidate = baseline_by_fp[fingerprint]
                elapsed, result, error = execute_sql(spark, candidate.sql)
                matches = not error and results_match(result, reference)

                row = make_row(
                    "measured",
                    repetition,
                    candidate,
                    elapsed,
                    result,
                    matches,
                    error,
                )
                all_runs.append(row)
                save_csv(runs_path, all_runs, RUN_FIELDS)

                print(
                    f"{index:02}/{len(order):02} "
                    f"{candidate.strategy:38} {elapsed:7.3f}s "
                    f"equivalent={matches} error={'yes' if error else 'no'}"
                )

        # A plan is eligible only if every measured repetition succeeded
        # and produced a result equivalent to the original query.
        by_fp = {}
        for row in all_runs:
            by_fp.setdefault(row["fingerprint"], []).append(row)

        medians = {}
        plan_errors = {}
        plan_equivalent = {}

        for fingerprint in fingerprints:
            rows = by_fp.get(fingerprint, [])
            errors = sum(bool(row["error"]) for row in rows)
            equivalent_all = (
                len(rows) == args.repetitions
                and errors == 0
                and all(row["result_equivalent"] for row in rows)
            )

            plan_errors[fingerprint] = errors
            plan_equivalent[fingerprint] = equivalent_all

            if equivalent_all:
                medians[fingerprint] = statistics.median(
                    float(row["execution_seconds"]) for row in rows
                )

        if not medians:
            raise RuntimeError(
                "No plan completed all repetitions with equivalent results."
            )

        # Find the best measured plan in each candidate set using comparable
        # repeated medians, rather than one-shot screening times.
        best_by_mode = {}

        for mode, _limit in configs:
            eligible = [
                fp for fp in candidate_sets[mode]
                if fp in medians
            ]
            if eligible:
                best_by_mode[mode] = min(
                    eligible,
                    key=lambda fp: medians[fp],
                )

        if "baseline" not in best_by_mode:
            raise RuntimeError("No fully valid baseline plan was measured.")

        baseline_best = best_by_mode["baseline"]
        baseline_best_time = medians[baseline_best]
        summary_rows = []

        for mode, limit in configs:
            plan_set = candidate_sets[mode]
            best = best_by_mode.get(mode)
            best_time = medians.get(best) if best else None

            latency_change = ""
            if best_time is not None:
                latency_change = (
                    f"{100 * (best_time - baseline_best_time) / baseline_best_time:.2f}"
                )

            summary_rows.append({
                "mode": mode,
                "candidate_analysis_limit": "" if limit is None else limit,
                "candidate_plan_count": len(plan_set),
                "valid_measured_plan_count": sum(
                    fp in medians for fp in plan_set
                ),
                "best_strategy": (
                    baseline_by_fp[best].strategy if best else ""
                ),
                "best_fingerprint": best or "",
                "best_median_seconds": (
                    f"{best_time:.6f}" if best_time is not None else ""
                ),
                "baseline_best_fingerprint_retained":
                    baseline_best in plan_set,
                "latency_change_percent_vs_baseline": latency_change,
                "all_candidate_results_equivalent": all(
                    plan_equivalent.get(fp, False)
                    for fp in plan_set
                ),
                "execution_errors": sum(
                    plan_errors.get(fp, 0) for fp in plan_set
                ),
                "repetitions_per_plan": args.repetitions,
            })

        save_csv(summary_path, summary_rows, SUMMARY_FIELDS)

        print(f"\nSaved measured runs: {runs_path}")
        print(f"Saved summary: {summary_path}")
        print("\n===== Q05 ALL-PLAN QUALITY SUMMARY =====")

        for row in summary_rows:
            print(
                f"{row['mode']:12} "
                f"plans={row['candidate_plan_count']:>2} "
                f"valid={row['valid_measured_plan_count']:>2} "
                f"best_median={row['best_median_seconds']:>9}s "
                f"baseline_best_retained="
                f"{row['baseline_best_fingerprint_retained']} "
                f"latency_change="
                f"{row['latency_change_percent_vs_baseline']}% "
                f"equivalent={row['all_candidate_results_equivalent']} "
                f"errors={row['execution_errors']}"
            )

        print(
            "\nCaveat: these measurements use the five-table representative "
            "dataset, not full SF10."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
