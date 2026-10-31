import argparse
import contextlib
import csv
import io
import math
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession

import mocap.plans.generator as generator
from mocap.query.parser import QueryRequest


ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "benchmarks/tpcds/representative/parquet"
QUERY_PATH = ROOT / "benchmarks/tpcds/e3/queries/q05_5_tables.sql"
OUTPUT = ROOT / "results/e3/exhaustive_prescoring_q05_runs.csv"
SUMMARY = ROOT / "results/e3/exhaustive_prescoring_q05_summary.csv"

TABLE_PATHS = {
    "store_sales": REP / "store_sales",
    "item": REP / "item",
    "store": REP / "store",
    "date_dim": REP / "date_dim",
    "customer_demographics": REP / "customer_demographics",
}

MODES = [
    ("exhaustive", "exhaustive", None),
    ("prescore_12", "beam", 12),
    ("prescore_20", "beam", 20),
]

BEAM_WIDTH = 10
DEFAULT_REPETITIONS = 5
PLANNING_REPETITIONS = 3
SEED = 20261028
ABS_TOL = 1e-5
REL_TOL = 1e-9

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
    "planning_repetitions",
    "median_planning_seconds",
    "median_table_statistics_seconds",
    "median_sql_generation_seconds",
    "median_prescoring_seconds",
    "generated_sql_variant_count",
    "analyzed_sql_variant_count",
    "unique_physical_plan_count",
    "successfully_measured_plan_count",
    "best_strategy",
    "best_fingerprint",
    "best_median_execution_seconds",
    "exhaustive_best_retained",
    "pruning_loss_percent",
    "all_candidate_results_equivalent",
    "execution_errors",
    "execution_repetitions",
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


def results_equivalent(actual, reference):
    if actual is None or reference is None:
        return False
    if actual[0] != reference[0]:
        return False
    if actual[1] is None or reference[1] is None:
        return actual[1] is None and reference[1] is None

    return math.isclose(
        actual[1],
        reference[1],
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    )


def execute_sql(spark, sql):
    started = time.perf_counter()

    try:
        rows = spark.sql(sql).collect()
        elapsed = time.perf_counter() - started

        if len(rows) != 1:
            raise ValueError(
                f"Expected one aggregate result row; got {len(rows)}"
            )

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


def generate_candidates_measured(spark, query, mode, limit):
    """Instrument the production generator without changing its behavior."""
    metrics = {
        "planning_seconds": 0.0,
        "table_statistics_seconds": 0.0,
        "sql_generation_seconds": 0.0,
        "prescoring_seconds": 0.0,
        "generated_sql_variant_count": 0,
        "analyzed_sql_variant_count": 0,
    }

    original_estimator = generator._estimate_table_size_bytes
    original_generate = generator._generate_all_sql_variants
    original_prescore = generator._pre_score_sql_variants

    def timed_estimator(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_estimator(*args, **kwargs)
        finally:
            metrics["table_statistics_seconds"] += (
                time.perf_counter() - started
            )

    def timed_generate(*args, **kwargs):
        started = time.perf_counter()
        try:
            variants = original_generate(*args, **kwargs)
            metrics["generated_sql_variant_count"] = len(variants)
            return variants
        finally:
            metrics["sql_generation_seconds"] += (
                time.perf_counter() - started
            )

    def timed_prescore(variants, *args, **kwargs):
        started = time.perf_counter()
        try:
            selected = original_prescore(variants, *args, **kwargs)
            metrics["analyzed_sql_variant_count"] = len(selected)
            return selected
        finally:
            metrics["prescoring_seconds"] += (
                time.perf_counter() - started
            )

    captured = io.StringIO()
    started = time.perf_counter()

    try:
        generator._estimate_table_size_bytes = timed_estimator
        generator._generate_all_sql_variants = timed_generate
        generator._pre_score_sql_variants = timed_prescore

        with contextlib.redirect_stdout(captured):
            candidates = generator.generate_join_candidates(
                query,
                spark,
                search_mode=mode,
                beam_width=BEAM_WIDTH,
                candidate_workers=1,
                candidate_analysis_limit=limit,
            )
    finally:
        generator._estimate_table_size_bytes = original_estimator
        generator._generate_all_sql_variants = original_generate
        generator._pre_score_sql_variants = original_prescore

    metrics["planning_seconds"] = time.perf_counter() - started

    if metrics["analyzed_sql_variant_count"] == 0:
        metrics["analyzed_sql_variant_count"] = (
            metrics["generated_sql_variant_count"]
        )

    errors = [
        line.strip()
        for line in captured.getvalue().splitlines()
        if "Skipping strategy" in line
    ]

    if errors:
        raise RuntimeError(
            f"{mode} candidate analysis skipped variants: "
            + "; ".join(errors[:3])
        )

    return candidates, metrics


def append_run(rows, candidate, phase, repetition, elapsed, result,
               reference, error):
    rows.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "mode": "all_exhaustive_plans",
        "repetition": repetition,
        "fingerprint": candidate.fingerprint,
        "strategy": candidate.strategy,
        "execution_seconds": f"{elapsed:.6f}",
        "row_count": "" if result is None else result[0],
        "total_sales": (
            "" if result is None or result[1] is None else result[1]
        ),
        "result_equivalent": (
            not error and results_equivalent(result, reference)
        ),
        "error": error,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Execution repetitions per exhaustive plan (default: 5).",
    )
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    missing = [
        str(path)
        for path in [*TABLE_PATHS.values(), QUERY_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("\n".join(missing))

    query = QueryRequest(
        query_id="e3_q05_exhaustive_quality",
        sql=QUERY_PATH.read_text(encoding="utf-8"),
        budget=100.0,
        deadline=300.0,
    )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Exhaustive-PreScoring-Quality")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    rng = random.Random(SEED)
    run_rows = []

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)
            print(f"Registered {table}")

        _, reference, reference_error = execute_sql(spark, query.sql)
        if reference_error:
            raise RuntimeError(
                f"Reference query failed: {reference_error}"
            )

        print(
            f"\nReference: row_count={reference[0]}, "
            f"total_sales={reference[1]}"
        )

        plans_by_mode = {}
        planning_samples = defaultdict(list)
        planning_metrics = {}

        print("\n===== CANDIDATE GENERATION AND PLANNING =====")
        for label, search_mode, limit in MODES:
            fingerprints_per_run = []
            latest_candidates = None
            latest_metrics = None

            for repetition in range(1, PLANNING_REPETITIONS + 1):
                candidates, metrics = generate_candidates_measured(
                    spark, query, search_mode, limit
                )

                fingerprints_per_run.append(
                    {candidate.fingerprint for candidate in candidates}
                )
                latest_candidates = candidates
                latest_metrics = metrics
                planning_samples[label].append(metrics["planning_seconds"])

                print(
                    f"{label:12} planning {repetition}/"
                    f"{PLANNING_REPETITIONS}: "
                    f"{metrics['planning_seconds']:.3f}s, "
                    f"variants={metrics['analyzed_sql_variant_count']}, "
                    f"unique plans={len(candidates)}"
                )

            if any(
                fingerprints != fingerprints_per_run[0]
                for fingerprints in fingerprints_per_run[1:]
            ):
                raise RuntimeError(
                    f"Candidate fingerprints were not stable for {label}"
                )

            plans_by_mode[label] = {
                candidate.fingerprint: candidate
                for candidate in latest_candidates
            }
            planning_metrics[label] = latest_metrics

        exhaustive = plans_by_mode["exhaustive"]
        exhaustive_fingerprints = set(exhaustive)

        for label, candidates in plans_by_mode.items():
            if not set(candidates).issubset(exhaustive_fingerprints):
                raise RuntimeError(
                    f"{label} contains a physical-plan fingerprint "
                    "absent from the exhaustive reference"
                )

        print("\n===== EXECUTING ALL UNIQUE EXHAUSTIVE PLANS =====")
        print(f"Exhaustive physical plans: {len(exhaustive)}")
        print(f"Measured repetitions per plan: {args.repetitions}")
        print("Each plan receives one warm-up before measured rounds.")

        all_plans = list(exhaustive.values())
        rng.shuffle(all_plans)

        for index, candidate in enumerate(all_plans, start=1):
            elapsed, result, error = execute_sql(spark, candidate.sql)
            append_run(
                run_rows, candidate, "warmup", 0,
                elapsed, result, reference, error,
            )
            if error or not results_equivalent(result, reference):
                print(
                    f"WARNING warm-up {index}/{len(all_plans)} "
                    f"{candidate.strategy}: execution failed or result differed"
                )
            else:
                print(
                    f"warm-up {index}/{len(all_plans)} "
                    f"{candidate.strategy:38} {elapsed:.3f}s"
                )

            save_csv(OUTPUT, run_rows, RUN_FIELDS)

        for repetition in range(1, args.repetitions + 1):
            order = list(exhaustive.values())
            rng.shuffle(order)

            print(
                f"\n===== EXECUTION REPETITION "
                f"{repetition}/{args.repetitions} ====="
            )

            for index, candidate in enumerate(order, start=1):
                elapsed, result, error = execute_sql(spark, candidate.sql)
                append_run(
                    run_rows, candidate, "measured", repetition,
                    elapsed, result, reference, error,
                )
                if error or not results_equivalent(result, reference):
                    print(
                        f"WARNING rep={repetition} {candidate.strategy}: "
                        "execution failed or result differed"
                    )
                else:
                    print(
                        f"{index:03}/{len(order):03} "
                        f"{candidate.strategy:38} {elapsed:.3f}s"
                    )

                # Flush partial results so interruption does not discard work.
                save_csv(OUTPUT, run_rows, RUN_FIELDS)

        by_fingerprint = defaultdict(list)
        for row in run_rows:
            if row["phase"] == "measured":
                by_fingerprint[row["fingerprint"]].append(row)

        medians = {}
        plan_validity = {}
        plan_errors = {}

        for fingerprint in exhaustive_fingerprints:
            rows = by_fingerprint[fingerprint]
            errors = sum(bool(row["error"]) for row in rows)
            valid = (
                len(rows) == args.repetitions
                and errors == 0
                and all(row["result_equivalent"] == "True" for row in rows)
            )
            plan_validity[fingerprint] = valid
            plan_errors[fingerprint] = errors

            if valid:
                medians[fingerprint] = statistics.median(
                    float(row["execution_seconds"]) for row in rows
                )

        valid_exhaustive = [
            fingerprint for fingerprint in exhaustive_fingerprints
            if fingerprint in medians
        ]
        if not valid_exhaustive:
            raise RuntimeError(
                "No exhaustive candidate completed every repetition "
                "with equivalent results."
            )

        exhaustive_best = min(
            valid_exhaustive,
            key=lambda fingerprint: medians[fingerprint],
        )
        exhaustive_best_time = medians[exhaustive_best]

        summary_rows = []
        for label, _search_mode, limit in MODES:
            candidate_fingerprints = set(plans_by_mode[label])
            valid_fingerprints = [
                fingerprint for fingerprint in candidate_fingerprints
                if fingerprint in medians
            ]

            best = (
                min(valid_fingerprints, key=lambda fp: medians[fp])
                if valid_fingerprints else None
            )

            loss = ""
            if best is not None:
                loss = (
                    100.0 * (medians[best] / exhaustive_best_time - 1.0)
                )

            metrics = planning_metrics[label]
            relevant_errors = sum(
                plan_errors.get(fingerprint, 0)
                for fingerprint in candidate_fingerprints
            )

            summary_rows.append({
                "mode": label,
                "candidate_analysis_limit": "" if limit is None else limit,
                "planning_repetitions": PLANNING_REPETITIONS,
                "median_planning_seconds": f"{statistics.median(planning_samples[label]):.6f}",
                "median_table_statistics_seconds": f"{metrics['table_statistics_seconds']:.6f}",
                "median_sql_generation_seconds": f"{metrics['sql_generation_seconds']:.6f}",
                "median_prescoring_seconds": f"{metrics['prescoring_seconds']:.6f}",
                "generated_sql_variant_count": metrics["generated_sql_variant_count"],
                "analyzed_sql_variant_count": metrics["analyzed_sql_variant_count"],
                "unique_physical_plan_count": len(candidate_fingerprints),
                "successfully_measured_plan_count": len(valid_fingerprints),
                "best_strategy": exhaustive[best].strategy if best else "",
                "best_fingerprint": best or "",
                "best_median_execution_seconds": (
                    f"{medians[best]:.6f}" if best else ""
                ),
                "exhaustive_best_retained": exhaustive_best in candidate_fingerprints,
                "pruning_loss_percent": (
                    f"{loss:.4f}" if loss != "" else ""
                ),
                "all_candidate_results_equivalent": all(
                    plan_validity.get(fp, False)
                    for fp in candidate_fingerprints
                ),
                "execution_errors": relevant_errors,
                "execution_repetitions": args.repetitions,
            })

        save_csv(SUMMARY, summary_rows, SUMMARY_FIELDS)

        print(f"\nSaved execution runs: {OUTPUT}")
        print(f"Saved summary: {SUMMARY}")
        print("\n===== EXHAUSTIVE VS PRE-SCORING QUALITY SUMMARY =====")

        for row in summary_rows:
            print(
                f"{row['mode']:12} "
                f"plans={row['unique_physical_plan_count']:>3} "
                f"valid={row['successfully_measured_plan_count']:>3} "
                f"planning={row['median_planning_seconds']:>8}s "
                f"best_exec={row['best_median_execution_seconds']:>8}s "
                f"retained={row['exhaustive_best_retained']} "
                f"loss={row['pruning_loss_percent']}% "
                f"equivalent={row['all_candidate_results_equivalent']} "
                f"errors={row['execution_errors']}"
            )

        print(
            "\nInterpret pruning loss only if all candidates in the relevant "
            "set completed every repetition with equivalent results."
        )
        print(
            "Dataset scope: five-table E3 query on representative data; "
            "not a full-SF10 execution-quality result."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
