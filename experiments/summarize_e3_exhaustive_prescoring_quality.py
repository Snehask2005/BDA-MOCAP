import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

from pyspark.sql import SparkSession

import mocap.plans.generator as generator
from mocap.query.parser import QueryRequest


ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "benchmarks/tpcds/representative/parquet"
QUERY_PATH = ROOT / "benchmarks/tpcds/e3/queries/q05_5_tables.sql"

RUNS = ROOT / "results/e3/exhaustive_prescoring_q05_runs.csv"
SUMMARY = ROOT / "results/e3/exhaustive_prescoring_q05_summary.csv"

TABLE_PATHS = {
    "store_sales": REP / "store_sales",
    "item": REP / "item",
    "store": REP / "store",
    "date_dim": REP / "date_dim",
    "customer_demographics": REP / "customer_demographics",
}

CONFIGS = [
    ("exhaustive", "exhaustive", None),
    ("prescore_12", "beam", 12),
    ("prescore_20", "beam", 20),
]

PLANNING_REPETITIONS = 3
EXECUTION_REPETITIONS = 5

SUMMARY_FIELDS = [
    "mode",
    "candidate_analysis_limit",
    "planning_repetitions",
    "median_planning_seconds",
    "generated_sql_variants",
    "analyzed_sql_variants",
    "unique_physical_plans",
    "valid_measured_plans",
    "best_strategy",
    "best_fingerprint",
    "best_median_execution_seconds",
    "exhaustive_best_retained",
    "pruning_loss_percent",
    "all_candidate_results_equivalent",
    "execution_errors",
    "execution_repetitions",
]


def generate_candidates_measured(spark, query, search_mode, limit):
    """Repeat planning only; do not execute any candidate query."""
    metrics = {
        "generated": 0,
        "analyzed": 0,
        "seconds": 0.0,
    }

    original_generate = generator._generate_all_sql_variants
    original_prescore = generator._pre_score_sql_variants

    def timed_generate(*args, **kwargs):
        variants = original_generate(*args, **kwargs)
        metrics["generated"] = len(variants)
        return variants

    def timed_prescore(variants, *args, **kwargs):
        selected = original_prescore(variants, *args, **kwargs)
        metrics["analyzed"] = len(selected)
        return selected

    start = time.perf_counter()

    try:
        generator._generate_all_sql_variants = timed_generate
        generator._pre_score_sql_variants = timed_prescore

        candidates = generator.generate_join_candidates(
            query,
            spark,
            search_mode=search_mode,
            beam_width=10,
            candidate_workers=1,
            candidate_analysis_limit=limit,
        )
    finally:
        generator._generate_all_sql_variants = original_generate
        generator._pre_score_sql_variants = original_prescore

    metrics["seconds"] = time.perf_counter() - start

    # The exhaustive path does not invoke the pre-scoring helper.
    if metrics["analyzed"] == 0:
        metrics["analyzed"] = metrics["generated"]

    return candidates, metrics


def main():
    if not RUNS.is_file():
        raise FileNotFoundError(
            f"Saved execution measurements not found: {RUNS}"
        )

    for path in [*TABLE_PATHS.values(), QUERY_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    with RUNS.open(newline="", encoding="utf-8") as f:
        run_rows = list(csv.DictReader(f))

    measured_rows = [
        row for row in run_rows if row["phase"] == "measured"
    ]

    print(f"Saved raw execution rows: {len(run_rows)}")
    print(f"Measured execution rows: {len(measured_rows)}")

    by_fingerprint = defaultdict(list)
    for row in measured_rows:
        by_fingerprint[row["fingerprint"]].append(row)

    print(f"Distinct executed fingerprints: {len(by_fingerprint)}")

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Exhaustive-Quality-Summary-Recovery")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)

        query = QueryRequest(
            query_id="e3_q05_exhaustive_quality",
            sql=QUERY_PATH.read_text(encoding="utf-8"),
            budget=100.0,
            deadline=300.0,
        )

        candidates_by_mode = {}
        planning_times = {}
        variant_counts = {}

        print("\n===== REBUILDING CANDIDATE SETS (PLANNING ONLY) =====")

        for label, search_mode, limit in CONFIGS:
            measurements = []
            latest_candidates = None
            expected_fingerprints = None

            for repetition in range(1, PLANNING_REPETITIONS + 1):
                candidates, metrics = generate_candidates_measured(
                    spark, query, search_mode, limit
                )

                fingerprints = {
                    candidate.fingerprint for candidate in candidates
                }

                if expected_fingerprints is None:
                    expected_fingerprints = fingerprints
                    latest_candidates = candidates
                    variant_counts[label] = metrics
                elif fingerprints != expected_fingerprints:
                    raise RuntimeError(
                        f"Candidate fingerprints changed between "
                        f"planning runs for {label}"
                    )

                measurements.append(metrics["seconds"])
                print(
                    f"{label:12} run {repetition}/"
                    f"{PLANNING_REPETITIONS}: "
                    f"{metrics['seconds']:.3f}s, "
                    f"SQL analyzed={metrics['analyzed']}, "
                    f"unique plans={len(candidates)}"
                )

            candidates_by_mode[label] = {
                candidate.fingerprint: candidate
                for candidate in latest_candidates
            }
            planning_times[label] = statistics.median(measurements)

        exhaustive_fps = set(candidates_by_mode["exhaustive"])

        for label, candidates in candidates_by_mode.items():
            if not set(candidates).issubset(exhaustive_fps):
                raise RuntimeError(
                    f"{label} produced fingerprints absent from "
                    "the exhaustive candidate set"
                )

        print("\n===== RECOVERING EXECUTION MEDIANS =====")

        medians = {}
        errors_by_fp = {}
        valid_by_fp = {}

        for fingerprint, rows in by_fingerprint.items():
            errors = sum(bool(row["error"].strip()) for row in rows)

            all_valid = (
                len(rows) == EXECUTION_REPETITIONS
                and errors == 0
                and all(
                    row["result_equivalent"].strip().lower() == "true"
                    for row in rows
                )
            )

            errors_by_fp[fingerprint] = errors
            valid_by_fp[fingerprint] = all_valid

            if all_valid:
                medians[fingerprint] = statistics.median(
                    float(row["execution_seconds"]) for row in rows
                )

        print(f"Plans with all five valid executions: {len(medians)}")
        print(
            "Candidate execution errors or result mismatches: "
            f"{sum(not valid for valid in valid_by_fp.values())}"
        )

        valid_exhaustive = [
            fp for fp in exhaustive_fps if fp in medians
        ]

        if not valid_exhaustive:
            raise RuntimeError(
                "No exhaustive plan has five successful, equivalent "
                "execution measurements. Inspect the raw CSV."
            )

        exhaustive_best = min(
            valid_exhaustive,
            key=lambda fp: medians[fp],
        )
        exhaustive_best_time = medians[exhaustive_best]

        summary_rows = []

        for label, _search_mode, limit in CONFIGS:
            candidate_set = set(candidates_by_mode[label])
            valid_candidates = [
                fp for fp in candidate_set if fp in medians
            ]

            best = (
                min(valid_candidates, key=lambda fp: medians[fp])
                if valid_candidates else None
            )

            pruning_loss = ""
            if best is not None:
                pruning_loss = (
                    100.0
                    * (medians[best] / exhaustive_best_time - 1.0)
                )

            metrics = variant_counts[label]

            summary_rows.append({
                "mode": label,
                "candidate_analysis_limit": (
                    "" if limit is None else limit
                ),
                "planning_repetitions": PLANNING_REPETITIONS,
                "median_planning_seconds": (
                    f"{planning_times[label]:.6f}"
                ),
                "generated_sql_variants": metrics["generated"],
                "analyzed_sql_variants": metrics["analyzed"],
                "unique_physical_plans": len(candidate_set),
                "valid_measured_plans": len(valid_candidates),
                "best_strategy": (
                    candidates_by_mode[label][best].strategy
                    if best else ""
                ),
                "best_fingerprint": best or "",
                "best_median_execution_seconds": (
                    f"{medians[best]:.6f}" if best else ""
                ),
                "exhaustive_best_retained": (
                    exhaustive_best in candidate_set
                ),
                "pruning_loss_percent": (
                    f"{pruning_loss:.4f}"
                    if pruning_loss != "" else ""
                ),
                "all_candidate_results_equivalent": all(
                    valid_by_fp.get(fp, False)
                    for fp in candidate_set
                ),
                "execution_errors": sum(
                    errors_by_fp.get(fp, 0)
                    for fp in candidate_set
                ),
                "execution_repetitions": EXECUTION_REPETITIONS,
            })

        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARY.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=SUMMARY_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nSaved recovered summary: {SUMMARY}")
        print("\n===== EXHAUSTIVE VS PRE-SCORING QUALITY SUMMARY =====")

        for row in summary_rows:
            print(
                f"{row['mode']:12} "
                f"plans={row['unique_physical_plans']:>3} "
                f"valid={row['valid_measured_plans']:>3} "
                f"planning={row['median_planning_seconds']:>8}s "
                f"best_exec={row['best_median_execution_seconds']:>8}s "
                f"retained={row['exhaustive_best_retained']} "
                f"loss={row['pruning_loss_percent']}% "
                f"equivalent={row['all_candidate_results_equivalent']} "
                f"errors={row['execution_errors']}"
            )

        print(
            "\nNo execution queries were run during summary recovery. "
            "Execution timings came from the saved CSV."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
