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

from mocap.plans.generator import generate_join_candidates
from mocap.query.parser import QueryRequest

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "benchmarks/tpcds/representative/parquet"
E3 = ROOT / "benchmarks/tpcds/e3/parquet"
QUERIES = ROOT / "benchmarks/tpcds/e3/queries"
RESULTS = ROOT / "results/e3/quality_validation"

TABLE_PATHS = {
    "store_sales": REP / "store_sales",
    "item": REP / "item",
    "store": REP / "store",
    "date_dim": REP / "date_dim",
    "customer_demographics": REP / "customer_demographics",
    "customer": E3 / "customer",
    "customer_address": E3 / "customer_address",
    "household_demographics": E3 / "household_demographics",
    "time_dim": E3 / "time_dim",
    "promotion": E3 / "promotion",
}

BEAM_WIDTH = 40
CAP = 48
ADDITIONAL_REPETITIONS = 2
SEED = 20261030

RUN_FIELDS = [
    "timestamp_utc",
    "query_file",
    "phase",
    "fingerprint",
    "strategy",
    "repetition",
    "execution_seconds",
    "row_count",
    "total_sales",
    "result_equivalent",
    "error",
    "measurement_source",
]


def execute_sql(spark, sql):
    started = time.perf_counter()

    try:
        rows = spark.sql(sql).collect()
        elapsed = time.perf_counter() - started

        if len(rows) != 1:
            raise ValueError(f"Expected one result row, got {len(rows)}")

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
        return time.perf_counter() - started, None, repr(exc)[:500]


def results_match(actual, expected):
    if actual is None or expected is None:
        return False
    if actual[0] != expected[0]:
        return False
    if actual[1] is None or expected[1] is None:
        return actual[1] is None and expected[1] is None

    return math.isclose(
        actual[1], expected[1], rel_tol=1e-9, abs_tol=1e-5
    )


def parse_saved_result(row):
    if not row.get("row_count") or not row.get("total_sales"):
        return None

    return int(row["row_count"]), float(row["total_sales"])


def generate(spark, query, limit):
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        candidates = generate_join_candidates(
            query,
            spark,
            search_mode="beam",
            beam_width=BEAM_WIDTH,
            candidate_workers=4,
            candidate_analysis_limit=limit,
        )

    warnings = [
        line.strip()
        for line in output.getvalue().splitlines()
        if "Skipping strategy" in line
    ]
    return candidates, warnings


def make_row(query_file, phase, fingerprint, strategy, repetition,
             elapsed, result, equivalent, error, source):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_file": query_file,
        "phase": phase,
        "fingerprint": fingerprint,
        "strategy": strategy,
        "repetition": repetition,
        "execution_seconds": f"{elapsed:.6f}",
        "row_count": "" if result is None else result[0],
        "total_sales": (
            "" if result is None or result[1] is None else result[1]
        ),
        "result_equivalent": equivalent,
        "error": error,
        "measurement_source": source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="q10_10_tables.sql")
    parser.add_argument("--prior-tag", default="q10_validation")
    parser.add_argument("--tag", default="q10_full_reference")
    args = parser.parse_args()

    query_path = QUERIES / args.query
    prior_path = RESULTS / f"{args.prior_tag}_runs.csv"
    runs_path = RESULTS / f"{args.tag}_runs.csv"
    summary_path = RESULTS / f"{args.tag}_summary.csv"

    for path in [query_path, prior_path, *TABLE_PATHS.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    if runs_path.exists() or summary_path.exists():
        raise FileExistsError(
            "Output already exists. Choose a new --tag to preserve old data."
        )

    with prior_path.open(newline="", encoding="utf-8") as f:
        prior_rows = [
            row for row in csv.DictReader(f)
            if row["phase"] == "screen"
        ]

    prior_by_fp = {row["fingerprint"]: row for row in prior_rows}

    if len(prior_rows) != len(prior_by_fp) or not prior_rows:
        raise ValueError("Saved screening data is empty or has duplicate plans.")

    query = QueryRequest(
        query_id=f"full_reference_{query_path.stem}",
        sql=query_path.read_text(encoding="utf-8"),
        budget=100.0,
        deadline=300.0,
    )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Full-Beam-Reference-Quality")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    rng = random.Random(SEED)
    all_rows = []

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)
            print(f"Registered {table}")

        _, expected, reference_error = execute_sql(spark, query.sql)
        if reference_error:
            raise RuntimeError(f"Original query failed: {reference_error}")

        reference_candidates, reference_warnings = generate(
            spark, query, None
        )
        capped_candidates, capped_warnings = generate(
            spark, query, CAP
        )

        reference_by_fp = {
            c.fingerprint: c for c in reference_candidates
        }
        capped_by_fp = {
            c.fingerprint: c for c in capped_candidates
        }

        if set(prior_by_fp) != set(reference_by_fp):
            raise RuntimeError(
                "Saved screening fingerprints do not match regenerated "
                "beam-40 reference candidates. Do not combine timings."
            )

        if not set(capped_by_fp) <= set(reference_by_fp):
            raise RuntimeError("Capped candidate set is not a reference subset.")

        for fingerprint, row in prior_by_fp.items():
            if row["error"]:
                raise RuntimeError(
                    f"Saved screening execution failed for {fingerprint}: "
                    f"{row['error']}"
                )
            if row["result_equivalent"].lower() != "true":
                raise RuntimeError(
                    f"Saved screening result was not equivalent for {fingerprint}"
                )
            if not results_match(parse_saved_result(row), expected):
                raise RuntimeError(
                    f"Saved result differs from current original-query result "
                    f"for {fingerprint}"
                )

        RESULTS.mkdir(parents=True, exist_ok=True)

        with runs_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=RUN_FIELDS, lineterminator="\n"
            )
            writer.writeheader()

            # Reuse one screening execution as repetition 1 for each plan.
            for fingerprint, old in prior_by_fp.items():
                candidate = reference_by_fp[fingerprint]
                row = make_row(
                    query_path.name,
                    "reference_measurement",
                    fingerprint,
                    candidate.strategy,
                    1,
                    float(old["execution_seconds"]),
                    parse_saved_result(old),
                    True,
                    "",
                    "reused_screen",
                )
                writer.writerow(row)
                f.flush()
                all_rows.append(row)

            print(
                f"\nReference plans: {len(reference_by_fp)}; "
                f"cap-{CAP} plans: {len(capped_by_fp)}"
            )
            print(
                f"Generation warnings: "
                f"{len(reference_warnings) + len(capped_warnings)}"
            )
            print(
                f"Using saved screening executions as repetition 1. "
                f"Running {ADDITIONAL_REPETITIONS} additional repetitions "
                f"for all reference plans."
            )

            for repetition in range(2, ADDITIONAL_REPETITIONS + 2):
                fingerprints = list(reference_by_fp)
                rng.shuffle(fingerprints)
                print(f"\nRepetition {repetition}/3")

                for index, fingerprint in enumerate(fingerprints, start=1):
                    candidate = reference_by_fp[fingerprint]
                    elapsed, result, error = execute_sql(
                        spark, candidate.sql
                    )
                    equivalent = results_match(result, expected) and not error

                    row = make_row(
                        query_path.name,
                        "reference_measurement",
                        fingerprint,
                        candidate.strategy,
                        repetition,
                        elapsed,
                        result,
                        equivalent,
                        error,
                        "new_execution",
                    )
                    writer.writerow(row)
                    f.flush()
                    all_rows.append(row)

                    if index % 20 == 0 or index == len(fingerprints):
                        print(
                            f"  completed {index}/{len(fingerprints)} "
                            f"plans; errors so far="
                            f"{sum(bool(r['error']) for r in all_rows)}"
                        )

        # Every reference candidate now has three samples.
        by_fp = {}
        for row in all_rows:
            by_fp.setdefault(row["fingerprint"], []).append(row)

        medians = {}
        invalid = set()

        for fingerprint, measurements in by_fp.items():
            if (
                len(measurements) != 3
                or any(row["error"] for row in measurements)
                or any(
                    str(row["result_equivalent"]).lower() != "true"
                    for row in measurements
                )
            ):
                invalid.add(fingerprint)
                continue

            medians[fingerprint] = statistics.median(
                float(row["execution_seconds"]) for row in measurements
            )

        valid_reference = set(reference_by_fp) - invalid
        valid_capped = set(capped_by_fp) & valid_reference

        if not valid_reference or not valid_capped:
            raise RuntimeError("No valid measured reference/capped plans remain.")

        best_ref = min(valid_reference, key=lambda fp: medians[fp])
        best_cap = min(valid_capped, key=lambda fp: medians[fp])
        reference_time = medians[best_ref]
        capped_time = medians[best_cap]
        loss = 100.0 * (capped_time / reference_time - 1.0)

        summary = {
            "query_file": query_path.name,
            "reference_scope": "uncapped beam width 40; not exhaustive",
            "reference_plan_count": len(reference_by_fp),
            "capped_plan_count": len(capped_by_fp),
            "plans_with_three_valid_measurements": len(medians),
            "invalid_plan_count": len(invalid),
            "reference_best_strategy": reference_by_fp[best_ref].strategy,
            "reference_best_fingerprint": best_ref,
            "reference_best_median_seconds": f"{reference_time:.6f}",
            "capped_best_strategy": reference_by_fp[best_cap].strategy,
            "capped_best_fingerprint": best_cap,
            "capped_best_median_seconds": f"{capped_time:.6f}",
            "observed_pruning_loss_percent": f"{loss:.2f}",
            "meets_3_percent_vs_full_beam_reference": loss <= 3.0,
            "execution_errors": sum(bool(row["error"]) for row in all_rows),
            "result_mismatches": sum(
                str(row["result_equivalent"]).lower() != "true"
                for row in all_rows
            ),
            "query_execution_performed": True,
        }

        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(summary), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow(summary)

        print("\n===== FULL BEAM-REFERENCE QUALITY SUMMARY =====")
        for key, value in summary.items():
            print(f"{key}: {value}")

        print(f"\nSaved runs: {runs_path}")
        print(f"Saved summary: {summary_path}")
        print(
            "This compares all measured beam-40 reference candidates, "
            "not the true exhaustive search space."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
