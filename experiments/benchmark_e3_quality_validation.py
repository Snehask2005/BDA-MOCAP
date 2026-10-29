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
TOP_K = 5
FINAL_REPETITIONS = 5
SEED = 20261029

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

        if "row_count" not in values or "total_sales" not in values:
            raise ValueError(
                f"Unexpected output columns: {list(values)}"
            )

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
        actual[1],
        expected[1],
        rel_tol=1e-9,
        abs_tol=1e-5,
    )


def make_row(query_file, phase, candidate, repetition,
             elapsed, result, equivalent, error):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_file": query_file,
        "phase": phase,
        "fingerprint": candidate.fingerprint,
        "strategy": candidate.strategy,
        "repetition": repetition,
        "execution_seconds": f"{elapsed:.6f}",
        "row_count": "" if result is None else result[0],
        "total_sales": (
            "" if result is None or result[1] is None else result[1]
        ),
        "result_equivalent": equivalent,
        "error": error,
    }


def generate_candidates(spark, query, cap):
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        candidates = generate_join_candidates(
            query,
            spark,
            search_mode="beam",
            beam_width=BEAM_WIDTH,
            candidate_workers=4,
            candidate_analysis_limit=cap,
        )

    warnings = [
        line.strip()
        for line in captured.getvalue().splitlines()
        if "Skipping strategy" in line
    ]
    return candidates, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "q06_6_tables.sql",
            "q07_7_tables.sql",
            "q08_8_tables.sql",
            "q09_9_tables.sql",
            "q10_10_tables.sql",
        ],
        help="E3 query filenames to evaluate.",
    )
    parser.add_argument("--tag", default="q06_q10")
    args = parser.parse_args()

    query_paths = [QUERIES / name for name in args.queries]
    for path in [*TABLE_PATHS.values(), *query_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    RESULTS.mkdir(parents=True, exist_ok=True)
    runs_path = RESULTS / f"{args.tag}_runs.csv"
    summary_path = RESULTS / f"{args.tag}_summary.csv"

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Execution-Quality-Validation")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    rng = random.Random(SEED)
    summary_rows = []

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)
            print(f"Registered {table}")

        with runs_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=RUN_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            f.flush()

            for query_path in query_paths:
                print(f"\n===== {query_path.name} =====")
                query = QueryRequest(
                    query_id=f"quality_{query_path.stem}",
                    sql=query_path.read_text(encoding="utf-8"),
                    budget=100.0,
                    deadline=300.0,
                )

                # Establish the correctness reference.
                _, expected, reference_error = execute_sql(spark, query.sql)
                if reference_error:
                    raise RuntimeError(
                        f"Reference query failed for {query_path.name}: "
                        f"{reference_error}"
                    )

                print(
                    f"Reference: row_count={expected[0]}, "
                    f"total_sales={expected[1]}"
                )

                reference_candidates, reference_warnings = (
                    generate_candidates(spark, query, None)
                )
                capped_candidates, capped_warnings = (
                    generate_candidates(spark, query, CAP)
                )

                reference_by_fp = {
                    c.fingerprint: c for c in reference_candidates
                }
                capped_by_fp = {
                    c.fingerprint: c for c in capped_candidates
                }

                if not set(capped_by_fp) <= set(reference_by_fp):
                    raise RuntimeError(
                        f"{query_path.name}: capped plans are not a "
                        "subset of the beam-40 reference plans."
                    )

                print(
                    f"Reference plans={len(reference_by_fp)}, "
                    f"cap-{CAP} plans={len(capped_by_fp)}, "
                    f"generation warnings="
                    f"{len(reference_warnings) + len(capped_warnings)}"
                )

                # Screen each unique reference plan once.
                screen = {}
                candidates = list(reference_by_fp.values())
                rng.shuffle(candidates)

                print("Screening reference plans...")
                for index, candidate in enumerate(candidates, start=1):
                    elapsed, result, error = execute_sql(
                        spark, candidate.sql
                    )
                    equivalent = results_match(result, expected) and not error

                    row = make_row(
                        query_path.name, "screen", candidate, 1,
                        elapsed, result, equivalent, error,
                    )
                    writer.writerow(row)
                    f.flush()

                    screen[candidate.fingerprint] = {
                        "seconds": elapsed,
                        "valid": equivalent,
                        "error": error,
                        "result": result,
                        "strategy": candidate.strategy,
                    }

                    if index % 20 == 0 or index == len(candidates):
                        print(
                            f"  screened {index}/{len(candidates)} plans"
                        )

                valid_reference = [
                    fp for fp, data in screen.items() if data["valid"]
                ]
                valid_capped = [
                    fp for fp in capped_by_fp
                    if fp in screen and screen[fp]["valid"]
                ]

                valid_reference.sort(
                    key=lambda fp: (screen[fp]["seconds"], fp)
                )
                valid_capped.sort(
                    key=lambda fp: (screen[fp]["seconds"], fp)
                )

                reference_top = valid_reference[:TOP_K]
                capped_top = valid_capped[:TOP_K]
                finalists = sorted(set(reference_top) | set(capped_top))

                if not reference_top or not capped_top:
                    raise RuntimeError(
                        f"{query_path.name}: insufficient valid finalists."
                    )

                print(
                    f"Screening valid plans: {len(valid_reference)}/"
                    f"{len(reference_by_fp)}; "
                    f"finalists to repeat: {len(finalists)}"
                )
                print(
                    f"Fastest single-screen reference plan retained: "
                    f"{valid_reference[0] in capped_by_fp}"
                )

                # Repeat the union of the fastest reference and capped plans.
                final_runs = {fp: [] for fp in finalists}

                for repetition in range(1, FINAL_REPETITIONS + 1):
                    order = list(finalists)
                    rng.shuffle(order)
                    print(
                        f"Finalist repetition "
                        f"{repetition}/{FINAL_REPETITIONS}"
                    )

                    for fp in order:
                        candidate = reference_by_fp[fp]
                        elapsed, result, error = execute_sql(
                            spark, candidate.sql
                        )
                        equivalent = results_match(result, expected) and not error

                        row = make_row(
                            query_path.name, "finalist", candidate,
                            repetition, elapsed, result, equivalent, error,
                        )
                        writer.writerow(row)
                        f.flush()
                        final_runs[fp].append(row)

                # Only use candidates with all repetitions valid.
                medians = {}
                for fp, runs in final_runs.items():
                    if (
                        len(runs) == FINAL_REPETITIONS
                        and all(
                            not run["error"]
                            and str(run["result_equivalent"]).lower() == "true"
                            for run in runs
                        )
                    ):
                        medians[fp] = statistics.median(
                            float(run["execution_seconds"]) for run in runs
                        )

                reference_measured = [
                    fp for fp in reference_top if fp in medians
                ]
                capped_measured = [
                    fp for fp in capped_top if fp in medians
                ]

                ref_best_fp = min(
                    reference_measured,
                    key=lambda fp: medians[fp],
                    default=None,
                )
                cap_best_fp = min(
                    capped_measured,
                    key=lambda fp: medians[fp],
                    default=None,
                )

                ref_best = medians.get(ref_best_fp) if ref_best_fp else None
                cap_best = medians.get(cap_best_fp) if cap_best_fp else None
                loss = (
                    100.0 * (cap_best / ref_best - 1.0)
                    if ref_best is not None and cap_best is not None
                    else None
                )

                screen_errors = sum(
                    bool(data["error"]) for data in screen.values()
                )
                screen_mismatches = sum(
                    not data["error"] and not data["valid"]
                    for data in screen.values()
                )
                finalist_errors = sum(
                    bool(run["error"])
                    for runs in final_runs.values() for run in runs
                )
                finalist_mismatches = sum(
                    not run["error"]
                    and str(run["result_equivalent"]).lower() != "true"
                    for runs in final_runs.values() for run in runs
                )

                summary_rows.append({
                    "query_file": query_path.name,
                    "table_count": int(query_path.name[1:3]),
                    "reference_scope": "uncapped beam width 40; not exhaustive",
                    "reference_candidate_count": len(reference_by_fp),
                    "capped_candidate_count": len(capped_by_fp),
                    "screened_valid_reference_count": len(valid_reference),
                    "reference_top_k": len(reference_top),
                    "capped_top_k": len(capped_top),
                    "repeated_finalist_count": len(finalists),
                    "final_repetitions": FINAL_REPETITIONS,
                    "screening_best_fingerprint": valid_reference[0],
                    "screening_best_retained": valid_reference[0] in capped_by_fp,
                    "reference_finalist_best_strategy":
                        screen[ref_best_fp]["strategy"] if ref_best_fp else "",
                    "capped_finalist_best_strategy":
                        screen[cap_best_fp]["strategy"] if cap_best_fp else "",
                    "reference_finalist_best_median_seconds":
                        "" if ref_best is None else f"{ref_best:.6f}",
                    "capped_finalist_best_median_seconds":
                        "" if cap_best is None else f"{cap_best:.6f}",
                    "finalist_pruning_loss_percent":
                        "" if loss is None else f"{loss:.2f}",
                    "screen_errors": screen_errors,
                    "screen_result_mismatches": screen_mismatches,
                    "finalist_errors": finalist_errors,
                    "finalist_result_mismatches": finalist_mismatches,
                    "meets_3_percent_among_screened_finalists":
                        loss is not None and loss <= 3.0,
                    "query_execution_performed": True,
                })

                result_row = summary_rows[-1]
                print("\nQuality summary:")
                for key in (
                    "reference_candidate_count",
                    "capped_candidate_count",
                    "screening_best_retained",
                    "reference_finalist_best_median_seconds",
                    "capped_finalist_best_median_seconds",
                    "finalist_pruning_loss_percent",
                    "screen_errors",
                    "screen_result_mismatches",
                    "finalist_errors",
                    "finalist_result_mismatches",
                ):
                    print(f"  {key}: {result_row[key]}")

        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(summary_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nSaved raw runs: {runs_path}")
        print(f"Saved summary: {summary_path}")
        print(
            "\nInterpretation note: finalist pruning loss compares repeated "
            "medians among the top plans selected by a one-run screen. "
            "It is not the true exhaustive-search loss."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
