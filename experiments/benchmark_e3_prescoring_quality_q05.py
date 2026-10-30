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
LIMITS = (None, 12, 20)
REPETITIONS = 5
TOP_K = 3
SEED = 20261028

TABLE_PATHS = {
    "store_sales": REP / "store_sales",
    "item": REP / "item",
    "store": REP / "store",
    "date_dim": REP / "date_dim",
    "customer_demographics": REP / "customer_demographics",
}

RUN_FIELDS = [
    "timestamp_utc", "phase", "mode", "repetition",
    "fingerprint", "strategy", "execution_seconds",
    "row_count", "total_sales", "result_equivalent", "error",
]

SUMMARY_FIELDS = [
    "mode", "candidate_analysis_limit", "unique_physical_plans",
    "screened_finalists", "best_finalist_strategy",
    "best_finalist_fingerprint", "best_finalist_median_seconds",
    "baseline_best_fingerprint_retained",
    "latency_change_percent_vs_baseline",
    "all_screened_results_equivalent", "execution_errors",
    "measured_repetitions_per_finalist",
]


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def execute_sql(spark, sql):
    started = time.perf_counter()
    try:
        rows = spark.sql(sql).collect()
        elapsed = time.perf_counter() - started
        if len(rows) != 1:
            raise ValueError(f"Expected one result row, got {len(rows)}")

        values = {
            str(k).lower(): v for k, v in rows[0].asDict().items()
        }
        result = (
            int(values["row_count"]),
            None if values["total_sales"] is None
            else float(values["total_sales"]),
        )
        return elapsed, result, ""
    except Exception as exc:
        return time.perf_counter() - started, None, repr(exc)[:400]


def equivalent(actual, reference):
    if actual is None or reference is None or actual[0] != reference[0]:
        return False
    if actual[1] is None or reference[1] is None:
        return actual[1] is None and reference[1] is None

    return abs(actual[1] - reference[1]) <= max(
        1e-5, 1e-9 * abs(reference[1])
    )


def make_row(phase, mode, repetition, candidate, elapsed, result,
             matches, error):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "mode": mode,
        "repetition": repetition,
        "fingerprint": candidate.fingerprint,
        "strategy": candidate.strategy,
        "execution_seconds": f"{elapsed:.6f}",
        "row_count": "" if result is None else result[0],
        "total_sales": "" if result is None or result[1] is None
        else result[1],
        "result_equivalent": matches,
        "error": error,
    }


def main():
    missing = [str(p) for p in TABLE_PATHS.values() if not p.is_dir()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    if not QUERY_FILE.is_file():
        raise FileNotFoundError(QUERY_FILE)

    query = QueryRequest(
        query_id="e3_q05_prescoring_quality",
        sql=QUERY_FILE.read_text(encoding="utf-8"),
        budget=100.0,
        deadline=300.0,
    )

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-PreScoring-Quality-Q05")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    runs_path = RESULTS / "prescoring_q05_quality_runs.csv"
    summary_path = RESULTS / "prescoring_q05_quality_summary.csv"
    rng = random.Random(SEED)
    runs = []

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(path.resolve().as_uri()).createOrReplaceTempView(table)
            print(f"Registered {table}")

        print("\nRunning an original-query warm-up for the reference result...")
        _, reference, reference_error = execute_sql(spark, query.sql)
        if reference_error:
            raise RuntimeError(f"Reference query failed: {reference_error}")
        print(f"Reference result: row_count={reference[0]}, total_sales={reference[1]}")

        modes = [
            ("baseline", None),
            ("prescore_12", 12),
            ("prescore_20", 20),
        ]
        candidates_by_mode = {}

        print("\n===== GENERATING CANDIDATE SETS =====")
        for mode, limit in modes:
            candidates = generator.generate_join_candidates(
                query,
                spark,
                search_mode="beam",
                beam_width=10,
                candidate_workers=1,
                candidate_analysis_limit=limit,
            )
            candidates_by_mode[mode] = candidates
            print(f"{mode:12} unique physical plans={len(candidates)}")

        baseline = candidates_by_mode["baseline"]
        baseline_by_fp = {c.fingerprint: c for c in baseline}
        baseline_fps = set(baseline_by_fp)

        plan_sets = {
            mode: {c.fingerprint for c in candidates}
            for mode, candidates in candidates_by_mode.items()
        }
        for mode, fingerprints in plan_sets.items():
            if not fingerprints.issubset(baseline_fps):
                raise RuntimeError(
                    f"{mode} contains plans absent from the baseline set"
                )

        print("\n===== SCREENING EVERY BASELINE PLAN ONCE =====")
        screen_by_fp = {}
        screen_candidates = list(baseline)
        rng.shuffle(screen_candidates)

        for index, candidate in enumerate(screen_candidates, start=1):
            elapsed, result, error = execute_sql(spark, candidate.sql)
            matches = not error and equivalent(result, reference)

            row = make_row(
                "screen", "baseline_all_plans", 0, candidate,
                elapsed, result, matches, error,
            )
            runs.append(row)
            screen_by_fp[candidate.fingerprint] = row
            save_csv(runs_path, runs, RUN_FIELDS)

            print(
                f"{index:02}/{len(screen_candidates):02} "
                f"{candidate.strategy:38} {elapsed:7.3f}s "
                f"equivalent={matches} error={'yes' if error else 'no'}"
            )

        valid_screen = {
            fp: row for fp, row in screen_by_fp.items()
            if row["result_equivalent"] and not row["error"]
        }
        if not valid_screen:
            raise RuntimeError("No baseline candidates produced valid results.")

        baseline_best_fp = min(
            valid_screen,
            key=lambda fp: float(valid_screen[fp]["execution_seconds"]),
        )

        finalists_by_mode = {}
        all_finalist_fps = set()

        print("\n===== SELECTING TOP SCREENED FINALISTS =====")
        for mode, _limit in modes:
            ranked = [
                fp for fp in plan_sets[mode] if fp in valid_screen
            ]
            ranked.sort(
                key=lambda fp: float(valid_screen[fp]["execution_seconds"])
            )
            finalists = ranked[:TOP_K]
            finalists_by_mode[mode] = finalists
            all_finalist_fps.update(finalists)
            print(f"{mode:12}: {len(finalists)} finalists")

        # Warm each finalist once before repeated timings.
        print("\n===== FINALIST WARM-UP =====")
        for fp in sorted(all_finalist_fps):
            candidate = baseline_by_fp[fp]
            _, result, error = execute_sql(spark, candidate.sql)
            print(
                f"{candidate.strategy:38} "
                f"equivalent={not error and equivalent(result, reference)} "
                f"error={'yes' if error else 'no'}"
            )

        finalist_runs = {fp: [] for fp in all_finalist_fps}
        print("\n===== REPEATED FINALIST EXECUTIONS =====")

        for repetition in range(1, REPETITIONS + 1):
            order = list(all_finalist_fps)
            rng.shuffle(order)

            for fp in order:
                candidate = baseline_by_fp[fp]
                elapsed, result, error = execute_sql(spark, candidate.sql)
                matches = not error and equivalent(result, reference)
                row = make_row(
                    "finalist", "shared_finalists", repetition,
                    candidate, elapsed, result, matches, error,
                )
                runs.append(row)
                finalist_runs[fp].append(row)
                save_csv(runs_path, runs, RUN_FIELDS)
                print(
                    f"rep={repetition}/{REPETITIONS} "
                    f"{candidate.strategy:38} {elapsed:7.3f}s "
                    f"equivalent={matches}"
                )

        medians = {}
        for fp, fp_runs in finalist_runs.items():
            valid_times = [
                float(row["execution_seconds"])
                for row in fp_runs
                if row["result_equivalent"] and not row["error"]
            ]
            if valid_times:
                medians[fp] = statistics.median(valid_times)

        if baseline_best_fp not in medians:
            raise RuntimeError(
                "The fastest screened baseline candidate lacked valid "
                "repeated measurements; cannot compare finalist medians."
            )

        baseline_best_median = medians[baseline_best_fp]
        summary = []

        for mode, limit in modes:
            chosen = finalists_by_mode[mode]
            eligible = [fp for fp in chosen if fp in medians]
            best_fp = min(eligible, key=lambda fp: medians[fp]) if eligible else None

            if best_fp is None:
                best_median = None
                latency_change = ""
                best_strategy = ""
            else:
                best_median = medians[best_fp]
                latency_change = (
                    f"{100 * (best_median - baseline_best_median) / baseline_best_median:.2f}"
                )
                best_strategy = baseline_by_fp[best_fp].strategy

            screened_for_mode = [
                screen_by_fp[fp] for fp in plan_sets[mode]
                if fp in screen_by_fp
            ]

            summary.append({
                "mode": mode,
                "candidate_analysis_limit": "" if limit is None else limit,
                "unique_physical_plans": len(plan_sets[mode]),
                "screened_finalists": "; ".join(
                    baseline_by_fp[fp].strategy for fp in chosen
                ),
                "best_finalist_strategy": best_strategy,
                "best_finalist_fingerprint": best_fp or "",
                "best_finalist_median_seconds": (
                    "" if best_median is None else f"{best_median:.6f}"
                ),
                "baseline_best_fingerprint_retained":
                    baseline_best_fp in plan_sets[mode],
                "latency_change_percent_vs_baseline": latency_change,
                "all_screened_results_equivalent": all(
                    row["result_equivalent"] and not row["error"]
                    for row in screened_for_mode
                ),
                "execution_errors": sum(
                    bool(row["error"]) for row in screened_for_mode
                ) + sum(
                    bool(row["error"]) for fp in chosen
                    for row in finalist_runs[fp]
                ),
                "measured_repetitions_per_finalist": REPETITIONS,
            })

        save_csv(summary_path, summary, SUMMARY_FIELDS)
        print(f"\nSaved runs: {runs_path}")
        print(f"Saved summary: {summary_path}")
        print("\n===== Q05 PRE-SCORING QUALITY SUMMARY =====")
        for row in summary:
            print(
                f"{row['mode']:12} plans={row['unique_physical_plans']:>2} "
                f"best_median={row['best_finalist_median_seconds']:>8}s "
                f"baseline_best_retained="
                f"{row['baseline_best_fingerprint_retained']} "
                f"latency_change="
                f"{row['latency_change_percent_vs_baseline']}% "
                f"equivalent={row['all_screened_results_equivalent']}"
            )

        print(
            "\nCaveat: this is a five-table representative-data screening "
            "experiment. Baseline candidates are screened once, then only "
            "the top three per configuration are repeated. It is not a "
            "full-SF10 execution-quality result."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
