import argparse
import contextlib
import csv
import io
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
FULL = ROOT / "benchmarks/tpcds/full_sf10/parquet"
E3 = ROOT / "benchmarks/tpcds/e3/parquet"
QUERIES = ROOT / "benchmarks/tpcds/e3/queries"
RESULTS = ROOT / "results/e3"

REPETITIONS_DEFAULT = 10
SEED = 20261027
BEAM_WIDTH = 10
ANALYSIS_LIMITS = (None, 12, 20)

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
    "query_file",
    "table_count",
    "join_count",
    "mode",
    "beam_width",
    "candidate_analysis_limit",
    "repetition",
    "generated_sql_variant_count",
    "analyzed_sql_variant_count",
    "unique_physical_plan_count",
    "analysis_error_count",
    "analysis_error_sample",
    "table_statistics_seconds",
    "sql_variant_generation_seconds",
    "prescoring_seconds",
    "spark_analysis_seconds",
    "total_planning_seconds",
    "spark_version",
    "spark_master",
    "spark_aqe_enabled",
    "spark_cbo_enabled",
    "spark_cbo_join_reorder_enabled",
    "auto_broadcast_join_threshold",
    "query_execution_performed",
]


def build_configurations(query_paths):
    configs = []

    for path in query_paths:
        table_count = int(path.name[1:3])

        for limit in ANALYSIS_LIMITS:
            mode = "baseline" if limit is None else f"prescore_{limit}"
            configs.append({
                "path": path,
                "table_count": table_count,
                "mode": mode,
                "limit": limit,
            })

    return configs


def run_one(spark, config):
    """Benchmark the production candidate-generation entry point."""
    path = config["path"]
    limit = config["limit"]
    mode = config["mode"]

    query = QueryRequest(
        query_id=f"e3_{path.stem}_{mode}",
        sql=path.read_text(encoding="utf-8"),
        budget=100.0,
        deadline=300.0,
    )

    metrics = {
        "table_statistics_seconds": 0.0,
        "sql_variant_generation_seconds": 0.0,
        "prescoring_seconds": 0.0,
        "generated_sql_variant_count": 0,
        "analyzed_sql_variant_count": None,
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
        result = original_generate(*args, **kwargs)
        metrics["sql_variant_generation_seconds"] += (
            time.perf_counter() - started
        )
        metrics["generated_sql_variant_count"] = len(result)
        return result

    def timed_prescore(sql_variants, *args, **kwargs):
        started = time.perf_counter()
        try:
            result = original_prescore(sql_variants, *args, **kwargs)
            metrics["analyzed_sql_variant_count"] = len(result)
            return result
        finally:
            metrics["prescoring_seconds"] += (
                time.perf_counter() - started
            )

    # Instrument the real generator without modifying production code.
    generator._estimate_table_size_bytes = timed_estimator
    generator._generate_all_sql_variants = timed_generate
    generator._pre_score_sql_variants = timed_prescore

    captured_output = io.StringIO()
    total_start = time.perf_counter()

    try:
        with contextlib.redirect_stdout(captured_output):
            candidates = generator.generate_join_candidates(
                query,
                spark,
                search_mode="beam",
                beam_width=BEAM_WIDTH,
                candidate_workers=1,
                candidate_analysis_limit=limit,
            )
    finally:
        generator._estimate_table_size_bytes = original_estimator
        generator._generate_all_sql_variants = original_generate
        generator._pre_score_sql_variants = original_prescore

    total_seconds = time.perf_counter() - total_start

    if metrics["analyzed_sql_variant_count"] is None:
        # The baseline intentionally does not invoke the pre-scoring helper.
        metrics["analyzed_sql_variant_count"] = (
            metrics["generated_sql_variant_count"]
        )

    # This residual includes Spark physical-plan analysis, fingerprinting,
    # deduplication, feature extraction, and small pipeline overheads.
    analysis_seconds = (
        total_seconds
        - metrics["table_statistics_seconds"]
        - metrics["sql_variant_generation_seconds"]
        - metrics["prescoring_seconds"]
    )

    error_lines = [
        line.strip()
        for line in captured_output.getvalue().splitlines()
        if "Skipping strategy" in line
    ]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_file": path.name,
        "table_count": config["table_count"],
        "join_count": config["table_count"] - 1,
        "mode": mode,
        "beam_width": BEAM_WIDTH,
        "candidate_analysis_limit": "" if limit is None else limit,
        "repetition": "",
        "generated_sql_variant_count":
            metrics["generated_sql_variant_count"],
        "analyzed_sql_variant_count":
            metrics["analyzed_sql_variant_count"],
        "unique_physical_plan_count": len(candidates),
        "analysis_error_count": len(error_lines),
        "analysis_error_sample": error_lines[0][:300] if error_lines else "",
        "table_statistics_seconds":
            f"{metrics['table_statistics_seconds']:.6f}",
        "sql_variant_generation_seconds":
            f"{metrics['sql_variant_generation_seconds']:.6f}",
        "prescoring_seconds": f"{metrics['prescoring_seconds']:.6f}",
        "spark_analysis_seconds": f"{analysis_seconds:.6f}",
        "total_planning_seconds": f"{total_seconds:.6f}",
        "spark_version": spark.version,
        "spark_master": "local[2]",
        "spark_aqe_enabled": False,
        "spark_cbo_enabled": False,
        "spark_cbo_join_reorder_enabled": False,
        "auto_broadcast_join_threshold": -1,
        "query_execution_performed": False,
    }


def write_summary(rows, output_path):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["query_file"],
            row["mode"],
            str(row["candidate_analysis_limit"]),
        )
        groups[key].append(row)

    baseline_medians = {}
    for (query_file, mode, _limit), group in groups.items():
        if mode == "baseline":
            baseline_medians[query_file] = statistics.median(
                float(row["total_planning_seconds"]) for row in group
            )

    summary_rows = []

    for (query_file, mode, limit), group in sorted(groups.items()):
        median_total = statistics.median(
            float(row["total_planning_seconds"]) for row in group
        )
        baseline = baseline_medians[query_file]
        saved_percent = 100.0 * (baseline - median_total) / baseline

        summary_rows.append({
            "query_file": query_file,
            "table_count": group[0]["table_count"],
            "join_count": group[0]["join_count"],
            "mode": mode,
            "candidate_analysis_limit": limit,
            "generated_sql_variant_count":
                group[0]["generated_sql_variant_count"],
            "analyzed_sql_variant_count":
                group[0]["analyzed_sql_variant_count"],
            "unique_physical_plan_count":
                group[0]["unique_physical_plan_count"],
            "measured_repetitions": len(group),
            "median_table_statistics_seconds": f"{statistics.median(float(r['table_statistics_seconds']) for r in group):.6f}",
            "median_sql_variant_generation_seconds": f"{statistics.median(float(r['sql_variant_generation_seconds']) for r in group):.6f}",
            "median_prescoring_seconds": f"{statistics.median(float(r['prescoring_seconds']) for r in group):.6f}",
            "median_spark_analysis_seconds": f"{statistics.median(float(r['spark_analysis_seconds']) for r in group):.6f}",
            "median_total_planning_seconds": f"{median_total:.6f}",
            "planning_time_saved_percent_vs_baseline": f"{saved_percent:.2f}",
            "runs_with_analysis_errors": sum(
                int(row["analysis_error_count"]) > 0 for row in group
            ),
            "query_execution_performed": False,
        })

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {output_path}")
    print("\n===== E3 PRE-SCORING SUMMARY =====")

    for row in summary_rows:
        print(
            f"{row['query_file']:22} "
            f"{row['mode']:12} "
            f"generated={row['generated_sql_variant_count']:>3} "
            f"analyzed={row['analyzed_sql_variant_count']:>3} "
            f"plans={row['unique_physical_plan_count']:>3} "
            f"analysis={float(row['median_spark_analysis_seconds']):.3f}s "
            f"total={float(row['median_total_planning_seconds']):.3f}s "
            f"saved={float(row['planning_time_saved_percent_vs_baseline']):+.1f}%"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-queries",
        action="store_true",
        help="Benchmark all E3 queries, from 3 through 10 tables.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS_DEFAULT,
    )
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    missing = [
        str(path) for path in TABLE_PATHS.values()
        if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing E3 input directories:\n" + "\n".join(missing)
        )

    if args.all_queries:
        query_paths = sorted(QUERIES.glob("q*.sql"))
        if len(query_paths) != 8:
            raise RuntimeError(
                f"Expected 8 E3 queries; found {len(query_paths)}"
            )
        output_tag = "all"
    else:
        query_path = QUERIES / "q10_10_tables.sql"
        if not query_path.is_file():
            raise FileNotFoundError(query_path)
        query_paths = [query_path]
        output_tag = "q10"

    configs = build_configurations(query_paths)
    rng = random.Random(SEED)

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-PreScoring-Benchmark")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    output_runs = RESULTS / f"prescoring_{output_tag}_runs.csv"
    output_summary = RESULTS / f"prescoring_{output_tag}_summary.csv"

    try:
        for table, path in TABLE_PATHS.items():
            spark.read.parquet(
                path.resolve().as_uri()
            ).createOrReplaceTempView(table)
            print(f"Registered {table}")

        print(f"\nSpark version: {spark.version}")
        print("Master: local[2]; AQE=false; CBO=false; join reorder=false")
        print(f"Queries: {len(query_paths)}")
        print(f"Configurations: {len(configs)}")
        print(f"Measured repetitions per configuration: {args.repetitions}")
        print("Candidate workers: 1")
        print("Queries will not be executed.")

        print("\n===== WARM-UP (NOT RECORDED) =====")
        for index, config in enumerate(configs, start=1):
            result = run_one(spark, config)
            print(
                f"warm-up {index}/{len(configs)} "
                f"{result['query_file']} {result['mode']} "
                f"analyzed={result['analyzed_sql_variant_count']} "
                f"total={float(result['total_planning_seconds']):.3f}s "
                f"errors={result['analysis_error_count']}"
            )

        all_rows = []
        RESULTS.mkdir(parents=True, exist_ok=True)

        with output_runs.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=RUN_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()

            for repetition in range(1, args.repetitions + 1):
                order = list(configs)
                rng.shuffle(order)
                print(
                    f"\n===== MEASURED REPETITION "
                    f"{repetition}/{args.repetitions} ====="
                )

                for config in order:
                    row = run_one(spark, config)
                    row["repetition"] = repetition
                    writer.writerow(row)
                    file.flush()
                    all_rows.append(row)

                    print(
                        f"{row['query_file']:22} "
                        f"{row['mode']:12} "
                        f"analyzed={row['analyzed_sql_variant_count']:>3} "
                        f"plans={row['unique_physical_plan_count']:>3} "
                        f"total={float(row['total_planning_seconds']):.3f}s "
                        f"errors={row['analysis_error_count']}"
                    )

        print(f"\nSaved raw runs: {output_runs}")
        write_summary(all_rows, output_summary)

        error_runs = sum(
            int(row["analysis_error_count"]) > 0 for row in all_rows
        )
        if error_runs:
            print(
                f"\nWARNING: {error_runs} measured runs had analysis errors. "
                "Inspect the raw CSV before interpreting the results."
            )
        else:
            print("\nAll measured runs completed without candidate-analysis errors.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
