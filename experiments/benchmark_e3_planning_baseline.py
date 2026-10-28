import csv
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
OUTPUT = ROOT / "results/e3/planning_runs.csv"
SUMMARY = ROOT / "results/e3/planning_summary.csv"

REPETITIONS = 10
SEED = 20261027
BEAM_WIDTHS = (10, 20)

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
    "search_mode",
    "beam_width",
    "repetition",
    "candidate_workers",
    "sql_variant_count",
    "unique_physical_plan_count",
    "analysis_error_count",
    "analysis_error_sample",
    "table_statistics_seconds",
    "sql_variant_generation_seconds",
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


def build_configurations():
    query_files = sorted(QUERIES.glob("q*.sql"))

    if len(query_files) != 8:
        raise RuntimeError(
            f"Expected 8 E3 queries; found {len(query_files)}"
        )

    configs = []

    for path in query_files:
        table_count = int(path.name[1:3])

        # Exhaustive search is the reference only where supported.
        if table_count <= 5:
            configs.append({
                "path": path,
                "table_count": table_count,
                "mode": "exhaustive",
                "width": "",
            })

        for width in BEAM_WIDTHS:
            configs.append({
                "path": path,
                "table_count": table_count,
                "mode": "beam",
                "width": width,
            })

    return configs


def run_one(spark, config):
    """Time query-template generation and Spark planning separately."""
    path = config["path"]
    table_count = config["table_count"]
    mode = config["mode"]
    width = config["width"] or 20
    sql = path.read_text(encoding="utf-8")

    query = QueryRequest(
        query_id=f"e3_{path.stem}_{mode}_{width}",
        sql=sql,
        budget=100.0,
        deadline=300.0,
    )

    total_start = time.perf_counter()

    # Beam search uses Spark size estimates. Record this separately.
    stats_seconds = 0.0
    table_size_estimates = None

    if mode == "beam":
        stage_start = time.perf_counter()
        table_size_estimates = generator._estimate_table_size_bytes(
            query, spark
        )
        stats_seconds = time.perf_counter() - stage_start

    # Generate the complete set of SQL variants without analyzing them.
    stage_start = time.perf_counter()
    variants = generator._generate_all_sql_variants(
        query,
        search_mode=mode,
        beam_width=width,
        table_size_estimates=table_size_estimates,
        join_search_statistics=None,
    )
    generation_seconds = time.perf_counter() - stage_start

    # Analyze every variant, deduplicate fingerprints, and extract features.
    # These steps mirror the sequential production candidate pipeline.
    stage_start = time.perf_counter()
    seen_fingerprints = set()
    analysis_errors = 0
    first_error = ""

    for strategy, candidate_sql in variants:
        try:
            dataframe = spark.sql(candidate_sql)
            physical_plan = generator.get_physical_plan(dataframe)
            fingerprint = generator.create_plan_fingerprint(physical_plan)

            if fingerprint in seen_fingerprints:
                continue

            generator.extract_physical_plan_features(physical_plan)
            seen_fingerprints.add(fingerprint)

        except Exception as exc:
            analysis_errors += 1
            if not first_error:
                first_error = f"{strategy}: {exc}"[:300]

    analysis_seconds = time.perf_counter() - stage_start
    total_seconds = time.perf_counter() - total_start

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_file": path.name,
        "table_count": table_count,
        "join_count": table_count - 1,
        "search_mode": mode,
        "beam_width": config["width"],
        "candidate_workers": 1,
        "sql_variant_count": len(variants),
        "unique_physical_plan_count": len(seen_fingerprints),
        "analysis_error_count": analysis_errors,
        "analysis_error_sample": first_error,
        "table_statistics_seconds": f"{stats_seconds:.6f}",
        "sql_variant_generation_seconds": f"{generation_seconds:.6f}",
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


def write_summary(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["query_file"],
            row["table_count"],
            row["search_mode"],
            str(row["beam_width"]),
        )
        groups[key].append(row)

    summary_rows = []

    for (query_file, table_count, mode, width), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    ):
        totals = [
            float(row["total_planning_seconds"]) for row in group
        ]

        summary_rows.append({
            "query_file": query_file,
            "table_count": table_count,
            "join_count": table_count - 1,
            "search_mode": mode,
            "beam_width": width,
            "sql_variant_count": group[0]["sql_variant_count"],
            "unique_physical_plan_count":
                group[0]["unique_physical_plan_count"],
            "measured_repetitions": len(group),
            "median_table_statistics_seconds": f"{statistics.median(float(r['table_statistics_seconds']) for r in group):.6f}",
            "median_sql_variant_generation_seconds": f"{statistics.median(float(r['sql_variant_generation_seconds']) for r in group):.6f}",
            "median_spark_analysis_seconds": f"{statistics.median(float(r['spark_analysis_seconds']) for r in group):.6f}",
            "median_total_planning_seconds": f"{statistics.median(totals):.6f}",
            "min_total_planning_seconds": f"{min(totals):.6f}",
            "max_total_planning_seconds": f"{max(totals):.6f}",
            "runs_with_analysis_errors": sum(
                int(row["analysis_error_count"]) > 0 for row in group
            ),
            "query_execution_performed": False,
        })

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    with SUMMARY.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {SUMMARY}")

    print("\n===== E3 MEDIAN PLANNING SUMMARY =====")
    for row in summary_rows:
        print(
            f"{row['query_file']:22} "
            f"{row['search_mode']:9} "
            f"width={str(row['beam_width']):>2} "
            f"variants={row['sql_variant_count']:>4} "
            f"plans={row['unique_physical_plan_count']:>4} "
            f"generation={float(row['median_sql_variant_generation_seconds']):.3f}s "
            f"analysis={float(row['median_spark_analysis_seconds']):.3f}s "
            f"total={float(row['median_total_planning_seconds']):.3f}s"
        )


def main():
    missing = [str(path) for path in TABLE_PATHS.values()
               if not path.is_dir()]

    if missing:
        raise FileNotFoundError(
            "Missing E3 input directories:\n" + "\n".join(missing)
        )

    configs = build_configurations()
    rng = random.Random(SEED)

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Planning-Baseline")
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
            print(f"Registered {table}")

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSpark version: {spark.version}")
        print("Master: local[2]; AQE=false; CBO=false; join reorder=false")
        print(f"Configurations: {len(configs)}")
        print(f"Measured repetitions per configuration: {REPETITIONS}")
        print("Workers: 1 (sequential baseline)")

        print("\n===== WARM-UP (NOT RECORDED) =====")
        for index, config in enumerate(configs, start=1):
            result = run_one(spark, config)
            print(
                f"warm-up {index}/{len(configs)} "
                f"{result['query_file']} {result['search_mode']} "
                f"width={result['beam_width'] or '-'} "
                f"variants={result['sql_variant_count']} "
                f"total={float(result['total_planning_seconds']):.3f}s "
                f"errors={result['analysis_error_count']}"
            )

        all_rows = []

        # Save each measured row immediately so partial results survive
        # an interrupted experiment.
        with OUTPUT.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=RUN_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()

            for repetition in range(1, REPETITIONS + 1):
                order = list(configs)
                rng.shuffle(order)

                print(f"\n===== MEASURED REPETITION {repetition}/{REPETITIONS} =====")

                for config in order:
                    row = run_one(spark, config)
                    row["repetition"] = repetition
                    writer.writerow(row)
                    file.flush()
                    all_rows.append(row)

                    print(
                        f"{row['query_file']:22} "
                        f"{row['search_mode']:9} "
                        f"width={str(row['beam_width'] or '-'):>2} "
                        f"variants={row['sql_variant_count']:>4} "
                        f"generation={float(row['sql_variant_generation_seconds']):.3f}s "
                        f"analysis={float(row['spark_analysis_seconds']):.3f}s "
                        f"total={float(row['total_planning_seconds']):.3f}s "
                        f"errors={row['analysis_error_count']}"
                    )

        print(f"\nSaved raw runs: {OUTPUT}")

        write_summary(all_rows)

        error_runs = sum(
            int(row["analysis_error_count"]) > 0 for row in all_rows
        )
        if error_runs:
            print(
                f"\nWARNING: {error_runs} measured runs had analysis errors. "
                "Inspect the raw CSV before interpreting the results."
            )
        else:
            print("\nAll measured runs analyzed every candidate successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
