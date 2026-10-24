from __future__ import annotations

import csv
import math
import random
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from mocap.plans.generator import _generate_join_order_variants
from mocap.plans.representation import create_plan_fingerprint
from mocap.plans.search_statistics import JoinSearchStatistics
from mocap.query.parser import QueryRequest


WIDTHS = (2, 3, 5, 10)
REPEATS = 20
WARMUPS = 1
SEED = 42

SIZE_ESTIMATES = {
    "f": 720000.0,
    "a": 9000.0,
    "b": 18000.0,
    "c": 45000.0,
}


def build_workload(spark):
    fact = spark.range(30000).select(
        F.col("id").alias("row_id"),
        F.pmod(
            F.xxhash64("id", F.lit(11)), F.lit(1000)
        ).alias("ka"),
        F.pmod(
            F.xxhash64("id", F.lit(22)), F.lit(2000)
        ).alias("kb"),
        F.pmod(
            F.xxhash64("id", F.lit(33)), F.lit(5000)
        ).alias("kc"),
    )
    fact.createOrReplaceTempView("fact_sales")

    for name, size, salt, modulus in [
        ("dim_a", 1000, 101, 10),
        ("dim_b", 2000, 202, 2),
        ("dim_c", 5000, 303, 100),
    ]:
        dimension = spark.range(size).select(
            F.col("id"),
            (
                F.pmod(
                    F.xxhash64("id", F.lit(salt)),
                    F.lit(modulus),
                ) == 0
            ).alias("keep"),
        )
        dimension.createOrReplaceTempView(name)

    sql = """
        SELECT COUNT(*) AS result_count
        FROM fact_sales f
        JOIN dim_a a ON f.ka = a.id
        JOIN dim_b b ON f.kb = b.id
        JOIN dim_c c ON f.kc = c.id
        WHERE a.keep = true
          AND b.keep = true
          AND c.keep = true
    """

    return QueryRequest(
        query_id="BEAM_SCORER_AB",
        sql=sql,
        budget=1000.0,
    )


def collect_synthetic_statistics(spark):
    """Measure filtered row counts and pairwise join selectivities."""
    fact_rows = spark.table("fact_sales").count()
    relation_rows = {"f": float(fact_rows)}
    selectivities = {}

    for alias in ("a", "b", "c"):
        table = f"dim_{alias}"

        filtered_rows = spark.sql(
            f"SELECT COUNT(*) AS n FROM {table} WHERE keep = true"
        ).first()["n"]

        matched_rows = spark.sql(
            f"""
            SELECT COUNT(*) AS n
            FROM fact_sales f
            JOIN {table} d ON f.k{alias} = d.id
            WHERE d.keep = true
            """
        ).first()["n"]

        relation_rows[alias] = float(filtered_rows)
        denominator = fact_rows * filtered_rows

        selectivities[("f", alias)] = (
            float(matched_rows) / denominator
            if denominator
            else 0.0
        )

    return JoinSearchStatistics(
        relation_rows=relation_rows,
        join_selectivities=selectivities,
    )


def fingerprint(spark, sql):
    plan = (
        spark.sql(sql)
        ._jdf.queryExecution()
        .executedPlan()
        .toString()
    )
    return create_plan_fingerprint(plan)


def execute_once(spark, sql):
    dataframe = spark.sql(sql)
    plan = dataframe._jdf.queryExecution().executedPlan().toString()
    fp = create_plan_fingerprint(plan)

    start = time.perf_counter()
    rows = dataframe.collect()
    elapsed = time.perf_counter() - start

    if len(rows) != 1:
        raise AssertionError(f"Expected one aggregate row, got {len(rows)}")

    return int(rows[0][0]), elapsed, fp


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Compare-Beam-Scorers")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        query = build_workload(spark)
        stats = collect_synthetic_statistics(spark)

        print("Measured synthetic relation rows:", dict(stats.relation_rows))
        print(
            "Measured pairwise selectivities:",
            dict(stats.join_selectivities),
        )

        configurations = [("exhaustive", "exhaustive", None)]

        for width in WIDTHS:
            configurations.append(
                (f"size_beam_{width}", "size", width)
            )
            configurations.append(
                (f"stats_beam_{width}", "stats", width)
            )

        mode_data = {}
        sql_by_fingerprint = {}

        for name, scoring, width in configurations:
            kwargs = {}

            if scoring != "exhaustive":
                kwargs = {
                    "search_mode": "beam",
                    "beam_width": width,
                    "table_size_estimates": SIZE_ESTIMATES,
                }

                if scoring == "stats":
                    kwargs["join_search_statistics"] = stats

            variants = _generate_join_order_variants(query, **kwargs)
            candidates = [("original", query.sql)] + variants
            fingerprints = {}

            for label, sql in candidates:
                fp = fingerprint(spark, sql)
                fingerprints.setdefault(fp, label)
                sql_by_fingerprint.setdefault(fp, sql)

            mode_data[name] = {
                "width": width,
                "sql_count": len(candidates),
                "fingerprints": fingerprints,
            }

        expected, _, _ = execute_once(spark, query.sql)
        print(f"\nExpected result: {expected}")
        print(f"Unique physical plans across configurations: {len(sql_by_fingerprint)}")

        rng = random.Random(SEED)
        all_fingerprints = list(sql_by_fingerprint)
        samples = {fp: [] for fp in all_fingerprints}

        warmup_order = list(all_fingerprints)
        rng.shuffle(warmup_order)

        for fp in warmup_order:
            count, _, observed_fp = execute_once(
                spark, sql_by_fingerprint[fp]
            )
            if count != expected or observed_fp != fp:
                raise AssertionError(f"Warm-up verification failed for {fp}")

        # Each unique physical plan is timed once per repetition.
        # Configurations sharing that plan reuse the same samples.
        for repetition in range(1, REPEATS + 1):
            order = list(all_fingerprints)
            rng.shuffle(order)
            print(f"Measurement repetition {repetition}/{REPEATS}")

            for fp in order:
                count, elapsed, observed_fp = execute_once(
                    spark, sql_by_fingerprint[fp]
                )

                if count != expected:
                    raise AssertionError(f"Result mismatch for {fp}")
                if observed_fp != fp:
                    raise AssertionError(f"Physical plan changed for {fp}")

                samples[fp].append(elapsed)

        medians = {
            fp: statistics.median(values)
            for fp, values in samples.items()
        }

        exhaustive_fps = mode_data["exhaustive"]["fingerprints"]
        exhaustive_best = min(exhaustive_fps, key=lambda fp: medians[fp])
        exhaustive_latency = medians[exhaustive_best]

        summary_rows = []
        candidate_rows = []
        run_rows = []

        for repetition in range(REPEATS):
            for fp, values in samples.items():
                run_rows.append({
                    "query_id": query.query_id,
                    "physical_fingerprint": fp,
                    "repetition": repetition + 1,
                    "latency_seconds": values[repetition],
                    "result_count": expected,
                    "warmups": WARMUPS,
                    "spark_master": "local[2]",
                    "aqe_enabled": False,
                    "cbo_enabled": False,
                    "auto_broadcast_join_threshold": -1,
                })

        for name, info in mode_data.items():
            fingerprints = info["fingerprints"]
            best_fp = min(fingerprints, key=lambda fp: medians[fp])
            best_latency = medians[best_fp]
            loss = (
                (best_latency - exhaustive_latency)
                / exhaustive_latency * 100.0
            )

            summary_rows.append({
                "query_id": query.query_id,
                "mode": name,
                "beam_width": info["width"] if info["width"] else "",
                "candidate_sql_count": info["sql_count"],
                "unique_physical_plans": len(fingerprints),
                "best_candidate": fingerprints[best_fp],
                "best_fingerprint": best_fp,
                "best_median_latency_seconds": best_latency,
                "pruning_loss_percent": loss,
                "exhaustive_best_retained": exhaustive_best in fingerprints,
                "repetitions": REPEATS,
            })

            for fp, label in fingerprints.items():
                candidate_rows.append({
                    "query_id": query.query_id,
                    "mode": name,
                    "beam_width": info["width"] if info["width"] else "",
                    "candidate_label": label,
                    "physical_fingerprint": fp,
                    "shared_median_latency_seconds": medians[fp],
                    "shared_mean_latency_seconds": statistics.mean(samples[fp]),
                    "repetitions": REPEATS,
                })

        out = Path("results")
        out.mkdir(exist_ok=True)

        for filename, rows in [
            ("beam_scorer_comparison_summary.csv", summary_rows),
            ("beam_scorer_comparison_candidates.csv", candidate_rows),
            ("beam_scorer_comparison_runs.csv", run_rows),
        ]:
            with (out / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(rows[0])
                )
                writer.writeheader()
                writer.writerows(rows)

        print("\n=== SCORER COMPARISON ===")
        for row in summary_rows:
            print(
                f"{row['mode']:16s} "
                f"candidates={row['candidate_sql_count']:2d} "
                f"unique={row['unique_physical_plans']:2d} "
                f"best={row['best_median_latency_seconds']:.6f}s "
                f"loss={row['pruning_loss_percent']:.2f}% "
                f"retained={row['exhaustive_best_retained']}"
            )

        print("\nSaved scorer comparison CSV files under results/.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
