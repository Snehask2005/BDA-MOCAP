from __future__ import annotations

import csv
import random
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from mocap.plans.generator import _generate_join_order_variants
from mocap.plans.representation import create_plan_fingerprint
from mocap.query.parser import QueryRequest


BEAM_WIDTHS = (2, 3, 5, 10)
REPEATS = 20
WARMUPS = 1
SEED = 42


def build_workload(spark):
    fact = spark.range(30_000).select(
        F.col("id").alias("row_id"),
        F.pmod(F.xxhash64("id", F.lit(11)), F.lit(1000)).alias("ka"),
        F.pmod(F.xxhash64("id", F.lit(22)), F.lit(2000)).alias("kb"),
        F.pmod(F.xxhash64("id", F.lit(33)), F.lit(5000)).alias("kc"),
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
                F.pmod(F.xxhash64("id", F.lit(salt)), F.lit(modulus)) == 0
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
        query_id="BEAM_STAR_4",
        sql=sql,
        budget=1000.0,
    )


def fingerprint(spark, sql):
    plan = spark.sql(sql)._jdf.queryExecution().executedPlan().toString()
    return create_plan_fingerprint(plan)


def execute_once(spark, sql):
    df = spark.sql(sql)
    plan = df._jdf.queryExecution().executedPlan().toString()
    fp = create_plan_fingerprint(plan)

    start = time.perf_counter()
    rows = df.collect()
    elapsed = time.perf_counter() - start

    if len(rows) != 1:
        raise AssertionError(f"Expected one aggregate row, got {len(rows)}")

    return int(rows[0][0]), elapsed, fp


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Beam-Quality-Shared-Timing")
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
        estimates = {"f": 720000.0, "a": 9000.0, "b": 18000.0, "c": 45000.0}

        modes = [("exhaustive", None)]
        modes.extend(("beam", width) for width in BEAM_WIDTHS)

        # Map each mode to its physical-plan fingerprints.
        mode_candidates = {}
        sql_by_fp = {}

        for mode, width in modes:
            if mode == "exhaustive":
                variants = _generate_join_order_variants(
                    query, search_mode="exhaustive"
                )
            else:
                variants = _generate_join_order_variants(
                    query,
                    search_mode="beam",
                    beam_width=width,
                    table_size_estimates=estimates,
                )

            candidates = [("original", query.sql)] + variants
            mapping = {}

            for label, sql in candidates:
                fp = fingerprint(spark, sql)
                mapping.setdefault(fp, label)
                sql_by_fp.setdefault(fp, sql)

            mode_name = "exhaustive" if width is None else f"beam_{width}"
            mode_candidates[mode_name] = {
                "width": width,
                "sql_count": len(candidates),
                "fingerprints": mapping,
            }

        expected, _, original_fp = execute_once(spark, query.sql)
        print(f"Original query result: {expected}")
        print(f"Unique physical plans across all modes: {len(sql_by_fp)}")

        unique_fps = list(sql_by_fp)
        rng = random.Random(SEED)

        # One warm-up per unique physical plan.
        warmups = list(unique_fps)
        rng.shuffle(warmups)
        for fp in warmups:
            count, _, observed_fp = execute_once(spark, sql_by_fp[fp])
            if count != expected or observed_fp != fp:
                raise AssertionError(f"Warm-up validation failed for {fp}")

        # Measure each unique physical plan exactly once per repetition.
        # Modes that share a fingerprint reuse the same measured samples.
        samples = {fp: [] for fp in unique_fps}
        run_rows = []

        for repetition in range(1, REPEATS + 1):
            order = list(unique_fps)
            rng.shuffle(order)
            print(f"Repetition {repetition}/{REPEATS}")

            for fp in order:
                count, elapsed, observed_fp = execute_once(spark, sql_by_fp[fp])

                if count != expected:
                    raise AssertionError(
                        f"Result mismatch for {fp}: {count} != {expected}"
                    )
                if observed_fp != fp:
                    raise AssertionError(f"Physical plan changed for {fp}")

                samples[fp].append(elapsed)
                run_rows.append({
                    "query_id": query.query_id,
                    "physical_fingerprint": fp,
                    "repetition": repetition,
                    "latency_seconds": elapsed,
                    "result_count": count,
                    "warmups": WARMUPS,
                    "spark_master": "local[2]",
                    "aqe_enabled": False,
                    "cbo_enabled": False,
                    "cache_note": "local Spark; execution order randomized",
                })

        medians = {
            fp: statistics.median(values)
            for fp, values in samples.items()
        }

        candidate_rows = []
        summary_rows = []

        exhaustive = mode_candidates["exhaustive"]
        exhaustive_best_fp = min(
            exhaustive["fingerprints"],
            key=lambda fp: medians[fp],
        )
        exhaustive_best_latency = medians[exhaustive_best_fp]

        for mode_name, info in mode_candidates.items():
            fingerprints = info["fingerprints"]
            best_fp = min(fingerprints, key=lambda fp: medians[fp])
            best_latency = medians[best_fp]

            loss = (
                (best_latency - exhaustive_best_latency)
                / exhaustive_best_latency * 100.0
            )

            summary_rows.append({
                "query_id": query.query_id,
                "mode": mode_name,
                "beam_width": info["width"] if info["width"] is not None else "",
                "candidate_sql_count": info["sql_count"],
                "unique_physical_plans": len(fingerprints),
                "best_candidate": fingerprints[best_fp],
                "best_fingerprint": best_fp,
                "best_median_latency_seconds": best_latency,
                "pruning_loss_percent": loss,
                "exhaustive_best_retained": exhaustive_best_fp in fingerprints,
                "repetitions_per_unique_plan": REPEATS,
                "warmups_per_unique_plan": WARMUPS,
            })

            for fp, label in fingerprints.items():
                candidate_rows.append({
                    "query_id": query.query_id,
                    "mode": mode_name,
                    "beam_width": info["width"] if info["width"] is not None else "",
                    "candidate_label": label,
                    "physical_fingerprint": fp,
                    "shared_median_latency_seconds": medians[fp],
                    "shared_mean_latency_seconds": statistics.mean(samples[fp]),
                    "repetitions": REPEATS,
                })

        out = Path("results")
        out.mkdir(exist_ok=True)

        outputs = [
            ("beam_quality_runs.csv", run_rows),
            ("beam_quality_candidates.csv", candidate_rows),
            ("beam_quality_summary.csv", summary_rows),
        ]

        for filename, rows in outputs:
            with (out / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        print("\n=== SUMMARY ===")
        for row in summary_rows:
            print(
                f"{row['mode']:10s} candidates={row['candidate_sql_count']:2d} "
                f"unique={row['unique_physical_plans']:2d} "
                f"best median={row['best_median_latency_seconds']:.6f}s "
                f"loss={row['pruning_loss_percent']:.2f}% "
                f"best retained={row['exhaustive_best_retained']}"
            )

        print("\nSaved results/beam_quality_runs.csv")
        print("Saved results/beam_quality_candidates.csv")
        print("Saved results/beam_quality_summary.csv")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
