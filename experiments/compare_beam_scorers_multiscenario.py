from __future__ import annotations

import csv
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
REPEATS = 10
WARMUPS = 1
SEED = 42

SIZE_ESTIMATES = {
    "f": 720000.0,
    "a": 9000.0,
    "b": 18000.0,
    "c": 45000.0,
}

# Each profile changes which dimension is most selective.
SCENARIOS = {
    "a_selective": (2, 10, 100),
    "b_selective": (100, 2, 10),
    "c_selective": (10, 100, 2),
    "balanced": (5, 5, 5),
}


def register_workload(spark, scenario: str, moduli: tuple[int, int, int]):
    fact = spark.range(30000).select(
        F.col("id").alias("row_id"),
        F.pmod(F.xxhash64("id", F.lit(11)), F.lit(1000)).alias("ka"),
        F.pmod(F.xxhash64("id", F.lit(22)), F.lit(2000)).alias("kb"),
        F.pmod(F.xxhash64("id", F.lit(33)), F.lit(5000)).alias("kc"),
    )
    fact.createOrReplaceTempView("fact_sales")

    for (name, size, salt), modulus in zip(
        [
            ("dim_a", 1000, 101),
            ("dim_b", 2000, 202),
            ("dim_c", 5000, 303),
        ],
        moduli,
    ):
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

    query = QueryRequest(
        query_id=f"BEAM_SCORER_{scenario}",
        sql=sql,
        budget=1000.0,
    )
    return query


def collect_statistics(spark):
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
    physical_plan = dataframe._jdf.queryExecution().executedPlan().toString()
    fp = create_plan_fingerprint(physical_plan)

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
        .appName("MOCAP-Beam-Scorer-MultiScenario")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    summary_rows = []
    candidate_rows = []
    run_rows = []

    try:
        # Trigger Spark startup before timing any workload.
        spark.range(100).count()
        rng = random.Random(SEED)

        configurations = [("exhaustive", "exhaustive", None)]
        for width in WIDTHS:
            configurations.append((f"size_beam_{width}", "size", width))
            configurations.append((f"stats_beam_{width}", "stats", width))

        for scenario, moduli in SCENARIOS.items():
            print(f"\n===== Scenario: {scenario}, moduli={moduli} =====")

            query = register_workload(spark, scenario, moduli)
            stats = collect_statistics(spark)

            print("Filtered relation rows:", dict(stats.relation_rows))
            print("Measured selectivities:", dict(stats.join_selectivities))

            mode_data = {}
            sql_by_fp = {}

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
                    sql_by_fp.setdefault(fp, sql)

                mode_data[name] = {
                    "width": width,
                    "sql_count": len(candidates),
                    "fingerprints": fingerprints,
                }

            expected, _, _ = execute_once(spark, query.sql)
            print(f"Expected result count: {expected}")
            print(f"Unique physical plans: {len(sql_by_fp)}")

            samples = {fp: [] for fp in sql_by_fp}

            warmup_order = list(sql_by_fp)
            rng.shuffle(warmup_order)

            for fp in warmup_order:
                count, _, observed_fp = execute_once(spark, sql_by_fp[fp])
                if count != expected or observed_fp != fp:
                    raise AssertionError(f"Warm-up failed for {scenario}: {fp}")

            for repetition in range(1, REPEATS + 1):
                order = list(sql_by_fp)
                rng.shuffle(order)
                print(f"Repetition {repetition}/{REPEATS}")

                for fp in order:
                    count, elapsed, observed_fp = execute_once(
                        spark, sql_by_fp[fp]
                    )

                    if count != expected:
                        raise AssertionError(
                            f"Result mismatch for {scenario}, {fp}"
                        )
                    if observed_fp != fp:
                        raise AssertionError(
                            f"Physical plan changed for {scenario}, {fp}"
                        )

                    samples[fp].append(elapsed)
                    run_rows.append({
                        "scenario": scenario,
                        "query_id": query.query_id,
                        "physical_fingerprint": fp,
                        "repetition": repetition,
                        "latency_seconds": elapsed,
                        "result_count": count,
                        "warmups": WARMUPS,
                        "spark_master": "local[2]",
                        "aqe_enabled": False,
                        "cbo_enabled": False,
                    })

            medians = {
                fp: statistics.median(values)
                for fp, values in samples.items()
            }

            exhaustive_fps = mode_data["exhaustive"]["fingerprints"]
            exhaustive_best = min(exhaustive_fps, key=lambda fp: medians[fp])
            exhaustive_latency = medians[exhaustive_best]

            for name, info in mode_data.items():
                fingerprints = info["fingerprints"]
                best_fp = min(fingerprints, key=lambda fp: medians[fp])
                best_latency = medians[best_fp]
                loss = (
                    (best_latency - exhaustive_latency)
                    / exhaustive_latency * 100.0
                )

                summary_rows.append({
                    "scenario": scenario,
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
                        "scenario": scenario,
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

        outputs = [
            ("beam_scorer_multiscenario_summary.csv", summary_rows),
            ("beam_scorer_multiscenario_candidates.csv", candidate_rows),
            ("beam_scorer_multiscenario_runs.csv", run_rows),
        ]

        for filename, rows in outputs:
            with (out / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(rows[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)

        print("\n===== MULTI-SCENARIO SUMMARY =====")
        for row in summary_rows:
            print(
                f"{row['scenario']:12s} | {row['mode']:14s} | "
                f"candidates={row['candidate_sql_count']:2d} | "
                f"loss={row['pruning_loss_percent']:7.3f}% | "
                f"best retained={row['exhaustive_best_retained']}"
            )

        print("\nSaved:")
        for filename, _ in outputs:
            print(f"  results/{filename}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
