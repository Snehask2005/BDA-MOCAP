from __future__ import annotations

import csv
import math
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from mocap.plans.generator import generate_join_candidates
from mocap.query.parser import QueryRequest


TABLE_COUNTS = (3, 4, 5, 6, 8, 10)
BEAM_WIDTH = 10
WARMUPS = 2
REPEATS = 7
ROWS_PER_TABLE = 100


def register_chain(spark, n_tables: int) -> QueryRequest:
    table_names = [
        f"beam_scale_{n_tables}_table_{i}"
        for i in range(n_tables)
    ]
    aliases = [f"t{i}" for i in range(n_tables)]

    spark.range(ROWS_PER_TABLE).select(
        F.col("id")
    ).createOrReplaceTempView(table_names[0])

    for i in range(1, n_tables):
        frame = spark.range(ROWS_PER_TABLE).select(
            F.col("id"),
            F.pmod(
                F.col("id") - F.lit(1),
                F.lit(ROWS_PER_TABLE),
            ).alias("prev_id"),
        )
        frame.createOrReplaceTempView(table_names[i])

    sql = f"""
        SELECT {aliases[0]}.id AS result_id
        FROM {table_names[0]} {aliases[0]}
    """

    for i in range(1, n_tables):
        sql += (
            f" JOIN {table_names[i]} {aliases[i]}"
            f" ON {aliases[i-1]}.id = {aliases[i]}.prev_id"
        )

    return QueryRequest(
        query_id=f"BEAM_CATALYST_{n_tables}",
        sql=sql,
        budget=1_000_000.0,
    )


def generate(spark, query, mode: str):
    kwargs = {"search_mode": mode}

    if mode == "beam":
        kwargs["beam_width"] = BEAM_WIDTH

    start = time.perf_counter()
    candidates = generate_join_candidates(query, spark, **kwargs)
    elapsed = time.perf_counter() - start

    if not candidates:
        raise AssertionError(
            f"No candidates for {query.query_id}, mode={mode}"
        )

    return len(candidates), elapsed


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-Beam-Catalyst-Scalability")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.cbo.enabled", "false")
        .config("spark.sql.cbo.joinReorder.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    results = []

    try:
        # Trigger Spark initialization before collecting measurements.
        spark.range(100).count()

        for n_tables in TABLE_COUNTS:
            query = register_chain(spark, n_tables)

            modes = (
                ("exhaustive", "beam")
                if n_tables <= 5
                else ("beam",)
            )

            for mode in modes:
                warmup_counts = [
                    generate(spark, query, mode)[0]
                    for _ in range(WARMUPS)
                ]

                measured_counts = []
                times = []

                for _ in range(REPEATS):
                    count, elapsed = generate(spark, query, mode)
                    measured_counts.append(count)
                    times.append(elapsed)

                all_counts = warmup_counts + measured_counts
                if len(set(all_counts)) != 1:
                    raise AssertionError(
                        f"Candidate count changed across runs: "
                        f"{query.query_id}, {mode}, {all_counts}"
                    )

                ordered = sorted(times)
                p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]

                result = {
                    "query_id": query.query_id,
                    "table_count": n_tables,
                    "search_mode": mode,
                    "beam_width": BEAM_WIDTH if mode == "beam" else "",
                    "rows_per_table": ROWS_PER_TABLE,
                    "physical_candidates_after_dedup": measured_counts[0],
                    "generation_catalyst_median_seconds": round(
                        statistics.median(times), 6
                    ),
                    "generation_catalyst_mean_seconds": round(
                        statistics.mean(times), 6
                    ),
                    "generation_catalyst_p95_seconds": round(p95, 6),
                    "warmups": WARMUPS,
                    "measured_repetitions": REPEATS,
                    "spark_master": "local[2]",
                    "aqe_enabled": False,
                    "cbo_enabled": False,
                    "auto_broadcast_join_threshold": -1,
                    "executed_query": False,
                }
                results.append(result)

                print(
                    f"{n_tables:2d} tables | {mode:10s} | "
                    f"candidates={measured_counts[0]:2d} | "
                    f"median={statistics.median(times):.3f}s | "
                    f"P95={p95:.3f}s"
                )

        output = Path("results/beam_catalyst_scalability.csv")
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(results[0]),
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved: {output}")
        print(
            "These are candidate-generation and Catalyst-planning "
            "measurements, not query-execution timings."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
