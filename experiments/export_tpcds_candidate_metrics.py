import csv
import os
import sys
import time

from pyspark.sql import SparkSession

# Allow imports from project root
sys.path.insert(0, os.path.abspath("."))

from mocap.query.parser import QueryRequest
from mocap.cost.statistics import extract_plan_statistics
from mocap.pipeline import MOCAPPipeline
from mocap.plans.generator import generate_join_candidates


BASE = "benchmarks/tpcds/representative/parquet"
QUERY_FILE = "benchmarks/tpcds/representative/queries/q5_join_order.sql"
OUTPUT = "results/tpcds_candidate_metrics.csv"


def load_tables(spark):
    tables = {
        "store_sales": "store_sales",
        "item": "item",
        "store": "store",
        "date_dim": "date_dim",
        "customer_demographics": "customer_demographics",
    }

    for name, directory in tables.items():
        path = os.path.join(BASE, directory)

        df = spark.read.parquet(path)
        df.createOrReplaceTempView(name)

        print(f"{name:25s} rows = {df.count():,}")


def read_query():
    with open(QUERY_FILE) as f:
        return f.read().strip()


def main():

    spark = (
        SparkSession.builder
        .appName("MOCAP-TPCDS-Candidate-Metrics")
        .master("local[2]")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("\nLoading TPC-DS-derived tables...")
        load_tables(spark)

        sql = read_query()

        query = QueryRequest(
            query_id="tpcds_derived_q5",
            sql=sql,
            budget=1.0,
        )

        print("\nGenerating candidates...")
        start = time.perf_counter()

        candidates = generate_join_candidates(
            query,
            spark,
        )

        generation_time = time.perf_counter() - start

        print(f"Candidates generated: {len(candidates)}")
        print(f"Generation time: {generation_time:.6f} s")

        # ---------------------------------------------------------
        # Candidate-specific estimation
        # ---------------------------------------------------------

        pipeline = MOCAPPipeline(
            spark,
            num_cores=2,
        )

        print("\nEstimating candidate-specific metrics...")

        rows = []

        start = time.perf_counter()

        metrics = pipeline.estimate_candidates(candidates)

        estimation_time = time.perf_counter() - start

        # estimate_candidates() already returns:
        # Dict[str, PlanMetrics], keyed by plan_id.
        metric_map = metrics

        for candidate in candidates:

            metric = metric_map.get(candidate.plan_id)

            if metric is None:
                print(
                    f"WARNING: No metric found for "
                    f"{candidate.plan_id}"
                )
                continue

            # Extract Catalyst statistics for this specific candidate.
            candidate_df = spark.sql(candidate.sql)
            statistics = extract_plan_statistics(candidate_df)

            rows.append(
                {
                    "query_id": candidate.query_id,
                    "plan_id": candidate.plan_id,
                    "strategy": candidate.strategy,
                    "fingerprint": candidate.fingerprint,

                    "input_bytes":
                        statistics.input_bytes,

                    "intermediate_bytes":
                        statistics.intermediate_bytes,

                    "row_count":
                        statistics.row_count,

                    "estimated_cost":
                        metric.estimated_cost,

                    "estimated_latency":
                        metric.estimated_latency,

                    "cpu_component":
                        metric.cpu_component,

                    "io_component":
                        metric.io_component,

                    "shuffle_component":
                        metric.shuffle_component,

                    "cpu_seconds":
                        metric.cpu_seconds,

                    "bytes_scanned":
                        metric.bytes_scanned,

                    "bytes_shuffled":
                        metric.bytes_shuffled,

                }
            )

        # ---------------------------------------------------------
        # Save CSV
        # ---------------------------------------------------------

        os.makedirs(
            os.path.dirname(OUTPUT),
            exist_ok=True,
        )

        fieldnames = [
            "query_id",
            "plan_id",
            "strategy",
            "fingerprint",
            "input_bytes",
            "intermediate_bytes",
            "row_count",
            "estimated_cost",
            "estimated_latency",
            "cpu_component",
            "io_component",
            "shuffle_component",
            "cpu_seconds",
            "bytes_scanned",
            "bytes_shuffled",
        ]

        with open(
            OUTPUT,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        # ---------------------------------------------------------
        # Summary
        # ---------------------------------------------------------

        print("\n" + "=" * 70)
        print("TPC-DS-DERIVED CANDIDATE METRICS")
        print("=" * 70)

        print(f"Candidates:              {len(candidates)}")
        print(f"Metric records:           {len(rows)}")
        print(
            "Unique fingerprints:      "
            f"{len(set(r['fingerprint'] for r in rows))}"
        )

        print(
            "Unique bytes scanned:     "
            f"{len(set(r['bytes_scanned'] for r in rows))}"
        )

        print(
            "Unique bytes shuffled:    "
            f"{len(set(r['bytes_shuffled'] for r in rows))}"
        )

        print(
            "Unique predicted costs:   "
            f"{len(set(r['estimated_cost'] for r in rows))}"
        )

        print(
            "Unique predicted latency: "
            f"{len(set(r['estimated_latency'] for r in rows))}"
        )

        print(
            f"\nGeneration time:          "
            f"{generation_time:.6f} s"
        )

        print(
            f"Estimation time:          "
            f"{estimation_time:.6f} s"
        )

        print(
            f"Total planning/estimation:"
            f" {generation_time + estimation_time:.6f} s"
        )

        # ---------------------------------------------------------
        # Strategy summary
        # ---------------------------------------------------------

        print("\nStrategy counts:")

        strategy_counts = {}

        for row in rows:
            strategy = row["strategy"]
            strategy_counts[strategy] = (
                strategy_counts.get(strategy, 0) + 1
            )

        for strategy, count in sorted(
            strategy_counts.items()
        ):
            print(
                f"  {strategy:45s} {count}"
            )

        print(
            f"\nSaved to: {OUTPUT}"
        )

        print("\nDONE.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
