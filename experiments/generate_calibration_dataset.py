"""
Generate a multi-query calibration dataset for MOCAP.

Each query produces multiple candidate plans. Every candidate is
executed once and paired with its pre-execution prediction.

The resulting dataset is suitable for query-level train/test
calibration validation.
"""

from __future__ import annotations

import csv
import os

from pyspark.sql import SparkSession

from mocap.calibration.service import CalibrationService
from mocap.cost.estimator import collect_execution_telemetry
from mocap.cost.predictor import predict_candidates
from mocap.cost.statistics import extract_plan_statistics
from mocap.plans.generator import generate_join_candidates
from mocap.query.parser import QueryRequest


OUTPUT = "results/calibration_multi_query.csv"


def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPCalibrationDataset")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def create_tables(spark: SparkSession, scale: int, suffix: str) -> None:
    n = spark.range(0, scale).toDF("n_id")
    c = spark.range(0, scale * 10).toDF("c_id")
    o = spark.range(0, scale * 100).toDF("o_id")

    n.createOrReplaceTempView(f"n_{suffix}")
    c.createOrReplaceTempView(f"c_{suffix}")
    o.createOrReplaceTempView(f"o_{suffix}")


def main() -> None:
    spark = make_spark()

    try:
        os.makedirs(
            os.path.dirname(OUTPUT),
            exist_ok=True,
        )

        if os.path.exists(OUTPUT):
            os.remove(OUTPUT)

        calibration = CalibrationService()

        fieldnames = [
            "plan_id",
            "query_id",
            "cpu_component",
            "io_component",
            "shuffle_component",
            "estimated_cost",
            "actual_cost",
            "estimated_latency",
            "actual_latency",
        ]

        all_rows = []

        # Different workloads give the calibration model
        # observations from multiple independent queries.
        workloads = [
            ("CALIB_Q1", 50),
            ("CALIB_Q2", 75),
            ("CALIB_Q3", 100),
            ("CALIB_Q4", 150),
        ]

        for query_id, scale in workloads:
            suffix = query_id.lower()

            create_tables(
                spark=spark,
                scale=scale,
                suffix=suffix,
            )

            sql = f"""
                SELECT *
                FROM n_{suffix} n
                JOIN c_{suffix} c
                    ON n.n_id = c.c_id
                JOIN o_{suffix} o
                    ON c.c_id = o.o_id
            """

            query = QueryRequest(
                query_id=query_id,
                sql=sql,
                budget=100.0,
                deadline=100.0,
                accuracy_tolerance=0.0,
            )

            candidates = generate_join_candidates(
                query=query,
                spark=spark,
            )

            # We only need a small, controlled candidate subset
            # for calibration-data generation.
            candidates = candidates[:3]

            if len(candidates) < 3:
                raise RuntimeError(
                    f"{query_id}: expected at least 3 candidates, "
                    f"got {len(candidates)}"
                )

            for candidate in candidates:
                dataframe = spark.sql(candidate.sql)

                statistics = extract_plan_statistics(
                    dataframe
                )

                prediction_map = predict_candidates(
                    [candidate],
                    statistics=statistics,
                )

                prediction = prediction_map[
                    candidate.plan_id
                ]

                telemetry = collect_execution_telemetry(
                    candidate=candidate,
                    spark=spark,
                    num_cores=2,
                )

                sample = calibration.record(
                    prediction=prediction,
                    telemetry=telemetry,
                )

                all_rows.append(
                    {
                        "plan_id": sample.plan_id,
                        "query_id": sample.query_id,
                        "cpu_component": sample.cpu_component,
                        "io_component": sample.io_component,
                        "shuffle_component": sample.shuffle_component,
                        "estimated_cost": sample.estimated_cost,
                        "actual_cost": sample.actual_cost,
                        "estimated_latency": sample.estimated_latency,
                        "actual_latency": sample.actual_latency,
                    }
                )

                print(
                    f"{query_id} | "
                    f"{candidate.plan_id} | "
                    f"predicted={prediction.estimated_cost:.8f} | "
                    f"actual={telemetry.actual_cost:.8f} | "
                    f"latency={telemetry.actual_latency:.4f}s"
                )

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
            writer.writerows(all_rows)

        print()
        print("=" * 60)
        print("Calibration dataset generated")
        print("=" * 60)
        print(f"Output: {OUTPUT}")
        print(f"Samples: {len(all_rows)}")
        print(
            "Queries:",
            sorted(
                {
                    row["query_id"]
                    for row in all_rows
                }
            ),
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
