"""
Compare MOCAP planning with and without cost calibration.

Controlled experiment:

    Same query
        |
        +--> structural predictor only
        |
        +--> calibrated predictor
        |
        +--> compare predictions and selected plans

No candidate query is executed.
"""

from __future__ import annotations

import csv
import os

from pyspark.sql import SparkSession

from mocap.cost.learned import CalibrationModel
from mocap.pipeline import MOCAPPipeline
from mocap.query.parser import QueryRequest


MODEL_PATH = (
    "results/end_to_end_calibration_model.json"
)

OUTPUT = (
    "results/calibration_impact_comparison.csv"
)


def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPCalibrationImpact")
        .config(
            "spark.sql.adaptive.enabled",
            "false",
        )
        .config(
            "spark.sql.autoBroadcastJoinThreshold",
            "-1",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def create_query(spark: SparkSession) -> QueryRequest:
    n = spark.range(0, 80).toDF("n_id")
    c = spark.range(0, 800).toDF("c_id")
    o = spark.range(0, 8000).toDF("o_id")

    n.createOrReplaceTempView("impact_n")
    c.createOrReplaceTempView("impact_c")
    o.createOrReplaceTempView("impact_o")

    return QueryRequest(
        query_id="CALIBRATION_IMPACT_Q1",
        sql="""
            SELECT *
            FROM impact_n n
            JOIN impact_c c
                ON n.n_id = c.c_id
            JOIN impact_o o
                ON c.c_id = o.o_id
        """,
        budget=100.0,
        deadline=100.0,
        accuracy_tolerance=0.0,
    )


def print_run(label, result):
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(
        f"Candidates: {len(result.candidates)}"
    )

    print(
        f"Feasible: {len(result.feasible)}"
    )

    print(
        f"Pareto: {len(result.pareto_front)}"
    )

    if result.selected_plan is None:
        print("Selected plan: NONE")
        print(
            "Failure:",
            result.failure_reason,
        )
    else:
        selected = result.selected_plan
        metrics = result.metrics[selected.plan_id]

        print(
            f"Selected plan: {selected.plan_id}"
        )

        print(
            f"Expected cost: "
            f"{metrics.estimated_cost:.10f}"
        )

        print(
            f"Expected latency: "
            f"{metrics.estimated_latency:.6f}s"
        )


def main() -> None:
    spark = make_spark()

    try:
        query = create_query(spark)

        # ---------------------------------------------------------
        # 1. Uncalibrated MOCAP
        # ---------------------------------------------------------

        uncalibrated_pipeline = MOCAPPipeline(
            spark=spark,
            calibration_model=None,
            num_cores=2,
        )

        uncalibrated = uncalibrated_pipeline.run(
            query
        )

        print_run(
            "UNCALIBRATED MOCAP",
            uncalibrated,
        )

        # ---------------------------------------------------------
        # 2. Load calibration model
        # ---------------------------------------------------------

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Calibration model not found: {MODEL_PATH}"
            )

        model = CalibrationModel.load(
            MODEL_PATH
        )

        print()
        print(
            "Calibration weights "
            "[bias, cpu, io, shuffle]:"
        )
        print(model.weights)

        # ---------------------------------------------------------
        # 3. Calibrated MOCAP
        # ---------------------------------------------------------

        calibrated_pipeline = MOCAPPipeline(
            spark=spark,
            calibration_model=model,
            num_cores=2,
        )

        calibrated = calibrated_pipeline.run(
            query
        )

        print_run(
            "CALIBRATED MOCAP",
            calibrated,
        )

        # ---------------------------------------------------------
        # 4. Compare candidate predictions
        # ---------------------------------------------------------

        candidate_ids = sorted(
            set(uncalibrated.metrics)
            | set(calibrated.metrics)
        )

        rows = []

        for plan_id in candidate_ids:
            raw = uncalibrated.metrics.get(plan_id)
            cal = calibrated.metrics.get(plan_id)

            if raw is None or cal is None:
                continue

            rows.append(
                {
                    "query_id": query.query_id,
                    "plan_id": plan_id,

                    "uncalibrated_cost":
                        raw.estimated_cost,

                    "calibrated_cost":
                        cal.estimated_cost,

                    "cost_change":
                        cal.estimated_cost
                        - raw.estimated_cost,

                    "uncalibrated_latency":
                        raw.estimated_latency,

                    "calibrated_latency":
                        cal.estimated_latency,

                    "latency_change":
                        cal.estimated_latency
                        - raw.estimated_latency,

                    "cost_changed":
                        int(
                            abs(
                                cal.estimated_cost
                                - raw.estimated_cost
                            ) > 1e-15
                        ),

                    "latency_changed":
                        int(
                            abs(
                                cal.estimated_latency
                                - raw.estimated_latency
                            ) > 1e-12
                        ),
                }
            )

        # ---------------------------------------------------------
        # 5. Print comparison summary
        # ---------------------------------------------------------

        changed_cost = sum(
            row["cost_changed"]
            for row in rows
        )

        changed_latency = sum(
            row["latency_changed"]
            for row in rows
        )

        raw_selected = (
            uncalibrated.selected_plan.plan_id
            if uncalibrated.selected_plan
            else None
        )

        calibrated_selected = (
            calibrated.selected_plan.plan_id
            if calibrated.selected_plan
            else None
        )

        print()
        print("=" * 70)
        print("CALIBRATION IMPACT SUMMARY")
        print("=" * 70)

        print(
            f"Candidates compared: {len(rows)}"
        )

        print(
            f"Candidates with changed predicted cost: "
            f"{changed_cost}"
        )

        print(
            f"Candidates with changed predicted latency: "
            f"{changed_latency}"
        )

        print(
            f"Uncalibrated selected plan: "
            f"{raw_selected}"
        )

        print(
            f"Calibrated selected plan: "
            f"{calibrated_selected}"
        )

        print(
            "Selection changed:",
            raw_selected != calibrated_selected,
        )

        # ---------------------------------------------------------
        # 6. Save detailed results
        # ---------------------------------------------------------

        os.makedirs(
            os.path.dirname(OUTPUT),
            exist_ok=True,
        )

        with open(
            OUTPUT,
            "w",
            newline="",
        ) as f:

            fieldnames = [
                "query_id",
                "plan_id",
                "uncalibrated_cost",
                "calibrated_cost",
                "cost_change",
                "uncalibrated_latency",
                "calibrated_latency",
                "latency_change",
                "cost_changed",
                "latency_changed",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        print()
        print(
            f"Saved comparison to {OUTPUT}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
