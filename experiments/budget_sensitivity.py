"""
Budget sensitivity experiment for MOCAP.

Runs the same query and candidate-generation process under different
budgets using the same calibrated prediction model.

The experiment measures how hard budget constraints affect:
    - feasible candidate count
    - Pareto frontier size
    - selected plan
    - predicted cost
    - predicted latency
"""

from __future__ import annotations

import csv
import os

from pyspark.sql import SparkSession

from mocap.cost.learned import CalibrationModel
from mocap.pipeline import MOCAPPipeline
from mocap.query.parser import QueryRequest


MODEL_PATH = "results/end_to_end_calibration_model.json"
OUTPUT = "results/budget_sensitivity.csv"


# Chosen around the observed candidate-cost thresholds.
BUDGETS = [
    0.000020,
    0.000023,
    0.000024,
    0.000026,
    0.000030,
    0.000040,
    0.000060,
    0.000100,
    0.000140,
]


def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPBudgetSensitivity")
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


def create_query(
    spark: SparkSession,
    budget: float,
) -> QueryRequest:

    n = spark.range(0, 80).toDF("n_id")
    c = spark.range(0, 800).toDF("c_id")
    o = spark.range(0, 8000).toDF("o_id")

    n.createOrReplaceTempView("budget_n")
    c.createOrReplaceTempView("budget_c")
    o.createOrReplaceTempView("budget_o")

    return QueryRequest(
        query_id="BUDGET_SENSITIVITY_Q1",
        sql="""
            SELECT *
            FROM budget_n n
            JOIN budget_c c
                ON n.n_id = c.c_id
            JOIN budget_o o
                ON c.c_id = o.o_id
        """,
        budget=budget,
        deadline=100.0,
        accuracy_tolerance=0.0,
    )


def main() -> None:
    spark = make_spark()

    try:
        model = CalibrationModel.load(
            MODEL_PATH
        )

        rows = []

        print()
        print("=" * 80)
        print("MOCAP BUDGET SENSITIVITY")
        print("=" * 80)

        for budget in BUDGETS:

            query = create_query(
                spark,
                budget,
            )

            pipeline = MOCAPPipeline(
                spark=spark,
                calibration_model=model,
                num_cores=2,
            )

            result = pipeline.run(query)

            if result.selected_plan is None:

                selected_plan = ""
                selected_cost = ""
                selected_latency = ""

            else:

                selected_plan = (
                    result.selected_plan.plan_id
                )

                selected_metrics = result.metrics[
                    selected_plan
                ]

                selected_cost = (
                    selected_metrics.estimated_cost
                )

                selected_latency = (
                    selected_metrics.estimated_latency
                )

            row = {
                "budget": budget,
                "candidate_count": len(
                    result.candidates
                ),
                "feasible_count": len(
                    result.feasible
                ),
                "pareto_count": len(
                    result.pareto_front
                ),
                "selected_plan": selected_plan,
                "selected_cost": selected_cost,
                "selected_latency": selected_latency,
                "failure_reason": (
                    result.failure_reason or ""
                ),
            }

            rows.append(row)

            print(
                f"budget={budget:.8f} | "
                f"feasible={len(result.feasible):2d} | "
                f"pareto={len(result.pareto_front):2d} | "
                f"selected={selected_plan or 'NONE'}"
            )

        # ---------------------------------------------------------
        # Save results
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
                "budget",
                "candidate_count",
                "feasible_count",
                "pareto_count",
                "selected_plan",
                "selected_cost",
                "selected_latency",
                "failure_reason",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        print()
        print(
            f"Saved results to {OUTPUT}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
