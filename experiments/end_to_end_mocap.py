"""
End-to-end MOCAP experiment.

Flow:

    calibration observations
        ↓
    train calibration model
        ↓
    new query
        ↓
    MOCAP candidate generation
        ↓
    calibrated prediction
        ↓
    budget/deadline filtering
        ↓
    Pareto + Lagrangian selection
        ↓
    Spark execution
        ↓
    actual telemetry

This experiment demonstrates that the calibrated model is actually
connected to the MOCAP planning and execution path.
"""

from __future__ import annotations

import csv
import os

from pyspark.sql import SparkSession

from mocap.calibration.trainer import train_and_save
from mocap.cost.predictor import predict_candidates
from mocap.execution.executor import SparkExecutor
from mocap.pipeline import MOCAPPipeline
from mocap.query.parser import QueryRequest


CALIBRATION_DATASET = (
    "results/calibration_multi_query.csv"
)

MODEL_PATH = (
    "results/end_to_end_calibration_model.json"
)

OUTPUT = (
    "results/end_to_end_mocap.csv"
)


def make_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPEndToEnd")
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


def main() -> None:
    spark = make_spark()

    try:
        # ---------------------------------------------------------
        # 1. Train calibration model
        # ---------------------------------------------------------

        model = train_and_save(
            csv_path=CALIBRATION_DATASET,
            model_path=MODEL_PATH,
        )

        print("Calibration model trained.")
        print(
            "Weights [bias, cpu, io, shuffle]:",
            model.weights,
        )

        # ---------------------------------------------------------
        # 2. Create a new workload
        # ---------------------------------------------------------

        n = spark.range(0, 80).toDF("n_id")
        c = spark.range(0, 800).toDF("c_id")
        o = spark.range(0, 8000).toDF("o_id")

        n.createOrReplaceTempView(
            "e2e_n"
        )

        c.createOrReplaceTempView(
            "e2e_c"
        )

        o.createOrReplaceTempView(
            "e2e_o"
        )

        query = QueryRequest(
            query_id="E2E_MOCAP_Q1",
            sql="""
                SELECT *
                FROM e2e_n n
                JOIN e2e_c c
                    ON n.n_id = c.c_id
                JOIN e2e_o o
                    ON c.c_id = o.o_id
            """,
            budget=100.0,
            deadline=100.0,
            accuracy_tolerance=0.0,
        )

        # ---------------------------------------------------------
        # 3. Run MOCAP planning with calibration
        # ---------------------------------------------------------

        pipeline = MOCAPPipeline(
            spark=spark,
            calibration_model=model,
            num_cores=2,
        )

        result = pipeline.run(query)

        print()
        print("=" * 60)
        print("MOCAP PLANNING")
        print("=" * 60)

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
            raise RuntimeError(
                result.failure_reason
                or "MOCAP did not select a plan."
            )

        selected = result.selected_plan

        prediction = result.metrics[
            selected.plan_id
        ]

        print(
            f"Selected plan: {selected.plan_id}"
        )

        print(
            f"Expected cost: "
            f"{prediction.estimated_cost:.10f}"
        )

        print(
            f"Expected latency: "
            f"{prediction.estimated_latency:.6f}s"
        )

        # ---------------------------------------------------------
        # 4. Execute selected plan
        # ---------------------------------------------------------

        executor = SparkExecutor(
            spark=spark,
        )

        execution = executor.execute(
            selected,
            prediction=prediction,
        )

        print()
        print("=" * 60)
        print("MOCAP EXECUTION")
        print("=" * 60)

        print(
            f"Actual cost: "
            f"{execution.actual_cost:.10f}"
        )

        print(
            f"Actual latency: "
            f"{execution.actual_latency:.6f}s"
        )

        print(
            f"Result: "
            f"{execution.result_summary}"
        )

        # ---------------------------------------------------------
        # 5. Save experiment result
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

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query_id",
                    "candidate_count",
                    "feasible_count",
                    "pareto_count",
                    "selected_plan",
                    "predicted_cost",
                    "actual_cost",
                    "predicted_latency",
                    "actual_latency",
                    "budget",
                    "deadline",
                    "execution_mode",
                ],
            )

            writer.writeheader()

            writer.writerow(
                {
                    "query_id": query.query_id,
                    "candidate_count": len(
                        result.candidates
                    ),
                    "feasible_count": len(
                        result.feasible
                    ),
                    "pareto_count": len(
                        result.pareto_front
                    ),
                    "selected_plan": selected.plan_id,
                    "predicted_cost": (
                        prediction.estimated_cost
                    ),
                    "actual_cost": (
                        execution.actual_cost
                    ),
                    "predicted_latency": (
                        prediction.estimated_latency
                    ),
                    "actual_latency": (
                        execution.actual_latency
                    ),
                    "budget": query.budget,
                    "deadline": query.deadline,
                    "execution_mode": (
                        execution.execution_mode
                    ),
                }
            )

        print()
        print(
            f"Saved: {OUTPUT}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()