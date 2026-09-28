from pathlib import Path

from pyspark.sql import SparkSession

from mocap.calibration.dataset import load_dataset
from mocap.calibration.service import CalibrationService
from mocap.calibration.trainer import train_from_csv
from mocap.cost.estimator import collect_execution_telemetry
from mocap.pipeline import MOCAPPipeline
from mocap.query.parser import QueryRequest


def test_real_spark_calibration_feedback_loop(
    tmp_path: Path,
):
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPCalibrationTest")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        customers = [
            (i, f"customer_{i}")
            for i in range(20)
        ]

        orders = [
            (
                i,
                i % 20,
                float(i % 10 + 1),
            )
            for i in range(100)
        ]

        spark.createDataFrame(
            customers,
            ["id", "name"],
        ).createOrReplaceTempView("customer")

        spark.createDataFrame(
            orders,
            ["id", "customer_id", "amount"],
        ).createOrReplaceTempView("orders")

        query = QueryRequest(
            query_id="CALIBRATION_TEST",
            sql="""
                SELECT
                    c.id,
                    COUNT(o.id) AS order_count,
                    SUM(o.amount) AS total_amount
                FROM customer c
                JOIN orders o
                    ON c.id = o.customer_id
                GROUP BY c.id
                ORDER BY total_amount DESC
            """,
            budget=100.0,
        )

        # ---------------------------------------------------------
        # 1. MOCAP planning
        # ---------------------------------------------------------

        pipeline = MOCAPPipeline(spark)

        result = pipeline.run(query)

        assert result.selected_plan is not None

        selected_plan = result.selected_plan

        prediction = result.metrics[
            selected_plan.plan_id
        ]

        assert prediction.estimated_cost >= 0
        assert prediction.estimated_latency >= 0

        # ---------------------------------------------------------
        # 2. Execute selected candidate and collect telemetry
        # ---------------------------------------------------------

        candidate = next(
            candidate
            for candidate in result.candidates
            if candidate.plan_id == selected_plan.plan_id
        )

        telemetry = collect_execution_telemetry(
            candidate=candidate,
            spark=spark,
            num_cores=2,
        )

        assert telemetry.plan_id == selected_plan.plan_id
        assert telemetry.query_id == query.query_id
        assert telemetry.actual_cost >= 0
        assert telemetry.actual_latency > 0

        # ---------------------------------------------------------
        # 3. Feed prediction + observation into calibration
        # ---------------------------------------------------------

        calibration = CalibrationService()

        csv_path = tmp_path / "calibration.csv"

        sample = calibration.record_to_dataset(
            prediction=prediction,
            telemetry=telemetry,
            csv_path=str(csv_path),
        )

        assert sample.plan_id == selected_plan.plan_id
        assert calibration.sample_count == 1
        assert csv_path.exists()

        # ---------------------------------------------------------
        # 4. Verify persisted calibration data
        # ---------------------------------------------------------

        rows = load_dataset(
            str(csv_path)
        )

        assert len(rows) == 1

        assert rows[0]["plan_id"] == selected_plan.plan_id
        assert rows[0]["query_id"] == query.query_id

        assert rows[0]["estimated_cost"] == prediction.estimated_cost
        assert rows[0]["actual_cost"] == telemetry.actual_cost

        # ---------------------------------------------------------
        # 5. Train learned cost model
        #
        # The existing trainer intentionally requires at least
        # four independent observations.
        # ---------------------------------------------------------

        # The single real observation above proves the feedback
        # connection. Training is tested separately because a
        # statistically meaningful model needs multiple diverse
        # observations.
        #
        # Therefore we only verify the dataset here.

    finally:
        spark.stop()