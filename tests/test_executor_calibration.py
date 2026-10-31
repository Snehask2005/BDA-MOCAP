import csv

from pyspark.sql import SparkSession

from mocap.calibration.service import CalibrationService
from mocap.cost.analytical import PlanMetrics
from mocap.execution.executor import SparkExecutor
from mocap.interfaces import SelectedPlan


def test_executor_records_calibration_sample(tmp_path):
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPExecutorCalibrationTest")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.createDataFrame(
            [
                (1, "Alice"),
                (2, "Bob"),
                (3, "Charlie"),
            ],
            ["id", "name"],
        )

        df.createOrReplaceTempView("calibration_executor_test")

        plan = SelectedPlan(
            plan_id="calibration_plan",
            query_id="calibration_query",
            selected_strategy="default",
            expected_cost=0.5,
            expected_latency=1.0,
            selection_reason="calibration integration test",
            physical_plan_sql="SELECT * FROM calibration_executor_test",
            budget=10.0,
        )

        prediction = PlanMetrics(
            plan_id="calibration_plan",
            query_id="calibration_query",
            estimated_cost=0.5,
            estimated_latency=1.0,
            cpu_component=0.2,
            io_component=0.1,
            shuffle_component=0.05,
        )

        calibration = CalibrationService()

        dataset_path = str(
            tmp_path / "calibration_dataset.csv"
        )

        executor = SparkExecutor(
            spark=spark,
            calibration_service=calibration,
            calibration_dataset_path=dataset_path,
        )

        result = executor.execute(
            plan,
            prediction=prediction,
        )

        assert result.query_id == "calibration_query"
        assert result.plan_id == "calibration_plan"

        assert calibration.sample_count == 1

        sample = calibration.samples[0]

        assert sample.plan_id == "calibration_plan"
        assert sample.query_id == "calibration_query"
        assert sample.estimated_cost == 0.5
        assert sample.actual_cost == result.actual_cost

        with open(dataset_path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["plan_id"] == "calibration_plan"
        assert rows[0]["query_id"] == "calibration_query"

    finally:
        spark.stop()
