from pyspark.sql import SparkSession

from mocap.pipeline import MOCAPPipeline
from mocap.query.parser import QueryRequest


def test_mocap_end_to_end_spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPTest")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    customers = [(i, f"customer_{i}") for i in range(100)]
    orders = [(i, i % 100, float(i % 50 + 1)) for i in range(1000)]

    spark.createDataFrame(
        customers, ["id", "name"]
    ).createOrReplaceTempView("customer")

    spark.createDataFrame(
        orders, ["id", "customer_id", "amount"]
    ).createOrReplaceTempView("orders")

    query = QueryRequest(
        query_id="TEST_01",
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
            LIMIT 20
        """,
        budget=1.0,
    )

    result = MOCAPPipeline(spark).run(query)

    assert len(result.candidates) >= 2
    assert len(result.metrics) == len(result.candidates)
    assert result.selected_plan is not None
    assert result.selected_plan.plan_id in result.metrics
    assert result.selected_plan.expected_cost <= query.budget

    spark.stop()


def test_pipeline_uses_calibration_model():
    from mocap.cost.learned import CalibrationModel

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPCalibrationTest")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    customers = [(i, f"customer_{i}") for i in range(20)]
    orders = [(i, i % 20, float(i + 1)) for i in range(100)]

    spark.createDataFrame(
        customers, ["id", "name"]
    ).createOrReplaceTempView("customer")

    spark.createDataFrame(
        orders, ["id", "customer_id", "amount"]
    ).createOrReplaceTempView("orders")

    query = QueryRequest(
        query_id="CALIBRATION_01",
        sql="""
            SELECT c.id, COUNT(o.id) AS order_count
            FROM customer c
            JOIN orders o
                ON c.id = o.customer_id
            GROUP BY c.id
        """,
        budget=1.0,
    )

    baseline = MOCAPPipeline(spark).run(query)

    calibration_model = CalibrationModel(
        weights=[1.0, 2.0, 1.0, 1.0],
    )

    calibrated = MOCAPPipeline(
        spark,
        calibration_model=calibration_model,
    ).run(query)

    assert baseline.metrics
    assert calibrated.metrics

    for plan_id in baseline.metrics:
        assert plan_id in calibrated.metrics

        assert (
            calibrated.metrics[plan_id].estimated_cost
            != baseline.metrics[plan_id].estimated_cost
        )

    spark.stop()
