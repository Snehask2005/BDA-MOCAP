from pyspark.sql import SparkSession

from mocap.cost.statistics import (
    _MAX_REASONABLE_INPUT_BYTES,
    _valid_input_size,
    extract_plan_statistics,
)


def test_valid_input_size_accepts_reasonable_values():
    assert _valid_input_size(0) == 0.0
    assert _valid_input_size(1024) == 1024.0
    assert _valid_input_size(_MAX_REASONABLE_INPUT_BYTES) == _MAX_REASONABLE_INPUT_BYTES


def test_valid_input_size_rejects_invalid_values():
    assert _valid_input_size(-1) is None
    assert _valid_input_size(float("inf")) is None
    assert _valid_input_size(float("nan")) is None
    assert _valid_input_size(_MAX_REASONABLE_INPUT_BYTES * 10) is None


def test_extract_plan_statistics():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPStatisticsTest")
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

        stats = extract_plan_statistics(df)

        assert stats.input_bytes >= 0
        assert (
            stats.row_count is None
            or stats.row_count >= 0
        )

    finally:
        spark.stop()


def test_extract_plan_statistics_rejects_pathological_temp_view_stats():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAPPathologicalStatisticsTest")
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

        df.createOrReplaceTempView("stats_test_view")

        query_df = spark.sql(
            """
            SELECT id, name
            FROM stats_test_view
            WHERE id > 0
            """
        )

        stats = extract_plan_statistics(query_df)

        assert stats.input_bytes <= _MAX_REASONABLE_INPUT_BYTES

    finally:
        spark.stop()