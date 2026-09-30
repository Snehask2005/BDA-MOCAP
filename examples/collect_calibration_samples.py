"""
Collect independent calibration samples across query sizes and strategies.

Seeds synthetic customer/orders tables at three scale factors (so
Catalyst's input-size statistics genuinely differ, not just the SQL
text), generates join-strategy candidates for each, predicts + executes
+ records each one, and appends everything to a calibration dataset CSV.

Run from the repo root (after `pip install -e .`):
    python examples/collect_calibration_samples.py
"""
from __future__ import annotations

import random

from pyspark.sql import SparkSession

from mocap.calibration.service import CalibrationService
from mocap.cost.validation import validate_candidate
from mocap.plans.generator import generate_join_candidates
from mocap.query.parser import QueryRequest

SQL_TEMPLATE = """
SELECT
    c.c_mktsegment,
    COUNT(*) AS order_count
FROM customer c
JOIN orders o
    ON c.c_custkey = o.o_custkey
WHERE o.o_totalprice > 1000
GROUP BY c.c_mktsegment
"""

# (label, num_customers, num_orders) -- three distinct workload sizes.
SCALE_FACTORS = [
    ("small", 50, 200),
    ("medium", 500, 2000),
    ("large", 5000, 20000),
]

SEGMENTS = ["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"]


def _seed_tables(spark: SparkSession, num_customers: int, num_orders: int, seed: int = 0) -> None:
    rng = random.Random(seed)

    customers = [
        (cust_id, rng.choice(SEGMENTS))
        for cust_id in range(1, num_customers + 1)
    ]
    spark.createDataFrame(customers, ["c_custkey", "c_mktsegment"]).createOrReplaceTempView("customer")

    orders = [
        (order_id, rng.randint(1, num_customers), round(rng.uniform(50.0, 5000.0), 2))
        for order_id in range(1, num_orders + 1)
    ]
    spark.createDataFrame(orders, ["o_orderkey", "o_custkey", "o_totalprice"]).createOrReplaceTempView("orders")


def main() -> None:
    spark = SparkSession.builder.appName("mocap-calibration-collection").master("local[2]").getOrCreate()
    service = CalibrationService()
    csv_path = "results/calibration_dataset.csv"

    for label, num_customers, num_orders in SCALE_FACTORS:
        print(f"\n=== scale: {label} ({num_customers} customers, {num_orders} orders) ===")
        _seed_tables(spark, num_customers, num_orders)

        query = QueryRequest(query_id=f"Q_{label}", sql=SQL_TEMPLATE)
        candidates = generate_join_candidates(query, spark)

        for candidate in candidates:
            result = validate_candidate(candidate, spark)

            service.record_to_dataset(
                prediction=result.prediction,
                telemetry=result.telemetry,
                csv_path=csv_path,
            )

            print(
                f"  strategy={result.strategy:20s} "
                f"predicted_cost={result.predicted_cost:.6f} "
                f"actual_cost={result.actual_cost:.6f} "
                f"cost_abs_err={result.cost_absolute_error:.6f}"
            )

    print(f"\nCollected {service.sample_count} samples this run -> appended to {csv_path}")
    spark.stop()


if __name__ == "__main__":
    main()
