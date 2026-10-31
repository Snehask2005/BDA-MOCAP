"""
Live Spark smoke test: does overrun really cancel the job and degrade?

PASS looks like:
  adaptive_action = degraded
  log: S1 adaptive cancelled None   (cancelled attempt, no cost recorded)
       S1_degraded degraded ok <cost>
  cancelled_partial_cost > 0
"""

import json
import os

from pyspark.sql import SparkSession

from mocap.execution.adaptive import AdaptiveConfig, AdaptiveController
from mocap.execution.executor import SparkExecutor
from mocap.interfaces import SelectedPlan

LOG = "results/smoke_log.jsonl"

if os.path.exists(LOG):
    os.remove(LOG)

spark = (
    SparkSession.builder.master("local[2]")
    .appName("MOCAPSmoke")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    ex = SparkExecutor(spark, log_path=LOG, num_cores=2, allow_simulation=False)

    plan = SelectedPlan(
        plan_id="S1",
        query_id="SMOKE",
        selected_strategy="x",
        expected_cost=1e-4,
        expected_latency=1.0,
        selection_reason="smoke",
        physical_plan_sql=(
            "SELECT sum(id) FROM "
            "(SELECT /*+ REPARTITION(8) */ id FROM range(0, 400000000, 1, 200)) t"
        ),
        budget=1e-9,  # impossibly small -> must trigger overrun
    )

    ctl = AdaptiveController(
        ex,
        spark=spark,
        config=AdaptiveConfig(
            poll_interval_s=0.2,
            min_elapsed_s=0.5,
            min_progress=0.0,
            sample_fraction=0.01,
        ),
    )

    res = ctl.run(plan)

    print("\nRESULT plan:", res.plan_id, "mode:", res.execution_mode)
    print("RUNTIME METRICS:", res.runtime_metrics)
    print("\nEXECUTION LOG:")
    for line in open(LOG):
        r = json.loads(line)
        print(" ", r["plan_id"], r["mode"], r["status"], r["actual_cost"])
finally:
    spark.stop()
