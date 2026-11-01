"""
MOCAP Stage 4 -- Cluster Execution, Telemetry & Empirical Evaluation.

Owner: Anaswara (Lead: Cluster Execution & Tests)

Modules
-------
config      shared Spark environment (configs/spark_env.yaml)
telemetry   unified telemetry schema, logger and validator
run_plan    plan runner that executes SQL and logs telemetry
data        parquet registration, ANALYZE statistics, synthetic data
workload    experiment query workloads and families
baselines   Spark default, Greedy cheapest/fastest, Random, Oracle, MOCAP
stats       medians, bootstrap CIs, Wilcoxon tests, regret, violation rates
harness     measure candidate pools once; evaluate every strategy on them
cloud       cloud pricing profiles and bill reconciliation
"""
