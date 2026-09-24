"""
Real end-to-end MOCAP demonstration.

Creates a deterministic local Spark dataset, generates
alternative physical plans, evaluates them, applies the
budget constraint, and selects a plan using MOCAP.
"""

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.pipeline import run_mocap


def main():

    spark = (
        SparkSession.builder
        .appName("MOCAP-Demo")
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # ---------------------------------------------------------
    # 1. Create deterministic benchmark data
    # ---------------------------------------------------------

    customers = [
        (i, f"customer_{i}")
        for i in range(1, 1001)
    ]

    orders = [
        (
            i,
            ((i - 1) % 1000) + 1,
            float((i * 17) % 500) + 10.0,
        )
        for i in range(1, 10001)
    ]

    customer_df = spark.createDataFrame(
        customers,
        ["id", "name"],
    )

    orders_df = spark.createDataFrame(
        orders,
        ["id", "customer_id", "amount"],
    )

    customer_df.createOrReplaceTempView("customer")
    orders_df.createOrReplaceTempView("orders")

    # ---------------------------------------------------------
    # 2. Define query + user budget
    # ---------------------------------------------------------

    sql = """
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
    """

    query = QueryRequest(
        query_id="MOCAP_DEMO_01",
        sql=sql,
        budget=0.0000125,
    )

    print()
    print("=" * 70)
    print("MOCAP - Multi-Objective Cost-Aware Planner")
    print("=" * 70)

    print(f"Query ID : {query.query_id}")
    print(f"Budget   : ${query.budget:.6f}")

    # ---------------------------------------------------------
    # 3. Run MOCAP
    # ---------------------------------------------------------

    result = run_mocap(
        query=query,
        spark=spark,
        num_cores=1,
    )

    # ---------------------------------------------------------
    # 4. Candidate plans
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("CANDIDATE PLANS")
    print("-" * 70)

    for plan in result.candidates:
        print(
            f"{plan.plan_id:20s} "
            f"{plan.strategy:15s} "
            f"{str(plan.actual_join_strategy):25s}"
        )

    # ---------------------------------------------------------
    # 5. Estimated / measured metrics
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("PLAN METRICS")
    print("-" * 70)

    for plan in result.candidates:
        m = result.metrics[plan.plan_id]

        print(
            f"{plan.plan_id:20s} "
            f"cost=${m.estimated_cost:.8f}  "
            f"latency={m.estimated_latency:.4f}s  "
            f"shuffle={m.bytes_shuffled:.0f}B"
        )

    # ---------------------------------------------------------
    # 6. Feasible plans
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("FEASIBLE PLANS")
    print("-" * 70)

    if result.feasible:
        for plan, metrics in result.feasible:
            print(
                f"{plan.plan_id:20s} "
                f"cost=${metrics.estimated_cost:.8f}  "
                f"latency={metrics.estimated_latency:.4f}s"
            )
    else:
        print("No feasible plan.")

    # ---------------------------------------------------------
    # 7. Pareto frontier
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("PARETO FRONTIER")
    print("-" * 70)

    for plan, metrics in result.pareto_front:
        print(
            f"{plan.plan_id:20s} "
            f"{plan.actual_join_strategy}  "
            f"cost=${metrics.estimated_cost:.8f}  "
            f"latency={metrics.estimated_latency:.4f}s"
        )

    # ---------------------------------------------------------
    # 8. Final MOCAP selection
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("MOCAP SELECTION")
    print("-" * 70)

    if result.selected_plan is None:
        print("MOCAP could not find a feasible plan.")
        print(f"Reason: {result.failure_reason}")
    else:
        selected = result.selected_plan

        print(f"Selected plan : {selected.plan_id}")
        print(f"Strategy      : {selected.selected_strategy}")
        print(f"Expected cost : ${selected.expected_cost:.8f}")
        print(f"Expected time : {selected.expected_latency:.4f}s")
        print(f"Score         : {result.selected_score:.8f}")
        print(f"Reason        : {selected.selection_reason}")

    print()
    print("=" * 70)

    spark.stop()


if __name__ == "__main__":
    main()
