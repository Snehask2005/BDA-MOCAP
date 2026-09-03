from typing import List

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest
from mocap.plans.representation import (
    CandidatePlan,
    create_plan_fingerprint,
)


def get_physical_plan(df) -> str:
    """
    Extract the physical execution plan as text.
    """

    return df._jdf.queryExecution().executedPlan().toString()


def generate_join_candidates(
    query: QueryRequest,
    spark: SparkSession,
) -> List[CandidatePlan]:
    """
    Generate alternative join-strategy candidates for a query.

    Initial strategies:
        - default
        - broadcast
        - merge
        - shuffle_hash

    Duplicate physical plans are removed automatically.
    """

    # Candidate SQL variants.
    candidates = [
        ("default", query.sql),

        (
            "broadcast",
            query.sql.replace(
                "SELECT",
                "SELECT /*+ BROADCAST(c) */",
                1,
            ),
        ),

        (
            "merge",
            query.sql.replace(
                "SELECT",
                "SELECT /*+ MERGE(c, o) */",
                1,
            ),
        ),

        (
            "shuffle_hash",
            query.sql.replace(
                "SELECT",
                "SELECT /*+ SHUFFLE_HASH(c, o) */",
                1,
            ),
        ),
    ]

    unique_plans = {}
    plan_counter = 1

    for strategy, sql in candidates:

        try:
            df = spark.sql(sql)

            physical_plan = get_physical_plan(df)

            fingerprint = create_plan_fingerprint(
                physical_plan
            )

            if fingerprint in unique_plans:
                continue

            plan_id = f"P{plan_counter:02d}"
            plan_counter += 1

            candidate = CandidatePlan(
                query_id=query.query_id,
                plan_id=plan_id,
                strategy=strategy,
                sql=sql,
                physical_plan=physical_plan,
                fingerprint=fingerprint,
            )

            unique_plans[fingerprint] = candidate

        except Exception as exc:
            print(
                f"Skipping strategy '{strategy}': {exc}"
            )

    return list(unique_plans.values())