from typing import List

from pyspark.sql import SparkSession

from mocap.query.parser import QueryRequest, extract_query_structure
from mocap.plans.representation import (
    CandidatePlan,
    create_plan_fingerprint,
    extract_physical_plan_features,
)


def get_physical_plan(df) -> str:
    """
    Extract the physical execution plan as text.
    """
    return df._jdf.queryExecution().executedPlan().toString()


def _build_hint_sql(sql: str, hint: str) -> str:
    """
    Insert a Spark SQL join hint immediately after SELECT.

    Example:
        SELECT * FROM customer c JOIN orders o ...

    becomes:
        SELECT /*+ BROADCAST(c) */ * FROM customer c JOIN orders o ...
    """

    select_position = sql.upper().find("SELECT")

    if select_position == -1:
        raise ValueError("SQL query must contain SELECT.")

    insert_position = select_position + len("SELECT")

    return (
        sql[:insert_position]
        + f" /*+ {hint} */"
        + sql[insert_position:]
    )


def _generate_sql_variants(
    query: QueryRequest,
) -> List[tuple[str, str]]:
    """
    Generate controlled SQL variants for candidate physical plans.

    Table aliases are extracted from the query itself instead
    of being hard-coded.
    """

    structure = extract_query_structure(query.sql)

    variants: List[tuple[str, str]] = [
        ("default", query.sql)
    ]

    # A query without joins does not need join-strategy variants.
    if not structure.joins:
        return variants

    aliases = [table.alias for table in structure.tables]

    if not aliases:
        return variants

    # Candidate 1: broadcast the first table.
    broadcast_alias = aliases[0]

    variants.append(
        (
            "broadcast",
            _build_hint_sql(
                query.sql,
                f"BROADCAST({broadcast_alias})",
            ),
        )
    )

    # Candidate 2 and 3: explicitly request
    # merge and shuffled-hash joins.
    if len(aliases) >= 2:
        left_alias = aliases[0]
        right_alias = aliases[1]

        variants.append(
            (
                "merge",
                _build_hint_sql(
                    query.sql,
                    f"MERGE({left_alias}, {right_alias})",
                ),
            )
        )

        variants.append(
            (
                "shuffle_hash",
                _build_hint_sql(
                    query.sql,
                    f"SHUFFLE_HASH({left_alias}, {right_alias})",
                ),
            )
        )

    return variants


def generate_join_candidates(
    query: QueryRequest,
    spark: SparkSession,
) -> List[CandidatePlan]:
    """
    Generate alternative physical-plan candidates for a query.

    Candidate strategies:
        - default
        - broadcast
        - merge
        - shuffle_hash

    Each SQL variant is submitted to Spark so that its actual
    physical execution plan can be inspected.

    Candidates producing identical physical plans are removed.
    """

    candidates = _generate_sql_variants(query)

    unique_plans = {}
    plan_counter = 1

    for strategy, sql in candidates:

        try:
            # Ask Spark to analyze the candidate SQL.
            df = spark.sql(sql)

            # Extract the actual physical plan selected by Spark.
            physical_plan = get_physical_plan(df)

            # Create an identifier for the physical plan.
            fingerprint = create_plan_fingerprint(
                physical_plan
            )

            # Skip duplicate physical plans.
            if fingerprint in unique_plans:
                continue

            # Extract structural characteristics of the plan.
            physical_features = extract_physical_plan_features(
                physical_plan
            )

            plan_id = f"{query.query_id}_P{plan_counter:02d}"
            plan_counter += 1

            candidate = CandidatePlan(
                query_id=query.query_id,
                plan_id=plan_id,
                strategy=strategy,
                sql=sql,
                physical_plan=physical_plan,
                fingerprint=fingerprint,
                actual_join_strategy=physical_features[
                    "actual_join_strategy"
                ],
                num_joins=physical_features[
                    "num_joins"
                ],
                num_exchanges=physical_features[
                    "num_exchanges"
                ],
                num_broadcast_exchanges=physical_features[
                    "num_broadcast_exchanges"
                ],
                num_sorts=physical_features[
                    "num_sorts"
                ],
            )

            unique_plans[fingerprint] = candidate

        except Exception as exc:
            print(
                f"Skipping strategy '{strategy}': {exc}"
            )

    return list(unique_plans.values())