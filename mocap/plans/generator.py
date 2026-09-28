from __future__ import annotations

import itertools
from typing import List

import sqlglot
from sqlglot import expressions as exp
from pyspark.sql import SparkSession

from mocap.query.parser import (
    QueryRequest,
    extract_query_structure,
)

from mocap.plans.representation import (
    CandidatePlan,
    create_plan_fingerprint,
    extract_physical_plan_features,
)


def get_physical_plan(df) -> str:
    """Extract the physical execution plan as text."""
    return (
        df._jdf
        .queryExecution()
        .executedPlan()
        .toString()
    )


def _build_hint_sql(
    sql: str,
    hint: str,
) -> str:
    """
    Insert a Spark SQL join hint immediately after SELECT.
    """

    select_position = sql.upper().find("SELECT")

    if select_position == -1:
        raise ValueError(
            "SQL query must contain SELECT."
        )

    insert_position = (
        select_position + len("SELECT")
    )

    return (
        sql[:insert_position]
        + f" /*+ {hint} */"
        + sql[insert_position:]
    )


def _generate_sql_variants(
    query: QueryRequest,
) -> List[tuple[str, str]]:
    """
    Generate controlled physical-strategy SQL variants.

    The default candidate preserves the user's original SQL. Join
    strategy hints are generated from aliases discovered from the query.
    """

    structure = extract_query_structure(
        query.sql
    )

    variants: List[tuple[str, str]] = [
        ("default", query.sql)
    ]

    if not structure.joins:
        return variants

    aliases = [
        table.alias
        for table in structure.tables
    ]

    if not aliases:
        return variants

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


def _table_alias(table: exp.Table) -> str:
    """Return the SQL-visible alias/name of a table expression."""
    return table.alias_or_name


def _is_inner_join(join: exp.Join) -> bool:
    """
    Return True only for INNER JOIN semantics.

    SQLGlot represents LEFT/RIGHT/FULL joins using the `side` argument,
    while ordinary INNER JOIN may have neither `side` nor `kind`.
    """

    side = join.args.get("side")
    kind = join.args.get("kind")

    if side is not None:
        return str(side).upper() == "INNER"

    if kind is not None:
        return str(kind).upper() == "INNER"

    return True


def _table_alias(table: exp.Table) -> str:
    """Return the SQL-visible alias/name of a table."""
    return table.alias_or_name


def _join_condition_aliases(
    condition: exp.Expression | None,
) -> set[str]:
    """Extract table aliases referenced by a join predicate."""

    if condition is None:
        return set()

    aliases: set[str] = set()

    for column in condition.find_all(exp.Column):
        if column.table:
            aliases.add(column.table)

    return aliases


def _find_connecting_condition(
    alias: str,
    joined_aliases: set[str],
    join_edges: list[
        tuple[
            exp.Expression | None,
            set[str],
            str,
        ]
    ],
) -> tuple[exp.Expression | None, str] | None:
    """
    Find a join predicate that connects `alias` to the current join tree.

    A predicate is usable when it references the table being added and at
    least one table that is already present in the join tree.
    """

    for condition, aliases, join_type in join_edges:
        if alias not in aliases:
            continue

        if aliases & joined_aliases:
            return condition, join_type

    return None


def _build_join_order_sql(
    query: QueryRequest,
    order: tuple[str, ...],
    tables_by_alias: dict[str, exp.Table],
    join_edges: list[
        tuple[
            exp.Expression | None,
            set[str],
            str,
        ]
    ],
) -> str:
    """
    Build one SQL query with a requested left-deep join order.

    A join predicate is attached to the point where its two referenced
    relations become connected.

    Only INNER JOIN semantics are supported here.
    """

    tree = sqlglot.parse_one(query.sql)

    if not order:
        raise ValueError("Join order cannot be empty.")

    base_alias = order[0]

    if base_alias not in tables_by_alias:
        raise ValueError(
            f"Unknown base table alias: {base_alias}"
        )

    tree.set(
        "from_",
        exp.From(
            this=tables_by_alias[base_alias].copy()
        ),
    )

    tree.set(
        "joins",
        [],
    )

    joined_aliases = {base_alias}
    used_edges: set[int] = set()
    new_joins: list[exp.Join] = []

    for alias in order[1:]:
        if alias not in tables_by_alias:
            raise ValueError(
                f"Unknown table alias: {alias}"
            )

        selected_edge = None

        for edge_index, (
            condition,
            aliases,
            join_type,
        ) in enumerate(join_edges):

            if edge_index in used_edges:
                continue

            if alias not in aliases:
                continue

            if not (aliases & joined_aliases):
                continue

            selected_edge = (
                edge_index,
                condition,
                join_type,
            )
            break

        if selected_edge is None:
            raise ValueError(
                f"No valid join predicate connects "
                f"'{alias}' to the current join tree."
            )

        edge_index, condition, join_type = selected_edge

        if join_type != "INNER":
            raise ValueError(
                "Join-order generation only supports INNER JOIN."
            )

        new_joins.append(
            exp.Join(
                this=tables_by_alias[alias].copy(),
                on=(
                    condition.copy()
                    if condition is not None
                    else None
                ),
                kind="INNER",
            )
        )

        used_edges.add(edge_index)
        joined_aliases.add(alias)

    tree.set(
        "joins",
        new_joins,
    )

    return tree.sql()


def _generate_join_order_variants(
    query: QueryRequest,
) -> List[tuple[str, str]]:
    """
    Generate valid left-deep join-order alternatives.

    Only queries whose joins are all INNER JOINs are reordered.

    For N tables, all permutations are considered up to five tables.
    Invalid permutations that would require a Cartesian product are
    discarded.
    """

    tree = sqlglot.parse_one(query.sql)

    from_clause = tree.args.get("from_")
    joins = tree.args.get("joins") or []

    if from_clause is None or not joins:
        return []

    # Never reorder a query containing an outer join.
    if not all(
        _is_inner_join(join)
        for join in joins
    ):
        return []

    base_table = from_clause.this

    if not isinstance(base_table, exp.Table):
        return []

    tables_by_alias = {
        _table_alias(base_table): base_table
    }

    join_edges: list[
        tuple[
            exp.Expression | None,
            set[str],
            str,
        ]
    ] = []

    for join in joins:
        table = join.this

        if not isinstance(table, exp.Table):
            return []

        alias = _table_alias(table)

        tables_by_alias[alias] = table

        condition = join.args.get("on")

        join_type = (
            str(
                join.args.get("side")
                or join.args.get("kind")
                or "INNER"
            )
            .upper()
        )

        condition_aliases = _join_condition_aliases(
            condition
        )

        # A join predicate should connect at least two relations.
        if condition is not None and len(condition_aliases) < 2:
            return []

        join_edges.append(
            (
                condition,
                condition_aliases,
                join_type,
            )
        )

    aliases = list(tables_by_alias)

    if len(aliases) < 3:
        return []

    # Prevent factorial explosion during the prototype phase.
    if len(aliases) > 5:
        return []

    original_order = tuple(aliases)
    variants: List[tuple[str, str]] = []

    for permutation in itertools.permutations(aliases):

        if permutation == original_order:
            continue

        try:
            sql = _build_join_order_sql(
                query=query,
                order=permutation,
                tables_by_alias=tables_by_alias,
                join_edges=join_edges,
            )
        except ValueError:
            continue

        label = (
            "join_order_"
            + "_".join(permutation)
        )

        variants.append(
            (label, sql)
        )

    return variants


def _generate_all_sql_variants(
    query: QueryRequest,
) -> List[tuple[str, str]]:
    """
    Generate the complete SQL candidate space.

    Candidate dimensions:
        1. Original/default physical plan.
        2. Physical join-strategy alternatives.
        3. Valid join-order alternatives.
        4. Physical join strategies applied to each join order.

    Catalyst is responsible for producing the final physical plan.
    Duplicate physical plans are removed later using fingerprints.
    """

    variants: List[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Original query + physical strategy variants
    # ------------------------------------------------------------------

    variants.extend(
        _generate_sql_variants(query)
    )

    # ------------------------------------------------------------------
    # Join-order variants
    # ------------------------------------------------------------------

    join_orders = _generate_join_order_variants(query)

    for order_label, order_sql in join_orders:

        # Keep the pure join-order candidate.
        variants.append(
            (
                order_label,
                order_sql,
            )
        )

        # Apply each physical join strategy to this join order.
        for strategy, hint in (
            ("broadcast", "BROADCAST"),
            ("merge", "MERGE"),
            ("shuffle_hash", "SHUFFLE_HASH"),
        ):

            # Extract all aliases from this particular SQL query.
            structure = extract_query_structure(
                order_sql
            )

            aliases = [
                table.alias
                for table in structure.tables
            ]

            if not aliases:
                continue

            hinted_sql = _build_hint_sql(
                order_sql,
                f"{hint}({', '.join(aliases)})",
            )

            variants.append(
                (
                    f"{order_label}_{strategy}",
                    hinted_sql,
                )
            )

    return variants


def generate_join_candidates(
    query: QueryRequest,
    spark: SparkSession,
) -> List[CandidatePlan]:
    """
    Generate alternative physical-plan candidates.

    Candidate generation is delegated to Spark/Catalyst: MOCAP supplies
    controlled SQL alternatives and extracts the resulting physical plans.

    No candidate query is executed here.
    """

    sql_variants = _generate_all_sql_variants(
        query
    )

    unique_plans = {}
    plan_counter = 1

    for strategy, sql in sql_variants:

        try:
            df = spark.sql(sql)

            physical_plan = get_physical_plan(
                df
            )

            fingerprint = create_plan_fingerprint(
                physical_plan
            )

            if fingerprint in unique_plans:
                continue

            features = extract_physical_plan_features(
                physical_plan
            )

            plan_id = (
                f"{query.query_id}_P"
                f"{plan_counter:02d}"
            )

            plan_counter += 1

            candidate = CandidatePlan(
                query_id=query.query_id,
                plan_id=plan_id,
                strategy=strategy,
                sql=sql,
                physical_plan=physical_plan,
                fingerprint=fingerprint,

                actual_join_strategy=(
                    features.actual_join_strategy
                ),

                num_joins=features.num_joins,
                num_broadcast_joins=(
                    features.num_broadcast_joins
                ),
                num_shuffle_hash_joins=(
                    features.num_shuffle_hash_joins
                ),
                num_sort_merge_joins=(
                    features.num_sort_merge_joins
                ),
                num_nested_loop_joins=(
                    features.num_nested_loop_joins
                ),

                num_exchanges=(
                    features.num_exchanges
                ),
                num_broadcast_exchanges=(
                    features.num_broadcast_exchanges
                ),
                num_sorts=features.num_sorts,
                num_aggregates=(
                    features.num_aggregates
                ),
                num_filters=features.num_filters,
                num_scans=features.num_scans,
                plan_depth=features.plan_depth,
            )

            unique_plans[fingerprint] = candidate

        except Exception as exc:
            print(
                f"Skipping strategy "
                f"'{strategy}': {exc}"
            )

    return list(
        unique_plans.values()
    )
