from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import itertools
import math
from collections.abc import Mapping
from statistics import median
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
from mocap.plans.beam_search import beam_search_join_orders
from mocap.plans.search_statistics import (
    JoinSearchStatistics,
    build_cardinality_scorer,
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



def _estimate_table_size_bytes(
    query: QueryRequest,
    spark: SparkSession,
) -> dict[str, float]:
    """Read available Spark size estimates for the query's base tables."""
    tree = sqlglot.parse_one(query.sql)
    from_clause = tree.args.get("from_")

    if from_clause is None:
        return {}

    relations: list[exp.Table] = []
    if isinstance(from_clause.this, exp.Table):
        relations.append(from_clause.this)

    for join in tree.args.get("joins") or []:
        if isinstance(join.this, exp.Table):
            relations.append(join.this)

    estimates: dict[str, float] = {}

    for table in relations:
        parts = [
            str(part)
            for part in (table.catalog, table.db, table.name)
            if part
        ]
        if not parts:
            continue

        try:
            dataframe = spark.table(".".join(parts))
            logical_plan = (
                dataframe._jdf
                .queryExecution()
                .optimizedPlan()
            )
            size_bytes = float(
                logical_plan.stats().sizeInBytes()
            )
        except Exception:
            # Missing or unresolved statistics are handled by the scorer.
            continue

        if (
            math.isfinite(size_bytes)
            and 0.0 <= size_bytes <= 1e18
        ):
            estimates[_table_alias(table)] = size_bytes

    return estimates


def _size_only_prefix_scorer(
    aliases: tuple[str, ...],
    table_size_estimates: Mapping[str, float] | None,
):
    """Build a cheap size-based score for partial join orders.

    This preliminary heuristic is not a join-cardinality estimator
    and does not predict execution latency.
    """
    valid: dict[str, float] = {}

    for alias in aliases:
        try:
            size = float((table_size_estimates or {}).get(alias, math.nan))
        except (TypeError, ValueError, OverflowError):
            continue

        if math.isfinite(size) and 0.0 <= size <= 1e18:
            valid[alias] = size

    fallback = median(list(valid.values())) if valid else 1.0
    sizes = {
        alias: valid.get(alias, fallback)
        for alias in aliases
    }
    scale = max(max(sizes.values(), default=1.0), 1.0)
    normalized = {
        alias: sizes[alias] / scale
        for alias in aliases
    }

    def score_prefix(order: tuple[str, ...]) -> float:
        cumulative_size = 0.0
        score = 0.0

        for alias in order:
            cumulative_size += normalized[alias]
            score += math.log1p(cumulative_size)

        return score

    return score_prefix


def _generate_join_order_variants(
    query: QueryRequest,
    *,
    search_mode: str = "exhaustive",
    beam_width: int = 20,
    table_size_estimates: Mapping[str, float] | None = None,
    join_search_statistics: JoinSearchStatistics | None = None,
) -> List[tuple[str, str]]:
    """Generate valid left-deep join-order alternatives.

    Exhaustive mode preserves the existing five-table limit and serves
    as the reference baseline. Beam mode uses a bounded search and can
    explore connected INNER JOIN queries with more than five tables.
    """
    if search_mode not in {"exhaustive", "beam"}:
        raise ValueError(
            "search_mode must be either 'exhaustive' or 'beam'"
        )

    if (
        not isinstance(beam_width, int)
        or isinstance(beam_width, bool)
        or beam_width < 1
    ):
        raise ValueError("beam_width must be a positive integer")

    tree = sqlglot.parse_one(query.sql)
    from_clause = tree.args.get("from_")
    joins = tree.args.get("joins") or []

    if from_clause is None or not joins:
        return []

    if not all(_is_inner_join(join) for join in joins):
        return []

    base_table = from_clause.this
    if not isinstance(base_table, exp.Table):
        return []

    tables_by_alias = {_table_alias(base_table): base_table}
    join_edges: list[
        tuple[exp.Expression | None, set[str], str]
    ] = []

    for join in joins:
        table = join.this
        if not isinstance(table, exp.Table):
            return []

        alias = _table_alias(table)
        tables_by_alias[alias] = table

        condition = join.args.get("on")
        join_type = str(
            join.args.get("side")
            or join.args.get("kind")
            or "INNER"
        ).upper()

        condition_aliases = _join_condition_aliases(condition)

        if condition is not None and len(condition_aliases) != 2:
            return []

        join_edges.append(
            (condition, condition_aliases, join_type)
        )

    aliases = tuple(tables_by_alias)
    if len(aliases) < 3:
        return []

    original_order = aliases

    if search_mode == "exhaustive":
        # Preserve the previous reference implementation.
        if len(aliases) > 5:
            return []
        candidate_orders = itertools.permutations(aliases)
    else:
        alias_set = set(aliases)
        adjacency: dict[str, set[str]] = {
            alias: set() for alias in aliases
        }

        for _condition, condition_aliases, _join_type in join_edges:
            connected = condition_aliases & alias_set
            for alias in connected:
                adjacency[alias].update(connected - {alias})

        score_prefix = build_cardinality_scorer(
            aliases,
            adjacency,
            join_search_statistics,
        )

        if score_prefix is None:
            score_prefix = _size_only_prefix_scorer(
                aliases,
                table_size_estimates,
            )

        # Request one extra order because the original order may be
        # included in the beam and is removed from alternative variants.
        candidate_orders = beam_search_join_orders(
            aliases,
            adjacency,
            score_prefix=score_prefix,
            beam_width=beam_width + 1,
        )

    variants: List[tuple[str, str]] = []
    seen_orders: set[tuple[str, ...]] = set()

    for candidate_order in candidate_orders:
        order = tuple(candidate_order)

        if order == original_order or order in seen_orders:
            continue
        seen_orders.add(order)

        try:
            sql = _build_join_order_sql(
                query=query,
                order=order,
                tables_by_alias=tables_by_alias,
                join_edges=join_edges,
            )
        except ValueError:
            continue

        label = "join_order_" + "_".join(order)
        variants.append((label, sql))

        if search_mode == "beam" and len(variants) >= beam_width:
            break

    return variants



def _generate_all_sql_variants(
    query: QueryRequest,
    *,
    search_mode: str = "exhaustive",
    beam_width: int = 20,
    table_size_estimates: Mapping[str, float] | None = None,
    join_search_statistics: JoinSearchStatistics | None = None,
) -> List[tuple[str, str]]:
    """Generate base strategy variants and join-order strategy variants."""
    if search_mode not in {"exhaustive", "beam"}:
        raise ValueError(
            "search_mode must be either 'exhaustive' or 'beam'"
        )

    variants: List[tuple[str, str]] = []
    variants.extend(_generate_sql_variants(query))

    join_orders = _generate_join_order_variants(
        query,
        search_mode=search_mode,
        beam_width=beam_width,
        table_size_estimates=table_size_estimates,
        join_search_statistics=join_search_statistics,
    )

    for order_label, order_sql in join_orders:
        variants.append((order_label, order_sql))

        structure = extract_query_structure(order_sql)
        aliases = [table.alias for table in structure.tables]

        if not aliases:
            continue

        for strategy, hint in (
            ("broadcast", "BROADCAST"),
            ("merge", "MERGE"),
            ("shuffle_hash", "SHUFFLE_HASH"),
        ):
            hinted_sql = _build_hint_sql(
                order_sql,
                f"{hint}({', '.join(aliases)})",
            )
            variants.append(
                (f"{order_label}_{strategy}", hinted_sql)
            )

    return variants


def generate_join_candidates(
    query: QueryRequest,
    spark: SparkSession,
    *,
    search_mode: str = "exhaustive",
    beam_width: int = 20,
    join_search_statistics: JoinSearchStatistics | None = None,
    candidate_workers: int = 1,
) -> List[CandidatePlan]:
    """
    Generate alternative physical-plan candidates.

    Candidate generation is delegated to Spark/Catalyst: MOCAP supplies
    controlled SQL alternatives and extracts the resulting physical plans.

    No candidate query is executed here. ``candidate_workers`` controls
    bounded concurrent physical-plan analysis; the default of 1 preserves
    sequential analysis. Deduplication and candidate IDs remain ordered.
    """

    if search_mode not in {"exhaustive", "beam"}:
        raise ValueError(
            "search_mode must be either 'exhaustive' or 'beam'"
        )

    if (
        not isinstance(beam_width, int)
        or isinstance(beam_width, bool)
        or beam_width < 1
    ):
        raise ValueError("beam_width must be a positive integer")

    if (
        not isinstance(candidate_workers, int)
        or isinstance(candidate_workers, bool)
        or candidate_workers < 1
    ):
        raise ValueError("candidate_workers must be a positive integer")

    table_size_estimates = (
        _estimate_table_size_bytes(query, spark)
        if search_mode == "beam"
        else None
    )

    sql_variants = _generate_all_sql_variants(
        query,
        search_mode=search_mode,
        beam_width=beam_width,
        table_size_estimates=table_size_estimates,
        join_search_statistics=join_search_statistics,
    )

    def analyze_variant(item):
        strategy, sql = item
        try:
            dataframe = spark.sql(sql)
            physical_plan = get_physical_plan(dataframe)
            fingerprint = create_plan_fingerprint(physical_plan)
            return strategy, sql, physical_plan, fingerprint, None
        except Exception as exc:
            return strategy, sql, None, None, exc

    if candidate_workers == 1:
        analyses = [
            analyze_variant(item)
            for item in sql_variants
        ]
    else:
        with ThreadPoolExecutor(max_workers=candidate_workers) as executor:
            # executor.map preserves input order even when tasks finish
            # in a different order.
            analyses = list(executor.map(analyze_variant, sql_variants))

    unique_plans = {}
    plan_counter = 1

    for strategy, sql, physical_plan, fingerprint, analysis_error in analyses:
        if analysis_error is not None:
            print(
                f"Skipping strategy '{strategy}': {analysis_error}"
            )
            continue

        if fingerprint in unique_plans:
            continue

        try:
            features = extract_physical_plan_features(physical_plan)

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
                actual_join_strategy=features.actual_join_strategy,
                num_joins=features.num_joins,
                num_broadcast_joins=features.num_broadcast_joins,
                num_shuffle_hash_joins=features.num_shuffle_hash_joins,
                num_sort_merge_joins=features.num_sort_merge_joins,
                num_nested_loop_joins=features.num_nested_loop_joins,
                num_exchanges=features.num_exchanges,
                num_broadcast_exchanges=features.num_broadcast_exchanges,
                num_sorts=features.num_sorts,
                num_aggregates=features.num_aggregates,
                num_filters=features.num_filters,
                num_scans=features.num_scans,
                plan_depth=features.plan_depth,
            )

            unique_plans[fingerprint] = candidate

        except Exception as exc:
            print(
                f"Skipping strategy '{strategy}': {exc}"
            )

    return list(unique_plans.values())
