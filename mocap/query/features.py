from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp


@dataclass
class QueryFeatures:
    query_id: str
    num_tables: int
    num_joins: int
    num_filters: int
    has_group_by: bool
    has_order_by: bool
    has_aggregation: bool


def extract_query_features(
    query_id: str,
    sql: str,
) -> QueryFeatures:
    if not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    try:
        tree = sqlglot.parse_one(sql)
    except Exception as exc:
        raise ValueError(f"Unable to parse SQL query: {exc}") from exc

    tables = list(tree.find_all(exp.Table))
    joins = list(tree.find_all(exp.Join))
    filters = list(tree.find_all(exp.Where))
    aggregations = list(tree.find_all(exp.AggFunc))

    return QueryFeatures(
        query_id=query_id,
        num_tables=len(tables),
        num_joins=len(joins),
        num_filters=len(filters),
        has_group_by=tree.args.get("group") is not None,
        has_order_by=tree.args.get("order") is not None,
        has_aggregation=len(aggregations) > 0,
    )