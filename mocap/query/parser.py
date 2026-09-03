from dataclasses import dataclass
from typing import Optional

import sqlglot
from sqlglot import expressions as exp

from mocap.query.structure import (
    JoinReference,
    QueryStructure,
    TableReference,
)


@dataclass
class QueryRequest:
    """
    Represents a user query together with its optimization constraints.
    """

    query_id: str
    sql: str
    budget: float
    deadline: Optional[float] = None
    accuracy_tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id cannot be empty.")

        if not self.sql.strip():
            raise ValueError("SQL query cannot be empty.")

        if self.budget <= 0:
            raise ValueError("Budget must be greater than zero.")

        if self.deadline is not None and self.deadline <= 0:
            raise ValueError("Deadline must be greater than zero.")

        if (
            self.accuracy_tolerance is not None
            and not 0 <= self.accuracy_tolerance <= 100
        ):
            raise ValueError(
                "Accuracy tolerance must be between 0 and 100."
            )


def extract_query_structure(sql: str) -> QueryStructure:
    """
    Extract table and join structure from an SQL query.

    This information is later used by the MOCAP
    candidate-plan generator.
    """

    if not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    try:
        tree = sqlglot.parse_one(sql)
    except Exception as exc:
        raise ValueError(
            f"Unable to parse SQL query: {exc}"
        ) from exc

    tables = []

    for table in tree.find_all(exp.Table):
        tables.append(
            TableReference(
                name=table.name,
                alias=table.alias_or_name,
            )
        )

    joins = []

    for join in tree.find_all(exp.Join):
        joined_table = join.this

        if not isinstance(joined_table, exp.Table):
            continue

        condition = join.args.get("on")

        joins.append(
            JoinReference(
                table=TableReference(
                    name=joined_table.name,
                    alias=joined_table.alias_or_name,
                ),
                condition=(
                    condition.sql()
                    if condition
                    else None
                ),
                join_type=(
                    join.args.get("kind") or "INNER"
                ).upper(),
            )
        )

    return QueryStructure(
        tables=tables,
        joins=joins,
    )