from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TableReference:
    """Represents one table and its SQL alias."""

    name: str
    alias: str


@dataclass(frozen=True)
class JoinReference:
    """Represents one SQL join."""

    table: TableReference
    condition: Optional[str]
    join_type: str


@dataclass(frozen=True)
class QueryStructure:
    """Structural information extracted from an SQL query."""

    tables: list[TableReference]
    joins: list[JoinReference]