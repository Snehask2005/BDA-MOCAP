"""
Spark CBO (Cost-Based Optimizer) Statistics Integration & Verification.
Part of Phase 1 Deliverables for Hridhika (Learned Predictor Lead).

Responsibilities:
1. Enable Spark CBO flags (`spark.sql.cbo.enabled=true`, star-schema optimization).
2. Configure safe hint testing (`spark.sql.cbo.joinReorder.enabled=false`) to prevent
   Spark CBO from silently overriding injected join order hints.
3. Automate `ANALYZE TABLE` execution with column statistics.
4. Parse and validate `EXPLAIN COST` plans, verifying non-degenerate row/byte estimates.
5. Verify physical plan operators match the intended optimization hints under AQE.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CBOCostStats:
    """Parsed CBO statistical estimates for a plan node or relation."""

    node_name: str
    row_count: Optional[float] = None
    size_in_bytes: Optional[float] = None
    has_stats: bool = False
    is_valid: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def is_degenerate(self) -> bool:
        """True if statistics are missing, zero, or default single-value placeholders."""
        if not self.has_stats or self.row_count is None or self.size_in_bytes is None:
            return True
        return self.row_count <= 0 or self.size_in_bytes <= 0


def enable_cbo(spark_session: Any) -> Dict[str, str]:
    """
    Configure Spark Session with production CBO flags.
    Returns the applied configuration dictionary.
    """
    configs = {
        "spark.sql.cbo.enabled": "true",
        "spark.sql.cbo.joinReorder.enabled": "true",
        "spark.sql.cbo.starSchemaOptimization": "true",
        "spark.sql.statistics.histogram.enabled": "true",
    }
    for k, v in configs.items():
        try:
            spark_session.conf.set(k, v)
        except Exception:
            pass
    return configs


def enforce_hint_testing_mode(spark_session: Any) -> Dict[str, str]:
    """
    Disable automatic CBO join reordering during hint tests.
    Master Audit Checklist Requirement:
      'Never trust raw join hints; turn off spark.sql.cbo.joinReorder.enabled
       and verify executed physical plan against intended plan.'
    """
    configs = {
        "spark.sql.cbo.enabled": "true",
        "spark.sql.cbo.joinReorder.enabled": "false",
    }
    for k, v in configs.items():
        try:
            spark_session.conf.set(k, v)
        except Exception:
            pass
    return configs


def analyze_table(
    spark_session: Any,
    table_name: str,
    columns: Optional[List[str]] = None,
    compute_histograms: bool = True,
) -> str:
    """
    Run ANALYZE TABLE to collect table and column-level statistics.
    Returns the executed SQL command.
    """
    base_sql = f"ANALYZE TABLE {table_name} COMPUTE STATISTICS"
    if columns:
        cols_str = ", ".join(columns)
        col_sql = f"ANALYZE TABLE {table_name} COMPUTE STATISTICS FOR COLUMNS {cols_str}"
        spark_session.sql(col_sql)
        return col_sql

    if compute_histograms:
        col_sql = f"ANALYZE TABLE {table_name} COMPUTE STATISTICS FOR ALL COLUMNS"
        try:
            spark_session.sql(col_sql)
            return col_sql
        except Exception:
            # Fall back to base table statistics if column collection is restricted
            spark_session.sql(base_sql)
            return base_sql
    else:
        spark_session.sql(base_sql)
        return base_sql


# Regex patterns for Spark EXPLAIN COST output
# Examples:
#   Statistics(sizeInBytes=120.0 KiB, rowCount=1.00E+3)
#   Statistics(sizeInBytes=8000, rowCount=100)
#   sizeInBytes=1048576, rowCount=20000
#   sizeInBytes=64.0 B
_STATS_BLOCK_RE = re.compile(
    r"(?:Statistics\(|sizeInBytes\s*=\s*)(.*?)(?:\)|$)", re.IGNORECASE
)
_SIZE_RE = re.compile(
    r"sizeInBytes\s*=\s*([0-9.]+)\s*([KMGT]?i?B)?", re.IGNORECASE
)
_ROWS_RE = re.compile(
    r"rowCount\s*=\s*([0-9.E+-]+)", re.IGNORECASE
)

_MULTIPLIERS = {
    "B": 1.0,
    "KB": 1000.0,
    "KIB": 1024.0,
    "MB": 1000.0 ** 2,
    "MIB": 1024.0 ** 2,
    "GB": 1000.0 ** 3,
    "GIB": 1024.0 ** 3,
    "TB": 1000.0 ** 4,
    "TIB": 1024.0 ** 4,
}


def _parse_size_bytes(val_str: str, unit_str: Optional[str]) -> float:
    base = float(val_str)
    if not unit_str:
        return base
    unit = unit_str.upper()
    return base * _MULTIPLIERS.get(unit, 1.0)


def parse_explain_cost(explain_cost_text: str) -> List[CBOCostStats]:
    """
    Parse the text output of Spark EXPLAIN COST / df.explain("cost").
    Extracts node-level and table-level statistics including sizeInBytes and rowCount.
    """
    results: List[CBOCostStats] = []
    lines = explain_cost_text.splitlines()

    current_node = "Root"
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue

        # Extract node name (e.g. +- Filter, :- HashAggregate, Scan parquet ...)
        node_match = re.search(r"[:\+\-\s]*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)", cleaned)
        if node_match:
            potential_name = node_match.group(1).strip()
            if potential_name not in ("Statistics", "sizeInBytes", "rowCount"):
                current_node = potential_name

        # Look for Statistics(...)
        stats_match = _STATS_BLOCK_RE.search(cleaned)
        if stats_match:
            block = stats_match.group(1) or stats_match.group(0)
            row_count: Optional[float] = None
            size_in_bytes: Optional[float] = None

            size_m = _SIZE_RE.search(block)
            if size_m:
                try:
                    size_in_bytes = _parse_size_bytes(size_m.group(1), size_m.group(2))
                except Exception:
                    size_in_bytes = None

            rows_m = _ROWS_RE.search(block)
            if rows_m:
                try:
                    row_count = float(rows_m.group(1))
                except Exception:
                    row_count = None

            has_stats = (row_count is not None) or (size_in_bytes is not None)
            is_valid = has_stats and (
                (row_count is None or row_count > 0) and
                (size_in_bytes is None or size_in_bytes > 0)
            )

            results.append(
                CBOCostStats(
                    node_name=current_node,
                    row_count=row_count,
                    size_in_bytes=size_in_bytes,
                    has_stats=has_stats,
                    is_valid=is_valid,
                    details={"raw_block": block, "line": cleaned},
                )
            )

    return results


def verify_plan_matches_hint(
    physical_plan_text: str,
    expected_strategy: str,
) -> bool:
    """
    Verify that the physical plan generated by Spark actually contains the
    join operator intended by the join hint (or strategy specification).

    Supported strategy names:
    - 'broadcast': BroadcastHashJoinExec
    - 'merge': SortMergeJoinExec
    - 'shuffle_hash': ShuffledHashJoinExec
    - 'default': Any valid physical join
    """
    strat = expected_strategy.lower()
    plan_lower = physical_plan_text.lower()

    if strat in ("broadcast", "bhj"):
        return "broadcasthashjoin" in plan_lower or "broadcastnestedloopjoin" in plan_lower
    elif strat in ("merge", "smj", "sort_merge"):
        return "sortmergejoin" in plan_lower
    elif strat in ("shuffle_hash", "shj", "hash_join"):
        return "shuffledhashjoin" in plan_lower or "hashjoin" in plan_lower
    elif strat in ("default", "any"):
        return "join" in plan_lower
    return True


def verify_cbo_active(spark_session: Any, table_name: str) -> bool:
    """
    Verify that CBO is actively calculating real row and byte statistics
    for the given table, rather than relying on unanalyzed defaults.
    """
    try:
        df = spark_session.sql(f"SELECT * FROM {table_name}")
        explain_str = df._jdf.queryExecution().optimizedPlan().stats().toString()
        stats = parse_explain_cost(explain_str)
        return any(s.is_valid and not s.is_degenerate() for s in stats)
    except Exception:
        return False
