"""
Catalyst Plan Feature Extraction Module.
Part of WP2 and Phase 1 Deliverables for Hridhika (Learned Predictor Lead).

Extracts fine-grained execution-plan features from Apache Spark Catalyst plans:
- Scan bytes and input volume
- Shuffle bytes across exchange nodes
- Peak / maximum intermediate relation sizes
- Operator counts: Exchange, BroadcastExchange, Sort, Aggregate, Filter, Project
- Join type decomposition: BroadcastHashJoin, SortMergeJoin, ShuffleHashJoin, BroadcastNestedLoopJoin
- Physical operator tree depth and node count
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from predictor.cbo_stats import parse_explain_cost


@dataclass
class PlanFeatures:
    """Vectorized representation of a Spark Catalyst physical plan."""

    query_id: str
    plan_id: str

    # Volume and I/O features (bytes)
    scan_bytes: float = 0.0
    shuffle_bytes: float = 0.0
    max_intermediate_size: float = 0.0
    broadcast_join_sizes: float = 0.0

    # Exchange & Shuffle operators
    exchange_count: int = 0
    broadcast_exchange_count: int = 0

    # Join operator types
    bhj_count: int = 0      # BroadcastHashJoin
    smj_count: int = 0      # SortMergeJoin
    shj_count: int = 0      # ShuffleHashJoin / HashJoin
    bnlj_count: int = 0     # BroadcastNestedLoopJoin / Cartesian

    # Relational operators
    filter_count: int = 0
    project_count: int = 0
    aggregate_count: int = 0
    sort_count: int = 0

    # Structural tree complexity
    tree_depth: int = 1
    num_nodes: int = 1

    # CBO statistical priors
    estimated_rows: float = 0.0

    # Strategy tag
    strategy: str = "default"

    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "scan_bytes",
            "shuffle_bytes",
            "max_intermediate_size",
            "broadcast_join_sizes",
            "exchange_count",
            "broadcast_exchange_count",
            "bhj_count",
            "smj_count",
            "shj_count",
            "bnlj_count",
            "filter_count",
            "project_count",
            "aggregate_count",
            "sort_count",
            "tree_depth",
            "num_nodes",
            "estimated_rows",
        ]

    def to_vector(self) -> np.ndarray:
        """Return 1D NumPy feature vector for model training/inference."""
        return np.array(
            [
                float(self.scan_bytes),
                float(self.shuffle_bytes),
                float(self.max_intermediate_size),
                float(self.broadcast_join_sizes),
                float(self.exchange_count),
                float(self.broadcast_exchange_count),
                float(self.bhj_count),
                float(self.smj_count),
                float(self.shj_count),
                float(self.bnlj_count),
                float(self.filter_count),
                float(self.project_count),
                float(self.aggregate_count),
                float(self.sort_count),
                float(self.tree_depth),
                float(self.num_nodes),
                float(self.estimated_rows),
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.feature_names()}
        d["query_id"] = self.query_id
        d["plan_id"] = self.plan_id
        d["strategy"] = self.strategy
        return d


# Substrings for metric mapping
_SCAN_METRIC_KEYS = ("size of files read", "bytes read", "scan bytes")
_SHUFFLE_METRIC_KEYS = ("shuffle bytes written", "remote bytes read", "local bytes read", "shuffle bytes")


def _compute_tree_structure(lines: List[str]) -> Tuple[int, int]:
    """Calculate operator tree depth and total node count from indented plan text."""
    max_depth = 1
    node_count = 0

    for line in lines:
        cleaned = line.rstrip()
        if not cleaned:
            continue
        # Tree indentation markers: '   ', ':-', '+-', '| '
        match = re.match(r"^([\s\|\:\+\-]+)(.*)", cleaned)
        if match:
            prefix, content = match.group(1), match.group(2).strip()
            if content and not content.startswith(("Arguments:", "Output:", "Input:")):
                node_count += 1
                # Estimate depth based on prefix depth
                indent_level = len(prefix) // 3 + 1
                if indent_level > max_depth:
                    max_depth = indent_level
        else:
            if cleaned and not cleaned.startswith(("Arguments:", "Output:", "Input:")):
                node_count += 1

    return max(1, max_depth), max(1, node_count)


def extract_features_from_plan_text(
    query_id: str,
    plan_id: str,
    plan_text: str,
    strategy: str = "default",
    telemetry_metrics: Optional[Dict[str, float]] = None,
) -> PlanFeatures:
    """
    Extract PlanFeatures by parsing a Spark physical plan string (EXPLAIN COST or toString()).
    """
    lines = plan_text.splitlines()
    tree_depth, num_nodes = _compute_tree_structure(lines)

    # Operator counters
    exchange_count = 0
    broadcast_exchange_count = 0
    bhj_count = 0
    smj_count = 0
    shj_count = 0
    bnlj_count = 0
    filter_count = 0
    project_count = 0
    aggregate_count = 0
    sort_count = 0

    scan_bytes = 0.0
    shuffle_bytes = 0.0
    max_intermediate = 0.0
    broadcast_join_sizes = 0.0
    estimated_rows = 0.0

    # Parse CBO statistics blocks if present in the plan text
    cbo_stats_list = parse_explain_cost(plan_text)
    for cbo in cbo_stats_list:
        if cbo.size_in_bytes:
            if cbo.size_in_bytes > max_intermediate:
                max_intermediate = cbo.size_in_bytes
        if cbo.row_count:
            if cbo.row_count > estimated_rows:
                estimated_rows = cbo.row_count

    # Analyze lines for physical operators
    for line in lines:
        lower = line.lower()

        # Join operators
        if "broadcasthashjoin" in lower:
            bhj_count += 1
        elif "sortmergejoin" in lower:
            smj_count += 1
        elif "shuffledhashjoin" in lower or "hashjoin" in lower:
            shj_count += 1
        elif "broadcastnestedloopjoin" in lower or "cartesianproduct" in lower:
            bnlj_count += 1

        # Exchange operators
        if "broadcastexchange" in lower:
            broadcast_exchange_count += 1
        elif "exchange" in lower or "shuffleexchange" in lower:
            exchange_count += 1

        # Relational operators
        if "filter" in lower:
            filter_count += 1
        if "project" in lower:
            project_count += 1
        if "aggregate" in lower or "hashaggregate" in lower or "sortaggregate" in lower:
            aggregate_count += 1
        if "sort " in lower or lower.startswith("sort"):
            sort_count += 1

        # Scan and data size hints in the text
        if "filesize:" in lower or "sizeinbytes:" in lower or "data size:" in lower:
            size_match = re.search(r"(?:filesize:|sizeinbytes:|data size:)\s*([0-9.]+)\s*([KMGT]?i?B)?", line, re.I)
            if size_match:
                try:
                    val = float(size_match.group(1))
                    unit = (size_match.group(2) or "").upper()
                    mult = {"KB": 1e3, "KIB": 1024, "MB": 1e6, "MIB": 1024**2, "GB": 1e9, "GIB": 1024**3}.get(unit, 1.0)
                    scan_bytes += val * mult
                except Exception:
                    pass

    # Merge execution telemetry metrics if provided
    if telemetry_metrics:
        for k, v in telemetry_metrics.items():
            kl = k.lower()
            if any(key in kl for key in _SCAN_METRIC_KEYS):
                scan_bytes = max(scan_bytes, float(v))
            elif any(key in kl for key in _SHUFFLE_METRIC_KEYS):
                shuffle_bytes = max(shuffle_bytes, float(v))

    # If shuffle_bytes wasn't directly in telemetry, estimate from exchanges and scans
    if shuffle_bytes == 0.0 and exchange_count > 0:
        # Default shuffle heuristic: exchange nodes shuffle fraction of scanned bytes
        shuffle_bytes = float(exchange_count * max(scan_bytes * 0.4, 65536.0))

    if scan_bytes == 0.0:
        # Fallback to estimated intermediate size or standard baseline if no stats attached
        scan_bytes = max_intermediate if max_intermediate > 0 else 1024.0 * 1024.0

    if max_intermediate == 0.0:
        max_intermediate = max(scan_bytes, shuffle_bytes)

    if bhj_count > 0:
        # Estimate broadcast relation size
        broadcast_join_sizes = min(scan_bytes * 0.15, 10.0 * 1024 * 1024)

    return PlanFeatures(
        query_id=query_id,
        plan_id=plan_id,
        scan_bytes=float(scan_bytes),
        shuffle_bytes=float(shuffle_bytes),
        max_intermediate_size=float(max_intermediate),
        broadcast_join_sizes=float(broadcast_join_sizes),
        exchange_count=exchange_count,
        broadcast_exchange_count=broadcast_exchange_count,
        bhj_count=bhj_count,
        smj_count=smj_count,
        shj_count=shj_count,
        bnlj_count=bnlj_count,
        filter_count=filter_count,
        project_count=project_count,
        aggregate_count=aggregate_count,
        sort_count=sort_count,
        tree_depth=tree_depth,
        num_nodes=num_nodes,
        estimated_rows=float(estimated_rows),
        strategy=strategy,
    )


def extract_plan_features(
    candidate_or_plan: Any,
    query_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    strategy: Optional[str] = None,
    telemetry_metrics: Optional[Dict[str, float]] = None,
) -> PlanFeatures:
    """
    Universal feature extractor: handles CandidatePlan objects, raw strings,
    or PySpark Java SparkPlan objects.
    """
    # 1. CandidatePlan object
    if hasattr(candidate_or_plan, "physical_plan") and hasattr(candidate_or_plan, "query_id"):
        qid = candidate_or_plan.query_id
        pid = getattr(candidate_or_plan, "plan_id", "P01")
        strat = getattr(candidate_or_plan, "strategy", "default")
        plan_str = candidate_or_plan.physical_plan
        return extract_features_from_plan_text(
            query_id=qid,
            plan_id=pid,
            plan_text=plan_str,
            strategy=strat,
            telemetry_metrics=telemetry_metrics,
        )

    # 2. PySpark Java plan object
    if hasattr(candidate_or_plan, "children") and hasattr(candidate_or_plan, "metrics"):
        plan_str = candidate_or_plan.toString()
        qid = query_id or "Q_UNKNOWN"
        pid = plan_id or "P_UNKNOWN"
        strat = strategy or "default"
        return extract_features_from_plan_text(
            query_id=qid,
            plan_id=pid,
            plan_text=plan_str,
            strategy=strat,
            telemetry_metrics=telemetry_metrics,
        )

    # 3. String representation
    plan_str = str(candidate_or_plan)
    qid = query_id or "Q01"
    pid = plan_id or "P01"
    strat = strategy or "default"
    return extract_features_from_plan_text(
        query_id=qid,
        plan_id=pid,
        plan_text=plan_str,
        strategy=strat,
        telemetry_metrics=telemetry_metrics,
    )
