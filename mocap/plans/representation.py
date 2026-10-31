from dataclasses import dataclass
from typing import Optional
import hashlib
import re


@dataclass
class PlanFeatures:
    actual_join_strategy: str | None
    num_joins: int
    num_broadcast_joins: int
    num_shuffle_hash_joins: int
    num_sort_merge_joins: int
    num_nested_loop_joins: int
    num_exchanges: int
    num_broadcast_exchanges: int
    num_sorts: int
    num_aggregates: int
    num_filters: int
    num_scans: int
    plan_depth: int

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class CandidatePlan:
    """
    Represents one physical execution-plan candidate generated for
    a query.

    The physical-plan features describe what Spark/Catalyst actually
    produced. They can be consumed by MOCAP's cost/latency model.
    """

    query_id: str
    plan_id: str
    strategy: str
    sql: str
    physical_plan: str
    fingerprint: str

    # Physical-plan characteristics
    actual_join_strategy: Optional[str] = None

    num_joins: int = 0
    num_broadcast_joins: int = 0
    num_shuffle_hash_joins: int = 0
    num_sort_merge_joins: int = 0
    num_nested_loop_joins: int = 0

    num_exchanges: int = 0
    num_broadcast_exchanges: int = 0
    num_sorts: int = 0
    num_aggregates: int = 0
    num_filters: int = 0
    num_scans: int = 0

    plan_depth: int = 0

    # Values populated later by cost/execution modules.
    estimated_cost: Optional[float] = None
    estimated_latency: Optional[float] = None
    actual_cost: Optional[float] = None
    actual_latency: Optional[float] = None


def create_plan_fingerprint(physical_plan: str) -> str:
    """
    Create a stable fingerprint for a Spark physical plan.

    Volatile expression IDs, plan IDs and whole-stage codegen IDs are
    normalized before hashing.
    """

    normalized = physical_plan

    normalized = re.sub(
        r"plan_id=\d+",
        "plan_id=X",
        normalized,
    )

    normalized = re.sub(
        r"#\d+[A-Za-z]*",
        "#X",
        normalized,
    )

    normalized = re.sub(
        r"\*\(\d+\)",
        "*",
        normalized,
    )

    normalized = " ".join(normalized.split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]


def _count_operator(plan: str, operator: str) -> int:
    """
    Count physical-plan operators in Spark's textual plan.

    Spark prefixes many operators with whole-stage codegen markers
    such as ``*(5)``. Tree prefixes such as ``+-`` and ``:-`` may
    also appear before the operator.
    """
    count = 0

    target = operator.lower()

    for line in plan.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        # Remove Spark tree prefixes.
        stripped = re.sub(
            r"^[\s:+\-]+",
            "",
            stripped,
        )

        # Remove whole-stage codegen markers such as:
        # *(1)
        # *(5)
        stripped = re.sub(
            r"^\*\(\d+\)\s*",
            "",
            stripped,
        )

        # The first token is now the physical operator.
        first_token = (
            stripped.split(None, 1)[0]
            if stripped
            else ""
        )

        if first_token.lower() == target:
            count += 1

    return count


def _calculate_plan_depth(physical_plan: str) -> int:
    """
    Estimate physical-plan tree depth from Spark's indentation.

    Spark's textual plan uses indentation to represent nesting.
    This is a structural feature, not a runtime measurement.
    """

    lines = [
        line
        for line in physical_plan.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0

    depths = []

    for line in lines:
        stripped = line.lstrip()

        # Number of leading spaces is a useful approximation of
        # nesting depth in Spark's textual physical plan.
        indentation = len(line) - len(stripped)

        depths.append(indentation // 2)

    return max(depths) + 1


def extract_physical_plan_features(
    physical_plan: str,
) -> PlanFeatures:
    """
    Extract structural characteristics from a Spark physical plan.

    This function does not estimate monetary cost or latency.
    """
    plan = physical_plan.lower()

    broadcast_hash = _count_operator(
        plan,
        "broadcasthashjoin",
    )

    shuffle_hash = _count_operator(
        plan,
        "shuffledhashjoin",
    )

    sort_merge = _count_operator(
        plan,
        "sortmergejoin",
    )

    broadcast_nested = _count_operator(
        plan,
        "broadcastnestedloopjoin",
    )

    # Count only non-broadcast nested-loop joins here.
    nested_loop = (
        _count_operator(plan, "nestedloopjoin")
        - broadcast_nested
    )

    num_joins = (
        broadcast_hash
        + shuffle_hash
        + sort_merge
        + broadcast_nested
        + nested_loop
    )

    if broadcast_hash:
        actual_join_strategy = "BroadcastHashJoin"
    elif shuffle_hash:
        actual_join_strategy = "ShuffledHashJoin"
    elif sort_merge:
        actual_join_strategy = "SortMergeJoin"
    elif broadcast_nested:
        actual_join_strategy = "BroadcastNestedLoopJoin"
    elif nested_loop:
        actual_join_strategy = "NestedLoopJoin"
    else:
        actual_join_strategy = None

    # BroadcastExchange is a type of exchange.
    num_broadcast_exchanges = _count_operator(
        plan,
        "broadcastexchange",
    )

    num_exchanges = (
        _count_operator(plan, "exchange")
        + num_broadcast_exchanges
    )

    return PlanFeatures(
        actual_join_strategy=actual_join_strategy,

        num_joins=num_joins,
        num_broadcast_joins=(
            broadcast_hash + broadcast_nested
        ),
        num_shuffle_hash_joins=shuffle_hash,
        num_sort_merge_joins=sort_merge,
        num_nested_loop_joins=(
            nested_loop + broadcast_nested
        ),

        num_exchanges=num_exchanges,
        num_broadcast_exchanges=num_broadcast_exchanges,

        num_sorts=_count_operator(
            plan,
            "sort",
        ),

        num_aggregates=(
            _count_operator(plan, "hashaggregate")
            + _count_operator(plan, "sortaggregate")
        ),

        num_filters=_count_operator(
            plan,
            "filter",
        ),

        num_scans=(
            _count_operator(plan, "scan")
            + _count_operator(plan, "filescan")
        ),

        plan_depth=_calculate_plan_depth(
            physical_plan
        ),
    )