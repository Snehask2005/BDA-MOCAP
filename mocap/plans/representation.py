from dataclasses import dataclass
from typing import Optional
import hashlib


@dataclass
class CandidatePlan:
    """
    Represents one physical execution-plan candidate
    generated for a query.

    The physical-plan features describe what Spark/Catalyst
    actually produced. They can later be consumed by the
    MOCAP cost and latency model.
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
    num_exchanges: int = 0
    num_broadcast_exchanges: int = 0
    num_sorts: int = 0

    # Values populated later by the cost/execution modules.
    estimated_cost: Optional[float] = None
    estimated_latency: Optional[float] = None
    actual_cost: Optional[float] = None
    actual_latency: Optional[float] = None


def create_plan_fingerprint(physical_plan: str) -> str:
    """
    Create a stable fingerprint for a Spark physical plan.

    Spark physical-plan strings contain volatile identifiers such as
    expression IDs and plan IDs. These identifiers do not represent
    meaningful execution-strategy differences, so they are normalized
    before hashing.
    """

    import re

    normalized = physical_plan

    # Remove Spark plan identifiers:
    #   plan_id=63
    normalized = re.sub(
        r"plan_id=\d+",
        "plan_id=X",
        normalized,
    )

    # Remove expression IDs such as:
    #   c_name#1
    #   total_spent#14
    normalized = re.sub(
        r"#\d+[A-Za-z]*",
        "#X",
        normalized,
    )

    # Remove whole-stage codegen IDs such as:
    #   *(6)
    normalized = re.sub(
        r"\*\(\d+\)",
        "*",
        normalized,
    )

    # Normalize whitespace.
    normalized = " ".join(normalized.split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]

def extract_physical_plan_features(
    physical_plan: str,
) -> dict:
    """
    Extract basic structural characteristics from a Spark
    physical execution plan.

    This function intentionally performs structural extraction
    only. It does not estimate monetary cost or latency.
    """

    plan = physical_plan.lower()

    if "broadcasthashjoin" in plan:
        actual_join_strategy = "BroadcastHashJoin"
    elif "shuffledhashjoin" in plan:
        actual_join_strategy = "ShuffledHashJoin"
    elif "sortmergejoin" in plan:
        actual_join_strategy = "SortMergeJoin"
    elif "broadcastnestedloopjoin" in plan:
        actual_join_strategy = "BroadcastNestedLoopJoin"
    elif "nestedloopjoin" in plan:
        actual_join_strategy = "NestedLoopJoin"
    else:
        actual_join_strategy = None

    return {
        "actual_join_strategy": actual_join_strategy,
        "num_joins": (
            plan.count("join")
            if actual_join_strategy is not None
            else 0
        ),
        "num_exchanges": plan.count("exchange"),
        "num_broadcast_exchanges": plan.count(
            "broadcastexchange"
        ),
        "num_sorts": plan.count("sort "),
    }