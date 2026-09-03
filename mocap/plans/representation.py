from dataclasses import dataclass
from typing import Optional


@dataclass
class CandidatePlan:
    """
    Represents one physical execution-plan candidate
    generated for a query.
    """

    query_id: str
    plan_id: str
    strategy: str
    sql: str
    physical_plan: str
    fingerprint: str

    estimated_cost: Optional[float] = None
    estimated_latency: Optional[float] = None
    actual_cost: Optional[float] = None
    actual_latency: Optional[float] = None

import hashlib


def create_plan_fingerprint(physical_plan: str) -> str:
    """
    Create a stable identifier for a physical plan.

    The fingerprint allows us to detect duplicate candidates
    that produce the same physical execution plan.
    """

    normalized = " ".join(physical_plan.split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]