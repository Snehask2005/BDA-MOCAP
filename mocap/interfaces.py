"""
Shared data contracts for the MOCAP pipeline.

These dataclasses implement the "Shared Interfaces" agreed on Day 1
(Section 3 of the implementation plan) and the "Integration Contract"
(Section 11): Student 3 -> Student 4 hands off a SelectedPlan, and
Student 4 -> Student 2 hands off an ExecutionResult.

Every object is JSON-serializable so it can be logged, passed between
modules in-process, or written to disk for reproducibility (Section 16,
"Publication-Quality Implementation Rules").

NOTE: if the team already has a shared interfaces.py from someone else's
branch, merge this into it rather than keeping two copies -- there should
be exactly one source of truth for these contracts.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _dataclass_to_json(obj) -> str:
    return json.dumps(asdict(obj), default=str, indent=2)


@dataclass
class SelectedPlan:
    """Produced by Student 3's optimizer; consumed by Student 4's executor."""

    plan_id: str
    query_id: str
    selected_strategy: str          # e.g. "broadcast_join", "sort_merge_join"
    expected_cost: float
    expected_latency: float
    selection_reason: str
    physical_plan_sql: Optional[str] = None   # SQL / plan text to actually run
    budget: Optional[float] = None
    deadline: Optional[float] = None
    accuracy_tolerance: Optional[float] = None

    def to_json(self) -> str:
        return _dataclass_to_json(self)


@dataclass
class ExecutionResult:
    """Produced by Student 4's executor; consumed by Student 2's calibration loop."""

    query_id: str
    plan_id: str
    actual_cost: float
    actual_latency: float
    runtime_metrics: Dict[str, Any] = field(default_factory=dict)
    result_summary: Optional[str] = None
    result_path: Optional[str] = None
    execution_mode: str = "strict"   # "strict" | "adaptive" | "degraded"
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return _dataclass_to_json(self)


@dataclass
class PlanMetrics:
    """Round-trips with Student 2's cost model; used when Adaptive Mode replans."""

    plan_id: str
    estimated_cost: float
    estimated_latency: float
    cpu_component: float = 0.0
    io_component: float = 0.0
    shuffle_component: float = 0.0
    actual_cost: Optional[float] = None
    actual_latency: Optional[float] = None

    def to_json(self) -> str:
        return _dataclass_to_json(self)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
