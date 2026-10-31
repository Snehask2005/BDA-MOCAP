"""
Shared statistics representation for MOCAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PlanStatistics:
    """
    Pre-execution statistics obtained from Spark/Catalyst or catalog metadata.

    input_bytes:
        Sum of base/leaf relation sizes.

    intermediate_bytes:
        Largest reliable intermediate join relation size observed in the
        optimized Catalyst plan. Pathological Spark estimates are excluded
        by the statistics extractor.

    row_count:
        Best-effort Catalyst row-count estimate.
    """

    input_bytes: float = 0.0
    intermediate_bytes: float = 0.0
    row_count: Optional[float] = None
