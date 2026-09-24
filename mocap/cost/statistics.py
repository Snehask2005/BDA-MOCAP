from __future__ import annotations

import math
from typing import Optional

from mocap.cost.statistics_model import PlanStatistics


# Spark can sometimes expose pathological/unbounded statistics for
# temporary views or plans whose statistics are unavailable.
# Treat extremely large values as unknown rather than allowing them
# to dominate cost estimation.
_MAX_REASONABLE_INPUT_BYTES = 1e18


def _safe_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number) or number < 0:
        return None

    return number


def _valid_input_size(value) -> Optional[float]:
    number = _safe_float(value)

    if number is None:
        return None

    if number > _MAX_REASONABLE_INPUT_BYTES:
        return None

    return number


def extract_plan_statistics(dataframe) -> PlanStatistics:
    logical_plan = dataframe._jdf.queryExecution().optimizedPlan()
    statistics = logical_plan.stats()

    size_in_bytes = _valid_input_size(statistics.sizeInBytes())

    row_count = None

    try:
        row_count_option = statistics.rowCount()

        if row_count_option.isDefined():
            row_count = _safe_float(row_count_option.get())

    except Exception:
        row_count = None

    return PlanStatistics(
        input_bytes=size_in_bytes or 0.0,
        row_count=row_count,
    )
