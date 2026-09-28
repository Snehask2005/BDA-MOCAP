"""Spark Catalyst statistics extraction for MOCAP."""

from __future__ import annotations

import math
from typing import Optional

from mocap.cost.statistics_model import PlanStatistics


_MAX_REASONABLE_INPUT_BYTES = 1e18


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if result < 0:
        return None

    return result


def _valid_input_size(value) -> float | None:
    size = _safe_float(value)

    if size is None:
        return None

    if not math.isfinite(size):
        return None

    if size < 0:
        return None

    if size > _MAX_REASONABLE_INPUT_BYTES:
        return None

    return size


def _extract_base_input_sizes(plan) -> list[float]:
    """
    Recursively collect leaf/base-relation statistics.

    Join nodes often contain inflated cardinality estimates because Spark
    multiplies child relation sizes when join selectivity is unknown.
    For workload sizing we therefore use leaf relation sizes rather than
    intermediate join output sizes.
    """

    children = plan.children()

    if children.size() == 0:
        try:
            size = _valid_input_size(
                plan.stats().sizeInBytes()
            )
        except Exception:
            return []

        return [size] if size is not None else []

    sizes: list[float] = []

    for index in range(children.size()):
        child = children.apply(index)
        sizes.extend(
            _extract_base_input_sizes(child)
        )

    return sizes


def _extract_join_intermediate_sizes(plan) -> list[float]:
    """
    Recursively collect reliable intermediate join sizes.

    Spark can produce pathological join cardinality estimates when
    selectivity is unavailable. We therefore:

    1. inspect only Join nodes,
    2. accept only finite, bounded statistics,
    3. ignore the root/final join because it is frequently inflated,
    4. retain intermediate join estimates below the root.
    """

    children = plan.children()

    sizes: list[float] = []

    node_name = str(plan.nodeName()).lower()

    if node_name == "join":
        try:
            size = _valid_input_size(
                plan.stats().sizeInBytes()
            )
        except Exception:
            size = None

        if size is not None:
            sizes.append(size)

    for index in range(children.size()):
        child = children.apply(index)

        sizes.extend(
            _extract_join_intermediate_sizes(child)
        )

    return sizes


def _extract_row_count(plan) -> Optional[float]:
    """
    Recursively find the first valid row-count estimate.
    """

    try:
        row_count = plan.stats().rowCount()

        if row_count.isDefined():
            value = _safe_float(
                row_count.get()
            )

            if value is not None:
                return value
    except Exception:
        pass

    children = plan.children()

    for index in range(children.size()):
        value = _extract_row_count(
            children.apply(index)
        )

        if value is not None:
            return value

    return None


def _extract_intermediate_bytes(plan) -> float:
    """
    Return the largest reliable non-root join intermediate.

    The root Join is intentionally excluded because Spark's final join
    cardinality estimate can become pathological when join selectivity is
    unavailable.
    """

    children = plan.children()

    if children.size() == 0:
        return 0.0

    sizes: list[float] = []

    node_name = str(plan.nodeName()).lower()

    if node_name == "join":
        for index in range(children.size()):
            child = children.apply(index)

            try:
                child_size = _valid_input_size(
                    child.stats().sizeInBytes()
                )
            except Exception:
                child_size = None

            if child_size is not None:
                sizes.append(child_size)

    for index in range(children.size()):
        sizes.append(
            _extract_intermediate_bytes(
                children.apply(index)
            )
        )

    flattened = [
        value
        for value in sizes
        if isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        and value <= _MAX_REASONABLE_INPUT_BYTES
    ]

    return max(flattened, default=0.0)


def extract_plan_statistics(dataframe) -> PlanStatistics:
    """
    Extract workload statistics from a Spark DataFrame.

    Input bytes are calculated from leaf/base relations rather than
    intermediate join estimates.

    Intermediate bytes are derived from child statistics of Join nodes.
    This provides a bounded signal for comparing join orders without relying
    on Spark's often-pathological final join cardinality estimate.
    """

    logical_plan = (
        dataframe
        ._jdf
        .queryExecution()
        .optimizedPlan()
    )

    base_sizes = _extract_base_input_sizes(
        logical_plan
    )

    input_bytes = sum(base_sizes)

    intermediate_bytes = _extract_intermediate_bytes(
        logical_plan
    )

    row_count = _extract_row_count(
        logical_plan
    )

    return PlanStatistics(
        input_bytes=input_bytes,
        intermediate_bytes=intermediate_bytes,
        row_count=row_count,
    )
