"""
Calibration dataset utilities.

Calibration samples are created from a pre-execution PlanMetrics
prediction paired with post-execution ExecutionTelemetry.

This keeps prediction and ground truth separate and prevents
calibration data from being generated from estimated values alone.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, List

from mocap.calibration.service import CalibrationSample


_COLUMNS = [
    "plan_id",
    "query_id",
    "cpu_component",
    "io_component",
    "shuffle_component",
    "estimated_cost",
    "actual_cost",
    "estimated_latency",
    "actual_latency",
]


def append_samples(
    samples: Iterable[CalibrationSample],
    csv_path: str,
) -> None:
    """
    Append calibration observations to a CSV dataset.
    """

    os.makedirs(
        os.path.dirname(csv_path) or ".",
        exist_ok=True,
    )

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        for sample in samples:
            writer.writerow(
                {
                    "plan_id": sample.plan_id,
                    "query_id": sample.query_id,
                    "cpu_component": sample.cpu_component,
                    "io_component": sample.io_component,
                    "shuffle_component": sample.shuffle_component,
                    "estimated_cost": sample.estimated_cost,
                    "actual_cost": sample.actual_cost,
                    "estimated_latency": sample.estimated_latency,
                    "actual_latency": sample.actual_latency,
                }
            )


def load_dataset(csv_path: str) -> List[dict]:
    """Load calibration observations from CSV."""

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        rows = []

        for row in reader:
            parsed = dict(row)

            for key in (
                "cpu_component",
                "io_component",
                "shuffle_component",
                "estimated_cost",
                "actual_cost",
                "estimated_latency",
                "actual_latency",
            ):
                parsed[key] = float(row[key])

            rows.append(parsed)

        return rows