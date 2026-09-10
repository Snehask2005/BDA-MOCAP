"""
Build a calibration dataset from PlanMetrics that already have a
ground-truth actual_cost recorded (i.e. produced by
mocap.cost.estimator.estimate_from_execution across many runs).

Keep the queries used to build this dataset separate from whatever
queries you use for final evaluation, to avoid leakage (see the
plan's publication-quality rules, section 16).
"""
from __future__ import annotations

import csv
import os
from typing import Iterable, List

from mocap.cost.analytical import PlanMetrics

_COLUMNS = [
    "plan_id",
    "query_id",
    "cpu_component",
    "io_component",
    "shuffle_component",
    "estimated_cost",
    "actual_cost",
]


def append_records(metrics_list: Iterable[PlanMetrics], csv_path: str) -> None:
    """
    Append rows for every PlanMetrics in `metrics_list` that has an
    actual_cost recorded. Rows missing ground truth are skipped
    (and reported) since they can't be used for calibration.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if not file_exists:
            writer.writeheader()

        for m in metrics_list:
            if m.actual_cost is None:
                print(f"Skipping {m.plan_id}: no actual_cost recorded yet")
                continue
            writer.writerow(
                {
                    "plan_id": m.plan_id,
                    "query_id": m.query_id,
                    "cpu_component": m.cpu_component,
                    "io_component": m.io_component,
                    "shuffle_component": m.shuffle_component,
                    "estimated_cost": m.estimated_cost,
                    "actual_cost": m.actual_cost,
                }
            )


def load_dataset(csv_path: str) -> List[dict]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = dict(row)
            for key in ("cpu_component", "io_component", "shuffle_component", "estimated_cost", "actual_cost"):
                parsed[key] = float(row[key])
            rows.append(parsed)
        return rows
