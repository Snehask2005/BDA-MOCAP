from pathlib import Path

from mocap.calibration.dataset import (
    append_samples,
    load_dataset,
)
from mocap.calibration.service import CalibrationSample


def test_calibration_dataset_round_trip(tmp_path: Path):
    csv_path = tmp_path / "calibration.csv"

    sample = CalibrationSample(
        plan_id="plan-1",
        query_id="query-1",
        estimated_cost=10.0,
        actual_cost=12.0,
        estimated_latency=100.0,
        actual_latency=120.0,
        cpu_component=5.0,
        io_component=3.0,
        shuffle_component=2.0,
    )

    append_samples(
        [sample],
        str(csv_path),
    )

    rows = load_dataset(str(csv_path))

    assert len(rows) == 1

    assert rows[0]["plan_id"] == "plan-1"
    assert rows[0]["query_id"] == "query-1"
    assert rows[0]["estimated_cost"] == 10.0
    assert rows[0]["actual_cost"] == 12.0
    assert rows[0]["estimated_latency"] == 100.0
    assert rows[0]["actual_latency"] == 120.0