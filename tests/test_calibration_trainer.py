from pathlib import Path

import numpy as np

from mocap.calibration.dataset import append_samples
from mocap.calibration.service import CalibrationSample
from mocap.calibration.trainer import train_from_csv


def test_trainer_learns_known_linear_relationship(
    tmp_path: Path,
):
    csv_path = tmp_path / "calibration.csv"

    # Independent feature combinations.
    feature_rows = [
        (1.0, 2.0, 3.0),
        (2.0, 1.0, 4.0),
        (3.0, 4.0, 1.0),
        (4.0, 3.0, 2.0),
        (5.0, 2.0, 5.0),
        (2.0, 5.0, 3.0),
        (4.0, 1.0, 6.0),
        (6.0, 3.0, 1.0),
    ]

    samples = []

    for i, (cpu, io, shuffle) in enumerate(
        feature_rows,
        start=1,
    ):
        # Known relationship:
        #
        # cost = 1 + 2*cpu + 3*io + 4*shuffle
        actual_cost = (
            1.0
            + 2.0 * cpu
            + 3.0 * io
            + 4.0 * shuffle
        )

        samples.append(
            CalibrationSample(
                plan_id=f"plan-{i}",
                query_id=f"query-{i}",
                estimated_cost=actual_cost,
                actual_cost=actual_cost,
                estimated_latency=10.0,
                actual_latency=10.0,
                cpu_component=cpu,
                io_component=io,
                shuffle_component=shuffle,
            )
        )

    append_samples(
        samples,
        str(csv_path),
    )

    model = train_from_csv(
        str(csv_path),
    )

    assert np.allclose(
        model.weights,
        np.array([1.0, 2.0, 3.0, 4.0]),
        atol=1e-8,
    )


def test_trainer_requires_enough_examples(
    tmp_path: Path,
):
    csv_path = tmp_path / "calibration.csv"

    samples = [
        CalibrationSample(
            plan_id="plan-1",
            query_id="query-1",
            estimated_cost=1.0,
            actual_cost=2.0,
            estimated_latency=10.0,
            actual_latency=20.0,
            cpu_component=1.0,
            io_component=1.0,
            shuffle_component=1.0,
        )
    ]

    append_samples(
        samples,
        str(csv_path),
    )

    try:
        train_from_csv(str(csv_path))
        assert False
    except ValueError:
        pass

def test_trained_model_can_be_saved_and_reloaded(
    tmp_path: Path,
):
    csv_path = tmp_path / "calibration.csv"
    model_path = tmp_path / "calibration_model.json"

    samples = [
        CalibrationSample(
            plan_id=f"plan-{i}",
            query_id=f"query-{i}",
            estimated_cost=1.0,
            actual_cost=2.0 + i,
            estimated_latency=1.0,
            actual_latency=1.0,
            cpu_component=float(i),
            io_component=1.0,
            shuffle_component=1.0,
        )
        for i in range(1, 6)
    ]

    append_samples(samples, str(csv_path))

    model = train_from_csv(str(csv_path))
    model.save(str(model_path))

    from mocap.cost.learned import CalibrationModel

    loaded = CalibrationModel.load(str(model_path))

    assert np.allclose(
        loaded.weights,
        model.weights,
    )

    assert loaded.predict(
        cpu_component=2.0,
        io_component=1.0,
        shuffle_component=1.0,
    ) == model.predict(
        cpu_component=2.0,
        io_component=1.0,
        shuffle_component=1.0,
    )
