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

def test_train_test_evaluation_splits_by_query(tmp_path):
    from mocap.calibration.updater import evaluate_train_test

    csv_path = tmp_path / "calibration.csv"

    rows = [
        {
            "plan_id": f"Q1_P{i}",
            "query_id": "Q1",
            "cpu_component": 1.0 + i,
            "io_component": 0.5,
            "shuffle_component": 0.2,
            "estimated_cost": 10.0 + i,
            "actual_cost": 8.0 + i,
            "estimated_latency": 1.0,
            "actual_latency": 1.0,
        }
        for i in range(3)
    ]

    rows += [
        {
            "plan_id": f"Q2_P{i}",
            "query_id": "Q2",
            "cpu_component": 2.0 + i,
            "io_component": 0.7,
            "shuffle_component": 0.3,
            "estimated_cost": 12.0 + i,
            "actual_cost": 9.0 + i,
            "estimated_latency": 1.2,
            "actual_latency": 1.1,
        }
        for i in range(3)
    ]

    rows += [
        {
            "plan_id": f"Q3_P{i}",
            "query_id": "Q3",
            "cpu_component": 3.0 + i,
            "io_component": 0.9,
            "shuffle_component": 0.4,
            "estimated_cost": 14.0 + i,
            "actual_cost": 11.0 + i,
            "estimated_latency": 1.4,
            "actual_latency": 1.2,
        }
        for i in range(3)
    ]

    import csv

    fieldnames = [
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

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    result = evaluate_train_test(
        str(csv_path),
        test_fraction=1 / 3,
        random_seed=42,
    )

    assert result["train_samples"] == 6
    assert result["test_samples"] == 3

    assert set(result["train_queries"]).isdisjoint(
        result["test_queries"]
    )

    assert set(result["train_queries"]) | set(
        result["test_queries"]
    ) == {"Q1", "Q2", "Q3"}

    assert "MAE" in result["analytical"]
    assert "RMSE" in result["analytical"]
    assert "MAPE_pct" in result["analytical"]

    assert "MAE" in result["calibrated"]
    assert "RMSE" in result["calibrated"]
    assert "MAPE_pct" in result["calibrated"]
