from mocap.calibration.service import CalibrationService
from mocap.cost.analytical import PlanMetrics
from mocap.interfaces import ExecutionTelemetry
from pathlib import Path

def make_prediction():
    return PlanMetrics(
        plan_id="plan-1",
        query_id="query-1",
        estimated_cost=10.0,
        estimated_latency=100.0,
        cpu_component=5.0,
        io_component=3.0,
        shuffle_component=2.0,
        cpu_seconds=100.0,
        bytes_scanned=1024.0,
        bytes_shuffled=512.0,
    )


def make_telemetry():
    return ExecutionTelemetry(
        plan_id="plan-1",
        query_id="query-1",
        actual_cost=20.0,
        actual_latency=200.0,
        cpu_seconds=200.0,
        bytes_scanned=2048.0,
        bytes_shuffled=1024.0,
    )


def test_calibration_service_records_observation():
    service = CalibrationService()

    sample = service.record(
        make_prediction(),
        make_telemetry(),
    )

    assert service.sample_count == 1
    assert sample.plan_id == "plan-1"
    assert sample.query_id == "query-1"
    assert sample.estimated_cost == 10.0
    assert sample.actual_cost == 20.0


def test_training_rows_match_existing_trainer_schema():
    service = CalibrationService()

    service.record(
        make_prediction(),
        make_telemetry(),
    )

    rows = service.build_cost_training_rows()

    assert len(rows) == 1

    row = rows[0]

    assert row["cpu_component"] == 5.0
    assert row["io_component"] == 3.0
    assert row["shuffle_component"] == 2.0
    assert row["estimated_cost"] == 10.0
    assert row["actual_cost"] == 20.0


def test_calibration_requires_matching_plan():
    service = CalibrationService()

    prediction = make_prediction()

    telemetry = ExecutionTelemetry(
        plan_id="different-plan",
        query_id="query-1",
        actual_cost=20.0,
        actual_latency=200.0,
    )

    try:
        service.record(prediction, telemetry)
        assert False
    except ValueError:
        pass

def test_record_to_dataset(
    tmp_path: Path,
):
    service = CalibrationService()

    csv_path = tmp_path / "calibration.csv"

    sample = service.record_to_dataset(
        prediction=make_prediction(),
        telemetry=make_telemetry(),
        csv_path=str(csv_path),
    )

    assert sample.plan_id == "plan-1"
    assert service.sample_count == 1
    assert csv_path.exists()

    contents = csv_path.read_text()

    assert "plan-1" in contents
    assert "query-1" in contents
    assert "10.0" in contents
    assert "20.0" in contents