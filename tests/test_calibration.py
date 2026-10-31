from mocap.calibration.model import (
    CalibrationModel,
    CalibrationSample,
)


def test_calibration_identity_for_identity_data():
    samples = [
        CalibrationSample(
            predicted_cost=1.0,
            actual_cost=1.0,
            predicted_latency=10.0,
            actual_latency=10.0,
        ),
        CalibrationSample(
            predicted_cost=2.0,
            actual_cost=2.0,
            predicted_latency=20.0,
            actual_latency=20.0,
        ),
        CalibrationSample(
            predicted_cost=3.0,
            actual_cost=3.0,
            predicted_latency=30.0,
            actual_latency=30.0,
        ),
    ]

    model = CalibrationModel()
    model.fit(samples)

    assert abs(model.calibrate_cost(4.0) - 4.0) < 1e-9
    assert abs(model.calibrate_latency(40.0) - 40.0) < 1e-9


def test_calibration_learns_scale():
    samples = [
        CalibrationSample(
            predicted_cost=1.0,
            actual_cost=2.0,
            predicted_latency=10.0,
            actual_latency=20.0,
        ),
        CalibrationSample(
            predicted_cost=2.0,
            actual_cost=4.0,
            predicted_latency=20.0,
            actual_latency=40.0,
        ),
        CalibrationSample(
            predicted_cost=3.0,
            actual_cost=6.0,
            predicted_latency=30.0,
            actual_latency=60.0,
        ),
    ]

    model = CalibrationModel()
    model.fit(samples)

    assert abs(model.calibrate_cost(4.0) - 8.0) < 1e-9
    assert abs(model.calibrate_latency(40.0) - 80.0) < 1e-9


def test_invalid_values_are_rejected():
    model = CalibrationModel()

    try:
        model.add_sample(
            predicted_cost=-1.0,
            actual_cost=1.0,
            predicted_latency=1.0,
            actual_latency=1.0,
        )
        assert False
    except ValueError:
        pass