import numpy as np
from mocap.cost.predictor import (
    PlanStatistics,
    predict_plan_metrics,
)
from mocap.plans.representation import CandidatePlan


def test_predictor_uses_plan_statistics() -> None:
    candidate = CandidatePlan(
        query_id="Q1",
        plan_id="P1",
        strategy="default",
        sql="SELECT * FROM test",
        physical_plan="Scan ExistingRDD",
        fingerprint="test",
        num_scans=1,
    )

    statistics = PlanStatistics(
        input_bytes=1024 * 1024 * 1024,
        row_count=100000,
    )

    metrics = predict_plan_metrics(
        candidate,
        statistics=statistics,
    )

    assert metrics.bytes_scanned == 1024 * 1024 * 1024
    assert metrics.estimated_cost > 0
    assert metrics.estimated_latency > 0

def test_predictor_applies_calibration_model() -> None:
    candidate = CandidatePlan(
        query_id="Q1",
        plan_id="P1",
        strategy="default",
        sql="SELECT * FROM test",
        physical_plan="Scan ExistingRDD",
        fingerprint="test",
        num_scans=1,
    )

    statistics = PlanStatistics(
        input_bytes=1024 * 1024 * 1024,
        row_count=100000,
    )

    from mocap.cost.learned import CalibrationModel

    import numpy as np

    model = CalibrationModel(
        weights=np.array([0.5, 2.0, 1.0, 1.0]),
    )

    metrics = predict_plan_metrics(
        candidate,
        statistics=statistics,
        calibration_model=model,
    )

    assert metrics.estimated_cost > 0
    assert metrics.estimated_latency > 0


def test_predictor_without_calibration_preserves_baseline() -> None:
    candidate = CandidatePlan(
        query_id="Q1",
        plan_id="P1",
        strategy="default",
        sql="SELECT * FROM test",
        physical_plan="Scan ExistingRDD",
        fingerprint="test",
        num_scans=1,
    )

    statistics = PlanStatistics(
        input_bytes=1024 * 1024,
        row_count=1000,
    )

    from mocap.cost.learned import CalibrationModel

    baseline = predict_plan_metrics(
        candidate,
        statistics=statistics,
        calibration_model=CalibrationModel.identity(),
    )

    explicit_identity = predict_plan_metrics(
        candidate,
        statistics=statistics,
        calibration_model=CalibrationModel.identity(),
    )

    assert explicit_identity.estimated_cost == baseline.estimated_cost
    assert explicit_identity.estimated_latency == baseline.estimated_latency


def test_predictor_automatically_loads_saved_calibration_model(
    tmp_path,
    monkeypatch,
) -> None:
    from mocap.cost import learned
    from mocap.cost.learned import CalibrationModel

    candidate = CandidatePlan(
        query_id="Q1",
        plan_id="P1",
        strategy="default",
        sql="SELECT * FROM test",
        physical_plan="Scan ExistingRDD",
        fingerprint="test",
        num_scans=1,
    )

    statistics = PlanStatistics(
        input_bytes=1024 * 1024,
        row_count=1000,
    )

    baseline = predict_plan_metrics(
        candidate,
        statistics=statistics,
    )

    model_path = tmp_path / "calibration_model.json"

    model = CalibrationModel(
        weights=np.array([0.5, 2.0, 1.0, 1.0]),
    )
    model.save(str(model_path))

    monkeypatch.setattr(
        learned,
        "_DEFAULT_MODEL_JSON",
        str(model_path),
    )

    calibrated = predict_plan_metrics(
        candidate,
        statistics=statistics,
    )

    assert calibrated.estimated_cost != baseline.estimated_cost
    assert calibrated.estimated_latency == baseline.estimated_latency


def test_predictor_scales_cpu_with_workload_size() -> None:
    candidate = CandidatePlan(
        query_id="Q1",
        plan_id="P1",
        strategy="default",
        sql="SELECT * FROM test",
        physical_plan="Scan ExistingRDD",
        fingerprint="test",
        num_scans=1,
        num_joins=1,
    )

    small = predict_plan_metrics(
        candidate,
        statistics=PlanStatistics(
            input_bytes=64 * 1024 * 1024,
            row_count=1000,
        ),
        calibration_model=__import__(
            "mocap.cost.learned",
            fromlist=["CalibrationModel"],
        ).CalibrationModel.identity(),
    )

    large = predict_plan_metrics(
        candidate,
        statistics=PlanStatistics(
            input_bytes=1024 * 1024 * 1024,
            row_count=100000,
        ),
        calibration_model=__import__(
            "mocap.cost.learned",
            fromlist=["CalibrationModel"],
        ).CalibrationModel.identity(),
    )

    assert large.cpu_seconds > small.cpu_seconds
    assert large.estimated_latency > small.estimated_latency
    assert large.estimated_cost > small.estimated_cost


def test_predictor_scales_shuffle_with_workload_size() -> None:
    candidate = CandidatePlan(
        query_id="Q1",
        plan_id="P1",
        strategy="sort_merge_join",
        sql="SELECT * FROM test",
        physical_plan="SortMergeJoin",
        fingerprint="test",
        num_scans=2,
        num_joins=1,
        num_exchanges=1,
        num_sorts=1,
    )

    small = predict_plan_metrics(
        candidate,
        statistics=PlanStatistics(
            input_bytes=64 * 1024 * 1024,
            row_count=1000,
        ),
        calibration_model=__import__(
            "mocap.cost.learned",
            fromlist=["CalibrationModel"],
        ).CalibrationModel.identity(),
    )

    large = predict_plan_metrics(
        candidate,
        statistics=PlanStatistics(
            input_bytes=1024 * 1024 * 1024,
            row_count=100000,
        ),
        calibration_model=__import__(
            "mocap.cost.learned",
            fromlist=["CalibrationModel"],
        ).CalibrationModel.identity(),
    )

    assert large.bytes_shuffled > small.bytes_shuffled
    assert large.estimated_cost > small.estimated_cost


def test_predictor_strategy_factor_still_affects_prediction() -> None:
    statistics = PlanStatistics(
        input_bytes=256 * 1024 * 1024,
        row_count=10000,
    )

    broadcast = CandidatePlan(
        query_id="Q1",
        plan_id="B",
        strategy="broadcast",
        sql="SELECT * FROM test",
        physical_plan="BroadcastHashJoin",
        fingerprint="broadcast",
        num_scans=2,
        num_joins=1,
        num_exchanges=1,
        num_broadcast_exchanges=1,
    )

    sort_merge = CandidatePlan(
        query_id="Q1",
        plan_id="S",
        strategy="sort_merge_join",
        sql="SELECT * FROM test",
        physical_plan="SortMergeJoin",
        fingerprint="sort_merge",
        num_scans=2,
        num_joins=1,
        num_exchanges=1,
        num_sorts=1,
    )

    broadcast_metrics = predict_plan_metrics(
        broadcast,
        statistics=statistics,
        calibration_model=__import__(
            "mocap.cost.learned",
            fromlist=["CalibrationModel"],
        ).CalibrationModel.identity(),
    )

    sort_merge_metrics = predict_plan_metrics(
        sort_merge,
        statistics=statistics,
        calibration_model=__import__(
            "mocap.cost.learned",
            fromlist=["CalibrationModel"],
        ).CalibrationModel.identity(),
    )

    assert broadcast_metrics.cpu_seconds < sort_merge_metrics.cpu_seconds


def test_predictor_missing_statistics_preserves_baseline() -> None:
    candidate = CandidatePlan(
        query_id="Q1",
        plan_id="P1",
        strategy="default",
        sql="SELECT * FROM test",
        physical_plan="Scan ExistingRDD",
        fingerprint="test",
        num_scans=1,
        num_joins=1,
    )

    without_statistics = predict_plan_metrics(
        candidate,
        calibration_model=__import__(
            "mocap.cost.learned",
            fromlist=["CalibrationModel"],
        ).CalibrationModel.identity(),
    )

    with_zero_statistics = predict_plan_metrics(
        candidate,
        statistics=PlanStatistics(
            input_bytes=0.0,
            row_count=None,
        ),
        calibration_model=__import__(
            "mocap.cost.learned",
            fromlist=["CalibrationModel"],
        ).CalibrationModel.identity(),
    )

    assert (
        without_statistics.cpu_seconds
        == with_zero_statistics.cpu_seconds
    )
    assert (
        without_statistics.bytes_scanned
        == with_zero_statistics.bytes_scanned
    )
    assert (
        without_statistics.bytes_shuffled
        == with_zero_statistics.bytes_shuffled
    )
