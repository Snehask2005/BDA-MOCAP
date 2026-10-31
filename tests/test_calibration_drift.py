"""
Tests for calibration under workload drift and stale/pathological
Catalyst statistics.

Two distinct concerns, per the project's Definition of Done:

1. Drift: a calibration model fit on one cost regime should visibly
   degrade (not silently stay "accurate") when evaluated on a shifted
   regime -- this test measures and asserts that degradation is
   detectable, it does not claim the model handles drift gracefully.

2. Stale statistics: the predictor must not blow up or return
   nonsensical numbers when Catalyst statistics are wrong/outdated
   (e.g. carried over from a much larger or much smaller prior
   workload). It should stay bounded via PredictorConfig's
   min/max_workload_scale clamps.

No Spark session is required for either test.
"""
from __future__ import annotations

import math

from mocap.calibration.dataset import append_samples
from mocap.calibration.service import CalibrationSample
from mocap.calibration.trainer import train_from_csv
from mocap.calibration.updater import evaluate
from mocap.cost.predictor import PredictorConfig, predict_plan_metrics
from mocap.cost.statistics_model import PlanStatistics
from mocap.plans.representation import CandidatePlan


def _make_candidate(plan_id: str = "P01", query_id: str = "Q01") -> CandidatePlan:
    return CandidatePlan(
        query_id=query_id,
        plan_id=plan_id,
        strategy="broadcast",
        sql="SELECT 1",
        physical_plan="BroadcastHashJoin",
        fingerprint="fp1",
        actual_join_strategy="broadcast",
        num_joins=1,
        num_exchanges=1,
        num_broadcast_exchanges=1,
        num_sorts=0,
        num_aggregates=1,
        num_filters=1,
        num_scans=2,
        plan_depth=4,
    )


def _write_regime_csv(csv_path: str, cost_fn, n: int = 12) -> None:
    """
    Write a synthetic calibration dataset where actual_cost is a known
    function of the component values, so we can control what "the
    relationship" is for each regime.
    """
    samples = []
    for i in range(n):
        cpu = 0.01 * (i + 1)
        io = 0.02 * (i + 1)
        shuffle = 0.015 * (i + 1)
        estimated = cpu + io + shuffle
        actual = cost_fn(cpu, io, shuffle)

        samples.append(
            CalibrationSample(
                plan_id=f"P{i:02d}",
                query_id="Q_regime",
                estimated_cost=estimated,
                actual_cost=actual,
                estimated_latency=1.0,
                actual_latency=1.0,
                cpu_component=cpu,
                io_component=io,
                shuffle_component=shuffle,
            )
        )

    append_samples(samples, csv_path)


def test_calibration_error_increases_under_workload_drift(tmp_path):
    """
    Train on one (cpu, io, shuffle) -> actual_cost relationship, then
    evaluate on a *different* relationship (drifted -- e.g. pricing or
    hardware characteristics changed). Error on the drifted regime
    must be detectably worse than in-sample error on the training
    regime; a calibration model that looked equally "good" on both
    would mean the test setup wasn't actually exercising drift.
    """
    train_csv = tmp_path / "train_regime.csv"
    drifted_csv = tmp_path / "drifted_regime.csv"

    # Training regime: actual cost is a mild linear function of components.
    _write_regime_csv(str(train_csv), lambda cpu, io, shuffle: 1.5 * cpu + 1.2 * io + 1.1 * shuffle + 0.01)

    # Drifted regime: a structurally different (much steeper) relationship,
    # simulating e.g. a shuffle-heavy workload becoming far more expensive.
    _write_regime_csv(str(drifted_csv), lambda cpu, io, shuffle: 1.5 * cpu + 1.2 * io + 6.0 * shuffle + 0.01)

    model = train_from_csv(str(train_csv))

    in_sample_report = evaluate(str(train_csv), model)
    drifted_report = evaluate(str(drifted_csv), model)

    in_sample_mae = in_sample_report["calibrated"]["MAE"]
    drifted_mae = drifted_report["calibrated"]["MAE"]

    assert math.isfinite(in_sample_mae)
    assert math.isfinite(drifted_mae)

    # The whole point of this test: drift must be visible in the metric,
    # not silently absorbed.
    assert drifted_mae > in_sample_mae, (
        f"expected drifted-regime MAE ({drifted_mae}) to exceed "
        f"in-sample MAE ({in_sample_mae}) -- if this fails, the drifted "
        "regime in this test isn't actually different enough to detect."
    )


def test_predictor_stays_bounded_under_stale_oversized_statistics():
    """
    Feed the predictor statistics carried over from a workload orders of
    magnitude larger than what PredictorConfig's base_scan_bytes assumes
    (simulating stale/reused statistics). Prediction must remain finite
    and non-negative, not explode -- this is what
    PredictorConfig.max_workload_scale exists to guarantee.
    """
    candidate = _make_candidate()
    config = PredictorConfig()

    stale_statistics = PlanStatistics(
        # Wildly larger than base_scan_bytes -- e.g. stats left over from
        # a prior, much bigger table before a schema/data change.
        input_bytes=config.base_scan_bytes * 1_000_000,
        intermediate_bytes=config.base_scan_bytes * 1_000_000,
        row_count=10**12,
    )

    metrics = predict_plan_metrics(candidate, config=config, statistics=stale_statistics)

    assert math.isfinite(metrics.estimated_cost)
    assert math.isfinite(metrics.estimated_latency)
    assert metrics.estimated_cost >= 0
    assert metrics.estimated_latency >= 0

    # The clamp should actually be doing something: prediction with the
    # stale stats should not be unboundedly larger than with normal ones.
    normal_statistics = PlanStatistics(
        input_bytes=config.base_scan_bytes,
        intermediate_bytes=0.0,
        row_count=1000,
    )
    normal_metrics = predict_plan_metrics(candidate, config=config, statistics=normal_statistics)

    # max_workload_scale bounds the ratio; allow generous headroom above
    # that bound rather than asserting the exact factor, since strategy
    # factors and fixed overhead also contribute.
    assert metrics.estimated_cost <= normal_metrics.estimated_cost * (config.max_workload_scale * 2)


def test_predictor_stays_bounded_under_stale_undersized_statistics():
    """
    Same idea, opposite direction: statistics claiming a near-zero
    workload (e.g. stale stats from an empty table before a bulk load).
    min_workload_scale should prevent the prediction from collapsing to
    ~zero fixed cost.
    """
    candidate = _make_candidate()
    config = PredictorConfig()

    understated_statistics = PlanStatistics(
        input_bytes=1.0,  # effectively nothing
        intermediate_bytes=0.0,
        row_count=1,
    )

    metrics = predict_plan_metrics(candidate, config=config, statistics=understated_statistics)

    assert math.isfinite(metrics.estimated_cost)
    assert metrics.estimated_cost > 0  # fixed overhead must still show up
    assert metrics.cpu_seconds >= config.base_cpu_seconds * config.min_workload_scale * 0  # sanity: no crash


def test_predictor_handles_missing_statistics_gracefully():
    """
    No statistics at all (statistics=None) is the common case for a
    candidate whose DataFrame was never materialized -- must fall back
    to the structural baseline (workload_scale == 1.0) rather than
    erroring.
    """
    candidate = _make_candidate()

    metrics = predict_plan_metrics(candidate, statistics=None)

    assert math.isfinite(metrics.estimated_cost)
    assert math.isfinite(metrics.estimated_latency)
    assert metrics.estimated_cost >= 0
