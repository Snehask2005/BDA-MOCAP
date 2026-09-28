from mocap.cost.pricing import PricingConfig
from mocap.execution.monitor import ProgressSample, RuntimeMonitor


def make_pricing(cpu_price=0.048):
    return PricingConfig(
        currency="USD",
        cost_per_vcore_hour=cpu_price,
        cost_per_gb_scanned=0.005,
        cost_per_gb_shuffled=0.010,
        cost_per_gb_hour_memory=0.006,
        penalty_per_second=0.0,
    )


def test_project_total_cost():
    monitor = RuntimeMonitor(
        spark=None,
        budget=1.0,
        pricing=make_pricing(),
    )

    sample = ProgressSample(
        t=10.0,
        completed_tasks=25,
        total_tasks=100,
        elapsed_s=10.0,
    )

    projected = monitor.project_total_cost(sample)

    # 25% complete after 10 seconds -> projected 40 seconds.
    # 40 / 3600 * $0.048 per vCPU-hour.
    expected = (40.0 / 3600.0) * 0.048

    assert projected is not None
    assert abs(projected - expected) < 1e-6


def test_project_total_cost_returns_none_without_progress():
    monitor = RuntimeMonitor(
        spark=None,
        budget=1.0,
        pricing=make_pricing(),
    )

    sample = ProgressSample(
        t=10.0,
        completed_tasks=0,
        total_tasks=100,
        elapsed_s=10.0,
    )

    assert monitor.project_total_cost(sample) is None


def test_project_total_cost_returns_none_for_invalid_total():
    monitor = RuntimeMonitor(
        spark=None,
        budget=1.0,
        pricing=make_pricing(),
    )

    sample = ProgressSample(
        t=10.0,
        completed_tasks=10,
        total_tasks=0,
        elapsed_s=10.0,
    )

    assert monitor.project_total_cost(sample) is None


def test_custom_pricing_changes_projected_cost():
    cheap = RuntimeMonitor(
        spark=None,
        budget=1.0,
        pricing=make_pricing(cpu_price=0.048),
    )

    expensive = RuntimeMonitor(
        spark=None,
        budget=1.0,
        pricing=make_pricing(cpu_price=0.096),
    )

    sample = ProgressSample(
        t=10.0,
        completed_tasks=50,
        total_tasks=100,
        elapsed_s=10.0,
    )

    cheap_cost = cheap.project_total_cost(sample)
    expensive_cost = expensive.project_total_cost(sample)

    assert cheap_cost is not None
    assert expensive_cost is not None
    assert abs(expensive_cost - 2 * cheap_cost) < 1e-6


def test_monitor_without_spark_can_start_and_stop():
    monitor = RuntimeMonitor(
        spark=None,
        budget=1.0,
        poll_interval_s=0.01,
        pricing=make_pricing(),
    )

    monitor.start()
    samples = monitor.stop()

    assert isinstance(samples, list)
