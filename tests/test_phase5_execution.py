"""
Unit tests for Anaswara's execution/ package (no Spark/JVM required).
"""

from __future__ import annotations

import json
import math
import random

import pytest

from execution import baselines as B
from execution import harness as H
from execution.cloud import load_profile, spearman
from execution.config import load_env
from execution.stats import (
    bootstrap_ci,
    holm_correction,
    paired_tests,
    summarize_outcomes,
    wilcoxon_paired,
)
from execution.telemetry import (
    TELEMETRY_FIELDS,
    TelemetryLogger,
    TelemetryRecord,
    validate_file,
    validate_record,
)
from execution.workload import load_sql_dir, select_queries, star_join_workload, star_query_sql


# --------------------------------------------------------------------- helpers

def _record(**over):
    base = dict(
        query_id="Q1", plan_id="Q1_P01", strategy="default", latency_ms=12.5,
        bytes_scanned=100.0, bytes_shuffled=50.0, executor_seconds=0.02, run_number=1,
        cache_state="warm", aqe_enabled=False, cbo_enabled=True, num_cores=2,
        memory_gb=4.0, seed=7, timestamp=1_700_000_000.0,
    )
    base.update(over)
    return TelemetryRecord(**base)


def _cand(pid, pred_cost, pred_lat, act_cost, act_lat, n=5, noise=0.0, seed=0):
    rng = random.Random(seed)
    lat = [act_lat * (1 + noise * rng.uniform(-1, 1)) for _ in range(n)]
    cost = [act_cost * (1 + noise * rng.uniform(-1, 1)) for _ in range(n)]
    return H.CandidateRecord(plan_id=pid, strategy=pid, pred_cost=pred_cost, pred_latency=pred_lat,
                             latencies_s=lat, costs=cost)


def _measurement(qid, cands, family="fam"):
    m = H.QueryMeasurement(qid, family, 4, "mid")
    m.candidates = cands
    m.spark_defaults = [H.CandidateRecord(plan_id=f"{qid}_sd", strategy=H.SPARK_DEFAULT_REFERENCE,
                                          kind="spark_default", latencies_s=[2.0] * 3, costs=[1.0] * 3)]
    return m


def _workload(n_queries=12, seed=1):
    """Queries where a cheap-but-slow plan, a fast-but-pricey plan and a
    balanced plan compete; predictions are unbiased with mild noise."""
    rng = random.Random(seed)
    ms = []
    for q in range(n_queries):
        cands = []
        truth = [("cheap_slow", 1.0, 4.0), ("balanced", 1.3, 2.0), ("fast_pricey", 3.0, 1.0)]
        for pid, cost, lat in truth:
            err = math.exp(rng.gauss(0, 0.05))
            cands.append(_cand(f"Q{q}_{pid}", cost * err, lat * err, cost, lat, seed=q))
        ms.append(_measurement(f"Q{q}", cands))
    return ms


# ------------------------------------------------------------------- telemetry

def test_schema_has_the_agreed_15_fields_in_order():
    assert TELEMETRY_FIELDS == (
        "query_id", "plan_id", "strategy", "latency_ms", "bytes_scanned", "bytes_shuffled",
        "executor_seconds", "run_number", "cache_state", "aqe_enabled", "cbo_enabled",
        "num_cores", "memory_gb", "seed", "timestamp",
    )
    assert list(_record().to_dict())[:15] == list(TELEMETRY_FIELDS)


def test_valid_record_passes_and_bad_records_fail():
    assert validate_record(_record().to_dict()) == []
    bad = _record().to_dict()
    del bad["seed"]
    assert any("missing field 'seed'" in p for p in validate_record(bad))
    assert validate_record(_record(cache_state="lukewarm").to_dict())
    assert validate_record(_record(run_number=0).to_dict())
    assert validate_record(_record(latency_ms=-1.0).to_dict())
    assert validate_record({**_record().to_dict(), "num_cores": True})  # bool is not an int here


def test_failed_run_may_have_null_metrics_but_ok_run_may_not():
    assert validate_record(_record(status="failed", latency_ms=None, bytes_scanned=None).to_dict()) == []
    assert validate_record(_record(latency_ms=None).to_dict())


def test_logger_writes_valid_jsonl_and_rejects_invalid(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = TelemetryLogger(path)
    logger.log(_record())
    logger.log(_record(run_number=2))
    assert validate_file(path) == []
    assert len(path.read_text().strip().splitlines()) == 2
    with pytest.raises(ValueError):
        logger.log(_record(cache_state="hot"))
    path.write_text(path.read_text() + "{not json}\n")
    assert any("invalid JSON" in p for p in validate_file(path))


def test_telemetry_cli_check(tmp_path, capsys):
    from execution.telemetry import _main

    path = tmp_path / "ok.jsonl"
    TelemetryLogger(path).log(_record())
    assert _main(["check", str(path)]) == 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"query_id": "x"}) + "\n")
    assert _main(["check", str(bad)]) == 1


# ---------------------------------------------------------------------- config

def test_env_profiles_load_and_keep_hint_safe_defaults():
    env = load_env(profile="local")
    assert env.spark_conf["spark.sql.cbo.joinReorder.enabled"] == "false"
    assert env.spark_conf["spark.sql.autoBroadcastJoinThreshold"] == "-1"
    assert env.num_cores >= 1
    assert env.experiment["budget_multipliers"] == [1.0, 1.2, 1.5, 2.0]
    cluster = load_env(profile="cluster")
    assert cluster.master is None and cluster.num_cores > env.num_cores
    with pytest.raises(KeyError):
        load_env(profile="nope")


def test_spark_default_variants_cover_cbo_and_aqe_on_off():
    names = {n: o for n, o in H.spark_default_overrides(load_env(profile="local"))}
    assert len(names) == 4
    combos = {(o["spark.sql.cbo.enabled"], o["spark.sql.adaptive.enabled"]) for o in names.values()}
    assert combos == {("true", "true"), ("true", "false"), ("false", "true"), ("false", "false")}
    assert all(o["spark.sql.autoBroadcastJoinThreshold"] != "-1" for o in names.values())


# --------------------------------------------------------------------- workload

def test_star_workload_shapes():
    queries = star_join_workload()
    assert len(queries) == 8 * 3 + 3
    assert len({q.query_id for q in queries}) == len(queries)
    sql = star_query_sql(10, "d.d_year = 2000")
    assert sql.count("INNER JOIN") == 9
    assert {q.family for q in queries} == {"star_small", "star_medium", "star_large", "q5_groupby"}
    assert [q.query_id for q in select_queries(queries, ["star_small"])][:1] == ["E3_Q03_narrow"]
    with pytest.raises(ValueError):
        star_query_sql(2, "x")


def test_load_sql_dir(tmp_path):
    (tmp_path / "q1.sql").write_text("SELECT * FROM a JOIN b ON a.x = b.x;")
    queries = load_sql_dir(str(tmp_path))
    assert queries[0].query_id == "Q1" and not queries[0].sql.endswith(";")


# ------------------------------------------------------------------------ stats

def test_bootstrap_ci_contains_median():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lo, hi = bootstrap_ci(values, seed=1)
    assert lo <= 5.5 <= hi
    assert bootstrap_ci([]) != bootstrap_ci([])  # nan != nan
    assert bootstrap_ci([3.0]) == (3.0, 3.0)


def test_wilcoxon_detects_consistent_improvement_and_ties():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    b = [x + 1 for x in a]
    res = wilcoxon_paired(a, b)
    assert res["wins"] == 10 and res["losses"] == 0 and res["p_value"] < 0.01
    tie = wilcoxon_paired(a, a)
    assert tie["ties"] == 10 and tie["p_value"] == 1.0


def test_holm_is_monotone_and_bounded():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)
    assert all(0 <= p <= 1 for p in adj)
    assert adj[1] >= adj[2] - 1e-12 or adj[2] >= adj[0]


# -------------------------------------------------------------------- baselines

def test_baseline_rules():
    cands = [
        _cand("cheap_slow", 1.0, 4.0, 1.0, 4.0),
        _cand("balanced", 1.3, 2.0, 1.3, 2.0),
        _cand("fast_pricey", 3.0, 1.0, 3.0, 1.0),
    ]
    assert B.greedy_cheapest(cands, 2.0).plan_id == "cheap_slow"
    assert B.greedy_fastest(cands, 2.0).plan_id == "balanced"
    fb = B.greedy_fastest(cands, 0.5)
    assert fb.plan_id == "cheap_slow" and fb.infeasible_fallback
    assert B.oracle(cands, 2.0).plan_id == "balanced"
    assert B.oracle(cands, 10.0).plan_id == "fast_pricey"
    r1 = B.random_choice(cands, 2.0, seed=3, query_id="Q")
    r2 = B.random_choice(cands, 2.0, seed=3, query_id="Q")
    assert r1.plan_id == r2.plan_id


def test_oracle_never_uses_predictions():
    # predictions are wildly wrong; oracle must still pick by measurements
    cands = [_cand("a", 100.0, 100.0, 1.0, 1.0), _cand("b", 0.01, 0.01, 1.0, 5.0)]
    assert B.oracle(cands, 2.0).plan_id == "a"


def test_mocap_select_uses_robust_optimizer():
    from selector.conformal import ConformalQuantile

    def q(mult, metric):
        return ConformalQuantile(metric, 0.05, 0.95, 100, 95, math.log(mult), mult)

    cands = [
        _cand("cheap_slow", 1.0, 4.0, 1.0, 4.0),
        _cand("balanced", 1.3, 2.0, 1.3, 2.0),
        _cand("fast_pricey", 3.0, 1.0, 3.0, 1.0),
    ]
    tight = B.mocap_select(cands, 1.5, "Q", q(1.0, "cost"), q(1.0, "latency"))
    assert tight.plan_id == "balanced" and not tight.infeasible_fallback
    # a wide conformal bound excludes "balanced" (1.3 * 1.25 > 1.5)
    wide = B.mocap_select(cands, 1.5, "Q", q(1.25, "cost"), q(1.0, "latency"))
    assert wide.plan_id == "cheap_slow"
    point = B.mocap_select(cands, 1.5, "Q", q(1.25, "cost"), q(1.0, "latency"), use_conformal=False)
    assert point.plan_id == "balanced" and point.strategy == "mocap_point"


# --------------------------------------------------------------------- harness
def test_signed_quantile_corrects_overestimating_predictor():
    # predictor over-estimates by 4x with small noise
    rng = random.Random(0)
    preds = [4.0 * math.exp(rng.gauss(0, 0.05)) for _ in range(200)]
    acts = [1.0] * 200
    q = H._signed_quantile(preds, acts, 0.05, "cost")
    assert q.multiplier < 0.3
    covered = sum(a <= p * q.multiplier for p, a in zip(preds, acts)) / len(preds)
    assert covered >= 0.95

def test_signed_quantile_returns_infinity_when_rank_exceeds_sample():
    q = H._signed_quantile(
        [1.0] * 5,
        [1.1] * 5,
        0.05,
        "cost",
    )

    assert math.isinf(q.raw_quantile_q)
    assert math.isinf(q.multiplier)
    assert q.residuals_summary["rank_exceeds_n"] is True


def test_signed_quantile_rejects_invalid_delta():
    with pytest.raises(ValueError):
        H._signed_quantile([1.0], [1.1], 0.0, "cost")

    with pytest.raises(ValueError):
        H._signed_quantile([1.0], [1.1], 1.0, "cost")


def test_signed_quantile_rejects_empty_samples():
    with pytest.raises(ValueError):
        H._signed_quantile([], [], 0.05, "cost")


def test_signed_quantile_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        H._signed_quantile([1.0], [], 0.05, "cost")


def test_residual_mode_switch():
    H.set_residual_mode("absolute")
    try:
        assert H.RESIDUAL_MODE == "absolute"
    finally:
        H.set_residual_mode("signed")
    with pytest.raises(ValueError):
        H.set_residual_mode("both")


def test_loqo_calibration_excludes_the_query_itself():
    ms = _workload(6)
    # make Q0's predictions absurd; that must not affect Q0's own quantile
    for c in ms[0].candidates:
        c.pred_cost *= 1000.0
    qs = H.loqo_quantiles(ms, 0.05)
    assert set(qs) == {m.query_id for m in ms}
    assert qs["Q0"][0].multiplier == pytest.approx(
        H.calibrate(H.calibration_samples(ms[1:]), 0.05)[0].multiplier)


def test_evaluate_scores_all_strategies_and_mocap_beats_greedy_cheapest():
    ms = _workload(12)
    rows = H.evaluate(ms, [1.5], [0.05], seed=1)
    strategies = {r["strategy"] for r in rows}
    assert {"mocap", "mocap_point", "greedy_cheapest", "greedy_fastest", "random", "oracle",
            H.SPARK_DEFAULT_REFERENCE} <= strategies
    by = {(r["query_id"], r["strategy"]): r for r in rows}
    for m in ms:
        oracle = by[(m.query_id, "oracle")]
        assert oracle["oracle_regret"] == pytest.approx(0.0)
        assert by[(m.query_id, "mocap")]["latency_s"] <= by[(m.query_id, "greedy_cheapest")]["latency_s"]
    summary = {s["strategy"]: s for s in summarize_outcomes(rows, n_resamples=200)}
    assert summary["mocap"]["budget_violation_rate"] == 0.0
    assert summary["mocap"]["median_latency_s"] < summary["greedy_cheapest"]["median_latency_s"]
    tests = {t["baseline"]: t for t in paired_tests(rows, "mocap")}
    assert tests["greedy_cheapest"]["wins"] >= 10 and tests["greedy_cheapest"]["losses"] == 0
    assert tests["greedy_cheapest"]["p_value"] < 0.01


def test_perturbation_is_seen_by_predicted_strategies_only():
    ms = _workload(4)

    def inflate_fast(qid, pid, c, l):  # makes the fast plan look cheap
        return (0.1 * c, l) if pid.endswith("fast_pricey") else (c, l)

    rows = H.evaluate(ms, [1.5], [0.05], seed=1, perturb=inflate_fast,
                      quantile_source=ms, strategies=("greedy_fastest", "oracle"),
                      include_spark_defaults=False)
    by = {(r["query_id"], r["strategy"]): r for r in rows}
    assert by[("Q0", "greedy_fastest")]["plan_id"].endswith("fast_pricey")
    assert by[("Q0", "greedy_fastest")]["budget_violated"] is True
    assert by[("Q0", "oracle")]["plan_id"].endswith("balanced")


def test_measurement_json_round_trip(tmp_path):
    m = _workload(1)[0]
    path = tmp_path / "Q0.json"
    path.write_text(m.to_json())
    again = H.load_measurement(path)
    assert again.query_id == m.query_id
    assert again.candidates[1].actual_latency == pytest.approx(m.candidates[1].actual_latency)


def test_coverage_report_hits_target_on_unbiased_noise():
    rng = random.Random(3)
    ms = []
    for q in range(30):
        cands = []
        for p in range(6):
            cost = rng.uniform(1, 5)
            err = math.exp(rng.gauss(0, 0.3))
            cands.append(_cand(f"Q{q}_P{p}", cost * err, cost * err, cost, cost))
        ms.append(_measurement(f"Q{q}", cands))
    rep = H.coverage_report(ms, 0.10)
    assert rep["plans"] == 180
    assert 0.85 <= rep["conformal_coverage"] <= 0.97
    assert rep["point_coverage"] < rep["conformal_coverage"]


# ------------------------------------------------------------------------ cloud

def test_cloud_profiles_and_correlation():
    dp = load_profile("dataproc", "n2-standard-8")
    emr = load_profile("emr", "m5.xlarge")
    assert dp.run_cost(3600 * 8) == pytest.approx(dp.compute_per_hour + dp.service_per_hour)
    assert emr.per_vcpu_second > 0 and emr.run_cost(None) is None
    res = spearman([1, 2, 3, 4], [10, 20, 30, 45])
    assert res["spearman"] == pytest.approx(1.0)
    with pytest.raises(KeyError):
        load_profile("azure", "x")
