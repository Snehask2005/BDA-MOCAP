"""
Comprehensive Unit Tests for MOCAP Selector Package.

Roshini (Lead: Robust Optimizer & Theory) — Phases 2 & 3 Test Suite

Validates:
    1. ConformalCalibrator: Log-space residuals, finite-sample quantile formula k = ceil((n+1)(1-delta)),
       multiplier calculation, query-grouped split (zero data leakage).
    2. ConformalBoundsComputer & PlanBounds: Certified cost & latency upper/lower bounds, margins.
    3. IntervalFilter: Chance-constrained admissibility (C_hat^+ <= B), rejection classification.
    4. ParetoExtractor: 2D Pareto frontier extraction, dominance rules, tie handling, latency < 50ms.
    5. DecisionEngine: Fastest Under Budget, Normalized Scalarization, Cheapest Feasible.
    6. Lagrangian Collapse Proof: Empirical verification that lambda * max(0, C_hat - B) == 0 on feasible plans.
    7. InfeasibleEngine: Three-mode fallback (Min Cost Over Budget, Min Budget Reporting, Formal Refusal).
    8. RobustOptimizer: Unified end-to-end chance-constrained pipeline.
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import pytest

from selector.bounds import (
    ConformalBoundsComputer,
    FilterRejectionReason,
    IntervalFilter,
    IntervalFilterSummary,
    PlanBounds,
)
from selector.conformal import (
    CalibrationSample,
    ConformalCalibrator,
    ConformalQuantile,
)
from selector.decision import (
    DecisionEngine,
    DecisionOutcome,
    DecisionPolicyType,
    DecisionScore,
)
from selector.infeasible import (
    InfeasibleDiagnostics,
    InfeasibleEngine,
    InfeasibleMode,
    InfeasibleResolution,
)
from selector.pareto import (
    ParetoExtractor,
    ParetoFrontierResult,
    ParetoPlan,
)
from selector.robust_optimizer import (
    RobustOptimizer,
    RobustSelectionResult,
)


# ===========================================================================
# Fixtures & Helpers
# ===========================================================================

@pytest.fixture
def sample_calibrator() -> ConformalCalibrator:
    return ConformalCalibrator()


@pytest.fixture
def mock_calibration_samples() -> List[CalibrationSample]:
    samples = []
    # 5 queries, 4 plans each
    for q_idx in range(1, 6):
        q_id = f"Q{q_idx}"
        for p_idx in range(1, 5):
            p_id = f"{q_id}_P{p_idx}"
            pred_cost = 10.0 * p_idx
            # Actual cost with controlled multiplicative error: 1.1x, 1.2x, 0.9x, etc.
            mult = 1.0 + (p_idx * 0.05)
            act_cost = pred_cost * mult
            pred_lat = 5.0 * p_idx
            act_lat = pred_lat * mult
            samples.append(
                CalibrationSample(
                    query_id=q_id,
                    plan_id=p_id,
                    predicted_cost=pred_cost,
                    actual_cost=act_cost,
                    predicted_latency=pred_lat,
                    actual_latency=act_lat,
                )
            )
    return samples


def _make_plan_bounds(
    plan_id: str,
    point_cost: float,
    point_latency: float,
    multiplier_cost: float = 1.20,
    multiplier_latency: float = 1.15,
    query_id: str = "TEST_Q",
) -> PlanBounds:
    return PlanBounds(
        plan_id=plan_id,
        query_id=query_id,
        point_cost=point_cost,
        point_latency=point_latency,
        conformal_cost_upper=point_cost * multiplier_cost,
        conformal_latency_upper=point_latency * multiplier_latency,
        conformal_cost_lower=point_cost / multiplier_cost,
        conformal_latency_lower=point_latency / multiplier_latency,
        delta_cost=0.05,
        delta_latency=0.05,
        cost_multiplier=multiplier_cost,
        latency_multiplier=multiplier_latency,
    )


# ===========================================================================
# 1. Conformal Calibration Tests (Phase 2)
# ===========================================================================

def test_residual_calculation(sample_calibrator: ConformalCalibrator):
    pred = [10.0, 20.0, 100.0]
    act = [10.0, 24.0, 80.0]
    residuals = sample_calibrator.compute_residuals(pred, act)

    assert len(residuals) == 3
    assert pytest.approx(residuals[0], abs=1e-6) == 0.0
    assert pytest.approx(residuals[1], abs=1e-4) == abs(math.log(24.0 / 20.0))
    assert pytest.approx(residuals[2], abs=1e-4) == abs(math.log(80.0 / 100.0))


def test_residual_length_mismatch(sample_calibrator: ConformalCalibrator):
    with pytest.raises(ValueError, match="Length mismatch"):
        sample_calibrator.compute_residuals([1.0], [1.0, 2.0])


def test_finite_sample_quantile_formula(sample_calibrator: ConformalCalibrator):
    # n = 99 residuals
    residuals = [0.01 * i for i in range(1, 100)]
    # delta = 0.05 => 1 - delta = 0.95
    # k = ceil((99 + 1) * 0.95) = ceil(95.0) = 95
    # 0-indexed position: 94
    q_result = sample_calibrator.compute_quantile(residuals, delta=0.05, metric_name="cost")

    assert q_result.confidence_level == 0.95
    assert q_result.sample_size == 99
    assert q_result.quantile_index == 94
    assert pytest.approx(q_result.raw_quantile_q, abs=1e-4) == 0.95
    assert pytest.approx(q_result.multiplier, abs=1e-4) == math.exp(0.95)


def test_query_grouped_split_no_leakage(
    sample_calibrator: ConformalCalibrator,
    mock_calibration_samples: List[CalibrationSample],
):
    train, cal = sample_calibrator.split_query_grouped(
        mock_calibration_samples, cal_fraction=0.4, seed=42
    )

    train_queries = set(s.query_id for s in train)
    cal_queries = set(s.query_id for s in cal)

    # Disjoint query sets (strict no-leakage invariant)
    assert train_queries.isdisjoint(cal_queries)
    assert train_queries.union(cal_queries) == set(s.query_id for s in mock_calibration_samples)


def test_quantile_edge_cases(sample_calibrator: ConformalCalibrator):
    with pytest.raises(ValueError, match="empty"):
        sample_calibrator.compute_quantile([], delta=0.05)

    with pytest.raises(ValueError, match="delta must be in"):
        sample_calibrator.compute_quantile([0.1, 0.2], delta=1.5)

    # Very small sample size where ceil((n+1)(1-delta)) > n
    # n=2, delta=0.05 => ceil(3 * 0.95) = 3 > 2 -> should clamp or use max
    q_small = sample_calibrator.compute_quantile([0.1, 0.2], delta=0.05)
    assert q_small.quantile_index <= 2
    assert q_small.raw_quantile_q == 0.2


# ===========================================================================
# 2. Plan Bounds & Interval Filter Tests (Phase 2 & 3)
# ===========================================================================

def test_plan_bounds_computations():
    p = _make_plan_bounds("P1", point_cost=10.0, point_latency=20.0, multiplier_cost=1.25)
    assert p.conformal_cost_upper == 12.5
    assert p.is_cost_feasible(budget=13.0, use_conformal=True) is True
    assert p.is_cost_feasible(budget=12.0, use_conformal=True) is False
    # Under point-prediction, budget=12.0 is feasible
    assert p.is_cost_feasible(budget=12.0, use_conformal=False) is True
    assert pytest.approx(p.cost_margin(15.0, use_conformal=True)) == 2.5


def test_interval_filter_admissibility_and_reasons():
    plans = [
        _make_plan_bounds("P_PASS", point_cost=10.0, point_latency=5.0),    # C^+ = 12.0, L^+ = 5.75
        _make_plan_bounds("P_FAIL_COST", point_cost=20.0, point_latency=5.0), # C^+ = 24.0
        _make_plan_bounds("P_FAIL_LAT", point_cost=10.0, point_latency=50.0), # L^+ = 57.5
        _make_plan_bounds("P_FAIL_BOTH", point_cost=20.0, point_latency=50.0),# Both fail
    ]

    filter_engine = IntervalFilter(use_conformal=True)
    summary = filter_engine.filter_plans(plans=plans, budget=15.0, deadline=20.0)

    assert summary.total_candidates == 4
    assert summary.admitted_count == 1
    assert summary.admitted_plans[0].plan_id == "P_PASS"

    # Verify rejection classification
    reasons = {r.plan.plan_id: r.rejection_reason for r in summary.rejection_records}
    assert reasons["P_PASS"] == FilterRejectionReason.PASSED
    assert reasons["P_FAIL_COST"] == FilterRejectionReason.BUDGET_EXCEEDED
    assert reasons["P_FAIL_LAT"] == FilterRejectionReason.DEADLINE_EXCEEDED
    assert reasons["P_FAIL_BOTH"] == FilterRejectionReason.BOTH_EXCEEDED


# ===========================================================================
# 3. 2D Pareto Frontier Extraction Tests (Phase 3)
# ===========================================================================

def test_pareto_dominance_2d():
    plans = [
        _make_plan_bounds("P1", point_cost=10.0, point_latency=50.0), # C^+=12, L=50 (Pareto optimal)
        _make_plan_bounds("P2", point_cost=20.0, point_latency=20.0), # C^+=24, L=20 (Pareto optimal)
        _make_plan_bounds("P3", point_cost=30.0, point_latency=10.0), # C^+=36, L=10 (Pareto optimal)
        _make_plan_bounds("P4_DOM", point_cost=25.0, point_latency=30.0), # Dominated by P2 (24 <= 30 and 20 <= 30)
        _make_plan_bounds("P5_DOM", point_cost=40.0, point_latency=60.0), # Dominated by all
    ]

    extractor = ParetoExtractor(use_conformal_cost=True)
    result = extractor.extract_frontier(plans)

    assert result.total_candidates == 5
    assert result.frontier_count == 3
    assert result.dominated_count == 2

    frontier_ids = [p.plan.plan_id for p in result.frontier]
    assert "P1" in frontier_ids
    assert "P2" in frontier_ids
    assert "P3" in frontier_ids
    assert "P4_DOM" not in frontier_ids
    assert "P5_DOM" not in frontier_ids


def test_pareto_tie_and_single_plan():
    p1 = _make_plan_bounds("P1", point_cost=10.0, point_latency=20.0)
    p2 = _make_plan_bounds("P2", point_cost=10.0, point_latency=20.0) # Identical duplicate

    extractor = ParetoExtractor()
    result = extractor.extract_frontier([p1, p2])

    assert result.frontier_count == 1
    assert len(result.frontier[0].equivalent_plans) == 1

    # Single plan edge case
    single_res = extractor.extract_frontier([p1])
    assert single_res.frontier_count == 1
    assert single_res.frontier[0].plan.plan_id == "P1"


def test_pareto_performance_under_50ms():
    # Generate 150 candidate plans
    plans = [
        _make_plan_bounds(f"P{i}", point_cost=10.0 + (i % 20), point_latency=100.0 - (i % 25))
        for i in range(150)
    ]
    extractor = ParetoExtractor()

    t0 = time.perf_counter()
    result = extractor.extract_frontier(plans)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50.0  # Acceptance criteria: planning latency < 50ms
    assert result.frontier_count > 0


# ===========================================================================
# 4. Decision Engine & Inert Lagrangian Collapse Proof (Phase 3)
# ===========================================================================

def test_fastest_under_budget_policy():
    plans = [
        _make_plan_bounds("CHEAP_SLOW", point_cost=10.0, point_latency=80.0), # C^+=12, L=80
        _make_plan_bounds("MID_MID", point_cost=15.0, point_latency=40.0),     # C^+=18, L=40
        _make_plan_bounds("EXP_FAST", point_cost=20.0, point_latency=15.0),    # C^+=24, L=15
    ]
    frontier = [ParetoPlan(plan=p) for p in plans]

    engine = DecisionEngine(policy=DecisionPolicyType.FASTEST_UNDER_BUDGET)
    outcome = engine.select(frontier=frontier, budget=25.0)

    # Both are under budget 25.0, FASTEST_UNDER_BUDGET must choose EXP_FAST
    assert outcome.selected_plan.plan_id == "EXP_FAST"
    assert outcome.policy == DecisionPolicyType.FASTEST_UNDER_BUDGET


def test_normalized_scalarization_tradeoff():
    p1 = _make_plan_bounds("LOW_COST", point_cost=10.0, point_latency=100.0)
    p2 = _make_plan_bounds("LOW_LAT", point_cost=50.0, point_latency=10.0)
    frontier = [ParetoPlan(plan=p1), ParetoPlan(plan=p2)]

    # Cost-heavy preference (alpha=0.1 latency, beta=0.9 cost) -> Should choose LOW_COST
    cost_engine = DecisionEngine(
        policy=DecisionPolicyType.NORMALIZED_SCALARIZATION, alpha=0.1, beta=0.9
    )
    outcome_cost = cost_engine.select(frontier, budget=100.0)
    assert outcome_cost.selected_plan.plan_id == "LOW_COST"

    # Latency-heavy preference (alpha=0.9 latency, beta=0.1 cost) -> Should choose LOW_LAT
    lat_engine = DecisionEngine(
        policy=DecisionPolicyType.NORMALIZED_SCALARIZATION, alpha=0.9, beta=0.1
    )
    outcome_lat = lat_engine.select(frontier, budget=100.0)
    assert outcome_lat.selected_plan.plan_id == "LOW_LAT"


def test_lagrangian_penalty_collapse_proof():
    """
    Mathematical proof validation:
    For all feasible plans entering the decision engine, C_hat <= B.
    Therefore, max(0, C_hat - B) == 0 identically.
    Comparing lambda = 0 vs lambda = 10 must produce strictly identical scores
    and identical plan selection, demonstrating mathematical inertness of lambda.
    """
    plans = [
        _make_plan_bounds("P1", point_cost=10.0, point_latency=30.0),
        _make_plan_bounds("P2", point_cost=15.0, point_latency=20.0),
    ]
    frontier = [ParetoPlan(plan=p) for p in plans]

    # Evaluate with lambda = 0
    engine_lam0 = DecisionEngine(
        policy=DecisionPolicyType.LEGACY_LAGRANGIAN_ABLATION, alpha=0.5, beta=0.5, lagrangian_lambda=0.0
    )
    res_lam0 = engine_lam0.select(frontier=frontier, budget=25.0)

    # Evaluate with lambda = 10
    engine_lam10 = DecisionEngine(
        policy=DecisionPolicyType.LEGACY_LAGRANGIAN_ABLATION, alpha=0.5, beta=0.5, lagrangian_lambda=10.0
    )
    res_lam10 = engine_lam10.select(frontier=frontier, budget=25.0)

    assert res_lam0.selected_plan.plan_id == res_lam10.selected_plan.plan_id
    for s0, s10 in zip(res_lam0.ranked_candidates, res_lam10.ranked_candidates):
        assert s0.lagrangian_penalty == 0.0
        assert s10.lagrangian_penalty == 0.0
        assert s0.final_score == s10.final_score


# ===========================================================================
# 5. Three-Mode Infeasible Fallback Engine Tests (Phase 3)
# ===========================================================================

def test_infeasible_mode_1_min_cost_over_budget():
    plans = [
        _make_plan_bounds("P_EXPENSIVE", point_cost=50.0, point_latency=10.0), # C^+=60.0
        _make_plan_bounds("P_CHEAPER", point_cost=30.0, point_latency=30.0),   # C^+=36.0
    ]
    engine = InfeasibleEngine(mode=InfeasibleMode.MIN_COST_OVER_BUDGET)
    res = engine.resolve(all_candidates=plans, budget=20.0, query_id="Q_INFEAS")

    assert res.mode == InfeasibleMode.MIN_COST_OVER_BUDGET
    assert res.is_refusal is False
    assert res.fallback_plan is not None
    assert res.fallback_plan.plan_id == "P_CHEAPER"
    assert pytest.approx(res.overrun_amount) == 16.0  # 36.0 - 20.0
    assert res.diagnostics.min_budget_required == 36.0


def test_infeasible_mode_2_min_budget_reporting():
    plans = [
        _make_plan_bounds("P1", point_cost=40.0, point_latency=20.0), # C^+=48.0
    ]
    engine = InfeasibleEngine(mode=InfeasibleMode.MIN_BUDGET_REPORTING)
    res = engine.resolve(all_candidates=plans, budget=20.0, query_id="Q_INFEAS")

    assert res.mode == InfeasibleMode.MIN_BUDGET_REPORTING
    assert res.is_refusal is True
    assert res.fallback_plan is None
    assert res.diagnostics.min_budget_required == 48.0
    assert pytest.approx(res.diagnostics.recommended_budget_with_slack) == 48.0 * 1.05


def test_infeasible_mode_3_formal_refusal():
    plans = [
        _make_plan_bounds("P1", point_cost=50.0, point_latency=20.0),
    ]
    engine = InfeasibleEngine(mode=InfeasibleMode.FORMAL_REFUSAL)
    res = engine.resolve(all_candidates=plans, budget=20.0, query_id="Q_INFEAS")

    assert res.mode == InfeasibleMode.FORMAL_REFUSAL
    assert res.is_refusal is True
    assert res.fallback_plan is None
    assert "Infeasible Formal Refusal" in res.message


def test_infeasible_engine_zero_candidates():
    engine = InfeasibleEngine(mode=InfeasibleMode.MIN_COST_OVER_BUDGET)
    res = engine.resolve(all_candidates=[], budget=20.0, query_id="Q_EMPTY")

    assert res.is_refusal is True
    assert res.fallback_plan is None
    assert "Zero candidates provided" in res.message


# ===========================================================================
# 6. Unified Robust Optimizer End-to-End Tests
# ===========================================================================

def test_robust_optimizer_feasible_execution():
    optimizer = RobustOptimizer(
        cost_multiplier=1.20,
        latency_multiplier=1.10,
        decision_policy=DecisionPolicyType.FASTEST_UNDER_BUDGET,
    )

    candidates = [
        ("PLAN_A", 10.0, 50.0, None), # C^+=12.0, L=50
        ("PLAN_B", 15.0, 20.0, None), # C^+=18.0, L=20
        ("PLAN_C", 30.0, 10.0, None), # C^+=36.0, L=10 (exceeds budget 20.0)
    ]

    result = optimizer.select(
        candidate_items=candidates,
        budget=20.0,
        query_id="TEST_Q_FEASIBLE",
    )

    assert result.is_feasible is True
    assert result.candidate_count == 3
    assert result.feasible_count == 2
    assert result.selected_plan is not None
    assert result.selected_plan.plan_id == "PLAN_B"
    assert len(result.pareto_plans) == 2
    assert result.total_planning_time_ms < 50.0  # Planning latency < 50ms


def test_robust_optimizer_infeasible_fallback():
    optimizer = RobustOptimizer(
        cost_multiplier=1.20,
        decision_policy=DecisionPolicyType.FASTEST_UNDER_BUDGET,
        infeasible_mode=InfeasibleMode.MIN_COST_OVER_BUDGET,
    )

    candidates = [
        ("PLAN_A", 30.0, 50.0, None), # C^+=36.0
        ("PLAN_B", 40.0, 20.0, None), # C^+=48.0
    ]

    # Budget = 15.0, both plans violate budget
    result = optimizer.select(
        candidate_items=candidates,
        budget=15.0,
        query_id="TEST_Q_INFEASIBLE",
    )

    assert result.is_feasible is False
    assert result.feasible_count == 0
    assert result.selected_plan is not None
    assert result.selected_plan.plan_id == "PLAN_A"  # Min cost over budget
    assert result.infeasible_resolution is not None
    assert result.infeasible_resolution.overrun_amount == 21.0  # 36 - 15
