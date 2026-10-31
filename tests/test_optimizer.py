"""
Comprehensive tests for the MOCAP Core Optimizer.

Roshini (Member 3) — Core MOCAP Optimizer

Test coverage:
    - ConstraintsFilter: budget filtering, deadline filtering, infeasibility
      reasons, edge cases (zero candidates, missing metrics, boundary values)
    - ParetoSelector: dominance, ties, equivalence, single plan, all dominated
    - LagrangianRanker: score computation, normalization, weight validation,
      violation penalty, rank ordering, degenerate range
    - MOCAPSelector: end-to-end selection with all stages, ablation configs,
      infeasibility with rich failure reasons, SelectionResult correctness
    - Baselines: SparkDefaultBaseline, GreedyBudgetPruningBaseline
    - Ablations: all ablation variants instantiate and produce configs
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from mocap.optimizer.ablations import (
    ALL_ABLATIONS,
    AblationConfig,
    AblationVariant,
    get_ablation,
    list_ablations,
)
from mocap.optimizer.baselines import (
    BaselineResult,
    GreedyBudgetPruningBaseline,
    SparkDefaultBaseline,
)
from mocap.optimizer.constraints import ConstraintsFilter, InfeasibilityRecord
from mocap.optimizer.lagrangian import LagrangianRanker, ScoredPlan
from mocap.optimizer.pareto import ParetoSelector
from mocap.optimizer.selector import MOCAPSelector, SelectionResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _plan(
    plan_id: str,
    query_id: str = "Q1",
    strategy: str = "sort_merge_join",
    actual_join_strategy: Optional[str] = None,
    sql: str = "SELECT 1",
) -> SimpleNamespace:
    """Minimal CandidatePlan-like object."""
    p = SimpleNamespace(
        plan_id=plan_id,
        query_id=query_id,
        strategy=strategy,
        actual_join_strategy=actual_join_strategy,
        sql=sql,
    )
    return p


def _metrics(
    plan_id: str,
    query_id: str = "Q1",
    estimated_cost: float = 0.01,
    estimated_latency: float = 1.0,
) -> SimpleNamespace:
    """Minimal PlanMetrics-like object."""
    return SimpleNamespace(
        plan_id=plan_id,
        query_id=query_id,
        estimated_cost=estimated_cost,
        estimated_latency=estimated_latency,
    )


def _metrics_map(*items) -> Dict:
    """Build a plan_id -> metrics dict from (plan, metrics) pairs."""
    return {m.plan_id: m for _, m in items}


# ---------------------------------------------------------------------------
# ConstraintsFilter tests
# ---------------------------------------------------------------------------


class TestConstraintsFilter:

    def test_all_feasible_when_budget_generous(self):
        plans = [_plan("P1"), _plan("P2")]
        m = {
            "P1": _metrics("P1", estimated_cost=0.005),
            "P2": _metrics("P2", estimated_cost=0.009),
        }
        cf = ConstraintsFilter(budget=0.01)
        feasible = cf.filter_feasible_plans(plans, m)
        assert len(feasible) == 2
        assert not cf.infeasibility_log

    def test_budget_filtering_excludes_expensive_plans(self):
        plans = [_plan("P1"), _plan("P2"), _plan("P3")]
        m = {
            "P1": _metrics("P1", estimated_cost=0.005),
            "P2": _metrics("P2", estimated_cost=0.020),  # over budget
            "P3": _metrics("P3", estimated_cost=0.009),
        }
        cf = ConstraintsFilter(budget=0.01)
        feasible = cf.filter_feasible_plans(plans, m)
        assert len(feasible) == 2
        feasible_ids = {p.plan_id for p, _ in feasible}
        assert "P1" in feasible_ids
        assert "P3" in feasible_ids
        assert "P2" not in feasible_ids

    def test_infeasibility_reason_budget_exceeded(self):
        plans = [_plan("P1")]
        m = {"P1": _metrics("P1", estimated_cost=0.05)}
        cf = ConstraintsFilter(budget=0.01)
        cf.filter_feasible_plans(plans, m)
        assert len(cf.infeasibility_log) == 1
        assert cf.infeasibility_log[0].reason == "budget_exceeded"

    def test_deadline_filtering_excludes_slow_plans(self):
        plans = [_plan("P1"), _plan("P2")]
        m = {
            "P1": _metrics("P1", estimated_cost=0.005, estimated_latency=50.0),
            "P2": _metrics("P2", estimated_cost=0.005, estimated_latency=5.0),
        }
        cf = ConstraintsFilter(budget=1.0, deadline=10.0)
        feasible = cf.filter_feasible_plans(plans, m)
        assert len(feasible) == 1
        assert feasible[0][0].plan_id == "P2"
        assert cf.infeasibility_log[0].reason == "deadline_exceeded"

    def test_both_exceeded_reason(self):
        plans = [_plan("P1")]
        m = {"P1": _metrics("P1", estimated_cost=0.5, estimated_latency=999.0)}
        cf = ConstraintsFilter(budget=0.01, deadline=10.0)
        feasible = cf.filter_feasible_plans(plans, m)
        assert len(feasible) == 0
        assert cf.infeasibility_log[0].reason == "both_exceeded"

    def test_missing_metrics_recorded_as_no_metrics(self):
        plans = [_plan("P1"), _plan("P2")]
        m = {"P1": _metrics("P1")}
        cf = ConstraintsFilter(budget=1.0)
        feasible = cf.filter_feasible_plans(plans, m)
        assert len(feasible) == 1
        assert cf.infeasibility_log[0].reason == "no_metrics"

    def test_boundary_cost_equals_budget_is_feasible(self):
        plans = [_plan("P1")]
        m = {"P1": _metrics("P1", estimated_cost=0.01)}
        cf = ConstraintsFilter(budget=0.01)
        feasible = cf.filter_feasible_plans(plans, m)
        assert len(feasible) == 1

    def test_empty_candidate_list_returns_empty(self):
        cf = ConstraintsFilter(budget=1.0)
        feasible = cf.filter_feasible_plans([], {})
        assert feasible == []

    def test_infeasibility_summary_counts(self):
        plans = [_plan("P1"), _plan("P2"), _plan("P3"), _plan("P4")]
        m = {
            "P1": _metrics("P1", estimated_cost=0.5),        # budget exceeded
            "P2": _metrics("P2", estimated_latency=999.0),   # deadline exceeded
            "P3": _metrics("P3", estimated_cost=0.5, estimated_latency=999.0),  # both
            # P4 has no metrics
        }
        cf = ConstraintsFilter(budget=0.01, deadline=10.0)
        feasible = cf.filter_feasible_plans(plans, m)
        summary = cf.infeasibility_summary()
        assert summary["total_rejected"] == 4
        assert summary["budget_exceeded_count"] == 1
        assert summary["deadline_exceeded_count"] == 1
        assert summary["both_exceeded_count"] == 1
        assert summary["no_metrics_count"] == 1

    def test_failure_reason_budget_only(self):
        plans = [_plan("P1"), _plan("P2")]
        m = {
            "P1": _metrics("P1", estimated_cost=0.5),
            "P2": _metrics("P2", estimated_cost=1.0),
        }
        cf = ConstraintsFilter(budget=0.01)
        cf.filter_feasible_plans(plans, m)
        reason = cf.failure_reason(candidate_count=2, feasible_count=0)
        assert reason is not None
        assert "budget" in reason.lower()

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            ConstraintsFilter(budget=0)

    def test_invalid_deadline_raises(self):
        with pytest.raises(ValueError):
            ConstraintsFilter(budget=1.0, deadline=-5.0)


# ---------------------------------------------------------------------------
# ParetoSelector tests
# ---------------------------------------------------------------------------


class TestParetoSelector:

    def test_dominated_plan_excluded(self):
        """P2 dominates P1 (better cost AND latency)."""
        p1 = _plan("P1")
        p2 = _plan("P2")
        m1 = _metrics("P1", estimated_cost=0.10, estimated_latency=5.0)
        m2 = _metrics("P2", estimated_cost=0.05, estimated_latency=2.0)
        selector = ParetoSelector()
        front = selector.compute_front([(p1, m1), (p2, m2)])
        ids = {p.plan_id for p, _ in front}
        assert "P2" in ids
        assert "P1" not in ids

    def test_non_dominated_plans_both_on_front(self):
        """P1 cheaper but slower; P2 faster but more expensive — neither dominates."""
        p1 = _plan("P1")
        p2 = _plan("P2")
        m1 = _metrics("P1", estimated_cost=0.05, estimated_latency=5.0)
        m2 = _metrics("P2", estimated_cost=0.10, estimated_latency=2.0)
        selector = ParetoSelector()
        front = selector.compute_front([(p1, m1), (p2, m2)])
        ids = {p.plan_id for p, _ in front}
        assert "P1" in ids
        assert "P2" in ids

    def test_single_plan_is_pareto_optimal(self):
        p1 = _plan("P1")
        m1 = _metrics("P1")
        selector = ParetoSelector()
        front = selector.compute_front([(p1, m1)])
        assert len(front) == 1

    def test_empty_input_returns_empty(self):
        selector = ParetoSelector()
        assert selector.compute_front([]) == []

    def test_tied_plans_both_on_front(self):
        """Two plans with identical cost and latency — neither dominates."""
        p1 = _plan("P1")
        p2 = _plan("P2")
        m1 = _metrics("P1", estimated_cost=0.05, estimated_latency=3.0)
        m2 = _metrics("P2", estimated_cost=0.05, estimated_latency=3.0)
        selector = ParetoSelector()
        front = selector.compute_front([(p1, m1), (p2, m2)])
        ids = {p.plan_id for p, _ in front}
        assert "P1" in ids
        assert "P2" in ids
        assert len(selector.equivalent_groups) == 1

    def test_three_way_dominance_chain(self):
        """P3 dominates P2 which dominates P1 — only P3 on front."""
        plans_metrics = [
            (_plan("P1"), _metrics("P1", estimated_cost=0.30, estimated_latency=30.0)),
            (_plan("P2"), _metrics("P2", estimated_cost=0.20, estimated_latency=20.0)),
            (_plan("P3"), _metrics("P3", estimated_cost=0.10, estimated_latency=10.0)),
        ]
        selector = ParetoSelector()
        front = selector.compute_front(plans_metrics)
        assert len(front) == 1
        assert front[0][0].plan_id == "P3"
        assert len(selector.dominated_plans) == 2

    def test_static_convenience_method(self):
        p1 = _plan("P1")
        m1 = _metrics("P1", estimated_cost=0.05, estimated_latency=5.0)
        front = ParetoSelector.static_pareto([(p1, m1)])
        assert len(front) == 1

    def test_dominated_count_in_summary(self):
        plans_metrics = [
            (_plan("P1"), _metrics("P1", estimated_cost=0.30, estimated_latency=30.0)),
            (_plan("P2"), _metrics("P2", estimated_cost=0.10, estimated_latency=10.0)),
        ]
        selector = ParetoSelector()
        selector.compute_front(plans_metrics)
        summary = selector.dominance_summary()
        assert summary["dominated_count"] == 1

    def test_backward_compat_classmethod(self):
        """ParetoSelector.get_pareto_optimal(feasible) must still work."""
        p1 = _plan("P1")
        m1 = _metrics("P1", estimated_cost=0.05, estimated_latency=5.0)
        front = ParetoSelector.get_pareto_optimal([(p1, m1)])
        assert len(front) == 1


# ---------------------------------------------------------------------------
# LagrangianRanker tests
# ---------------------------------------------------------------------------


class TestLagrangianRanker:

    def test_lower_score_wins(self):
        """Plan with lower normalized cost+latency should rank first."""
        p1 = _plan("P1")
        p2 = _plan("P2")
        m1 = _metrics("P1", estimated_cost=0.10, estimated_latency=5.0)
        m2 = _metrics("P2", estimated_cost=0.02, estimated_latency=1.0)
        ranker = LagrangianRanker()
        ranked = ranker.rank_candidates([(p1, m1), (p2, m2)], budget=1.0)
        # P2 has lower cost and latency, so it wins.
        assert ranked[0][0].plan_id == "P2"

    def test_empty_input_returns_empty(self):
        ranker = LagrangianRanker()
        assert ranker.rank_candidates([], budget=1.0) == []

    def test_single_plan_ranked_first(self):
        p1 = _plan("P1")
        m1 = _metrics("P1")
        ranker = LagrangianRanker()
        ranked = ranker.rank_candidates([(p1, m1)], budget=1.0)
        assert ranked[0][0].plan_id == "P1"

    def test_degenerate_range_does_not_crash(self):
        """All plans have identical cost and latency — score should be 0."""
        plans_metrics = [
            (_plan("P1"), _metrics("P1", estimated_cost=0.05, estimated_latency=3.0)),
            (_plan("P2"), _metrics("P2", estimated_cost=0.05, estimated_latency=3.0)),
        ]
        ranker = LagrangianRanker()
        ranked = ranker.rank_candidates(plans_metrics, budget=1.0)
        assert len(ranked) == 2
        # Both should score 0.0
        assert ranked[0][2] == pytest.approx(0.0)

    def test_alpha_beta_weights_affect_ranking(self):
        """alpha=1/beta=0 should prefer low-latency; alpha=0/beta=1 low-cost."""
        p1 = _plan("P1")
        p2 = _plan("P2")
        m1 = _metrics("P1", estimated_cost=0.02, estimated_latency=10.0)  # cheap, slow
        m2 = _metrics("P2", estimated_cost=0.10, estimated_latency=1.0)   # expensive, fast

        latency_first = LagrangianRanker(alpha=1.0, beta=0.0)
        ranked = latency_first.rank_candidates([(p1, m1), (p2, m2)], budget=1.0)
        assert ranked[0][0].plan_id == "P2"  # latency wins

        cost_first = LagrangianRanker(alpha=0.0, beta=1.0)
        ranked = cost_first.rank_candidates([(p1, m1), (p2, m2)], budget=1.0)
        assert ranked[0][0].plan_id == "P1"  # cost wins

    def test_violation_penalty_inflates_score(self):
        """Plan exceeding budget should score higher (worse) with lambda > 0."""
        p1 = _plan("P1")
        p2 = _plan("P2")
        m1 = _metrics("P1", estimated_cost=0.20, estimated_latency=1.0)  # over budget
        m2 = _metrics("P2", estimated_cost=0.01, estimated_latency=5.0)

        budget = 0.05
        ranker_no_lambda = LagrangianRanker(lambda_mult=0.0)
        ranker_with_lambda = LagrangianRanker(lambda_mult=100.0)

        ranked_no = ranker_no_lambda.rank_candidates([(p1, m1), (p2, m2)], budget)
        ranked_with = ranker_with_lambda.rank_candidates([(p1, m1), (p2, m2)], budget)

        # With high lambda, P1 (budget violator) should rank last.
        assert ranked_with[-1][0].plan_id == "P1"

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            LagrangianRanker(alpha=-0.1)

    def test_describe_returns_dict(self):
        ranker = LagrangianRanker(alpha=0.3, beta=0.7)
        desc = ranker.describe()
        assert desc["alpha"] == 0.3
        assert desc["beta"] == 0.7
        assert "objective" in desc

    def test_rank_with_details_returns_scored_plans(self):
        p1 = _plan("P1")
        m1 = _metrics("P1", estimated_cost=0.05, estimated_latency=3.0)
        ranker = LagrangianRanker()
        details = ranker.rank_with_details([(p1, m1)], budget=1.0)
        assert len(details) == 1
        assert isinstance(details[0], ScoredPlan)
        assert details[0].plan_id == "P1"

    def test_normalize_min_equals_max_returns_zero(self):
        assert LagrangianRanker._normalize(5.0, 5.0, 5.0) == 0.0

    def test_normalize_range(self):
        assert LagrangianRanker._normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# MOCAPSelector (end-to-end) tests
# ---------------------------------------------------------------------------


class TestMOCAPSelector:

    def _make_candidates_and_metrics(
        self,
        n: int = 3,
        base_cost: float = 0.005,
        base_latency: float = 2.0,
    ):
        candidates = []
        metrics = {}
        for i in range(n):
            pid = f"P{i+1}"
            c = _plan(pid)
            m = _metrics(
                pid,
                estimated_cost=base_cost * (i + 1),
                estimated_latency=base_latency * (i + 1),
            )
            candidates.append(c)
            metrics[pid] = m
        return candidates, metrics

    def test_selects_best_plan(self):
        candidates, metrics = self._make_candidates_and_metrics(n=3, base_cost=0.002)
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
        )
        assert result.selected_plan is not None
        assert result.selected_plan.plan_id in [c.plan_id for c in candidates]
        assert result.selected_plan.expected_cost <= 1.0

    def test_no_feasible_returns_failure(self):
        candidates, metrics = self._make_candidates_and_metrics(
            n=2, base_cost=0.5
        )
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=0.0001,
        )
        assert result.selected_plan is None
        assert result.failure_reason is not None
        assert result.feasible_count == 0

    def test_selected_plan_cost_within_budget(self):
        candidates, metrics = self._make_candidates_and_metrics(
            n=4, base_cost=0.002
        )
        budget = 0.006
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=budget,
        )
        if result.selected_plan:
            assert result.selected_plan.expected_cost <= budget

    def test_stage_counts_are_consistent(self):
        candidates, metrics = self._make_candidates_and_metrics(n=3, base_cost=0.002)
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
        )
        assert result.candidate_count == 3
        assert result.feasible_count <= result.candidate_count
        assert result.pareto_count <= result.feasible_count

    def test_timing_fields_are_non_negative(self):
        candidates, metrics = self._make_candidates_and_metrics(n=2)
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
        )
        assert result.constraint_time_s >= 0
        assert result.pareto_time_s >= 0
        assert result.ranking_time_s >= 0
        assert result.total_time_s >= 0

    def test_to_dict_serializable(self):
        candidates, metrics = self._make_candidates_and_metrics(n=2)
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "query_id" in d
        assert "budget" in d
        assert "selected_plan_id" in d

    def test_empty_candidates_returns_failure(self):
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=[],
            metrics={},
            budget=1.0,
        )
        assert result.selected_plan is None
        assert result.candidate_count == 0

    def test_deadline_filtering_applied(self):
        candidates = [_plan("P1"), _plan("P2")]
        metrics = {
            "P1": _metrics("P1", estimated_cost=0.01, estimated_latency=100.0),
            "P2": _metrics("P2", estimated_cost=0.01, estimated_latency=5.0),
        }
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
            deadline=10.0,
        )
        assert result.selected_plan is not None
        assert result.selected_plan.plan_id == "P2"

    def test_ablation_no_pareto(self):
        """No-Pareto ablation should still return a valid selection."""
        from mocap.optimizer.ablations import ALL_ABLATIONS, AblationVariant
        candidates, metrics = self._make_candidates_and_metrics(n=3)
        cfg = ALL_ABLATIONS[AblationVariant.NO_PARETO]
        selector = MOCAPSelector(ablation=cfg)
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
        )
        assert result.selected_plan is not None
        assert result.ablation_variant == "no_pareto"

    def test_ablation_no_lagrangian(self):
        """No-Lagrangian ablation: falls back to min-cost selection."""
        from mocap.optimizer.ablations import ALL_ABLATIONS, AblationVariant
        candidates, metrics = self._make_candidates_and_metrics(n=3, base_cost=0.002)
        cfg = ALL_ABLATIONS[AblationVariant.NO_LAGRANGIAN]
        selector = MOCAPSelector(ablation=cfg)
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
        )
        assert result.selected_plan is not None

    def test_accuracy_tolerance_passed_through(self):
        candidates, metrics = self._make_candidates_and_metrics(n=1)
        selector = MOCAPSelector()
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
            accuracy_tolerance=5.0,
        )
        if result.selected_plan:
            assert result.selected_plan.accuracy_tolerance == 5.0

    def test_force_accuracy_tolerance_ablation(self):
        """NO_ADAPTIVE ablation should override accuracy_tolerance to 0."""
        from mocap.optimizer.ablations import ALL_ABLATIONS, AblationVariant
        candidates, metrics = self._make_candidates_and_metrics(n=1)
        cfg = ALL_ABLATIONS[AblationVariant.NO_ADAPTIVE]
        selector = MOCAPSelector(ablation=cfg)
        result = selector.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=1.0,
            accuracy_tolerance=20.0,  # should be overridden to 0.0
        )
        if result.selected_plan:
            assert result.selected_plan.accuracy_tolerance == 0.0


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------


class TestBaselines:

    def _candidates_and_metrics(self):
        candidates = [
            _plan("P1", strategy="broadcast_join"),
            _plan("P2", strategy="sort_merge_join"),
            _plan("P3", strategy="shuffle_hash_join"),
        ]
        metrics = {
            "P1": _metrics("P1", estimated_cost=0.003, estimated_latency=2.0),
            "P2": _metrics("P2", estimated_cost=0.010, estimated_latency=1.0),
            "P3": _metrics("P3", estimated_cost=0.007, estimated_latency=1.5),
        }
        return candidates, metrics

    # SparkDefaultBaseline

    def test_spark_default_picks_min_cost(self):
        candidates, metrics = self._candidates_and_metrics()
        baseline = SparkDefaultBaseline()
        result = baseline.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
        )
        assert result.selected_plan_id == "P1"  # lowest cost = 0.003
        assert result.baseline_name == "spark_default"

    def test_spark_default_no_candidates(self):
        baseline = SparkDefaultBaseline()
        result = baseline.select(
            query_id="Q1",
            candidates=[],
            metrics={},
        )
        assert result.selected_plan_id is None
        assert result.failure_reason is not None

    def test_spark_default_ignores_budget(self):
        """SparkDefaultBaseline should pick min-cost even if it exceeds budget."""
        candidates, metrics = self._candidates_and_metrics()
        baseline = SparkDefaultBaseline()
        result = baseline.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
            budget=0.001,  # all candidates exceed this
        )
        # Still selects min-cost P1 despite budget constraint.
        assert result.selected_plan_id == "P1"

    def test_spark_default_to_dict(self):
        candidates, metrics = self._candidates_and_metrics()
        baseline = SparkDefaultBaseline()
        result = baseline.select("Q1", candidates, metrics)
        d = result.to_dict()
        assert "baseline" in d
        assert "selected_plan" in d

    # GreedyBudgetPruningBaseline

    def test_greedy_picks_min_latency_within_budget(self):
        candidates, metrics = self._candidates_and_metrics()
        # Budget allows P1 (0.003) and P3 (0.007) but not P2 (0.010)
        baseline = GreedyBudgetPruningBaseline(budget=0.008)
        result = baseline.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
        )
        # Among feasible (P1, P3), P3 has lower latency (1.5 < 2.0)
        assert result.selected_plan_id == "P3"

    def test_greedy_returns_failure_when_no_feasible(self):
        candidates, metrics = self._candidates_and_metrics()
        baseline = GreedyBudgetPruningBaseline(budget=0.001)
        result = baseline.select(
            query_id="Q1",
            candidates=candidates,
            metrics=metrics,
        )
        assert result.selected_plan_id is None
        assert result.failure_reason is not None
        assert result.feasible_count == 0

    def test_greedy_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            GreedyBudgetPruningBaseline(budget=-1.0)

    def test_greedy_deadline_filtering(self):
        candidates, metrics = self._candidates_and_metrics()
        # P1 latency=2.0 > deadline=1.2, P3 latency=1.5 > deadline=1.2
        # Only P2 latency=1.0 <= 1.2 — but P2 cost 0.010 > budget 0.008
        baseline = GreedyBudgetPruningBaseline(budget=0.008, deadline=1.2)
        result = baseline.select("Q1", candidates, metrics)
        # No plan satisfies both budget and deadline
        assert result.selected_plan_id is None


# ---------------------------------------------------------------------------
# Ablation configuration tests
# ---------------------------------------------------------------------------


class TestAblations:

    def test_all_ablations_instantiate(self):
        """Every ablation variant must have a registered config."""
        for variant in AblationVariant:
            cfg = ALL_ABLATIONS[variant]
            assert isinstance(cfg, AblationConfig)

    def test_get_ablation_by_variant(self):
        cfg = get_ablation(AblationVariant.FULL_MOCAP)
        assert cfg.use_calibration is True
        assert cfg.use_pareto is True
        assert cfg.use_lagrangian is True

    def test_no_calibration_ablation(self):
        cfg = get_ablation(AblationVariant.NO_CALIBRATION)
        assert cfg.use_calibration is False

    def test_latency_only_ablation(self):
        cfg = get_ablation(AblationVariant.LATENCY_ONLY)
        assert cfg.alpha == 1.0
        assert cfg.beta == 0.0

    def test_cost_only_ablation(self):
        cfg = get_ablation(AblationVariant.COST_ONLY)
        assert cfg.alpha == 0.0
        assert cfg.beta == 1.0

    def test_high_lambda_ablation(self):
        cfg = get_ablation(AblationVariant.HIGH_LAMBDA)
        assert cfg.lambda_mult > 0

    def test_no_adaptive_ablation_sets_tolerance(self):
        cfg = get_ablation(AblationVariant.NO_ADAPTIVE)
        assert cfg.force_accuracy_tolerance == 0.0

    def test_list_ablations_returns_all(self):
        ablations = list_ablations()
        assert len(ablations) == len(AblationVariant)

    def test_ablation_to_dict(self):
        cfg = get_ablation(AblationVariant.FULL_MOCAP)
        d = cfg.to_dict()
        assert d["variant"] == "full_mocap"
        assert "use_calibration" in d
        assert "alpha" in d
