from types import SimpleNamespace

from mocap.execution.replanner import make_cost_replanner
from mocap.interfaces import SelectedPlan


def cand(pid, strategy="SortMergeJoin"):
    return SimpleNamespace(
        plan_id=pid, query_id="Q1", sql=f"SELECT {pid}",
        strategy=strategy, actual_join_strategy=strategy,
    )


def met(cost, latency=1.0):
    return SimpleNamespace(estimated_cost=cost, estimated_latency=latency)


def current(pid="P1", cost=0.5):
    return SelectedPlan(
        plan_id=pid, query_id="Q1", selected_strategy="SortMergeJoin",
        expected_cost=cost, expected_latency=1.0, selection_reason="t",
        physical_plan_sql="SELECT 1", budget=0.1, deadline=9.0,
    )


def test_picks_cheapest_cheaper_candidate_and_keeps_constraints():
    cands = [cand("P1"), cand("P2"), cand("P3"), cand("P4")]
    metrics = {"P1": met(0.5), "P2": met(0.3), "P3": met(0.1), "P4": met(0.9)}
    replan = make_cost_replanner(cands, metrics)

    new = replan(current())
    assert new.plan_id == "P3"
    assert new.expected_cost == 0.1
    assert new.budget == 0.1 and new.deadline == 9.0
    assert new.physical_plan_sql == "SELECT P3"


def test_never_returns_a_plan_twice():
    cands = [cand("P1"), cand("P2"), cand("P3")]
    metrics = {"P1": met(0.5), "P2": met(0.3), "P3": met(0.1)}
    replan = make_cost_replanner(cands, metrics)

    first = replan(current())
    assert first.plan_id == "P3"
    second = replan(first)          # P3 has nothing cheaper left
    assert second is None
    third = replan(current())       # P1 again: P3 already tried -> P2
    assert third.plan_id == "P2"


def test_none_when_no_cheaper_candidate():
    replan = make_cost_replanner([cand("P1"), cand("P2")],
                                 {"P1": met(0.1), "P2": met(0.2)})
    assert replan(current(cost=0.1)) is None
