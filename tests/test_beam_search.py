import pytest

from mocap.plans.beam_search import beam_search_join_orders


def test_returns_only_complete_connected_orders():
    aliases = ("a", "b", "c", "d")
    adjacency = {
        "a": {"b"},
        "b": {"a", "c"},
        "c": {"b", "d"},
        "d": {"c"},
    }

    orders = beam_search_join_orders(
        aliases,
        adjacency,
        score_prefix=lambda order: float(len(order)),
        beam_width=20,
    )

    assert orders
    assert len(orders) == len(set(orders))

    for order in orders:
        assert len(order) == len(aliases)
        assert set(order) == set(aliases)

        joined = {order[0]}

        for alias in order[1:]:
            assert adjacency[alias] & joined
            joined.add(alias)


def test_respects_beam_width():
    aliases = ("a", "b", "c", "d", "e", "f")
    adjacency = {
        alias: set(aliases) - {alias}
        for alias in aliases
    }

    orders = beam_search_join_orders(
        aliases,
        adjacency,
        score_prefix=lambda _order: 0.0,
        beam_width=3,
    )

    assert orders
    assert len(orders) <= 3
    assert all(len(order) == len(aliases) for order in orders)


def test_scorer_changes_order_ranking():
    aliases = ("a", "b", "c")
    adjacency = {
        alias: set(aliases) - {alias}
        for alias in aliases
    }
    weights = {"a": 100.0, "b": 10.0, "c": 1.0}

    def score(order):
        return sum(
            position * weights[alias]
            for position, alias in enumerate(order, start=1)
        )

    orders = beam_search_join_orders(
        aliases,
        adjacency,
        score_prefix=score,
        beam_width=6,
    )

    assert orders[0] == ("a", "b", "c")


def test_disconnected_graph_returns_no_complete_order():
    orders = beam_search_join_orders(
        ("a", "b", "c"),
        {
            "a": {"b"},
            "b": {"a"},
            "c": set(),
        },
        score_prefix=lambda order: float(len(order)),
        beam_width=10,
    )

    assert orders == []


def test_rejects_invalid_beam_width():
    with pytest.raises(ValueError, match="beam_width"):
        beam_search_join_orders(
            ("a",),
            {"a": set()},
            score_prefix=lambda _order: 0.0,
            beam_width=0,
        )


def test_rejects_duplicate_aliases():
    with pytest.raises(ValueError, match="unique"):
        beam_search_join_orders(
            ("a", "a"),
            {"a": set()},
            score_prefix=lambda _order: 0.0,
        )


def test_rejects_unknown_neighbor():
    with pytest.raises(ValueError, match="unknown aliases"):
        beam_search_join_orders(
            ("a", "b"),
            {"a": {"missing"}, "b": set()},
            score_prefix=lambda _order: 0.0,
        )


def test_rejects_non_finite_score():
    with pytest.raises(ValueError, match="finite"):
        beam_search_join_orders(
            ("a",),
            {"a": set()},
            score_prefix=lambda _order: float("nan"),
        )
