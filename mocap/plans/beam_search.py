from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence


JoinOrder = tuple[str, ...]
PrefixScorer = Callable[[JoinOrder], float]


def beam_search_join_orders(
    aliases: Sequence[str],
    adjacency: Mapping[str, Iterable[str]],
    *,
    score_prefix: PrefixScorer,
    beam_width: int = 20,
) -> list[JoinOrder]:
    """Generate bounded, valid left-deep join orders with beam search.

    The adjacency mapping represents an undirected join graph.
    A relation can be appended only if it connects to a relation
    already present in the partial join order.

    Lower scores are preferred. The scorer is injected so that it
    can later use real table statistics and join-selectivity estimates.

    Only complete orders are returned. Ties are resolved
    lexicographically for deterministic results.
    """
    if (
        not isinstance(beam_width, int)
        or isinstance(beam_width, bool)
        or beam_width < 1
    ):
        raise ValueError("beam_width must be a positive integer")

    aliases = tuple(aliases)

    if len(set(aliases)) != len(aliases):
        raise ValueError("aliases must be unique")

    if not aliases:
        return []

    known = set(aliases)
    unknown_keys = set(adjacency) - known

    if unknown_keys:
        raise ValueError(
            f"adjacency contains unknown aliases: {sorted(unknown_keys)}"
        )

    graph: dict[str, set[str]] = {}

    for alias in aliases:
        neighbors = set(adjacency.get(alias, ()))
        unknown_neighbors = neighbors - known

        if unknown_neighbors:
            raise ValueError(
                f"adjacency for {alias!r} contains unknown aliases: "
                f"{sorted(unknown_neighbors)}"
            )

        neighbors.discard(alias)
        graph[alias] = neighbors

    # Make the graph undirected even if only one direction was supplied.
    for alias, neighbors in tuple(graph.items()):
        for neighbor in tuple(neighbors):
            graph[neighbor].add(alias)

    def evaluate(order: JoinOrder) -> float:
        score = float(score_prefix(order))

        if not math.isfinite(score):
            raise ValueError(
                "score_prefix must return a finite numeric score"
            )

        return score

    # Seed the beam with possible starting relations.
    beam: list[tuple[JoinOrder, float]] = [
        ((alias,), evaluate((alias,)))
        for alias in aliases
    ]

    beam.sort(key=lambda item: (item[1], item[0]))
    beam = beam[:beam_width]

    # Extend each retained partial order by one connected relation.
    for target_size in range(2, len(aliases) + 1):
        expanded: list[tuple[JoinOrder, float]] = []

        for order, _score in beam:
            used = set(order)

            for alias in aliases:
                if alias in used:
                    continue

                if not (graph[alias] & used):
                    continue

                extended = order + (alias,)
                expanded.append((extended, evaluate(extended)))

        if not expanded:
            # The retained states cannot produce a complete valid order.
            return []

        # Keep the lowest-scoring version of each order.
        best_for_order: dict[JoinOrder, float] = {}

        for order, score in expanded:
            previous = best_for_order.get(order)

            if previous is None or score < previous:
                best_for_order[order] = score

        beam = sorted(
            best_for_order.items(),
            key=lambda item: (item[1], item[0]),
        )[:beam_width]

        if not all(len(order) == target_size for order, _ in beam):
            raise RuntimeError("Internal beam-search depth invariant failed")

    return [
        order
        for order, _score in beam
        if len(order) == len(aliases)
    ]
