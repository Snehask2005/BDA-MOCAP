from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


JoinOrder = tuple[str, ...]
PrefixScorer = Callable[[JoinOrder], float]
Pair = tuple[str, str]


@dataclass(frozen=True)
class JoinSearchStatistics:
    """Statistics supplied to the join-order pre-scorer.

    relation_rows:
        Estimated rows remaining in each relation after its local filters.

    join_selectivities:
        Estimated selectivity for each join edge, keyed by a pair of
        aliases. Keys are order-independent: ("a", "b") and ("b", "a")
        refer to the same edge.

    These values must come from measured or explicitly estimated
    statistics. This class does not invent missing estimates.
    """

    relation_rows: Mapping[str, float]
    join_selectivities: Mapping[Pair, float]


def build_cardinality_scorer(
    aliases: Iterable[str],
    adjacency: Mapping[str, Iterable[str]],
    statistics: JoinSearchStatistics | None,
) -> PrefixScorer | None:
    """Build a cardinality-based scorer when its required stats exist.

    The score sums log-scaled estimated intermediate row counts.
    Lower scores are preferred.

    Returns None when estimates are incomplete, allowing the caller
    to fall back to its existing size-only heuristic.
    """
    if statistics is None:
        return None

    aliases = tuple(aliases)
    known = set(aliases)

    if any(alias not in known for alias in adjacency):
        raise ValueError("adjacency contains an unknown alias")

    graph = {
        alias: set(adjacency.get(alias, ()))
        for alias in aliases
    }

    for alias, neighbors in graph.items():
        if not neighbors <= known:
            raise ValueError(
                f"adjacency for {alias!r} contains an unknown alias"
            )
        if alias in neighbors:
            raise ValueError("self-join graph edges are not supported")

    # Normalize the graph to undirected edges.
    for alias, neighbors in tuple(graph.items()):
        for neighbor in tuple(neighbors):
            graph[neighbor].add(alias)

    rows: dict[str, float] = {}

    for alias in aliases:
        if alias not in statistics.relation_rows:
            return None

        value = float(statistics.relation_rows[alias])

        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Invalid estimated row count for alias {alias!r}"
            )

        rows[alias] = value

    selectivities: dict[Pair, float] = {}

    for raw_pair, raw_value in statistics.join_selectivities.items():
        if len(raw_pair) != 2 or raw_pair[0] == raw_pair[1]:
            raise ValueError("Each selectivity key must contain two aliases")

        left, right = raw_pair

        if left not in known or right not in known:
            raise ValueError(
                f"Selectivity refers to an unknown alias: {raw_pair}"
            )

        value = float(raw_value)

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Invalid join selectivity for {raw_pair}"
            )

        pair = tuple(sorted((left, right)))

        if pair in selectivities and selectivities[pair] != value:
            raise ValueError(
                f"Conflicting selectivities supplied for {pair}"
            )

        selectivities[pair] = value

    required_edges = {
        tuple(sorted((alias, neighbor)))
        for alias, neighbors in graph.items()
        for neighbor in neighbors
    }

    if not required_edges <= selectivities.keys():
        return None

    def score_prefix(order: JoinOrder) -> float:
        if not order:
            return 0.0

        if any(alias not in known for alias in order):
            raise ValueError("join order contains an unknown alias")

        if len(set(order)) != len(order):
            raise ValueError("join order contains duplicate aliases")

        current_rows = rows[order[0]]
        score = math.log1p(current_rows)

        for index, alias in enumerate(order[1:], start=1):
            prefix = order[:index]

            connecting = [
                previous
                for previous in prefix
                if previous in graph[alias]
            ]

            if not connecting:
                raise ValueError(
                    f"Alias {alias!r} is disconnected from the prefix"
                )

            factors = [
                selectivities[tuple(sorted((alias, previous)))]
                for previous in connecting
            ]

            if current_rows == 0 or rows[alias] == 0 or 0.0 in factors:
                current_rows = 0.0
            else:
                log_rows = (
                    math.log(current_rows)
                    + math.log(rows[alias])
                    + sum(math.log(factor) for factor in factors)
                )
                # Guard against floating-point overflow on poor estimates.
                current_rows = math.exp(min(log_rows, 700.0))

            score += math.log1p(current_rows)

        return score

    return score_prefix
