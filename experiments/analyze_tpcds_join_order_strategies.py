import csv
from collections import defaultdict

INPUT = "results/tpcds_candidate_metrics.csv"
OUTPUT = "results/tpcds_join_order_strategy_analysis.csv"


def get_join_order(strategy):
    """
    Convert:

        join_order_ss_d_i_s_cd
        join_order_ss_d_i_s_cd_merge
        join_order_ss_d_i_s_cd_shuffle_hash

    into:

        ss_d_i_s_cd
    """

    name = strategy[len("join_order_"):]

    if name.endswith("_shuffle_hash"):
        name = name[:-len("_shuffle_hash")]
    elif name.endswith("_merge"):
        name = name[:-len("_merge")]

    return name


with open(INPUT) as f:
    rows = list(csv.DictReader(f))


join_rows = [
    r for r in rows
    if r["strategy"].startswith("join_order_")
]


groups = defaultdict(list)

for row in join_rows:
    join_order = get_join_order(row["strategy"])
    groups[join_order].append(row)


print("=" * 90)
print("TPC-DS-DERIVED JOIN ORDER / PHYSICAL STRATEGY ANALYSIS")
print("=" * 90)

print(f"Total candidates:       {len(rows)}")
print(f"Join-order candidates:   {len(join_rows)}")
print(f"Logical join-order groups: {len(groups)}")

print("\nStrategy counts per logical join order:")

strategy_counts = defaultdict(set)

for join_order, candidates in groups.items():
    for row in candidates:
        strategy_counts[join_order].add(row["strategy"])

counts = defaultdict(int)

for strategies in strategy_counts.values():
    counts[len(strategies)] += 1

for number, count in sorted(counts.items()):
    print(
        f"  {count} join orders have {number} strategy variants"
    )


# ------------------------------------------------------------
# Create compact analysis rows
# ------------------------------------------------------------

analysis_rows = []

for join_order, candidates in sorted(groups.items()):

    # All candidates belonging to this logical join order
    # should have the same candidate-specific intermediate
    # statistic in this experiment.
    intermediate_values = {
        round(float(r["intermediate_bytes"]), 3)
        for r in candidates
    }

    intermediate_value = min(intermediate_values)

    for row in sorted(
        candidates,
        key=lambda r: r["strategy"]
    ):

        analysis_rows.append(
            {
                "join_order": join_order,
                "strategy": row["strategy"],
                "intermediate_bytes": row["intermediate_bytes"],
                "bytes_scanned": row["bytes_scanned"],
                "bytes_shuffled": row["bytes_shuffled"],
                "estimated_cost": row["estimated_cost"],
                "estimated_latency": row["estimated_latency"],
            }
        )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

fieldnames = [
    "join_order",
    "strategy",
    "intermediate_bytes",
    "bytes_scanned",
    "bytes_shuffled",
    "estimated_cost",
    "estimated_latency",
]

with open(
    OUTPUT,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(analysis_rows)


# ------------------------------------------------------------
# Print representative groups
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("REPRESENTATIVE JOIN ORDERS")
print("=" * 90)

# Sort logical join orders by intermediate size
ordered_groups = sorted(
    groups.items(),
    key=lambda item:
        float(item[1][0]["intermediate_bytes"])
)

# Show smallest, middle, and largest
selected = []

if ordered_groups:
    selected.append(ordered_groups[0])

if len(ordered_groups) >= 3:
    selected.append(
        ordered_groups[len(ordered_groups) // 2]
    )

if len(ordered_groups) >= 2:
    selected.append(ordered_groups[-1])


seen = set()

for join_order, candidates in selected:

    if join_order in seen:
        continue

    seen.add(join_order)

    print(
        f"\nJoin order: {join_order}"
    )

    print(
        f"Intermediate bytes: "
        f"{float(candidates[0]['intermediate_bytes']):.6e}"
    )

    for row in sorted(
        candidates,
        key=lambda r: r["strategy"]
    ):

        print(
            f"  {row['strategy']:45s}"
            f" shuffle={float(row['bytes_shuffled']):.6e}"
            f" cost={float(row['estimated_cost']):.8f}"
            f" latency={float(row['estimated_latency']):.6f}"
        )


# ------------------------------------------------------------
# Compare strategies
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("STRATEGY-LEVEL SUMMARY")
print("=" * 90)

strategy_groups = defaultdict(list)

for row in join_rows:
    strategy = row["strategy"]

    if strategy.endswith("_shuffle_hash"):
        strategy_type = "shuffle_hash"
    elif strategy.endswith("_merge"):
        strategy_type = "merge"
    else:
        strategy_type = "default"

    strategy_groups[strategy_type].append(row)


for strategy_type, candidates in sorted(
    strategy_groups.items()
):

    costs = [
        float(r["estimated_cost"])
        for r in candidates
    ]

    latencies = [
        float(r["estimated_latency"])
        for r in candidates
    ]

    shuffled = [
        float(r["bytes_shuffled"])
        for r in candidates
    ]

    print(f"\n{strategy_type}")
    print(f"  candidates:       {len(candidates)}")
    print(f"  unique costs:     {len(set(costs))}")
    print(f"  unique latencies: {len(set(latencies))}")
    print(
        f"  shuffle min:      {min(shuffled):.6e}"
    )
    print(
        f"  shuffle max:      {max(shuffled):.6e}"
    )


print("\n" + "=" * 90)
print(f"Saved: {OUTPUT}")
print("=" * 90)
