import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/e3/beam40_cap48/parallel_workers_all_summary.csv"
OUTPUT = ROOT / "results/e3/beam40_cap48/parallel_worker_comparison.svg"

with SOURCE.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

values = {}
for row in rows:
    query = int(row["table_count"])
    workers = int(row["candidate_workers"])
    values[(query, workers)] = float(row["median_planning_seconds"])

tables = sorted({query for query, _ in values})
workers_list = [1, 2, 4]
colors = {1: "#687a90", 2: "#d49a35", 4: "#2b7bbb"}

W, H = 1150, 610
LEFT, RIGHT, TOP, BOTTOM = 90, 30, 100, 100
PLOT_W, PLOT_H = W - LEFT - RIGHT, H - TOP - BOTTOM
MAX_Y = 4.0

def y_pos(value):
    return TOP + PLOT_H - value / MAX_Y * PLOT_H

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:Arial,sans-serif;fill:#243247}'
    '.grid{stroke:#dce2ea;stroke-width:1}.axis{stroke:#526174;stroke-width:1.2}</style>',
    f'<text x="{W/2}" y="35" text-anchor="middle" font-size="23" font-weight="bold">Parallel Plan Analysis: E3 Queries</text>',
]

for tick in range(5):
    y = y_pos(tick)
    svg.extend([
        f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W-RIGHT}" y2="{y:.1f}" class="grid"/>',
        f'<text x="{LEFT-12}" y="{y+5:.1f}" text-anchor="end" font-size="14">{tick}</text>',
    ])

svg.extend([
    f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP+PLOT_H}" class="axis"/>',
    f'<line x1="{LEFT}" y1="{TOP+PLOT_H}" x2="{W-RIGHT}" y2="{TOP+PLOT_H}" class="axis"/>',
    f'<text transform="translate(25 {TOP+PLOT_H/2}) rotate(-90)" text-anchor="middle" font-size="16">Median planning time (seconds)</text>',
])

group = PLOT_W / len(tables)
bar_w = 21
offsets = {1: -bar_w * 1.15, 2: 0, 4: bar_w * 1.15}

for index, table in enumerate(tables):
    center = LEFT + group * (index + 0.5)

    for worker in workers_list:
        value = values[(table, worker)]
        x = center + offsets[worker] - bar_w / 2
        y = y_pos(value)
        height = TOP + PLOT_H - y

        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{height:.1f}" '
            f'rx="2" fill="{colors[worker]}"/>'
        )
        svg.append(
            f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" '
            f'font-size="11">{value:.3f}</text>'
        )

    label_y = TOP + PLOT_H + 25
    svg.append(
        f'<text x="{center:.1f}" y="{label_y}" text-anchor="middle" font-size="13">Q{table:02d}</text>'
    )
    svg.append(
        f'<text x="{center:.1f}" y="{label_y+18}" text-anchor="middle" font-size="12">{table} tables</text>'
    )

legend_x = [W/2 - 170, W/2 - 10, W/2 + 150]
for x, worker in zip(legend_x, workers_list):
    svg.extend([
        f'<rect x="{x}" y="62" width="15" height="15" rx="2" fill="{colors[worker]}"/>',
        f'<text x="{x+22}" y="75" font-size="14">{worker} worker{"s" if worker > 1 else ""}</text>',
    ])

svg.append("</svg>")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(svg), encoding="utf-8")
print(f"Saved chart: {OUTPUT}")
