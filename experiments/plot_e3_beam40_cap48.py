import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/e3/beam40_cap48/prescoring_all_summary.csv"
OUTPUT = ROOT / "results/e3/beam40_cap48/planning_time_comparison.svg"

with SOURCE.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

baseline = {}
capped = {}

for row in rows:
    table_count = int(row["table_count"])
    seconds = float(row["median_total_planning_seconds"])

    if row["mode"] == "baseline":
        baseline[table_count] = seconds
    elif row["mode"] == "prescore_48":
        capped[table_count] = seconds

tables = sorted(set(baseline) & set(capped))
if not tables:
    raise RuntimeError("No matching baseline and cap-48 results found.")

W, H = 1100, 590
LEFT, RIGHT, TOP, BOTTOM = 90, 35, 85, 105
PLOT_W = W - LEFT - RIGHT
PLOT_H = H - TOP - BOTTOM
MAX_Y = max(12, int(max(baseline[t] for t in tables) / 2 + 1) * 2)

def y_pos(value):
    return TOP + PLOT_H - value / MAX_Y * PLOT_H

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:Arial,sans-serif;fill:#243247}'
    '.grid{stroke:#dce2ea;stroke-width:1}'
    '.axis{stroke:#526174;stroke-width:1.2}</style>',
    f'<text x="{W/2}" y="35" text-anchor="middle" '
    'font-size="23" font-weight="bold">E3 Candidate Planning: Beam Width 40</text>',
]

for tick in range(0, MAX_Y + 1, 2):
    y = y_pos(tick)
    svg.append(
        f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W-RIGHT}" '
        f'y2="{y:.1f}" class="grid"/>'
    )
    svg.append(
        f'<text x="{LEFT-12}" y="{y+5:.1f}" text-anchor="end" '
        f'font-size="14">{tick}</text>'
    )

svg.extend([
    f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" '
    f'y2="{TOP+PLOT_H}" class="axis"/>',
    f'<line x1="{LEFT}" y1="{TOP+PLOT_H}" x2="{W-RIGHT}" '
    f'y2="{TOP+PLOT_H}" class="axis"/>',
    f'<text transform="translate(25 {TOP+PLOT_H/2}) rotate(-90)" '
    'text-anchor="middle" font-size="16">Median planning time (seconds)</text>',
])

group = PLOT_W / len(tables)
bar_w = min(30, group * 0.27)

for i, table in enumerate(tables):
    center = LEFT + group * (i + 0.5)

    for offset, value, color in [
        (-bar_w / 2, baseline[table], "#687a90"),
        (bar_w / 2, capped[table], "#2b7bbb"),
    ]:
        x = center + offset - bar_w / 2
        y = y_pos(value)
        height = TOP + PLOT_H - y
        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{height:.1f}" rx="2" fill="{color}"/>'
        )
        svg.append(
            f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" '
            f'text-anchor="middle" font-size="12">{value:.2f}</text>'
        )

    label_y = TOP + PLOT_H + 25
    svg.append(
        f'<text x="{center:.1f}" y="{label_y}" text-anchor="middle" '
        f'font-size="13">Q{table:02d}</text>'
    )
    svg.append(
        f'<text x="{center:.1f}" y="{label_y+18}" text-anchor="middle" '
        f'font-size="12">{table} tables</text>'
    )

legend_y = 60
for x, color, label in [
    (W/2-180, "#687a90", "Beam 40, uncapped"),
    (W/2+45, "#2b7bbb", "Beam 40, cap 48"),
]:
    svg.append(
        f'<rect x="{x}" y="{legend_y-13}" width="16" height="16" '
        f'rx="2" fill="{color}"/>'
    )
    svg.append(
        f'<text x="{x+24}" y="{legend_y}" font-size="14">{label}</text>'
    )

svg.append("</svg>")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(svg), encoding="utf-8")
print(f"Saved chart: {OUTPUT}")
