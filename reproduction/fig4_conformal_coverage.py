"""
Reproduction Script: Figure 4 - Conformal Coverage & Budget Violation Trade-Off.
Authored by Hridhika (Lead Learned Predictor / ML Lead).

Generates side-by-side comparison illustrating:
- Panel (a): Empirical Coverage vs Certified Target (1 - delta) on held-out queries.
- Panel (b): Budget Violation Rate under real-world noise: Point-prediction collapse vs Conformal stability.

Output:
- paper/figures/fig4_conformal_coverage.svg (and .pdf/.png if matplotlib is installed)
- reproduction/fig4_data.json
"""
from __future__ import annotations

import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def generate_svg_conformal(
    tradeoff_data: List[Dict[str, Any]],
    out_svg_path: str,
) -> None:
    """Generate high-resolution publication-quality SVG for Figure 4."""
    width, height = 900, 420
    pad = 60
    panel_w = 380
    panel_h = 300

    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;">',
        f'<text x="{width // 2}" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">Figure 4: Conformal Coverage Certification & Budget Violation Immunity</text>',
    ]

    # Panel A: Target vs Empirical Coverage
    x0_a, y0_a = pad, pad + 20
    svg.append(f'<rect x="{x0_a}" y="{y0_a}" width="{panel_w}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    svg.append(f'<text x="{x0_a + panel_w//2}" y="{y0_a - 8}" font-size="13" font-weight="600" text-anchor="middle" fill="#1e293b">(a) Certified vs Empirical Coverage</text>')

    # Axis labels
    svg.append(f'<text x="{x0_a + panel_w//2}" y="{y0_a + panel_h + 35}" font-size="11" text-anchor="middle" fill="#64748b">Theoretical Confidence Level 1 - δ (%)</text>')
    svg.append(f'<text x="{x0_a - 38}" y="{y0_a + panel_h//2}" font-size="11" text-anchor="middle" fill="#64748b" transform="rotate(-90 {x0_a - 38} {y0_a + panel_h//2})">Empirical Test Coverage (%)</text>')

    # Diagonal ideal line (coverage = target)
    svg.append(f'<line x1="{x0_a + 30}" y1="{y0_a + panel_h - 30}" x2="{x0_a + panel_w - 30}" y2="{y0_a + 30}" stroke="#94a3b8" stroke-dasharray="4,4" stroke-width="1.5"/>')
    svg.append(f'<text x="{x0_a + panel_w - 95}" y="{y0_a + 50}" font-size="9.5" fill="#64748b">Ideal Coverage</text>')

    # Tolerance band (+/- 2%)
    poly_pts = [
        f"{x0_a + 30},{y0_a + panel_h - 30 - 8}",
        f"{x0_a + panel_w - 30},{y0_a + 30 - 8}",
        f"{x0_a + panel_w - 30},{y0_a + 30 + 8}",
        f"{x0_a + 30},{y0_a + panel_h - 30 + 8}",
    ]
    svg.append(f'<polygon points="{" ".join(poly_pts)}" fill="#10b981" opacity="0.15"/>')

    # Plot empirical points
    for item in tradeoff_data:
        target = item["target_coverage_pct"]  # 80 to 99
        emp = item["empirical_coverage_pct"]
        # Map 75..100 -> axis
        px = x0_a + 30 + ((target - 75.0) / 25.0) * (panel_w - 60)
        py = y0_a + panel_h - 30 - ((emp - 75.0) / 25.0) * (panel_h - 60)
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="#059669" stroke="#ffffff" stroke-width="2"/>')
        svg.append(f'<text x="{px + 8:.1f}" y="{py + 4:.1f}" font-size="10" font-weight="600" fill="#065f46">δ={item["delta"]:.2f} ({emp:.1f}%)</text>')

    # Panel B: Budget Violation Rate Comparison
    x0_b = pad + panel_w + 40
    y0_b = y0_a
    svg.append(f'<rect x="{x0_b}" y="{y0_b}" width="{panel_w}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    svg.append(f'<text x="{x0_b + panel_w//2}" y="{y0_b - 8}" font-size="13" font-weight="600" text-anchor="middle" fill="#1e293b">(b) Budget Violation Rate Comparison</text>')

    svg.append(f'<text x="{x0_b + panel_w//2}" y="{y0_b + panel_h + 35}" font-size="11" text-anchor="middle" fill="#64748b">Risk Tolerance Level δ</text>')
    svg.append(f'<text x="{x0_b - 38}" y="{y0_b + panel_h//2}" font-size="11" text-anchor="middle" fill="#64748b" transform="rotate(-90 {x0_b - 38} {y0_b + panel_h//2})">Budget Violation Rate (%)</text>')

    # Bar chart for violations across delta
    bar_width = 30
    x_positions = [x0_b + 55, x0_b + 140, x0_b + 225, x0_b + 310]

    max_v = 30.0  # max percentage for y-axis

    for idx, item in enumerate(tradeoff_data):
        bx = x_positions[idx]
        p_viol = item["point_violation_rate_pct"]
        c_viol = item["conformal_violation_rate_pct"]

        h_point = (p_viol / max_v) * (panel_h - 60)
        h_conf = (c_viol / max_v) * (panel_h - 60)

        # Point prediction bar (Red)
        y_pt = y0_b + panel_h - 30 - h_point
        svg.append(f'<rect x="{bx - bar_width//2}" y="{y_pt:.1f}" width="{bar_width}" height="{h_point:.1f}" fill="#f87171" rx="2"/>')
        svg.append(f'<text x="{bx}" y="{y_pt - 4:.1f}" font-size="9" text-anchor="middle" fill="#991b1b">{p_viol:.1f}%</text>')

        # Conformal bar (Emerald)
        bx_c = bx + 32
        y_cf = y0_b + panel_h - 30 - max(2.0, h_conf)
        svg.append(f'<rect x="{bx_c - bar_width//2}" y="{y_cf:.1f}" width="{bar_width}" height="{max(2.0, h_conf):.1f}" fill="#10b981" rx="2"/>')
        svg.append(f'<text x="{bx_c}" y="{y_cf - 4:.1f}" font-size="9" font-weight="bold" text-anchor="middle" fill="#065f46">{c_viol:.1f}%</text>')

        # Label delta
        svg.append(f'<text x="{bx + 16}" y="{y0_b + panel_h - 12}" font-size="10" font-weight="600" text-anchor="middle" fill="#334155">δ={item["delta"]:.2f}</text>')

    # Legend in Panel B
    svg.append(f'<rect x="{x0_b + 20}" y="{y0_b + 15}" width="16" height="10" fill="#f87171"/>')
    svg.append(f'<text x="{x0_b + 42}" y="{y0_b + 24}" font-size="10" fill="#475569">Point Estimate (Ĉ ≤ B)</text>')

    svg.append(f'<rect x="{x0_b + 180}" y="{y0_b + 15}" width="16" height="10" fill="#10b981"/>')
    svg.append(f'<text x="{x0_b + 202}" y="{y0_b + 24}" font-size="10" font-weight="bold" fill="#065f46">Conformal (Ĉ⁺ ≤ B)</text>')

    svg.append("</svg>")

    os.makedirs(os.path.dirname(out_svg_path) or ".", exist_ok=True)
    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def main() -> None:
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results", "phase2", "tradeoff_curve.csv"
    )
    if not os.path.exists(csv_path):
        from experiments.run_phase2_conformal import run_phase2_experiments
        run_phase2_experiments()

    df = pd.read_csv(csv_path)
    data = df.to_dict(orient="records")

    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    svg_out = os.path.join(figures_dir, "fig4_conformal_coverage.svg")

    generate_svg_conformal(data, svg_out)
    print(f"Generated Figure 4 SVG: {svg_out}")

    data_out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reproduction", "fig4_data.json")
    with open(data_out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved Figure 4 summary data: {data_out}")


if __name__ == "__main__":
    main()
