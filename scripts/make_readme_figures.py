#!/usr/bin/env python
"""
The two figures used in the README: a grouped bar comparison and a plain results table.

    python scripts/make_readme_figures.py --out_dir results/figures

Colour assignment follows the reference palette's fixed slot order, which is the CVD-safety
mechanism rather than a style choice: series are ordered so that slots 1-6 are used in
sequence (worst adjacent pair dE 9.1 protan, 19.6 normal-vision). Three of those six slots
sit under 3:1 against the surface, so the palette's relief rule applies - the black-and-white
table published beside the chart is that relief.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# reference palette, slots 1-6 in documented order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

BENCH = ["MedBullets\n(op4)", "MedBullets\n(op5)", "MedXpert", "MedQA", "MedMCQA", "PubmedQA"]

# series order is what binds a model to a palette slot; ours takes slot 2 (orange), the
# colour it carries in every other figure of this project
CHART = [
    ("MedReason-8B",         [57.5, 55.5, 19.0, 71.8, 60.7, 79.4], False),
    ("BrainMed-8B (ours)",   [62.01, 54.87, 18.63, 76.67, 64.07, 79.10], True),
    ("Huatuo-o1-RL-8B",      [55.2, 51.3, 16.7, 72.6, 60.4, 79.2], False),
    ("Llama3.1-Instruct-8B", [43.2, 40.9, 14.3, 58.7, 56.0, 75.2], False),
    ("Qwen2.5-Instruct-7B",  [50.0, 41.6, 12.6, 57.0, 55.6, 72.7], False),
    ("BioMistral-7B",        [46.4, 33.1, 12.4, 45.0, 40.2, 66.9], False),
]

# full table, grouped as in the reference literature
TABLE = [
    ("general", [
        ("Llama3.1-Instruct-8B", [43.2, 40.9, 14.3, 58.7, 56.0, 75.2, 48.0]),
        ("Qwen2.5-Instruct-7B",  [50.0, 41.6, 12.6, 57.0, 55.6, 72.7, 48.2]),
        ("Mistral-Instruct-7B",  [43.5, 33.4, 11.4, 48.2, 44.9, 50.1, 38.6]),
    ]),
    ("medical", [
        ("Medical-Llama3-8B", [33.4, 25.3, 9.0, 40.3, 46.8, 48.0, 33.8]),
        ("OpenBioLLM-8B",     [39.2, 35.7, 10.7, 57.7, 54.1, 74.1, 45.3]),
        ("BioMistral-7B",     [46.4, 33.1, 12.4, 45.0, 40.2, 66.9, 40.7]),
    ]),
    ("reasoning", [
        ("Medical-CoT-8B",      [39.3, 34.1, 12.6, 49.0, 42.6, 68.0, 40.9]),
        ("DeepSeek-Distill-8B", [41.9, 35.1, 13.5, 55.4, 49.0, 73.9, 44.8]),
        ("Huatuo-o1-SFT-8B",    [53.3, 49.7, 17.3, 70.2, 58.2, 76.1, 54.1]),
        ("Huatuo-o1-RL-8B",     [55.2, 51.3, 16.7, 72.6, 60.4, 79.2, 55.9]),
    ]),
    ("ours", [
        ("MedReason-8B",        [57.5, 55.5, 19.0, 71.8, 60.7, 79.4, 57.3]),
        ("BrainMed-8B (ours)",  [62.01, 54.87, 18.63, 76.67, 64.07, 79.10, 59.23]),
    ]),
]
COLS = ["MedBullets\n(op4)", "MedBullets\n(op5)", "MedXpert", "MedQA", "MedMCQA",
        "PubmedQA", "Avg"]


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.join(out_dir, name)}.png/.pdf")


def grouped_bars(out_dir):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK2,
    })
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    n = len(CHART)
    w = 0.84 / n
    x = range(len(BENCH))
    for j, (label, vals, ours) in enumerate(CHART):
        off = (j - (n - 1) / 2) * (w + 0.006)
        ax.bar([i + off for i in x], vals, w, label=label, color=SLOTS[j],
               edgecolor=SURFACE, linewidth=0.8, zorder=2)
        if ours:                       # selective direct labels: the table carries the rest
            for i, v in enumerate(vals):
                ax.text(i + off, v + 1.1, f"{v:.1f}", ha="center", fontsize=8.6,
                        color=INK, fontweight="bold", zorder=3)
    ax.set_xticks(list(x), BENCH, fontsize=10)
    ax.set_ylim(0, max(max(v) for _, v, _ in CHART) * 1.20)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("BrainMed-8B against published 7B-8B medical LLMs",
                 loc="left", pad=16, color=INK, fontsize=13.5, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=9.5, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), columnspacing=1.6, handlelength=1.3)
    fig.text(0.008, -0.03,
             "Published rows transcribed from the reference literature; BrainMed-8B measured "
             "on this harness (greedy decoding, strict answer prompt). Exact values in the "
             "table below.", fontsize=8, color=MUTED)
    save(fig, out_dir, "fig_model_comparison")


def bw_table(out_dir):
    """Plain black-and-white results table, in the layout used by the reference papers."""
    plt.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "text.color": "black",
    })
    rows = [r for _, group in TABLE for r in group]
    best = [max(r[1][c] for r in rows) for c in range(len(COLS))]

    ncol = len(COLS) + 1
    wcol = [0.235] + [(1 - 0.235) / (ncol - 1)] * (ncol - 1)
    edges, acc = [], 0.0
    for w in wcol:
        edges.append((acc, acc + w))
        acc += w

    nrow = len(rows)
    row_h, head_h = 0.30, 0.62
    fh = 0.30 + head_h + nrow * row_h + 0.22 * (len(TABLE) - 1) + 0.30
    fig = plt.figure(figsize=(11.2, fh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def xpos(c):
        return edges[c][0] + 0.004 if c == 0 else (edges[c][0] + edges[c][1]) / 2

    y = 1 - 0.14 / fh
    ax.plot([0, 1], [y, y], color="black", lw=1.5)          # top rule
    y -= 0.16 / fh
    ax.text(xpos(0), y, "Model", fontsize=11.5, fontweight="bold", va="top", ha="left")
    for c, h in enumerate(COLS):
        for k, part in enumerate(h.split("\n")):
            ax.text(xpos(c + 1), y - k * 0.22 / fh, part, fontsize=11.5, fontweight="bold",
                    va="top", ha="center")
    y -= head_h / fh
    ax.plot([0, 1], [y, y], color="black", lw=1.2)          # rule under header

    for gi, (_, group) in enumerate(TABLE):
        if gi:
            ax.plot([0, 1], [y, y], color="black", lw=0.6)  # group separator
            y -= 0.22 / fh
        for name, vals in group:
            y -= row_h / fh
            ax.text(xpos(0), y + 0.20 / fh, name, fontsize=11, va="top", ha="left")
            for c, v in enumerate(vals):
                # one decimal throughout: the published rows are only reported to one, and
                # mixing precisions in the same column implies a precision we cannot compare
                bold = abs(v - best[c]) < 1e-9
                ax.text(xpos(c + 1), y + 0.20 / fh, f"{v:.1f}",
                        fontsize=11, va="top", ha="center",
                        fontweight="bold" if bold else "normal")
        y -= 0.06 / fh
    ax.plot([0, 1], [y, y], color="black", lw=1.5)          # bottom rule

    save(fig, out_dir, "table_results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures"))
    args = ap.parse_args()
    print(f"figures -> {args.out_dir}")
    grouped_bars(args.out_dir)
    bw_table(args.out_dir)


if __name__ == "__main__":
    main()
