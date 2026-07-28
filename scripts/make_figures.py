#!/usr/bin/env python
"""
Evaluation figures, saved as PNG (300 dpi) + PDF next to the report.

Four figures, in the order a reader needs them:
  1. average score across the paper's six-benchmark table - where we land overall
  2. controlled before/after on the same harness - what the fine-tune actually did
  3. head-to-head against the backbone and the paper's best model, per benchmark
  4. training and validation loss - evidence the run was healthy

Palette: categorical slots 1-3 of the reference palette, validated all-pairs for CVD
(worst deutan dE 9.2, worst normal-vision dE 24.0). Aqua sits under 3:1 on the light
surface, so every bar carries a visible value label and REPORT.md holds the table view.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Categorical slots 1, 3, 7 and 2 of the reference palette. Slot 4 (yellow) is skipped on
# purpose: yellow beside orange fails the all-pairs floors. This four-colour set passes all
# six checks under --pairs all (worst deutan dE 9.2, worst normal-vision dE 16.3).
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRIT = "#006300", "#d03b3b"

TABLE4 = ["medbullets_op4", "medbullets_op5", "medxpertqa", "medqa_4opt", "medmcqa_val", "pubmedqa_test"]
COMMON = ["medqa_4opt", "medmcqa_val", "pubmedqa_test", "mmlu_pro_medical"]
CHALLENGING = ["medbullets_op4", "medbullets_op5", "medxpertqa", "hle_med"]
ALL8 = TABLE2 = COMMON + CHALLENGING          # MedReason Table 2 column order
SHORT = {"medqa_4opt": "MedQA", "medmcqa_val": "MedMCQA", "pubmedqa_test": "PubMedQA",
         "mmlu_pro_medical": "MMLU-Pro\n(Med)", "medbullets_op4": "MedBullets\nop4",
         "medbullets_op5": "MedBullets\nop5", "medxpertqa": "MedXpert", "hle_med": "HLE (med)",
         "gpqa_medical": "GPQA\n(Med)", "avg": "Average"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 120,
})


def style(ax, xgrid=False):
    """Recessive chrome: one grid direction only, no box."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(False)  # rcParams turns both axes on; opt back in to a single direction
    ax.grid(True, axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(out_dir, name)}.png/.pdf")


def avg(v):
    return round(sum(v) / len(v), 2) if v else None


# ======================================================================================
def fig_avg_ranking(runs, t4, out_dir):
    """Horizontal bars: overall average, published models + ours."""
    # kind: 0 = published, 1 = our backbone measured here, 2 = our fine-tune
    items = [(n, v[-1], 0) for n, v in t4.items()]
    for name, s in runs.items():
        got = [s["benchmarks"].get(k, {}).get("accuracy") for k in TABLE4]
        if all(g is not None for g in got):
            items.append((f"{name} (ours)", avg(got), 1 if name.startswith("base-") else 2))
    items.sort(key=lambda x: x[1])

    palette = {0: BLUE, 1: AQUA, 2: ORANGE}
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(items) + 1.6))
    ys = range(len(items))
    ax.barh(list(ys), [v for _, v, _ in items],
            color=[palette[k] for _, _, k in items], height=0.62)
    ax.set_yticks(list(ys), [n for n, _, _ in items], fontsize=9.5)
    for y, (_, v, k) in zip(ys, items):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=9.5,
                color=INK if k else INK2, fontweight="bold" if k == 2 else "normal")
    ax.set_xlim(0, max(v for _, v, _ in items) * 1.12)
    ax.set_xlabel("Average accuracy over 6 benchmarks (%)")
    ax.set_title("Overall performance, 7B-8B medical LLMs", loc="left", pad=14, color=INK)
    style(ax, xgrid=True)
    present = sorted({k for _, _, k in items})
    labels = {0: "Published (MedReason, Table 4)", 1: "Backbone, re-measured here",
              2: "Ours, fine-tuned"}
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in present],
              labels=[labels[k] for k in present],
              frameon=False, loc="lower right", fontsize=9)
    save(fig, out_dir, "fig1_average_ranking")


def fig_before_after(runs, out_dir):
    """Grouped bars: backbone vs fine-tune, identical harness."""
    base = next((n for n in runs if n.startswith("base-")), None)
    ft = next((n for n in runs if not n.startswith("base-")), None)
    if not base or not ft:
        print("  skipping fig2: need one base-* run and one fine-tuned run")
        return
    keys = [k for k in ALL8 if k in runs[base]["benchmarks"] and k in runs[ft]["benchmarks"]]
    b = [runs[base]["benchmarks"][k]["accuracy"] for k in keys]
    f = [runs[ft]["benchmarks"][k]["accuracy"] for k in keys]

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = range(len(keys))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], b, w, label=f"Before - {base}", color=BLUE)
    ax.bar([i + w / 2 + 0.01 for i in x], f, w, label=f"After - {ft}", color=ORANGE)
    for i, (bv, fv) in enumerate(zip(b, f)):
        ax.text(i - w / 2 - 0.01, bv + 1, f"{bv:.1f}", ha="center", fontsize=8.5, color=INK2)
        ax.text(i + w / 2 + 0.01, fv + 1, f"{fv:.1f}", ha="center", fontsize=8.5,
                color=INK, fontweight="bold")
        d = fv - bv
        ax.text(i, max(bv, fv) + 5.5, f"{d:+.1f}", ha="center", fontsize=9.5,
                color=GOOD if d > 0 else CRIT, fontweight="bold")
    ax.set_xticks(list(x), [SHORT[k] for k in keys], fontsize=9)
    ax.set_ylim(0, max(b + f) * 1.25)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Effect of fine-tuning on our corpus - same prompts, same scorer",
                 loc="left", pad=14, color=INK)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style(ax)
    save(fig, out_dir, "fig2_before_after")


def fig_head_to_head(runs, t4, out_dir):
    """Ours vs the backbone vs the paper's best, per benchmark."""
    ft = next((n for n in runs if not n.startswith("base-")), None)
    if not ft:
        return
    ours = [runs[ft]["benchmarks"].get(k, {}).get("accuracy") for k in TABLE4]
    if any(v is None for v in ours):
        print("  skipping fig3: fine-tuned run is missing benchmarks")
        return
    series = [("Huatuo-o1-RL-8B (backbone)", t4["Huatuo-o1-RL-8B"][:-1], BLUE),
              ("MedReason-8B (paper best)", t4["MedReason-8B"][:-1], AQUA),
              (f"{ft} (ours)", ours, ORANGE)]

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = range(len(TABLE4))
    w = 0.26
    for j, (label, vals, c) in enumerate(series):
        off = (j - 1) * (w + 0.015)
        ax.bar([i + off for i in x], vals, w, label=label, color=c)
        for i, v in enumerate(vals):
            ax.text(i + off, v + 1, f"{v:.1f}", ha="center", fontsize=8,
                    color=INK if j == 2 else INK2, fontweight="bold" if j == 2 else "normal")
    ax.set_xticks(list(x), [SHORT[k] for k in TABLE4], fontsize=9)
    ax.set_ylim(0, max(max(v) for _, v, _ in series) * 1.22)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Head-to-head against the backbone and the published best model",
                 loc="left", pad=14, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=3)
    style(ax)
    save(fig, out_dir, "fig3_head_to_head")


def fig_training_curves(metrics_path, out_dir):
    if not metrics_path or not os.path.exists(metrics_path):
        print("  skipping fig4: no metrics.jsonl")
        return
    rows = [json.loads(l) for l in open(metrics_path, encoding="utf-8") if l.strip()]
    tr = [(r["step"], r["train/loss"]) for r in rows if "train/loss" in r]
    va = [(r["step"], r["val/loss"]) for r in rows if "val/loss" in r]
    if not tr:
        return

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot([s for s, _ in tr], [v for _, v in tr], color=BLUE, lw=2, label="Training loss")
    if va:
        ax.plot([s for s, _ in va], [v for _, v in va], color=ORANGE, lw=2,
                marker="o", ms=4, label="Validation loss")
        bi = min(range(len(va)), key=lambda i: va[i][1])
        bs, bv = va[bi]
        ax.scatter([bs], [bv], s=110, facecolor=ORANGE, edgecolor=SURFACE, lw=2, zorder=5)
        # flip the callout inward when the best point sits near the right edge
        late = bs > 0.7 * max(s for s, _ in tr)
        ax.annotate(f"best checkpoint\nstep {bs}, loss {bv:.4f}", (bs, bv),
                    textcoords="offset points", xytext=(-14 if late else 12, 22),
                    ha="right" if late else "left", fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training run - loss and checkpoint selection", loc="left", pad=14, color=INK)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    save(fig, out_dir, "fig4_training_curves")


def fig_huatuo_core(runs, base, out_dir):
    """HuatuoGPT-o1 Table 1/2 column set: the five core medical benchmarks."""
    ft = next((n for n in runs if not n.startswith("base-")), None)
    if not ft:
        return
    keys = COMMON + ["gpqa_medical"]
    ours = [runs[ft]["benchmarks"].get(k, {}).get("accuracy") for k in keys]
    if any(v is None for v in ours):
        print("  skipping fig5: gpqa_medical missing (run the full bundle)")
        return
    ht = base["huatuo_table2_ablation"]
    series = [("LLaMA-3.1-8B-Instruct (published)", ht["LLaMA-3.1-8B-Instruct (base)"], BLUE),
              ("HuatuoGPT-o1-8B (published)", ht["SFT w/ Complex CoT + RL (PPO) = o1-8B"], AQUA),
              (f"{ft} (ours)", ours, ORANGE)]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(keys))
    w = 0.26
    for j, (label, vals, c) in enumerate(series):
        off = (j - 1) * (w + 0.015)
        ax.bar([i + off for i in x], vals, w, label=label, color=c)
        for i, v in enumerate(vals):
            ax.text(i + off, v + 1, f"{v:.1f}", ha="center", fontsize=8,
                    color=INK if j == 2 else INK2, fontweight="bold" if j == 2 else "normal")
    ax.set_xticks(list(x), [SHORT[k] for k in keys], fontsize=9)
    ax.set_ylim(0, max(max(v) for _, v, _ in series) * 1.22)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Core medical benchmarks (HuatuoGPT-o1 Table 2 column set)",
                 loc="left", pad=14, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=3)
    style(ax)
    save(fig, out_dir, "fig5_huatuo_core_benchmarks")


def fig_common_vs_challenging(runs, base, out_dir):
    """MedReason Table 3 framing: the two averages, published rows plus ours."""
    def wrap(s, width=17):
        out, line = [], ""
        for word in s.replace("-", "- ").split(" "):
            if len(line) + len(word) > width and line:
                out.append(line.rstrip())
                line = ""
            line += word + " "
        out.append(line.rstrip())
        return "\n".join(x.replace("- ", "-") for x in out)

    t3 = {k: v for k, v in base["table3_reasoning_models"].items() if not k.startswith("_")}
    labels, com, cha = [], [], []
    for n, v in t3.items():
        labels.append(wrap(n))
        com.append(v["common"][-1])
        cha.append(v["challenging"][-1])
    for name, s in runs.items():
        c = [s["benchmarks"].get(k, {}).get("accuracy") for k in COMMON]
        d = [s["benchmarks"].get(k, {}).get("accuracy") for k in CHALLENGING]
        if all(v is not None for v in c + d):
            labels.append(wrap(f"{name} (ours)"))
            com.append(avg(c))
            cha.append(avg(d))

    fig, ax = plt.subplots(figsize=(max(9.0, 1.95 * len(labels)), 5.0))
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], com, w, label="Common medical QA (4)", color=BLUE)
    ax.bar([i + w / 2 + 0.01 for i in x], cha, w, label="Challenging clinical (4)", color=ORANGE)
    for i, (a, b) in enumerate(zip(com, cha)):
        ax.text(i - w / 2 - 0.01, a + 0.8, f"{a:.1f}", ha="center", fontsize=8.5, color=INK2)
        ax.text(i + w / 2 + 0.01, b + 0.8, f"{b:.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_ylim(0, max(com + cha) * 1.18)
    ax.set_ylabel("Average accuracy (%)")
    ax.set_title("Common vs challenging benchmark averages (MedReason Table 3)",
                 loc="left", pad=14, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    style(ax)
    save(fig, out_dir, "fig6_common_vs_challenging")


def fig_data_ablation(runs, base, out_dir):
    """MedReason Table 2 framing: same backbone, different training corpus."""
    t2 = {k: v for k, v in base["table2_data_ablation"].items() if not k.startswith("_")}
    llama = {k: v for k, v in t2.items() if k.startswith("Llama3.1")}
    ours = None
    for name, s in runs.items():
        if "llama" not in name.lower() or name.startswith("base-"):
            continue
        got = [s["benchmarks"].get(k, {}).get("accuracy") for k in TABLE2]
        if all(v is not None for v in got):
            ours = (name, got + [avg(got)])
    if ours is None:
        print("  skipping fig7: needs a Llama-3.1 backbone run to be comparable to Table 2")
        return

    keys = TABLE2 + ["avg"]
    series = [("base", list(llama.values())[0], BLUE),
              ("+ Huatuo CoT", list(llama.values())[1], AQUA),
              ("+ MedReason", list(llama.values())[2], VIOLET),
              (f"+ ours ({ours[0]})", ours[1], ORANGE)]

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = range(len(keys))
    w = 0.2
    for j, (label, vals, c) in enumerate(series):
        off = (j - 1.5) * (w + 0.012)
        ax.bar([i + off for i in x], vals, w, label=label, color=c)
    for i, v in enumerate(series[-1][1]):
        ax.text(i + 1.5 * (w + 0.012), v + 1, f"{v:.1f}", ha="center", fontsize=8,
                color=INK, fontweight="bold")
    ax.set_xticks(list(x), [SHORT.get(k, "Avg") for k in keys], fontsize=9)
    ax.set_ylim(0, max(max(v) for _, v, _ in series) * 1.2)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Same backbone (Llama-3.1-8B-Instruct), different training corpus",
                 loc="left", pad=14, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=4)
    style(ax)
    save(fig, out_dir, "fig7_data_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--baselines", default=os.path.join(REPO_ROOT, "eval", "baselines.json"))
    ap.add_argument("--metrics", default=None, help="train_logs/<exp>/metrics.jsonl")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "figures"))
    args = ap.parse_args()

    base = json.load(open(args.baselines, encoding="utf-8"))
    t4 = {k: v for k, v in base["table4_sota_comparison"].items() if not k.startswith("_")}
    runs = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "*", "summary_*.json"))):
        s = json.load(open(p, encoding="utf-8"))
        runs[s["run"]] = s
    if not runs:
        raise SystemExit(f"no summary_*.json under {args.results_dir}")

    metrics = args.metrics
    if metrics is None:
        cand = sorted(glob.glob(os.path.join(REPO_ROOT, "train_logs", "*", "metrics.jsonl")))
        metrics = cand[-1] if cand else None

    print(f"figures -> {args.out_dir}")
    fig_avg_ranking(runs, t4, args.out_dir)
    fig_before_after(runs, args.out_dir)
    fig_head_to_head(runs, t4, args.out_dir)
    fig_training_curves(metrics, args.out_dir)
    fig_huatuo_core(runs, base, args.out_dir)
    fig_common_vs_challenging(runs, base, args.out_dir)
    fig_data_ablation(runs, base, args.out_dir)


if __name__ == "__main__":
    main()
