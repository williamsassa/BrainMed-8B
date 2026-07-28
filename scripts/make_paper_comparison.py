import argparse
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLE4 = ["medbullets_op4", "medbullets_op5", "medxpertqa", "medqa_4opt", "medmcqa_val", "pubmedqa_test"]
TABLE2 = ["medqa_4opt", "medmcqa_val", "pubmedqa_test", "mmlu_pro_medical",
          "medbullets_op4", "medbullets_op5", "medxpertqa", "hle_med"]
SHORT = {"medqa_4opt": "MedQA", "medmcqa_val": "MedMCQA", "pubmedqa_test": "PubMedQA",
         "mmlu_pro_medical": "MMLU-Pro", "medbullets_op4": "MB-op4", "medbullets_op5": "MB-op5",
         "medxpertqa": "MedXpert", "hle_med": "HLE(med)"}

# reference palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
HILITE = "#fdf0e9"          # tinted band for our rows, well below 3:1 so text stays readable
GOOD = "#006300"


def avg(v):
    return round(sum(v) / len(v), 2) if v else None


def fmt(x):
    return "" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


# ======================================================================================
# rendering
# ======================================================================================
def md_table(headers, rows, bold_rows=()):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for i, r in enumerate(rows):
        cells = [fmt(c) for c in r]
        if i in bold_rows:
            cells = [f"**{c}**" if c else c for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_table_image(title, headers, rows, out_path, highlight=(), aligns=None,
                       footnotes=(), subtitle=None):
    """Draw a table as a standalone figure: no chart chrome, just typography and rules."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    })
    cells = [[fmt(c) for c in r] for r in rows]
    ncol = len(headers)
    aligns = aligns or (["left"] + ["right"] * (ncol - 1))

    # column widths from the widest cell, with a floor so short numeric columns stay legible
    widths = [max(len(str(headers[c])), max((len(r[c]) for r in cells), default=0), 4)
              for c in range(ncol)]
    total = sum(widths)
    fw = min(max(0.135 * total + 0.8, 7.0), 17.0)
    nrow = len(cells)
    header_h, row_h = 0.40, 0.34
    fh = 0.95 + (0.28 if subtitle else 0) + header_h + nrow * row_h + 0.30 * len(footnotes) + 0.25

    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    lm, rm = 0.018, 0.018
    usable = 1 - lm - rm
    edges, acc = [], lm
    for w in widths:
        edges.append((acc, acc + usable * w / total))
        acc += usable * w / total

    def xpos(c, align):
        pad = 0.006
        return edges[c][0] + pad if align == "left" else edges[c][1] - pad

    y = 1 - (0.42 / fh)
    ax.text(lm, y, title, fontsize=13.5, color=INK, fontweight="bold", va="top")
    y -= 0.30 / fh
    if subtitle:
        ax.text(lm, y, subtitle, fontsize=9.5, color=INK2, va="top")
        y -= 0.30 / fh

    y -= 0.10 / fh
    for c, h in enumerate(headers):
        a = aligns[c]
        ax.text(xpos(c, a), y, str(h), fontsize=9.5, color=INK2, fontweight="bold",
                ha=a, va="top")
    y -= header_h / fh
    ax.plot([lm, 1 - rm], [y + 0.06 / fh, y + 0.06 / fh], color=AXIS, lw=1.1)

    for i, row in enumerate(cells):
        top, bot = y, y - row_h / fh
        if i in highlight:
            ax.add_patch(Rectangle((lm - 0.006, bot + 0.02 / fh), usable + 0.012,
                                   (row_h - 0.02) / fh, facecolor=HILITE, edgecolor="none",
                                   zorder=0))
            ax.add_patch(Rectangle((lm - 0.006, bot + 0.02 / fh), 0.0035,
                                   (row_h - 0.02) / fh, facecolor=ORANGE, edgecolor="none",
                                   zorder=1))
        else:
            ax.plot([lm, 1 - rm], [bot + 0.02 / fh, bot + 0.02 / fh], color=GRID, lw=0.6, zorder=0)
        for c, val in enumerate(row):
            a = aligns[c]
            bold = i in highlight
            color = INK if bold else (INK2 if c == 0 else INK)
            if val == "win":
                color, bold = GOOD, True
            ax.text(xpos(c, a), top - 0.075 / fh, val, fontsize=9.5, color=color,
                    fontweight="bold" if bold else "normal", ha=a, va="top", zorder=2)
        y = bot
    ax.plot([lm, 1 - rm], [y + 0.02 / fh, y + 0.02 / fh], color=AXIS, lw=1.1)

    y -= 0.16 / fh
    for note in footnotes:
        ax.text(lm, y, note, fontsize=8.2, color=MUTED, va="top")
        y -= 0.26 / fh

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  table image -> {out_path}.png/.pdf")


# ======================================================================================
def load_runs(results_dir, dedupe=True):
    runs = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "summary_*.json"))):
        s = json.load(open(p, encoding="utf-8"))
        runs[s["run"]] = s
    if not dedupe:
        return runs
    # the sweep re-evaluates the val-loss checkpoint under a second name; identical score
    # vectors are the same model twice and only pad the table
    seen, out = {}, {}
    for n, s in runs.items():
        key = tuple(sorted((k, v.get("accuracy")) for k, v in s["benchmarks"].items()))
        if key in seen:
            print(f"  dedupe: '{n}' is identical to '{seen[key]}', dropped")
            continue
        seen[key] = n
        out[n] = s
    return out


def scores(run, keys):
    return [run["benchmarks"].get(k, {}).get("accuracy") for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--baselines", default=os.path.join(REPO_ROOT, "eval", "baselines.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "COMPARISON.md"))
    ap.add_argument("--figures_dir", default=os.path.join(REPO_ROOT, "results", "figures"))
    ap.add_argument("--exclude", nargs="*", default=[], help="published rows to drop")
    ap.add_argument("--backbone_published", default="Huatuo-o1-RL-8B")
    ap.add_argument("--published_backbone_only", action="store_true",
                    help="omit our measured backbone row; use its published values instead")
    ap.add_argument("--no_dedupe", action="store_true",
                    help="keep runs whose scores are identical to another run")
    args = ap.parse_args()

    base = json.load(open(args.baselines, encoding="utf-8"))
    t4 = {k: v for k, v in base["table4_sota_comparison"].items() if not k.startswith("_")}
    t2 = {k: v for k, v in base["table2_data_ablation"].items() if not k.startswith("_")}
    runs = load_runs(args.results_dir, dedupe=not args.no_dedupe)
    if not runs:
        raise SystemExit(f"no summary_*.json under {args.results_dir}")

    ours = {n: s for n, s in runs.items() if not n.startswith("base-")}
    measured_base = next((s for n, s in runs.items() if n.startswith("base-")), None)
    kept = {k: v for k, v in t4.items() if k not in args.exclude}
    dropped = list(args.exclude)

    # ---- harness offset (still measured, even when the row is not displayed) -----------
    offset = None
    if measured_base and args.backbone_published in t4:
        pub = dict(zip(TABLE4, t4[args.backbone_published][:-1]))
        gaps = [measured_base["benchmarks"][k]["accuracy"] - pub[k]
                for k in TABLE4 if k in measured_base["benchmarks"]]
        offset = round(sum(gaps) / len(gaps), 2) if gaps else None

    notes = []
    if dropped:
        notes.append(f"Excluded from this table: {', '.join(dropped)}.")
    if args.published_backbone_only:
        notes.append(f"Backbone shown with its published values ({args.backbone_published}), "
                     f"not the score measured on this harness.")
    if offset is not None:
        notes.append(f"Harness offset measured on the untouched backbone: {offset:+.2f} pt vs the "
                     f"published protocol. Subtract it before reading our-vs-published gaps.")

    L = ["# Comparison against the published 7B-8B models", "",
         "Our rows are **measured by this pipeline**. Every other row is **transcribed from**",
         "MedReason (arXiv:2504.00993) and was not re-run.", ""]
    for n in notes:
        L += [f"> {n}", ""]

    fig_dir = args.figures_dir
    os.makedirs(fig_dir, exist_ok=True)

    # ---------------- Table A: ranking --------------------------------------------------
    entries = [(n, v[:-1], v[-1], False) for n, v in kept.items()]
    for n, s in ours.items():
        got = scores(s, TABLE4)
        if all(g is not None for g in got):
            entries.append((f"{n} (ours)", got, avg(got), True))
    if measured_base and not args.published_backbone_only:
        got = scores(measured_base, TABLE4)
        if all(g is not None for g in got):
            entries.append((f"{measured_base['run']} (ours, backbone)", got, avg(got), True))
    entries.sort(key=lambda r: -r[2])

    headers_a = ["#", "Model"] + [SHORT[k] for k in TABLE4] + ["Avg"]
    rows_a = [[i + 1, n] + list(v) + [a] for i, (n, v, a, _) in enumerate(entries)]
    hi_a = [i for i, e in enumerate(entries) if e[3]]
    L += ["## Table A — Ranking on the paper's six-benchmark suite", "",
          md_table(headers_a, rows_a, bold_rows=hi_a), ""]
    render_table_image(
        "Table A — Ranking on the paper's six-benchmark suite",
        headers_a, rows_a, os.path.join(fig_dir, "table_A_ranking"),
        highlight=hi_a, aligns=["right", "left"] + ["right"] * 7,
        subtitle="Average accuracy (%) over MedReason Table 4. Highlighted rows measured here; "
                 "all others transcribed from the paper.",
        footnotes=notes)

    # ---------------- Table B: head-to-head vs the published backbone -------------------
    if args.backbone_published in t4 and ours:
        pub = dict(zip(TABLE4, t4[args.backbone_published][:-1]))
        for n, s in ours.items():
            got = dict(zip(TABLE4, scores(s, TABLE4)))
            if any(v is None for v in got.values()):
                continue
            rows_b, wins = [], 0
            for k in TABLE4:
                d = got[k] - pub[k]
                # wins are labelled; everything else is left to the Delta column
                mark = "win" if d > 0.05 else ""
                wins += mark == "win"
                rows_b.append([SHORT[k], pub[k], got[k], f"{d:+.2f}", mark])
            a_o, a_p = avg(list(got.values())), t4[args.backbone_published][-1]
            rows_b.append(["Average", a_p, a_o, f"{a_o - a_p:+.2f}",
                           "win" if a_o > a_p else ""])
            headers_b = ["Benchmark", f"{args.backbone_published}\n(published)",
                         f"{n}\n(measured)", "Delta", ""]
            L += [f"## Table B — `{n}` vs published **{args.backbone_published}**", "",
                  md_table(headers_b, rows_b, bold_rows=[len(rows_b) - 1]), ""]
            nb = list(notes)
            nb.append(f"Wins on {wins} of {len(TABLE4)} benchmarks.")
            # one image per fine-tuned run, otherwise the second overwrites the first
            slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in n)
            suffix = f"_{slug}" if len(ours) > 1 else ""
            render_table_image(
                f"Table B — {n} vs {args.backbone_published}",
                [h.replace("\n", " ") for h in headers_b], rows_b,
                os.path.join(fig_dir, f"table_B_head_to_head{suffix}"),
                highlight=[len(rows_b) - 1],
                aligns=["left", "right", "right", "right", "left"],
                subtitle="Accuracy (%). Published values from MedReason Table 4; our column "
                         "measured by this pipeline.",
                footnotes=nb)

    # ---------------- Table D: eight-benchmark suite ------------------------------------
    rows_d, hi_d = [], []
    for n, v in t2.items():
        if n in args.exclude:
            continue
        rows_d.append([n] + v[:-1] + [v[-1]])
    for n, s in ours.items():
        got = scores(s, TABLE2)
        if all(g is not None for g in got):
            hi_d.append(len(rows_d))
            rows_d.append([f"{n} (ours)"] + got + [avg(got)])
    if measured_base and not args.published_backbone_only:
        got = scores(measured_base, TABLE2)
        if all(g is not None for g in got):
            hi_d.append(len(rows_d))
            rows_d.append([f"{measured_base['run']} (ours)"] + got + [avg(got)])
    headers_d = ["Model / data"] + [SHORT[k] for k in TABLE2] + ["Avg"]
    L += ["## Table D — Eight-benchmark suite (MedReason Table 2 columns)", "",
          md_table(headers_d, rows_d, bold_rows=hi_d), ""]
    render_table_image(
        "Table D — Eight-benchmark suite (MedReason Table 2 columns)",
        headers_d, rows_d, os.path.join(fig_dir, "table_D_eight_benchmarks"),
        highlight=hi_d, aligns=["left"] + ["right"] * 9,
        subtitle="Accuracy (%). Published rows use different backbones - compare within a "
                 "backbone family, not across.",
        footnotes=notes)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(L))
    print(f"\nmarkdown -> {args.out}")

    make_ranking_figure(entries, fig_dir, notes)


def make_ranking_figure(entries, out_dir, notes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 13, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2,
    })
    items = sorted(entries, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(8.4, 0.44 * len(items) + 1.9))
    ax.barh(range(len(items)), [r[2] for r in items],
            color=[ORANGE if r[3] else BLUE for r in items], height=0.62)
    ax.set_yticks(range(len(items)), [r[0] for r in items], fontsize=9.5)
    for y, r in enumerate(items):
        ax.text(r[2] + 0.5, y, f"{r[2]:.2f}", va="center", fontsize=9.5,
                color=INK if r[3] else INK2, fontweight="bold" if r[3] else "normal")
    ax.set_xlim(0, max(r[2] for r in items) * 1.13)
    ax.set_xlabel("Average accuracy over the 6 benchmarks of MedReason Table 4 (%)")
    ax.set_title("Ranking against published 7B-8B medical LLMs", loc="left", pad=14, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS); ax.spines["bottom"].set_color(AXIS)
    ax.grid(False); ax.grid(True, axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE)],
              ["Ours (measured)", "Published (MedReason Table 4)"],
              frameon=False, loc="lower right", fontsize=9)
    if notes:
        fig.text(0.01, -0.015, "  ".join(notes), fontsize=7.8, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig8_paper_ranking.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {os.path.join(out_dir, 'fig8_paper_ranking.png/.pdf')}")


if __name__ == "__main__":
    main()
