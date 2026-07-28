import argparse
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLE4 = ["medbullets_op4", "medbullets_op5", "medxpertqa", "medqa_4opt", "medmcqa_val", "pubmedqa_test"]
COMMON = ["medqa_4opt", "medmcqa_val", "pubmedqa_test", "mmlu_pro_medical"]
CHALLENGING = ["medbullets_op4", "medbullets_op5", "medxpertqa", "hle_med"]
TABLE2 = COMMON + CHALLENGING                       # MedReason Table 2 column order
HUATUO = COMMON + ["gpqa_medical"]                  # HuatuoGPT-o1 Table 2 column set

SHORT = {"medqa_4opt": "MedQA", "medmcqa_val": "MedMCQA", "pubmedqa_test": "PubMedQA",
         "mmlu_pro_medical": "MMLU-Pro", "medbullets_op4": "MB-op4", "medbullets_op5": "MB-op5",
         "medxpertqa": "MedXpert", "hle_med": "HLE(med)", "gpqa_medical": "GPQA(med)",
         "medqa_5opt": "MedQA-5opt"}


def avg(vals):
    return round(sum(vals) / len(vals), 2) if vals else None


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--baselines", default=os.path.join(REPO_ROOT, "eval", "baselines.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "REPORT.md"))
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="points a benchmark may drop vs the backbone before the gate fails")
    ap.add_argument("--baseline_published_row", default=None,
                    help="row in table4_sota_comparison matching the measured backbone "
                         "(auto-detected from the run name when omitted)")
    args = ap.parse_args()

    base = json.load(open(args.baselines, encoding="utf-8"))
    # keys starting with "_" are documentation, not models
    t4 = {k: v for k, v in base["table4_sota_comparison"].items() if not k.startswith("_")}

    runs = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "*", "summary_*.json"))):
        s = json.load(open(p, encoding="utf-8"))
        runs[s["run"]] = s
    if not runs:
        raise SystemExit(f"no summary_*.json under {args.results_dir}")

    L = ["# BrainMed reasoning SFT - evaluation report", "",
         "Rows marked **(ours)** were measured by this pipeline (greedy decoding, strict prompt,",
         "MedReason scorer with `<answer>` support, training system prompt applied at inference).",
         "All other rows are transcribed from MedReason (arXiv:2504.00993, Table 4) and were",
         "**not** re-run.", ""]

    # ---------------- Table 4 style comparison ----------------
    L += ["## Comparison with published 7B-8B models (MedReason Table 4)", ""]
    headers = ["Model"] + [SHORT[k] for k in TABLE4] + ["Avg"]
    rows = []
    for name, vals in t4.items():
        rows.append([name] + vals[:-1] + [f"**{vals[-1]}**"])
    for name, s in runs.items():
        got = [s["benchmarks"].get(k, {}).get("accuracy") for k in TABLE4]
        a = avg([v for v in got if v is not None]) if all(v is not None for v in got) else None
        rows.append([f"**{name} (ours)**"] + got + [f"**{a}**" if a else None])
    L += [md_table(headers, rows), ""]

    # ---------------- Table 2 style: data ablation ----------------
    L += ["", "## Table 2 style - effect of the training data (MedReason Table 2)", "",
          "Published rows compare the *same backbone* trained on Huatuo CoT vs MedReason.",
          "Our row adds the union corpus. Only rows sharing a backbone are comparable:",
          "read the Llama3.1 block against a Llama3.1 run, not against a Huatuo-o1 run.", ""]
    t2 = {k: v for k, v in base["table2_data_ablation"].items() if not k.startswith("_")}
    headers = ["Model / data"] + [SHORT[k] for k in TABLE2] + ["Avg"]
    rows = [[n] + v[:-1] + [f"**{v[-1]}**"] for n, v in t2.items()]
    for name, s in runs.items():
        got = [s["benchmarks"].get(k, {}).get("accuracy") for k in TABLE2]
        if all(v is not None for v in got):
            rows.append([f"**{name} + ours (union) (ours)**"] + got + [f"**{avg(got)}**"])
    L += [md_table(headers, rows), ""]

    # ---------------- Table 3 style: common vs challenging ----------------
    L += ["## Table 3 style - common vs challenging averages (MedReason Table 3)", ""]
    t3 = {k: v for k, v in base["table3_reasoning_models"].items() if not k.startswith("_")}
    headers = ["Model / data", "Avg common (4)", "Avg challenging (4)", "Avg overall (8)"]
    rows = []
    for n, v in t3.items():
        c, d = v["common"][-1], v["challenging"][-1]
        rows.append([n, c, d, round((c + d) / 2, 2)])
    for name, s in runs.items():
        c = [s["benchmarks"].get(k, {}).get("accuracy") for k in COMMON]
        d = [s["benchmarks"].get(k, {}).get("accuracy") for k in CHALLENGING]
        if all(v is not None for v in c + d):
            rows.append([f"**{name} (ours)**", avg(c), avg(d), avg(c + d)])
    L += [md_table(headers, rows), ""]

    # ---------------- Huatuo Table 1/2 style ----------------
    L += ["## HuatuoGPT-o1 Table 1/2 style - core medical benchmarks", "",
          "`MMLU-Pro (Med)` and `GPQA (Med)` are the merged medical tracks shipped in",
          "`eval_data.json`; Huatuo's Table 1 splits them per track, which the released files",
          "do not allow us to reproduce, so this uses their Table 2 granularity.", ""]
    ht2 = {k: v for k, v in base["huatuo_table2_ablation"].items() if not k.startswith("_")}
    headers = ["Model"] + [SHORT[k] for k in HUATUO] + ["Avg"]
    rows = [[n] + v + [f"**{avg(v)}**"] for n, v in ht2.items()]
    for name, s in runs.items():
        got = [s["benchmarks"].get(k, {}).get("accuracy") for k in HUATUO]
        if all(v is not None for v in got):
            rows.append([f"**{name} (ours)**"] + got + [f"**{avg(got)}**"])
    L += [md_table(headers, rows), ""]

    # ---------------- Table 5 style: our own ablations ----------------
    variants = {n: s for n, s in runs.items() if not n.startswith("base-")}
    L += ["## Table 5 style - pipeline ablations", "",
          "MedReason's Table 5 ablates quality filtering. The equivalent knobs here are",
          "decontamination (`--no_decontaminate`) and MCQ answer-format alignment",
          "(`--no_answer_alignment`). Each additional arm costs one more training run - the",
          "table below fills in from whatever runs exist.", ""]
    headers = ["Run"] + [SHORT[k] for k in TABLE2] + ["Avg (8)", "Avg (Table 4, 6)"]
    rows = []
    for name, s in variants.items():
        got = [s["benchmarks"].get(k, {}).get("accuracy") for k in TABLE2]
        six = [s["benchmarks"].get(k, {}).get("accuracy") for k in TABLE4]
        rows.append([name] + got +
                    [avg(got) if all(v is not None for v in got) else "-",
                     avg(six) if all(v is not None for v in six) else "-"])
    L += [md_table(headers, rows) if rows else "_no fine-tuned run yet_", "",
          "The two averages cover different column sets and must never be compared to each "
          "other; `Avg (Table 4, 6)` is the one the published tables use.", ""]

    # ---------------- every benchmark actually evaluated ----------------
    # Driven by what is in the summaries, not by a fixed list, so extras such as
    # gpqa_medical and medqa_5opt are reported instead of silently dropped.
    all_keys = []
    for s in runs.values():
        for k in s["benchmarks"]:
            if k not in all_keys:
                all_keys.append(k)
    ordered = [k for k in COMMON + CHALLENGING if k in all_keys] + \
              [k for k in all_keys if k not in COMMON + CHALLENGING]
    L += ["## All evaluated benchmarks", "",
          f"Every benchmark scored in this run ({len(ordered)} of them). The paper's averages",
          "cover only the eight it reports; the extras below are measured and kept for the",
          "record but excluded from those averages.", ""]
    headers = ["Benchmark", "n", "In paper avg"] + [f"{n}" for n in runs]
    rows = []
    for k in ordered:
        first = next((s["benchmarks"][k] for s in runs.values() if k in s["benchmarks"]), None)
        rows.append([SHORT.get(k, k), first["n"] if first else "",
                     "yes" if k in COMMON + CHALLENGING else "extra"] +
                    [s["benchmarks"].get(k, {}).get("accuracy", "-") for s in runs.values()])
    L += [md_table(headers, rows), ""]

    # ---------------- full 8-benchmark view ----------------
    L += ["## Paper-comparable suite (8 benchmarks)", ""]
    headers = ["Run"] + [SHORT[k] for k in COMMON] + ["Avg common"] + \
              [SHORT[k] for k in CHALLENGING] + ["Avg challenging", "Avg overall"]
    # An average over a subset of the columns is not comparable to one over all of them.
    # Partial rows (e.g. a QUICK sweep that skipped MMLU-Pro and HLE) get "-" rather than a
    # number that looks like the others but is computed on different ground.
    def strict_avg(vals):
        return avg(vals) if all(v is not None for v in vals) else "-"

    rows, partial = [], []
    for name, s in runs.items():
        c = [s["benchmarks"].get(k, {}).get("accuracy") for k in COMMON]
        d = [s["benchmarks"].get(k, {}).get("accuracy") for k in CHALLENGING]
        if any(v is None for v in c + d):
            partial.append(name)
        rows.append([name] + c + [strict_avg(c)] + d + [strict_avg(d)] + [strict_avg(c + d)])
    L += [md_table(headers, rows), ""]
    if partial:
        L += [f"`-` marks an average that cannot be computed: {', '.join(partial)} "
              f"were scored on a subset of the suite (QUICK sweep). Re-run the full bundle "
              f"before quoting an 8-benchmark average for them.", ""]

    # ---------------- win conditions ----------------
    L += ["## Win conditions", "",
          "Two things must both hold: the model beats the backbone it started from, and it",
          "beats every published 7B-8B model in MedReason Table 4. The backbone is compared",
          "on numbers measured here; the others on their published rows.", ""]
    rows = []
    for name, s in runs.items():
        if name.startswith("base-"):
            continue
        got = [s["benchmarks"].get(k, {}).get("accuracy") for k in TABLE4]
        if any(v is None for v in got):
            continue
        a = avg(got)

        # condition 1 - beats its own backbone, measured on this harness
        bs = [n for n in runs if n.startswith("base-")]
        if bs:
            bgot = [runs[bs[0]]["benchmarks"].get(k, {}).get("accuracy") for k in TABLE4]
            if all(v is not None for v in bgot):
                ba = avg(bgot)
                rows.append([name, f"beats backbone ({bs[0]}, measured)", ba, a,
                             f"{a - ba:+.2f}", "PASS" if a > ba else "**FAIL**"])

        # condition 2 - beats every published model
        beaten = [(m, v[-1]) for m, v in t4.items() if a > v[-1]]
        lost_to = [(m, v[-1]) for m, v in t4.items() if a <= v[-1]]
        best = max(t4.items(), key=lambda kv: kv[1][-1])
        rows.append([name, f"beats published best ({best[0]})", best[1][-1], a,
                     f"{a - best[1][-1]:+.2f}", "PASS" if a > best[1][-1] else "**FAIL**"])
        rows.append([name, f"beats all {len(t4)} published models", "-", a,
                     f"{len(beaten)}/{len(t4)}",
                     "PASS" if not lost_to else "**FAIL** (" +
                     ", ".join(m for m, _ in lost_to[:3]) + ")"])
    L += [md_table(["Run", "Condition", "Reference", "Ours", "Delta", "Verdict"], rows)
          if rows else "_no fine-tuned run with the full Table 4 suite yet_", ""]

    # ---------------- controlled before/after + no-regression gate ----------------
    ft = [n for n in runs if not n.startswith("base-")]
    bs = [n for n in runs if n.startswith("base-")]
    if ft and bs:
        b = runs[bs[0]]
        for ftn in ft:
            f = runs[ftn]
            L += [f"## Controlled before/after - same harness ({bs[0]} -> {ftn})", ""]
            rows, regressions = [], []
            for k in COMMON + CHALLENGING:
                bv = b["benchmarks"].get(k, {}).get("accuracy")
                fv = f["benchmarks"].get(k, {}).get("accuracy")
                if bv is None or fv is None:
                    continue
                d = fv - bv
                verdict = "OK" if d >= -args.tolerance else "**REGRESSION**"
                if d < -args.tolerance:
                    regressions.append((SHORT[k], round(d, 2)))
                rows.append([SHORT[k], bv, fv, f"{d:+.2f}", verdict])
            if rows:
                deltas = [float(r[3]) for r in rows]
                rows.append(["**Average**", avg([r[1] for r in rows]), avg([r[2] for r in rows]),
                             f"**{avg(deltas):+.2f}**", ""])
                L += [md_table(["Benchmark", "Before", "After", "Delta", "Gate"], rows), ""]
                if regressions:
                    detail = ", ".join(f"{n} {d:+.2f}" for n, d in regressions)
                    L += [f"**GATE FAILED** - {len(regressions)} benchmark(s) below the "
                          f"-{args.tolerance} pt tolerance: {detail}.", "",
                          "The fine-tune traded existing strengths for its gains. Before "
                          "shipping, either lower the learning rate / shorten the schedule, or "
                          "interpolate towards the backbone with `scripts/weight_soup.py` and "
                          "re-evaluate.", ""]
                else:
                    L += [f"**GATE PASSED** - no benchmark regressed by more than "
                          f"{args.tolerance} pt; the backbone's strengths are preserved.", ""]

    # ---------------- harness calibration ----------------
    # The published tables were produced on the upstream harness. Ours differs (an <answer>
    # branch in the scorer, a system prompt at inference, format-aligned targets). Measuring
    # the untouched backbone here and diffing against its published row separates "our data
    # helped" from "our harness scores differently" - without it, every gain is ambiguous.
    bs = [n for n in runs if n.startswith("base-")]
    if bs:
        pub_key = args.baseline_published_row
        if not pub_key:
            for cand in t4:
                tag = cand.lower().replace("-", "").replace("_", "")
                if tag in bs[0].lower().replace("-", "").replace("_", ""):
                    pub_key = cand
                    break
            pub_key = pub_key or ("Huatuo-o1-RL-8B" if "huatuo" in bs[0].lower() else None)
        if pub_key and pub_key in t4:
            L += ["## Harness calibration - backbone measured here vs its published row", "",
                  f"Backbone `{bs[0]}` against the published **{pub_key}**. Small gaps are",
                  "expected; a large one means the harness, not the training, is moving the",
                  "numbers, and every delta downstream has to be read with that in mind.", ""]
            pub = dict(zip(TABLE4, t4[pub_key][:-1]))
            rows, gaps = [], []
            for k in TABLE4:
                mv = runs[bs[0]]["benchmarks"].get(k, {}).get("accuracy")
                if mv is None:
                    continue
                gaps.append(mv - pub[k])
                rows.append([SHORT[k], pub[k], mv, f"{mv - pub[k]:+.2f}"])
            if rows:
                rows.append(["**Mean gap**", "", "", f"**{avg(gaps):+.2f}**"])
                L += [md_table(["Benchmark", "Published", "Measured here", "Gap"], rows), ""]
                worst = max(abs(g) for g in gaps)
                L += [f"Largest single-benchmark gap: **{worst:.2f} pt**. "
                      + ("Harness reproduces the published setup closely; published values for "
                         "the other models are safe to compare against."
                         if worst <= 2.0 else
                         "That is large enough to confound a data-driven gain - prefer the "
                         "measured before/after over comparisons with published rows."), ""]

    # ---------------- raw vs clean-subset scores ----------------
    has_clean = any("accuracy_clean" in v for s in runs.values() for v in s["benchmarks"].values())
    if has_clean:
        kept = next((s.get("training_kept_contaminated_rows") for s in runs.values()
                     if s.get("training_kept_contaminated_rows") is not None), None)
        L += ["## Raw score vs clean subset", ""]
        if kept:
            L += ["The training set **kept** the rows overlapping these benchmarks (the upstream",
                  "setting, which is what the published tables were produced under). `Raw` is",
                  "therefore the number comparable to the paper; `Clean` excludes the overlapping",
                  "items and is the number to quote as an unbiased estimate. A large gap between",
                  "the two means the score leans on memorised evaluation items.", ""]
        else:
            L += ["The overlapping rows were removed from the training set, so `Raw` and `Clean`",
                  "should agree closely; a gap here would indicate leakage from another route.", ""]
        headers = ["Run", "Benchmark", "n", "Contaminated", "Raw", "Clean", "Bias"]
        rows = []
        for name, s in runs.items():
            for k in COMMON + CHALLENGING:
                v = s["benchmarks"].get(k)
                if not v or not v.get("n_contaminated"):
                    continue
                rows.append([name, SHORT[k], v["n"], v["n_contaminated"],
                             v["accuracy"], v["accuracy_clean"], f"{v['contamination_bias']:+.2f}"])
        for name, s in runs.items():
            if "avg_overall" in s and "avg_overall_clean" in s:
                rows.append([f"**{name}**", "**Average (8)**", "", "",
                             f"**{s['avg_overall']}**", f"**{s['avg_overall_clean']}**",
                             f"**{s['avg_overall'] - s['avg_overall_clean']:+.2f}**"])
        L += [md_table(headers, rows) if rows else "_no contaminated items flagged_", ""]

    # ---------------- integrity ----------------
    man = os.path.join(REPO_ROOT, "data", "prepared", "manifest.json")
    if os.path.exists(man):
        m = json.load(open(man, encoding="utf-8"))["report"]
        L += ["## Data integrity", "",
              f"- training rows after decontamination: **{m['n_train_raw'] - m['decontamination']['rows_removed']}** "
              f"(removed {m['decontamination']['rows_removed']}, {m['decontamination']['pct_train']}%)", ""]
        rows = [[f, c["n_bench"], c["contaminated_bench_questions"], f"{c['pct_bench']}%", c["train_rows_flagged"]]
                for f, c in m.get("contamination", {}).items()]
        if rows:
            L += ["Contamination found in the *source* corpus and removed before training:", "",
                  md_table(["Benchmark", "n", "Leaked items", "%", "Train rows dropped"], rows), ""]

    # ---------------- format compliance ----------------
    L += ["## Answer-format compliance", "",
          "Share of outputs where the answer was read from an explicit `The answer is X.` rather",
          "than recovered by the scorer's fuzzy fallbacks. A low value means the accuracy above",
          "leans on string matching and should be treated with suspicion.", ""]
    rows = []
    for name, s in runs.items():
        for k in ordered:
            v = s["benchmarks"].get(k)
            if v:
                rows.append([name, SHORT.get(k, k), f"{v.get('format_compliance', 0)}%",
                             v.get("empty_outputs", 0)])
    L += [md_table(["Run", "Benchmark", "Strict-format %", "Empty outputs"], rows), ""]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
