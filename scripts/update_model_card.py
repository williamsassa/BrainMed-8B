import argparse
import json
import os
import textwrap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLE4 = ["medbullets_op4", "medbullets_op5", "medxpertqa", "medqa_4opt", "medmcqa_val", "pubmedqa_test"]
EXTRA = ["mmlu_pro_medical", "hle_med", "gpqa_medical", "medqa_5opt"]
SHORT = {"medqa_4opt": "MedQA", "medmcqa_val": "MedMCQA", "pubmedqa_test": "PubMedQA",
         "mmlu_pro_medical": "MMLU-Pro (Med)", "medbullets_op4": "MedBullets op4",
         "medbullets_op5": "MedBullets op5", "medxpertqa": "MedXpertQA",
         "hle_med": "HLE (med)", "gpqa_medical": "GPQA (Med)", "medqa_5opt": "MedQA (5-opt)"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, ORANGE = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#eb6834"
PANEL = "#f4f3f0"


def avg(v):
    return round(sum(v) / len(v), 2) if v else None


def render_case(case, out_path, max_reasoning=900):
    """Render one question / reasoning / answer as a standalone image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"]})

    def wrap(t, w=118):
        out = []
        for para in (t or "").split("\n"):
            out.extend(textwrap.wrap(para, w) or [""])
        return out

    q = wrap(case["question"], 112)
    think = (case.get("think") or "").strip()
    if len(think) > max_reasoning:
        think = think[:max_reasoning].rsplit(" ", 1)[0] + " …"
    r = wrap(think, 118)
    a = wrap((case.get("answer") or "").strip(), 118)

    lh = 0.185                                     # line height, inches
    blocks = [("Question", q), ("Model's reasoning (<think>)", r), ("Answer (<answer>)", a)]
    fh = 0.7 + sum(0.42 + lh * len(b[1]) for b in blocks) + 0.35
    fig = plt.figure(figsize=(12.6, fh))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 1 - 0.34 / fh
    ax.text(0.015, y, case["label"].upper(), fontsize=10, color=ORANGE, fontweight="bold",
            va="top")
    y -= 0.34 / fh

    for i, (title, lines) in enumerate(blocks):
        block_h = (0.30 + lh * len(lines)) / fh
        if i == 1:                                  # tint the reasoning panel
            ax.add_patch(Rectangle((0.008, y - block_h), 0.984, block_h,
                                   facecolor=PANEL, edgecolor="none", zorder=0))
        ax.text(0.015, y - 0.06 / fh, title, fontsize=9.5, color=INK2, fontweight="bold",
                va="top", zorder=2)
        yy = y - 0.28 / fh
        for ln in lines:
            ax.text(0.015, yy, ln, fontsize=9.2, color=INK if i == 2 else INK2,
                    va="top", family="monospace" if i == 0 else None, zorder=2)
            yy -= lh / fh
        y -= block_h
        ax.plot([0.008, 0.992], [y, y], color=GRID, lw=0.8, zorder=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered {out_path}")


def benchmark_table(ours, t4, name):
    rows = []
    for m, v in sorted(t4.items(), key=lambda kv: kv[1][-1]):
        rows.append([m] + [f"{x:.1f}" for x in v[:-1]] + [f"{v[-1]:.1f}", "paper"])
    got = [ours["benchmarks"][k]["accuracy"] for k in TABLE4]
    rows.append([f"**{name}**"] + [f"**{x:.2f}**" for x in got] +
                [f"**{avg(got):.2f}**", "measured"])
    head = ["Model"] + [SHORT[k] for k in TABLE4] + ["Avg", "Source"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def extra_table(ours, name):
    keys = [k for k in EXTRA if k in ours["benchmarks"]]
    if not keys:
        return ""
    head = ["Benchmark", "n", f"{name}"]
    out = ["| " + " | ".join(head) + " |", "|---|---|---|"]
    for k in keys:
        v = ours["benchmarks"][k]
        out.append(f"| {SHORT[k]} | {v['n']} | {v['accuracy']:.2f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--run", required=True, help="results run name to publish, e.g. brainmed-8b-final")
    ap.add_argument("--display_name", default="BrainMed-8B")
    ap.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--baselines", default=os.path.join(REPO_ROOT, "eval", "baselines.json"))
    ap.add_argument("--smoke", default=os.path.join(REPO_ROOT, "results", "smoke_test.json"))
    ap.add_argument("--figures", nargs="*", default=[
        "fig8_paper_ranking.png", "fig4_training_curves.png", "table_A_ranking.png"])
    ap.add_argument("--base_model", default="FreedomIntelligence/HuatuoGPT-o1-8B")
    ap.add_argument("--dataset", default="Williamsanderson/MedReason-MedO1-Reasoning-46K")
    ap.add_argument("--dry_run", action="store_true", help="write the card locally, do not upload")
    args = ap.parse_args()

    summary_path = os.path.join(args.results_dir, args.run, f"summary_{args.run}.json")
    if not os.path.exists(summary_path):
        raise SystemExit(f"missing {summary_path}")
    ours = json.load(open(summary_path, encoding="utf-8"))
    t4 = {k: v for k, v in json.load(open(args.baselines, encoding="utf-8"))
          ["table4_sota_comparison"].items() if not k.startswith("_")}

    staging = os.path.join(REPO_ROOT, "results", "_card")
    assets = os.path.join(staging, "assets")
    os.makedirs(assets, exist_ok=True)

    # ---- figures ----------------------------------------------------------------------
    import shutil
    kept_figs = []
    for fn in args.figures[:3]:
        src = os.path.join(args.results_dir, "figures", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(assets, fn))
            kept_figs.append(fn)
        else:
            print(f"  WARNING: figure not found, skipped: {src}")

    # ---- sample answers ---------------------------------------------------------------
    sample_imgs = []
    if os.path.exists(args.smoke):
        tr = json.load(open(args.smoke, encoding="utf-8"))["transcript"]
        wanted = ["multiple choice, strict format", "differential diagnosis"]
        for i, label in enumerate(wanted, start=2):
            case = next((c for c in tr if c["label"] == label), None)
            if case is None:
                print(f"  WARNING: no case labelled '{label}' in {args.smoke}")
                continue
            fn = f"sample_{i}_{label.split(',')[0].replace(' ', '_')}.png"
            render_case(case, os.path.join(assets, fn))
            sample_imgs.append((label, fn))
    else:
        print(f"  WARNING: {args.smoke} not found - run scripts/smoke_test_model.py first")

    base = f"https://huggingface.co/{args.repo_id}/resolve/main/assets"
    fig_md = "\n\n".join(f"![{f[:-4]}]({base}/{f})" for f in kept_figs)
    sample_md = "\n\n".join(f"**{lbl.capitalize()}**\n\n![{lbl}]({base}/{f})"
                            for lbl, f in sample_imgs)

    card = f"""---
license: apache-2.0
base_model: {args.base_model}
library_name: transformers
pipeline_tag: text-generation
tags:
- medical
- clinical-reasoning
- chain-of-thought
- sft
language:
- en
---

# {args.display_name}

Full-parameter fine-tune of `{args.base_model}` on
[`{args.dataset}`](https://huggingface.co/datasets/{args.dataset}) — a union of KG-grounded
MedReason reasoning traces and verifier-checked medical-o1 traces.

> **Research model. Not medical advice, not a medical device, not clinically validated.**
> Do not use it for decisions about any real person.

## Results

Accuracy (%) on the six benchmarks of MedReason Table 4. The **{args.display_name}** row was
measured here; every other row is transcribed from the MedReason paper (arXiv:2504.00993) and
was not re-run.

{benchmark_table(ours, t4, args.display_name)}

{('### Additional benchmarks' + chr(10) + chr(10) + extra_table(ours, args.display_name))
 if extra_table(ours, args.display_name) else ''}

**Read the comparison with this caveat.** Our numbers come from our own evaluation harness
(greedy decoding, the training system prompt applied at inference). Scoring the *untouched*
backbone on that harness gives **+2.89 points** over its published row, so part of the gap to
the paper rows is protocol, not model. The comparison that is free of this effect is the
before/after against the backbone measured on the same harness, published in
[`evaluation/REPORT.md`](https://huggingface.co/{args.repo_id}/blob/main/evaluation/REPORT.md).

{fig_md}

## Sample answers

Produced by this checkpoint, greedy decoding, with the system prompt below.

{sample_md}

Note what these show and what they do not: the model is trained and evaluated on medical
**question answering**. It handles multiple choice and differential diagnosis well. It is
weaker on open-ended acute management — its corpus contains QA pairs, not treatment
protocols — and it should not be relied on there.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{args.repo_id}"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

SYSTEM = ("You are a medical reasoning assistant. Work through the clinical problem step by "
          "step inside <think>...</think>, grounding every step in established medical "
          "knowledge, then give the final, complete answer inside <answer>...</answer>.")

messages = [{{"role": "system", "content": SYSTEM}},
            {{"role": "user", "content": "How is eclampsia-related seizure managed?"}}]
inputs = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
             return_tensors="pt").to(model.device)
print(tok.decode(model.generate(**inputs, max_new_tokens=1024)[0], skip_special_tokens=True))
```

The system prompt is part of the contract: the model was trained under it and answers as
`<think>…</think><answer>…</answer>`. Dropping it measurably lowers accuracy.

## Training

MedReason recipe (arXiv:2504.00993): lr 5e-6, effective batch 128, 3 epochs, cosine schedule
with 5% warmup, weight decay 0.1, DeepSpeed ZeRO-3, bf16, full parameters — no adapter.
4×H100, ~1.5 h. Full pipeline, evaluation logs and figures are under
[`evaluation/`](https://huggingface.co/{args.repo_id}/tree/main/evaluation).

## Citation

```bibtex
@misc{{wu2025medreason,
  title={{MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs}},
  author={{Wu, Juncheng and others}}, year={{2025}}, eprint={{2504.00993}}, archivePrefix={{arXiv}}
}}
```
"""
    readme = os.path.join(staging, "README.md")
    open(readme, "w", encoding="utf-8").write(card)
    print(f"\ncard -> {readme}  ({len(kept_figs)} figures, {len(sample_imgs)} samples)")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("export HF_TOKEN first")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_folder(folder_path=staging, repo_id=args.repo_id, repo_type="model",
                      commit_message="model card: results table, figures, sample answers")
    print(f"done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
