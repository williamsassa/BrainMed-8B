#!/usr/bin/env python
"""
Publish the selected checkpoint to the Hugging Face Hub.

Refuses to upload unless the checkpoint really is a full 8B model: it counts the
parameters in the safetensors shards and checks the config, so an adapter, a truncated
shard set or a half-written save cannot reach the Hub wearing the right name.

  python scripts/push_to_hf.py --checkpoint ./ckpts/brainmed-8b-v1/best \
      --repo_id BrainHealthAI/BrainMed-Reasoning-8B --results ./results --private
"""
import argparse
import json
import os
import shutil

MIN_PARAMS = 7.0e9
MAX_PARAMS = 9.0e9


def count_parameters(ckpt):
    """Sum tensor element counts across the safetensors shards, without loading weights."""
    from safetensors import safe_open

    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        files = sorted(set(json.load(open(idx, encoding="utf-8"))["weight_map"].values()))
    else:
        files = [f for f in os.listdir(ckpt) if f.endswith(".safetensors")]
    if not files:
        raise SystemExit(f"no safetensors in {ckpt} - refusing to upload")

    total, dtypes = 0, set()
    for fn in files:
        with safe_open(os.path.join(ckpt, fn), framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                n = 1
                for d in sl.get_shape():
                    n *= d
                total += n
                dtypes.add(sl.get_dtype())
    return total, files, sorted(dtypes)


def check_no_stray_single_file(ckpt, fix=False):
    """A sharded checkpoint must not also carry a single-file `model.safetensors`.

    `from_pretrained` resolves the single file *before* the shard index, so a stray one
    shadows the real weights and loads whatever it contains. Distributed saves are the
    usual source: ranks other than zero write a near-empty file next to rank 0's shards.
    """
    single = os.path.join(ckpt, "model.safetensors")
    index = os.path.join(ckpt, "model.safetensors.index.json")
    if not (os.path.exists(single) and os.path.exists(index)):
        return
    size_mb = os.path.getsize(single) / 1e6
    msg = (f"both model.safetensors ({size_mb:.1f} MB) and model.safetensors.index.json are "
           f"present: from_pretrained would load the single file and shadow the real shards")
    if not fix:
        raise SystemExit(f"{msg}\n  re-run with --fix_stray_weights to delete it")
    os.remove(single)
    print(f"  [fix] removed stray model.safetensors ({size_mb:.1f} MB)")


def verify(ckpt, fix_stray=False):
    print(f"[verify] {ckpt}")
    for req in ("config.json", "tokenizer_config.json"):
        if not os.path.exists(os.path.join(ckpt, req)):
            raise SystemExit(f"missing {req} - refusing to upload")
    check_no_stray_single_file(ckpt, fix=fix_stray)

    cfg = json.load(open(os.path.join(ckpt, "config.json"), encoding="utf-8"))
    for bad in ("peft_type", "base_model_name_or_path"):
        if bad in cfg:
            raise SystemExit(f"config.json contains '{bad}': this is an adapter, not a full model")

    n, files, dtypes = count_parameters(ckpt)
    size_gb = sum(os.path.getsize(os.path.join(ckpt, f)) for f in files) / 1e9
    print(f"  shards        : {len(files)} ({size_gb:.1f} GB on disk)")
    print(f"  dtypes        : {dtypes}")
    print(f"  parameters    : {n:,}  ({n / 1e9:.2f}B)")
    print(f"  architecture  : {cfg.get('architectures')} "
          f"L={cfg.get('num_hidden_layers')} d={cfg.get('hidden_size')} "
          f"vocab={cfg.get('vocab_size')}")
    if not (MIN_PARAMS <= n <= MAX_PARAMS):
        raise SystemExit(f"parameter count {n / 1e9:.2f}B outside the expected 8B band "
                         f"[{MIN_PARAMS / 1e9}, {MAX_PARAMS / 1e9}] - refusing to upload")
    print(f"  OK: full 8B model ({n / 1e9:.2f}B parameters)")
    return {"parameters": n, "parameters_b": round(n / 1e9, 2), "shards": len(files),
            "size_gb": round(size_gb, 1), "dtypes": [str(d) for d in dtypes],
            "architecture": cfg.get("architectures", [None])[0]}


def repair_remote(repo_id):
    """Remove a single-file model.safetensors that shadows the shard index on the Hub.

    `from_pretrained` resolves `model.safetensors` before `model.safetensors.index.json`, so
    a stray one - typically written by a non-zero rank during a distributed save - makes
    every download load a near-empty model while the real 16GB of shards sit unused beside
    it. Deleting the one file fixes the repo; nothing needs re-uploading.
    """
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("export HF_TOKEN first:  export HF_TOKEN=hf_...")
    api = HfApi(token=token)
    info = api.model_info(repo_id, files_metadata=True)
    names = {s.rfilename: (s.size or 0) for s in info.siblings}
    sharded = "model.safetensors.index.json" in names
    single = "model.safetensors" in names
    total = sum(v for k, v in names.items() if k.endswith(".safetensors")) / 1e9
    print(f"[{repo_id}] sharded index: {sharded} | single file: {single} | "
          f"safetensors total: {total:.1f} GB")

    if not (sharded and single):
        print("nothing to repair" if sharded else
              "WARNING: no shard index found - inspect this repo by hand")
        return
    mb = names["model.safetensors"] / 1e6
    api.delete_file("model.safetensors", repo_id=repo_id, repo_type="model",
                    commit_message="remove stray single-file weights shadowing the shards")
    info = api.model_info(repo_id, files_metadata=True)
    total = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9
    print(f"deleted model.safetensors ({mb:.1f} MB); safetensors remaining: {total:.1f} GB")
    if total < 10:
        raise SystemExit("less than 10 GB of weights remain - the repo is incomplete")
    print(f"repaired -> https://huggingface.co/{repo_id}")


def integrity_section(manifest_path):
    """State what was actually done to the corpus, read from the build manifest.

    This is a factual claim on a public artifact: it must reflect the run that produced
    these weights, not the pipeline's default. Keeping the overlapping rows is a legitimate
    choice - it matches the setting the published tables were produced under - but it has to
    be declared, together with the leakage it implies for the affected benchmarks.
    """
    if not os.path.exists(manifest_path):
        return ("_No build manifest was available when this card was written; the corpus "
                "preparation settings are unverified._")
    r = json.load(open(manifest_path, encoding="utf-8"))["report"]
    dec, align = r["decontamination"], r["answer_alignment"]
    kept = r["n_train_raw"] - dec["rows_removed"]
    leaks = {f[:-6]: c for f, c in r.get("contamination", {}).items()
             if c["contaminated_bench_questions"] > 0}

    lines = [f"Training rows: **{kept:,}**.", ""]
    if dec["enabled"]:
        lines += [f"The corpus was **decontaminated** by {r['ngram']}-gram overlap against every "
                  f"evaluation benchmark: {dec['rows_removed']} rows ({dec['pct_train']}%) were "
                  f"removed before training.", ""]
    else:
        lines += ["Rows overlapping the evaluation benchmarks were **kept**, matching the "
                  "upstream setting under which the published comparison tables were produced. "
                  "The overlap was measured and is declared below rather than removed; the "
                  "evaluation therefore reports a clean-subset score alongside the raw one, and "
                  "the clean figure is the unbiased estimate.", ""]
    if leaks:
        lines += ["| Benchmark | items | overlapping training rows |", "|---|---|---|"]
        lines += [f"| {b} | {c['n_bench']} | {c['contaminated_bench_questions']} "
                  f"({c['pct_bench']}%) |" for b, c in sorted(leaks.items())]
        lines += ["", "This overlap originates upstream: the MedReason corpus derives reasoning "
                  "traces from MMLU-medical, MedXpertQA and Humanity's Last Exam, three sets it "
                  "also evaluates on.", ""]
    if align["enabled"]:
        lines += [f"Answer-format alignment was applied to {align['train_rows_aligned']:,} rows "
                  f"({align['pct_train']}%): a canonical `The answer is X.` was appended where the "
                  f"letter was unambiguously recoverable from options already present in the "
                  f"question. Questions and reasoning traces are byte-identical to the source.", ""]
    return "\n".join(lines)


def model_card(args, facts, summary, meta, soup=None, integrity=""):
    rows = ""
    if summary:
        for k, v in summary.get("benchmarks", {}).items():
            rows += f"| {k} | {v['n']} | {v['accuracy']} |\n"
    avgs = {k: v for k, v in (summary or {}).items() if k.startswith("avg")}

    # Describe how this particular set of weights came about. A soup is not "the checkpoint
    # with the lowest validation loss", and saying so on the card would be false.
    if soup:
        a = soup["alpha"]
        provenance = (f"linear weight interpolation (WiSE-FT / model soup) between the backbone "
                      f"and the fine-tuned checkpoint `{os.path.basename(soup['finetuned'])}`, "
                      f"at alpha = {a}: `theta = {1 - a:.2f} * backbone + {a:.2f} * fine-tuned`")
    elif meta.get("val_loss") is not None:
        provenance = (f"lowest held-out validation loss (val loss {meta['val_loss']:.4f} at "
                      f"optimizer step {meta.get('opt_step')})")
    elif meta.get("final"):
        provenance = f"final checkpoint, end of epoch {meta.get('epoch', 0) + 1}"
    else:
        provenance = "end of a training epoch"
    return f"""---
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

# {args.repo_id.split('/')[-1]}

Full-parameter supervised fine-tune of `{args.base_model}` on
[`{args.dataset}`](https://huggingface.co/datasets/{args.dataset}), a union of
KG-grounded MedReason reasoning traces and verifier-checked medical-o1 traces.

- **Parameters:** {facts['parameters']:,} ({facts['parameters_b']}B) - full weights, not an adapter
- **Precision:** {', '.join(facts['dtypes'])}
- **Recipe:** MedReason (arXiv:2504.00993) - lr 5e-6, effective batch 128, 3 epochs,
  cosine schedule with 5% warmup, weight decay 0.1, DeepSpeed ZeRO-3, bf16
- **Weights obtained by:** {provenance}

## Output format

The model is trained to reason inside `<think>...</think>` and answer inside
`<answer>...</answer>`, under this system prompt:

```
{args.system_prompt}
```

## Evaluation

{'| Benchmark | n | Accuracy |' if rows else '_pending_'}
{'|---|---|---|' if rows else ''}
{rows}
{''.join(f'- **{k}**: {v}' + chr(10) for k, v in avgs.items())}

Scored with the MedReason evaluation harness (greedy decoding, strict answer prompt,
`max(head, tail)` extraction) on the benchmark files shipped by the MedReason and
HuatuoGPT-o1 repositories - no benchmark was rebuilt or resampled.

## Data integrity

{integrity}

## Intended use

Research and decision support. Not a medical device; not for autonomous clinical use.

## Citation

```bibtex
@misc{{wu2025medreason,
  title={{MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs}},
  author={{Wu, Juncheng and others}}, year={{2025}}, eprint={{2504.00993}}, archivePrefix={{arXiv}}
}}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="local checkpoint to publish (not needed with --repair_remote_only)")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--results", default=None, help="results dir to attach")
    ap.add_argument("--results_run", default=None,
                    help="run name whose scores go in the model card (default: inferred from "
                         "the checkpoint directory name)")
    ap.add_argument("--base_model", default="FreedomIntelligence/HuatuoGPT-o1-8B")
    ap.add_argument("--dataset", default="Williamsanderson/MedReason-MedO1-Reasoning-46K")
    ap.add_argument("--manifest", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "prepared", "manifest.json"),
        help="corpus build manifest; the data-integrity section is written from it")
    ap.add_argument("--system_prompt", default=(
        "You are a medical reasoning assistant. Work through the clinical problem step by step "
        "inside <think>...</think>, grounding every step in established medical knowledge, then "
        "give the final, complete answer inside <answer>...</answer>."))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--verify_only", action="store_true")
    ap.add_argument("--fix_stray_weights", action="store_true",
                    help="delete a single-file model.safetensors that shadows the shard index")
    ap.add_argument("--clean_remote", action="store_true",
                    help="also delete that file from the Hub repo if it is already there")
    ap.add_argument("--repair_remote_only", action="store_true",
                    help="inspect the Hub repo and remove a stray model.safetensors, without "
                         "re-uploading anything; --checkpoint is ignored")
    args = ap.parse_args()

    if args.repair_remote_only:
        repair_remote(args.repo_id)
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --repair_remote_only is given")

    facts = verify(args.checkpoint, fix_stray=args.fix_stray_weights)
    if args.verify_only:
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("export HF_TOKEN first")

    meta_path = os.path.join(args.checkpoint, "training_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}

    # Pick the summary that belongs to THIS checkpoint. Taking the alphabetically last one
    # silently attaches another candidate's scores to the card - with a sweep in the results
    # directory that is almost guaranteed to be the wrong model.
    summary = None
    if args.results:
        import glob
        cands = [p for p in glob.glob(os.path.join(args.results, "*", "summary_*.json"))
                 if "base-" not in os.path.basename(p)]
        want = args.results_run or os.path.basename(args.checkpoint.rstrip("/\\"))
        exact = [p for p in cands
                 if json.load(open(p, encoding="utf-8"))["run"].endswith(want)]
        if exact:
            summary = json.load(open(sorted(exact, key=len)[0], encoding="utf-8"))
            print(f"  attaching results from run '{summary['run']}'")
        elif args.results_run:
            raise SystemExit(f"no summary matching run '{args.results_run}' under {args.results}")
        elif cands:
            print(f"  WARNING: no summary matches checkpoint '{want}'; the card will carry no "
                  f"evaluation table. Pass --results_run to name the run explicitly.")

    soup_path = os.path.join(args.checkpoint, "soup_recipe.json")
    soup = json.load(open(soup_path, encoding="utf-8")) if os.path.exists(soup_path) else None

    integrity = integrity_section(args.manifest)
    card = os.path.join(args.checkpoint, "README.md")
    open(card, "w", encoding="utf-8").write(
        model_card(args, facts, summary, meta, soup, integrity))
    json.dump({"verification": facts, "training_meta": meta, "soup_recipe": soup,
               "results_run": (summary or {}).get("run")},
              open(os.path.join(args.checkpoint, "model_verification.json"), "w"), indent=2)

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    print(f"[upload] {args.checkpoint} -> {args.repo_id} (private={args.private})")
    api.upload_folder(folder_path=args.checkpoint, repo_id=args.repo_id, repo_type="model",
                      commit_message=f"full 8B SFT checkpoint ({facts['parameters_b']}B params)",
                      ignore_patterns=["*.pt", "optimizer*", "global_step*"])

    if args.results and os.path.isdir(args.results):
        staged = os.path.join(args.checkpoint, "_eval")
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(args.results, staged,
                        ignore=shutil.ignore_patterns("logs"))  # raw generations stay local
        api.upload_folder(folder_path=staged, repo_id=args.repo_id, repo_type="model",
                          path_in_repo="evaluation", commit_message="evaluation results and figures")
        shutil.rmtree(staged, ignore_errors=True)

    # post-upload readback: confirm the Hub really holds an 8B model and nothing shadows it
    info = api.model_info(args.repo_id, files_metadata=True)
    names = {s.rfilename: (s.size or 0) for s in info.siblings}
    remote_gb = sum(v for k, v in names.items() if k.endswith(".safetensors")) / 1e9
    print(f"[verify-remote] safetensors on the Hub: {remote_gb:.1f} GB")

    if "model.safetensors" in names and "model.safetensors.index.json" in names:
        mb = names["model.safetensors"] / 1e6
        if args.clean_remote:
            api.delete_file("model.safetensors", repo_id=args.repo_id, repo_type="model",
                            commit_message="remove stray single-file weights shadowing the shards")
            print(f"[verify-remote] deleted stray model.safetensors ({mb:.1f} MB) from the Hub")
            names.pop("model.safetensors")
            remote_gb = sum(v for k, v in names.items() if k.endswith(".safetensors")) / 1e9
        else:
            raise SystemExit(
                f"the Hub repo carries both model.safetensors ({mb:.1f} MB) and a shard index: "
                f"downloads would load the small file instead of the real weights.\n"
                f"  re-run with --clean_remote to delete it")

    if remote_gb < 10:
        raise SystemExit(f"only {remote_gb:.1f} GB uploaded - expected ~{facts['size_gb']} GB")
    print(f"done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
