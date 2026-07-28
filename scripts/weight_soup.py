#!/usr/bin/env python
"""
Linear weight interpolation between the backbone and the fine-tune (WiSE-FT / model soup).

    theta(a) = (1 - a) * theta_backbone + a * theta_finetuned

Why this matters when the corpus is largely a replay: a full fine-tune of a model on data
it has already seen tends to trade some of its existing strengths for a small gain. The
interpolation lets you buy back the strengths continuously and pick the point that is
above the backbone on the benchmarks you care about, and below it on none. It costs no
training - only one evaluation pass per alpha.

    python scripts/weight_soup.py --base UCSC-VLAA/MedReason-8B \
        --finetuned ./ckpts/brainmed-mr8b/best --alpha 0.3 --out ./ckpts/soup-a0.3

Then evaluate ./ckpts/soup-a0.3 like any other checkpoint. Sweep alpha over 0.2/0.3/0.5
and keep the one that clears the no-regression gate with the highest average.
"""
import argparse
import json
import os
import shutil

import torch


def load_shards(path):
    """Yield (shard_file, {tensor_name: tensor}) for a local dir or a Hub repo id."""
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    if not os.path.isdir(path):
        path = snapshot_download(path, token=os.environ.get("HF_TOKEN"),
                                 allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        files = sorted(set(json.load(open(idx, encoding="utf-8"))["weight_map"].values()))
    else:
        files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    return path, files, load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="backbone (repo id or local dir)")
    ap.add_argument("--finetuned", required=True, help="fine-tuned checkpoint dir")
    ap.add_argument("--alpha", type=float, required=True, help="weight of the fine-tune, 0..1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")

    from safetensors.torch import save_file

    base_dir, base_files, load = load_shards(args.base)
    ft_dir, ft_files, _ = load_shards(args.finetuned)
    if base_files != ft_files:
        raise SystemExit(f"shard layouts differ:\n  base {base_files}\n  ft   {ft_files}")

    os.makedirs(args.out, exist_ok=True)
    a = args.alpha
    n_mixed = n_skipped = 0

    for fn in base_files:
        wb = load(os.path.join(base_dir, fn))
        wf = load(os.path.join(ft_dir, fn))
        if wb.keys() != wf.keys():
            raise SystemExit(f"tensor names differ in {fn}")
        merged = {}
        for k, tb in wb.items():
            tf = wf[k]
            if tb.shape != tf.shape or not tb.is_floating_point():
                merged[k] = tf          # e.g. resized embeddings, integer buffers
                n_skipped += 1
                continue
            merged[k] = ((1 - a) * tb.float() + a * tf.float()).to(tb.dtype)
            n_mixed += 1
        save_file(merged, os.path.join(args.out, fn), metadata={"format": "pt"})
        print(f"  {fn}: {len(merged)} tensors", flush=True)
        del wb, wf, merged

    # config, tokenizer and the shard index come from the fine-tune
    for fn in os.listdir(ft_dir):
        if fn.endswith(".safetensors"):
            continue
        src = os.path.join(ft_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, fn))

    json.dump({"base": args.base, "finetuned": args.finetuned, "alpha": a,
               "tensors_interpolated": n_mixed, "tensors_copied_from_finetune": n_skipped},
              open(os.path.join(args.out, "soup_recipe.json"), "w"), indent=2)
    print(f"\ninterpolated {n_mixed} tensors at alpha={a} -> {args.out}")
    if n_skipped:
        print(f"({n_skipped} tensors copied verbatim from the fine-tune: shape or dtype mismatch)")


if __name__ == "__main__":
    main()
