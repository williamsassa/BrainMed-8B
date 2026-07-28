#!/usr/bin/env python
"""
Benchmark runner against an OpenAI-compatible completions endpoint (vLLM or SGLang).

Kept faithful to MedReason `src/evaluation/eval.py` on everything that moves the number
(strict prompt wording, greedy decoding, max(head,tail) scoring). Changed on purpose:

  * the training system prompt is injected at inference too. Our corpus is trained with a
    system turn defining the <think>/<answer> constitution; evaluating without it is a
    train/serve mismatch that costs accuracy and format compliance.
  * one result file per benchmark, plus a combined summary consumable by make_report.py.
  * bounded retries and chunked requests, so a 4000-question run does not die at 95%.

Usage:
  python eval/eval.py --model_path <hf-or-local> --port 30000 \
      --benchmarks eval/benchmarks --out_dir results/<run-name>
"""
import argparse
import json
import os
import sys
import time

import openai
from jinja2 import Template
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer import get_results, score  # noqa: E402

STRICT_PROMPT = ("Please answer the following multiple-choice questions, ensuring your response "
                 "concludes with the correct option in the format: 'The answer is A.'.\n"
                 "{question}\n{option_str}\n")
FREE_PROMPT = "Please answer the following multiple-choice question:\n{question}\n{option_str}\n"

DEFAULT_SYSTEM = (
    "You are a medical reasoning assistant. Work through the clinical problem step by step "
    "inside <think>...</think>, grounding every step in established medical knowledge, then "
    "give the final, complete answer inside <answer>...</answer>."
)


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="path/repo used to load the chat template")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--benchmarks", default="eval/benchmarks")
    ap.add_argument("--only", nargs="*", default=None, help="subset of benchmark names")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=2000)
    ap.add_argument("--chunk_size", type=int, default=256, help="prompts per HTTP request")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM,
                    help="must match training; pass '' to disable")
    ap.add_argument("--no_strict_prompt", action="store_true")
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--wandb_run_id", default=None, help="log results to an existing wandb run")
    args = ap.parse_args()

    run_name = args.run_name or os.path.basename(args.model_path.rstrip("/\\"))
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "result"), exist_ok=True)

    client = openai.Client(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    template = Template(tok.chat_template)
    query_prompt = FREE_PROMPT if args.no_strict_prompt else STRICT_PROMPT

    def render(user_text):
        msgs = []
        if args.system_prompt:
            msgs.append({"role": "system", "content": args.system_prompt})
        msgs.append({"role": "user", "content": user_text})
        return template.render(messages=msgs, bos_token=tok.bos_token, add_generation_prompt=True)

    def complete(prompts):
        last = None
        for attempt in range(args.max_retries):
            try:
                r = client.completions.create(model="default", prompt=prompts,
                                              temperature=args.temperature, top_p=args.top_p,
                                              max_tokens=args.max_new_tokens)
                # the server may reorder; `index` is authoritative
                texts = [""] * len(prompts)
                for c in r.choices:
                    texts[c.index] = c.text
                return texts
            except Exception as e:                                    # noqa: BLE001
                last = e
                wait = 5 * (attempt + 1)
                print(f"    request failed ({type(e).__name__}: {e}); retry in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"giving up after {args.max_retries} retries: {last}")

    # items whose question overlaps the training corpus, written by data/prepare_data.py
    ci_path = os.path.join(args.benchmarks, "contaminated_items.json")
    contaminated, kept_in_train = {}, None
    if os.path.exists(ci_path):
        blob = json.load(open(ci_path, encoding="utf-8"))
        contaminated = {k: set(v) for k, v in blob["items"].items()}
        kept_in_train = blob.get("training_set_kept_these_rows")
        flagged = sum(len(v) for v in contaminated.values())
        print(f"contamination map loaded: {flagged} flagged items across "
              f"{len(contaminated)} benchmarks; kept in training set = {kept_in_train}")
    else:
        print("no contaminated_items.json - clean-subset scores will not be reported")

    names = sorted(f[:-6] for f in os.listdir(args.benchmarks) if f.endswith(".jsonl"))
    if args.only:
        names = [n for n in names if n in args.only]
    if not names:
        sys.exit(f"no benchmarks found in {args.benchmarks}")

    summary = {"run": run_name, "model": args.model_path,
               "config": {k: v for k, v in vars(args).items() if k != "system_prompt"},
               "system_prompt": args.system_prompt, "benchmarks": {}}

    for name in names:
        rows = load_jsonl(os.path.join(args.benchmarks, f"{name}.jsonl"))
        print(f"\n=== {name}: {len(rows)} questions ===", flush=True)
        prompts = []
        for it in rows:
            it["option_str"] = "\n".join(f"{k}. {v}" for k, v in it["options"].items())
            it["input_str"] = query_prompt.format(question=it["question"], option_str=it["option_str"])
            prompts.append(render(it["input_str"]))
        if rows:
            print("--- example prompt ---\n" + prompts[0][:1200] + "\n---", flush=True)

        outputs = []
        for i in tqdm(range(0, len(prompts), args.chunk_size), desc=name):
            outputs.extend(complete(prompts[i:i + args.chunk_size]))

        for it, o in zip(rows, outputs):
            it["output"] = (o or "").lstrip().replace("</s>", "")

        log_path = os.path.join(args.out_dir, "logs", f"{run_name}__{name}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        res, _, _ = score(rows)
        merged = res.get(name) or list(res.values())[0]

        # second score on the items that do NOT overlap the training corpus
        bad = contaminated.get(name, set())
        if bad:
            clean = [r for i, r in enumerate(rows) if i not in bad]
            cres, _, _ = score(clean)
            cm = cres.get(name) or list(cres.values())[0]
            merged["accuracy_clean"] = cm["accuracy"]
            merged["n_clean"] = cm["n"]
            merged["n_contaminated"] = len(bad)
            merged["contamination_bias"] = round(merged["accuracy"] - cm["accuracy"], 2)
        else:
            merged["accuracy_clean"] = merged["accuracy"]
            merged["n_clean"] = merged["n"]
            merged["n_contaminated"] = 0
            merged["contamination_bias"] = 0.0

        summary["benchmarks"][name] = merged
        print(json.dumps({name: merged}, indent=2), flush=True)
        with open(os.path.join(args.out_dir, "result", f"result_{run_name}__{name}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    # paper-style averages, on both the raw and the clean-subset scores
    groups = {"common": ["medqa_4opt", "medmcqa_val", "pubmedqa_test", "mmlu_pro_medical"],
              "challenging": ["medbullets_op4", "medbullets_op5", "medxpertqa", "hle_med"]}
    groups["overall"] = groups["common"] + groups["challenging"]
    summary["training_kept_contaminated_rows"] = kept_in_train
    for g, keys in groups.items():
        for field, suffix in (("accuracy", ""), ("accuracy_clean", "_clean")):
            vals = [summary["benchmarks"][k][field] for k in keys if k in summary["benchmarks"]]
            if len(vals) == len(keys):
                summary[f"avg_{g}{suffix}"] = round(sum(vals) / len(vals), 2)

    sp = os.path.join(args.out_dir, f"summary_{run_name}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k.startswith("avg")}, indent=2))
    print(f"\nsummary -> {sp}")

    if args.wandb_run_id:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "brainmed-sft"),
                         id=args.wandb_run_id, resume="allow")
        run.log({f"eval/{k}": v["accuracy"] for k, v in summary["benchmarks"].items()})
        run.log({k: v for k, v in summary.items() if k.startswith("avg")})
        run.finish()


if __name__ == "__main__":
    main()
