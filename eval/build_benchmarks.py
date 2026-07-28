import argparse
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GIT_SOURCES = {
    "MedReason": "https://github.com/UCSC-VLAA/MedReason.git",
    "HuatuoGPT-o1": "https://github.com/FreedomIntelligence/HuatuoGPT-o1.git",
}

# name -> (source repo, relative path, key inside json or None for jsonl)
BUNDLE = {
    # --- common medical QA (scored on the HuatuoGPT-o1 bundle: matches published tables)
    "medqa_4opt":       ("huatuo", "evaluation/data/eval_data.json", "MedQA_USLME_test"),
    "medmcqa_val":      ("huatuo", "evaluation/data/eval_data.json", "MedMCQA_validation"),
    "pubmedqa_test":    ("huatuo", "evaluation/data/eval_data.json", "PubMedQA_test"),
    "mmlu_pro_medical": ("huatuo", "evaluation/data/eval_data.json", "MMLU-Pro_Medical_test"),
    # --- challenging clinical sets (shipped by MedReason)
    "medbullets_op4":   ("medreason", "eval_data/medbullets_op4.jsonl", None),
    "medbullets_op5":   ("medreason", "eval_data/medbullets_op5.jsonl", None),
    "medxpertqa":       ("medreason", "eval_data/MedXpertQA_test.jsonl", None),
    "hle_med":          ("medreason", "eval_data/HLE_biomed.jsonl", None),
    # --- extras (reported separately, not part of the paper's averages)
    "gpqa_medical":     ("huatuo", "evaluation/data/eval_data.json", "GPQA_Medical_test"),
    "medqa_5opt":       ("medreason", "eval_data/medqa_test.jsonl", None),
}

EXPECTED_ROWS = {
    "medqa_4opt": 1273, "medmcqa_val": 4183, "pubmedqa_test": 1000,
    "mmlu_pro_medical": 1535, "medbullets_op4": 308, "medbullets_op5": 308,
    "medxpertqa": 1449, "hle_med": 103, "gpqa_medical": 390, "medqa_5opt": 1273,
}

# benchmarks that make up the two averages reported in the paper
PAPER_GROUPS = {
    "common": ["medqa_4opt", "medmcqa_val", "pubmedqa_test", "mmlu_pro_medical"],
    "challenging": ["medbullets_op4", "medbullets_op5", "medxpertqa", "hle_med"],
}


def ensure_repo(path, url):
    if os.path.isdir(os.path.join(path, ".git")) or os.path.isdir(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"    cloning {url}", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", url, path], check=True)
    return path


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def normalise(rows, name):
    """Coerce every benchmark to {question, options{}, answer, answer_idx} and validate."""
    out, bad = [], 0
    for r in rows:
        # MedXpertQA ships `label` instead of `answer_idx`
        idx = r.get("answer_idx") or r.get("label")
        opts = r.get("options") or {}
        if not idx or not opts or idx not in opts:
            bad += 1
            continue
        rec = {
            "question": r["question"],
            "options": opts,
            "answer_idx": idx,
            "answer": r.get("answer", opts[idx]),
            "source": name,
        }
        if "id" in r:
            rec["id"] = r["id"]
        out.append(rec)
    return out, bad


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medreason_repo", default=os.path.join(REPO_ROOT, "..", "external", "MedReason"))
    ap.add_argument("--huatuo_repo", default=os.path.join(REPO_ROOT, "..", "external", "HuatuoGPT-o1"))
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "eval", "benchmarks"))
    args = ap.parse_args()

    repos = {
        "medreason": ensure_repo(os.path.abspath(args.medreason_repo), GIT_SOURCES["MedReason"]),
        "huatuo": ensure_repo(os.path.abspath(args.huatuo_repo), GIT_SOURCES["HuatuoGPT-o1"]),
    }
    os.makedirs(args.out_dir, exist_ok=True)

    json_cache = {}
    manifest = {"repos": {k: {"path": v, "commit": git_commit(v)} for k, v in repos.items()},
                "groups": PAPER_GROUPS, "benchmarks": {}}
    failures = []

    for name, (repo, rel, key) in BUNDLE.items():
        src = os.path.join(repos[repo], rel)
        if not os.path.exists(src):
            failures.append(f"{name}: missing {src}")
            continue
        if key is None:
            rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        else:
            if src not in json_cache:
                json_cache[src] = json.load(open(src, encoding="utf-8"))
            rows = json_cache[src][key]

        rows, bad = normalise(rows, name)
        exp = EXPECTED_ROWS.get(name)
        status = "OK" if exp is None or len(rows) == exp else f"MISMATCH (expected {exp})"
        if status != "OK":
            failures.append(f"{name}: {len(rows)} rows, {status}")

        dst = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(dst, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        manifest["benchmarks"][name] = {
            "rows": len(rows), "dropped_malformed": bad, "n_options": len(rows[0]["options"]) if rows else 0,
            "provenance": f"{repo}:{rel}" + (f"#{key}" if key else ""),
            "sha256": sha256_file(dst), "status": status,
        }
        print(f"  {name:18s} {len(rows):5d} rows  [{status}]  <- {repo}:{rel}", flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = sum(v["rows"] for v in manifest["benchmarks"].values())
    print(f"\n  {len(manifest['benchmarks'])} benchmarks, {total} questions -> {args.out_dir}")
    if failures:
        print("\n  FAILURES:")
        for x in failures:
            print("   -", x)
        sys.exit(1)


if __name__ == "__main__":
    main()
