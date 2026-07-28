import argparse
import collections
import hashlib
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEFAULT_DATASET = "Williamsanderson/MedReason-MedO1-Reasoning-46K"

SYSTEM_PROMPT = (
    "You are a medical reasoning assistant. Work through the clinical problem step by step "
    "inside <think>...</think>, grounding every step in established medical knowledge, then "
    "give the final, complete answer inside <answer>...</answer>."
)

# --------------------------------------------------------------------------------------
# text normalisation / n-gram contamination
# --------------------------------------------------------------------------------------


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def shingles(s: str, n: int = 13):
    w = norm(s).split()
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_index(train_rows, n=13):
    """Inverted n-gram index over the training questions, built once for all benchmarks."""
    idx = collections.defaultdict(set)
    sizes = []
    for i, r in enumerate(train_rows):
        s = shingles(r["question"], n)
        sizes.append(len(s))
        for g in s:
            idx[g].add(i)
    return idx, sizes


def find_contaminated(index, sizes, bench_questions, n=13, thresh=0.5):
    """Return (train row indices, benchmark item indices) that overlap.

    Flags both directions: benchmark-in-train (verbatim copy of the eval item) and
    train-in-benchmark (the eval item is a superset, e.g. options appended).
    """
    hit_rows, hit_bench = set(), []
    for bi, q in enumerate(bench_questions):
        bs = shingles(q, n)
        if not bs:
            continue
        cnt = collections.Counter()
        for g in bs:
            for i in index.get(g, ()):
                cnt[i] += 1
        flagged = False
        for i, c in cnt.items():
            if c / len(bs) >= thresh or c / max(1, sizes[i]) >= thresh:
                hit_rows.add(i)
                flagged = True
        if flagged:
            hit_bench.append(bi)
    return hit_rows, hit_bench


# --------------------------------------------------------------------------------------
# MCQ answer-format alignment
# --------------------------------------------------------------------------------------

OPT_RE = re.compile(r"(?m)^\s*([A-J])[\.\)]\s+(.+?)\s*$")


def recover_letter(question: str, answer: str):
    """Recover the option letter of the gold answer when the question carries inline options.

    Returns None when it cannot be resolved unambiguously - never guesses.
    """
    opts = OPT_RE.findall(question or "")
    if len(opts) < 2:
        return None
    a = norm(answer)
    if not a:
        return None
    exact = {L for L, t in opts if norm(t) and norm(t) == a}
    if len(exact) == 1:
        return exact.pop()
    starts = {L for L, t in opts if norm(t) and a.startswith(norm(t))}
    if len(starts) == 1:
        return starts.pop()
    inside = {L for L, t in opts if norm(t) and len(norm(t)) > 3 and norm(t) in a}
    if len(inside) == 1:
        return inside.pop()
    return None


def align_answer(question: str, answer: str):
    """Append a canonical 'The answer is X.' when the letter is unambiguous.

    Mirrors the extraction pattern the eval harness looks for, so the model's
    MCQ output format matches what the scorer parses first.
    """
    if re.search(r"answer is\s*[A-J]\b", answer or ""):
        return answer, False
    letter = recover_letter(question, answer)
    if letter is None:
        return answer, False
    return f"{answer.rstrip()}\n\nThe answer is {letter}.", True


# --------------------------------------------------------------------------------------


def build_messages(row, system_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["question"]},
        {"role": "assistant",
         "content": f"<think>\n{row['reasoning']}\n</think>\n<answer>\n{row['answer']}\n</answer>"},
    ]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--bench_dir", default=os.path.join(REPO_ROOT, "eval", "benchmarks"),
                    help="output of eval/build_benchmarks.py")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "data", "prepared"))
    ap.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    # Nothing is generated: no question, no reasoning, no answer is ever authored here.
    # `question` and `reasoning` are always byte-identical to the published dataset and the
    # run asserts it. The only text this pipeline may touch is the tail of `answer`, and
    # only to restate a letter already present in the row's own options.
    ap.add_argument("--decontaminate", action="store_true",
                    help="drop training rows overlapping an eval benchmark (the scan always "
                         "runs and is always reported; this only controls removal)")
    ap.add_argument("--answer_alignment", action="store_true", default=True,
                    help="append a canonical 'The answer is X.' when the letter is recoverable "
                         "unambiguously from the options already present in the question")
    ap.add_argument("--no_answer_alignment", dest="answer_alignment", action="store_false",
                    help="keep answers byte-identical to the source, as in the MedReason paper")
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--seed", type=int, default=2002)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load source corpus ------------------------------------------------------------
    from huggingface_hub import snapshot_download
    print(f"[1/5] downloading {args.dataset}", flush=True)
    local = snapshot_download(args.dataset, repo_type="dataset",
                              token=os.environ.get("HF_TOKEN"))
    splits = {}
    for sp in ("train", "val", "test"):
        p = os.path.join(local, f"{sp}.jsonl")
        splits[sp] = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print("    ", {k: len(v) for k, v in splits.items()}, flush=True)

    # ---- integrity gate ----------------------------------------------------------------
    print("[2/5] integrity checks", flush=True)
    problems = []
    seen_q = {}
    for sp, rows in splits.items():
        for r in rows:
            for k in ("id", "question", "reasoning", "answer", "source", "source_subset"):
                if not r.get(k):
                    problems.append(f"{sp}:{r.get('id')} missing {k}")
            nq = norm(r["question"])
            if nq in seen_q and seen_q[nq] != sp:
                problems.append(f"cross-split duplicate: {r['id']} also in {seen_q[nq]}")
            seen_q[nq] = sp
    if problems:
        print(f"    {len(problems)} integrity problems, first 5: {problems[:5]}")
    else:
        print("    OK: no missing fields, no cross-split duplicates", flush=True)

    # ---- decontamination ---------------------------------------------------------------
    train = splits["train"]
    report = {"dataset": args.dataset, "n_train_raw": len(train),
              "n_val": len(splits["val"]), "n_test": len(splits["test"]),
              "ngram": args.ngram, "contamination": {}}

    # The scan always runs, even when nothing is removed: knowing which *benchmark* items
    # overlap the training set is what lets the evaluation report a clean-subset score
    # alongside the raw one. Keeping the rows is a choice; not knowing is not.
    print("[3/5] contamination scan against the evaluation bundle", flush=True)
    if not os.path.isdir(args.bench_dir):
        sys.exit(f"missing {args.bench_dir} - run eval/build_benchmarks.py first")
    index, sizes = build_index(train, args.ngram)
    contaminated_items = {}
    removed = set()
    for fn in sorted(os.listdir(args.bench_dir)):
        if not fn.endswith(".jsonl"):
            continue
        bench = [json.loads(l) for l in
                 open(os.path.join(args.bench_dir, fn), encoding="utf-8") if l.strip()]
        rows, bench_idx = find_contaminated(index, sizes, [b["question"] for b in bench], args.ngram)
        nb = len(bench_idx)
        contaminated_items[fn[:-6]] = bench_idx
        if args.decontaminate:
            removed |= rows
        report["contamination"][fn] = {
            "n_bench": len(bench), "contaminated_bench_questions": nb,
            "pct_bench": round(100 * nb / max(1, len(bench)), 2),
            "train_rows_flagged": len(rows),
        }
        print(f"     {fn:32s} {nb:5d}/{len(bench):5d} bench items "
              f"({100 * nb / max(1, len(bench)):5.2f}%) <- {len(rows)} train rows", flush=True)

    # consumed by eval/eval.py to compute accuracy on the clean subset of each benchmark
    ci_path = os.path.join(args.bench_dir, "contaminated_items.json")
    with open(ci_path, "w", encoding="utf-8") as f:
        json.dump({"ngram": args.ngram, "threshold": 0.5,
                   "training_set_kept_these_rows": not args.decontaminate,
                   "items": contaminated_items}, f, indent=2)
    print(f"    contaminated benchmark items -> {ci_path}", flush=True)

    report["decontamination"] = {
        "enabled": args.decontaminate,
        "rows_removed": len(removed),
        "pct_train": round(100 * len(removed) / len(train), 3),
        "by_subset": dict(collections.Counter(train[i]["source_subset"] for i in removed)),
        "removed_ids": sorted(train[i]["id"] for i in removed),
    }
    kept = [r for i, r in enumerate(train) if i not in removed]
    if args.decontaminate:
        print(f"    removed {len(removed)} rows -> {len(kept)} kept", flush=True)
    else:
        print(f"    KEEPING all {len(kept)} rows: the training set matches the upstream "
              f"setting, so the evaluation reports clean-subset scores alongside the raw ones",
              flush=True)

    # ---- formatting --------------------------------------------------------------------
    # Nothing is generated here. `messages` is a re-packaging of question/reasoning/answer
    # into the chat schema the trainer consumes; the three text fields are copied verbatim.
    print("[4/5] formatting (extraction only)", flush=True)
    n_aligned = 0
    out_rows = {"train": [], "val": []}
    for sp, rows in (("train", kept), ("val", splits["val"])):
        for r in rows:
            ans = r["answer"]
            if args.answer_alignment:
                ans, did = align_answer(r["question"], ans)
                n_aligned += did and sp == "train"
            rec = dict(r)
            rec["answer"] = ans
            rec["messages"] = build_messages(rec, args.system_prompt)
            out_rows[sp].append({
                "id": rec["id"], "source": rec["source"], "source_subset": rec["source_subset"],
                "specialty": rec.get("specialty"), "question": rec["question"],
                "reasoning": rec["reasoning"], "answer": rec["answer"],
                "messages": rec["messages"],
            })
    report["answer_alignment"] = {
        "enabled": args.answer_alignment,
        "train_rows_aligned": int(n_aligned),
        "pct_train": round(100 * n_aligned / max(1, len(kept)), 2),
    }
    print(f"    answer-format alignment: "
          f"{'applied to %d train rows' % n_aligned if args.answer_alignment else 'OFF'}", flush=True)

    # ---- verbatim assertion ------------------------------------------------------------
    # Prove, rather than claim, what was and was not touched. `question` and `reasoning`
    # must be byte-identical to the source in every mode - no exception, no tolerance.
    # `answer` may differ only on the rows the alignment reports, and only by a suffix.
    src_by_id = {r["id"]: r for rows in splits.values() for r in rows}
    altered = collections.Counter()
    bad_suffix = []
    for rows in out_rows.values():
        for r in rows:
            s = src_by_id[r["id"]]
            for field in ("question", "reasoning"):
                if r[field] != s[field]:
                    altered[field] += 1
            if r["answer"] != s["answer"]:
                altered["answer"] += 1
                if not r["answer"].startswith(s["answer"].rstrip()):
                    bad_suffix.append(r["id"])

    n_aligned_all = altered["answer"]
    report["verbatim_check"] = {
        "rows_compared": sum(len(v) for v in out_rows.values()),
        "question_altered": altered["question"],
        "reasoning_altered": altered["reasoning"],
        "answer_altered": n_aligned_all,
        "answer_changes_are_suffix_only": not bad_suffix,
    }
    if altered["question"] or altered["reasoning"]:
        sys.exit(f"ABORT: question/reasoning must never be modified, found "
                 f"{altered['question']} / {altered['reasoning']} altered rows")
    if bad_suffix:
        sys.exit(f"ABORT: {len(bad_suffix)} answers were rewritten rather than suffixed, "
                 f"first: {bad_suffix[:3]}")
    if not args.answer_alignment and n_aligned_all:
        sys.exit(f"ABORT: {n_aligned_all} answers differ with alignment disabled")
    print(f"    verbatim check: question + reasoning byte-identical on all "
          f"{report['verbatim_check']['rows_compared']} rows; "
          f"answer suffixed on {n_aligned_all} rows, rewritten on 0", flush=True)

    # ---- write -------------------------------------------------------------------------
    print("[5/5] writing", flush=True)
    manifest = {"config": vars(args), "report": report, "files": {}}
    for sp, rows in out_rows.items():
        p = os.path.join(args.out_dir, f"{sp}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["files"][f"{sp}.jsonl"] = {"rows": len(rows), "sha256": sha256_file(p)}
        print(f"    {p}: {len(rows)} rows", flush=True)

    report["train_composition"] = dict(collections.Counter(r["source"] for r in out_rows["train"]))
    report["train_subsets"] = dict(collections.Counter(r["source_subset"] for r in out_rows["train"]))

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("    manifest.json written", flush=True)


if __name__ == "__main__":
    main()
