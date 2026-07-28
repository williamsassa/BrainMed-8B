import argparse
import glob
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "eval"))

TABLE4 = ["medbullets_op4", "medbullets_op5", "medxpertqa", "medqa_4opt", "medmcqa_val", "pubmedqa_test"]
SHORT = {"medqa_4opt": "MedQA", "medmcqa_val": "MedMCQA", "pubmedqa_test": "PubMedQA",
         "mmlu_pro_medical": "MMLU-Pro", "medbullets_op4": "MB-op4", "medbullets_op5": "MB-op5",
         "medxpertqa": "MedXpert", "hle_med": "HLE(med)", "gpqa_medical": "GPQA(med)",
         "medqa_5opt": "MedQA-5opt"}


def binom_two_sided_p(b, c):
    """Exact McNemar: P(|split| at least this lopsided) under a fair coin on b+c trials."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_items(results_dir, run, bench):
    """{question: correct?} for one run on one benchmark."""
    path = os.path.join(results_dir, run, "logs", f"{run}__{bench}.json")
    if not os.path.exists(path):
        return None
    from scorer import match_choice
    out = {}
    for it in json.load(open(path, encoding="utf-8")):
        gold = it["answer_idx"].lower()
        ans = it.get("ans")
        if ans is None:
            ans, _ = match_choice(it.get("output") or "", it["options"])
        # eval.py scores with max(head, tail); head is the answer the model committed to
        out[it["question"]] = (ans[0].lower() == gold)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--a", required=True, help="baseline run name")
    ap.add_argument("--b", required=True, help="candidate run name")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    benches = sorted({os.path.basename(p).split("__")[-1][:-5]
                      for p in glob.glob(os.path.join(args.results_dir, args.a, "logs", "*.json"))})
    if not benches:
        raise SystemExit(f"no per-item logs under {args.results_dir}/{args.a}/logs")

    L = [f"# Paired significance: `{args.b}` vs `{args.a}`", "",
         "McNemar's exact test on the items both runs answered. `only B` counts items the",
         "candidate gets right and the baseline gets wrong; `only A` the reverse. Items both",
         "get right or both get wrong are uninformative about which model is better.", ""]
    rows, n_sig_better, n_sig_worse, tot_b, tot_c = [], 0, 0, 0, 0

    for bench in benches:
        A = load_items(args.results_dir, args.a, bench)
        B = load_items(args.results_dir, args.b, bench)
        if not A or not B:
            continue
        shared = [q for q in A if q in B]
        if not shared:
            continue
        b = sum(1 for q in shared if A[q] and not B[q])   # only A correct
        c = sum(1 for q in shared if B[q] and not A[q])   # only B correct
        acc_a = 100 * sum(A[q] for q in shared) / len(shared)
        acc_b = 100 * sum(B[q] for q in shared) / len(shared)
        p = binom_two_sided_p(b, c)
        tot_b += b
        tot_c += c
        if p < args.alpha:
            if c > b:
                n_sig_better += 1
                verdict = "**better**"
            else:
                n_sig_worse += 1
                verdict = "**worse**"
        else:
            verdict = "no difference"
        rows.append([SHORT.get(bench, bench), len(shared), round(acc_a, 2), round(acc_b, 2),
                     f"{acc_b - acc_a:+.2f}", b, c, f"{p:.3f}", verdict])

    hdr = ["Benchmark", "n", args.a, args.b, "Delta", "only A", "only B", "p", f"verdict (a={args.alpha})"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    L += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]

    p_all = binom_two_sided_p(tot_b, tot_c)
    L += ["", "## Pooled across benchmarks", "",
          f"- items only `{args.a}` gets right: **{tot_b}**",
          f"- items only `{args.b}` gets right: **{tot_c}**",
          f"- McNemar exact p = **{p_all:.4g}**", ""]
    if p_all < args.alpha:
        better = args.b if tot_c > tot_b else args.a
        L += [f"Pooled, `{better}` is significantly better (p < {args.alpha}).", ""]
    else:
        L += [f"Pooled, the two runs are **statistically indistinguishable** "
              f"(p = {p_all:.3g}). Any average difference between them is within sampling noise.", ""]
    L += [f"Per benchmark: {n_sig_better} significantly better, {n_sig_worse} significantly "
          f"worse, {len(rows) - n_sig_better - n_sig_worse} indistinguishable.", "",
          "Pooling treats benchmarks as one sample and so weights them by size; read it",
          "alongside the per-benchmark rows, not instead of them.", ""]

    text = "\n".join(L)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(text)
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
