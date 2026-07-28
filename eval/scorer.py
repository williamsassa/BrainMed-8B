"""
Answer extraction + scoring.

Derived from MedReason `src/evaluation/scorer.py` (itself a fork of HuatuoGPT-o1's), kept
bit-compatible on the matching cascade so our numbers stay comparable with the published
tables. Two additions:

  1. `<answer>...</answer>` is recognised as a final-answer delimiter. Our models are
     trained on that constitution; without this branch the regex cascade scans the whole
     chain of thought and picks up option letters mentioned while reasoning.
  2. Extraction statistics are returned, so a collapse in format compliance is visible
     instead of being silently absorbed by the fuzzy fallbacks.
"""
import difflib
import json
import os
import re
from collections import defaultdict

# how the final answer was located, in decreasing order of reliability
MATCH_STRICT = 1   # "the answer is X"
MATCH_REGEX = 2    # a bare option letter
MATCH_TEXT = 3     # option text found verbatim in the output
MATCH_FUZZY = 4    # closest option by string similarity (last resort)

MATCH_NAMES = {MATCH_STRICT: "strict", MATCH_REGEX: "letter", MATCH_TEXT: "option_text", MATCH_FUZZY: "fuzzy"}


def str_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_most_similar_index(str_list, target):
    best_i, best = None, -1.0
    for i, s in enumerate(str_list):
        sim = str_similarity(s, target)
        if sim >= best:
            best, best_i = sim, i
    return best_i


def split_final(text):
    """Isolate the final-answer segment, dropping the reasoning trace."""
    if "<answer>" in text:                       # our constitution
        seg = text.split("<answer>")[-1]
        return seg.split("</answer>")[0]
    if "## Final Response\n\n" in text:           # HuatuoGPT-o1
        return text.split("## Final Response\n\n")[-1]
    if "## Final Answer\n\n" in text:             # MedReason
        return text.split("## Final Answer\n\n")[-1]
    if "</think>" in text:                        # generic reasoning models
        return text.split("</think>")[-1]
    return text


def match_choice(text, options):
    text = split_final(text)

    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        return [matches[0].group(2), matches[-1].group(2)], MATCH_STRICT

    match_options = "ABCDEFGHIJKLMN"[:len(options)]
    matches = list(re.finditer(
        r"([一-鿿]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )([" + match_options + r"])(\W|[一-鿿]|$)",
        text, re.S))
    if matches:
        return [matches[0].group(2), matches[-1].group(2)], MATCH_REGEX

    low = text.lower()
    hits = [(o, low.rindex(options[o].lower())) for o in options if options[o].lower() in low]
    if hits:
        last = sorted(hits, key=lambda x: x[1], reverse=True)[0][0]
        hits = [(o, low.index(options[o].lower())) for o in options if options[o].lower() in low]
        first = sorted(hits, key=lambda x: x[1], reverse=True)[0][0]
        return [first, last], MATCH_TEXT

    labels = list(options)
    i = find_most_similar_index([options[x].lower() for x in labels], low)
    return [labels[i], labels[i]], MATCH_FUZZY


def score(data):
    """Accuracy per source. Keeps the paper's max(head, tail) convention."""
    res = defaultdict(lambda: {"n": 0, "head": 0, "tail": 0,
                               "match_types": defaultdict(int), "empty": 0})
    wrong, correct = [], []

    for da in data:
        src = da.get("source", "unknown")
        r = res[src]
        r["n"] += 1
        out = da.get("output") or ""
        if not out.strip():
            r["empty"] += 1
            wrong.append(da)
            continue
        ans, kind = match_choice(out, da["options"])
        da["ans"], da["ans_type"] = ans, kind
        r["match_types"][MATCH_NAMES[kind]] += 1
        gold = da["answer_idx"].lower()
        if ans[0].lower() == gold:
            r["head"] += 1
            correct.append(da)
        else:
            wrong.append(da)
        if ans[1].lower() == gold:
            r["tail"] += 1

    out = {}
    for k, r in res.items():
        n = max(1, r["n"])
        head, tail = r["head"] / n, r["tail"] / n
        out[k] = {
            "accuracy": round(100 * max(head, tail), 2),
            "acc_head": round(100 * head, 2),
            "acc_tail": round(100 * tail, 2),
            "n": r["n"],
            "empty_outputs": r["empty"],
            "format_compliance": round(100 * r["match_types"]["strict"] / n, 2),
            "match_types": dict(r["match_types"]),
        }
    return out, wrong, correct


def get_results(res_path, save=True):
    data = json.load(open(res_path, encoding="utf-8"))
    res, wrong, correct = score(data)
    print(f"*Logging_file: {os.path.basename(res_path)}*")
    print(json.dumps(res, indent=4))
    if save:
        out = res_path.replace(os.sep + "logs" + os.sep, os.sep + "result" + os.sep)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        out = os.path.join(os.path.dirname(out), "result_" + os.path.basename(out))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    return res
