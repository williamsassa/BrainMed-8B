#!/usr/bin/env python
"""
Push the pipeline to a GitHub repository.

    GITHUB_TOKEN=ghp_... python scripts/push_to_github.py --repo <owner>/<name> --private

What goes: code, configs, manifests, the evaluation report, the comparison tables, the
figures. What does not, and why:

  * model weights - 16GB per checkpoint against a 100MB GitHub file limit. They live on the
    Hugging Face Hub; this repo links to them.
  * the prepared corpus (~300MB) and the benchmark jsonl files - both are *derived*, and
    `data/prepare_data.py` and `eval/build_benchmarks.py` rebuild them byte-for-byte from
    upstream. The manifests, with their SHA-256 digests, are committed so a rebuild can be
    verified. Redistributing another project's benchmark files is also a licensing question
    this repo does not need to answer.
  * raw generation logs - hundreds of MB of model outputs, archived alongside the model
    instead.

Before pushing, the script scans every staged file for credentials and refuses on a hit.
"""
import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GITIGNORE = """\
# ---- model weights -------------------------------------------------------------------
# 16GB per checkpoint; GitHub caps files at 100MB. The weights live on the Hugging Face Hub.
ckpts/
*.safetensors
*.bin
*.pt
*.pth

# ---- derived inputs, rebuilt by the pipeline -----------------------------------------
# data/prepare_data.py and eval/build_benchmarks.py regenerate these from upstream sources;
# the manifests (with SHA-256 digests) are committed so a rebuild can be checked.
data/prepared/*.jsonl
eval/benchmarks/*.jsonl
external/
.tokcache/
.hf/

# ---- run artefacts -------------------------------------------------------------------
results/*/logs/
results/_card/
train_logs/*/wandb/
logs/
state/
*.tgz
*.tar.gz

# ---- python --------------------------------------------------------------------------
__pycache__/
*.py[cod]
.venv/
venv/
.ipynb_checkpoints/

# ---- credentials ---------------------------------------------------------------------
.env
*.pem
*token*.json
"""

# patterns for things that must never be committed
SECRETS = [
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "Hugging Face token"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"), "GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b"), "GitHub fine-grained token"),
    (re.compile(r"\bwandb_v1_[A-Za-z0-9_-]{40,}\b"), "Weights & Biases key"),
    (re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"), "OpenAI-style key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key"),
]
MAX_FILE_MB = 90


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True,
                          capture_output=True, **kw).stdout.strip()


def staged_files():
    out = run(["git", "diff", "--cached", "--name-only"])
    return [f for f in out.splitlines() if f]


def scan(files):
    hits, big = [], []
    for f in files:
        p = os.path.join(REPO_ROOT, f)
        if not os.path.isfile(p):
            continue
        mb = os.path.getsize(p) / 1e6
        if mb > MAX_FILE_MB:
            big.append((f, round(mb, 1)))
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for rx, label in SECRETS:
            for m in rx.finditer(text):
                line = text[:m.start()].count("\n") + 1
                hits.append((f, line, label, m.group()[:12] + "…"))
    return hits, big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name, e.g. williamsassa/BrainMed-8B")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="BrainMed-8B: training and evaluation pipeline")
    ap.add_argument("--description", default=(
        "Full-parameter medical SFT pipeline (MedReason recipe) with reproducible "
        "evaluation on 10 benchmarks. Weights on the Hugging Face Hub."))
    ap.add_argument("--dry_run", action="store_true", help="stage and scan, do not push")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("export GITHUB_TOKEN first (a classic PAT with the 'repo' scope)")

    gi = os.path.join(REPO_ROOT, ".gitignore")
    if not os.path.exists(gi):
        open(gi, "w", encoding="utf-8").write(GITIGNORE)
        print("[git] wrote .gitignore")
    else:
        print("[git] .gitignore already present, left as is")

    if not os.path.isdir(os.path.join(REPO_ROOT, ".git")):
        run(["git", "init", "-b", args.branch])
        print(f"[git] initialised on branch {args.branch}")
    for k, v in (("user.name", "brainmed-pipeline"), ("user.email", "noreply@brainhealth.ai")):
        try:
            run(["git", "config", k])
        except subprocess.CalledProcessError:
            run(["git", "config", k, v])

    run(["git", "add", "-A"])
    files = staged_files()
    if not files:
        print("[git] nothing staged - already up to date?")
    total_mb = sum(os.path.getsize(os.path.join(REPO_ROOT, f)) / 1e6
                   for f in files if os.path.isfile(os.path.join(REPO_ROOT, f)))
    print(f"[git] {len(files)} files staged, {total_mb:.1f} MB")

    hits, big = scan(files)
    if big:
        print("\nFILES OVER THE GITHUB LIMIT:")
        for f, mb in big:
            print(f"  {f}  {mb} MB")
        raise SystemExit("add these to .gitignore, then `git rm --cached <file>` and re-run")
    if hits:
        print("\nCREDENTIALS FOUND IN STAGED FILES - refusing to push:")
        for f, line, label, frag in hits:
            print(f"  {f}:{line}  {label}  {frag}")
        raise SystemExit("remove them (and revoke the credential - it is already on disk)")
    print("[scan] no credentials, no oversized files")

    if args.dry_run:
        print("\ndry run: staged and scanned, nothing pushed")
        print("\n".join("  " + f for f in files[:40]))
        if len(files) > 40:
            print(f"  … and {len(files) - 40} more")
        return

    try:
        run(["git", "commit", "-m", args.message])
        print("[git] committed")
    except subprocess.CalledProcessError as e:
        if "nothing to commit" not in (e.stdout or "") + (e.stderr or ""):
            raise
        print("[git] nothing new to commit")

    owner, name = args.repo.split("/", 1)
    import json
    import urllib.error
    import urllib.request

    def api(path, data=None, method="GET"):
        req = urllib.request.Request(
            f"https://api.github.com{path}",
            data=json.dumps(data).encode() if data else None, method=method,
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "brainmed-pipeline"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or b"{}")

    try:
        api(f"/repos/{args.repo}")
        print(f"[github] {args.repo} already exists")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise SystemExit(f"github error {e.code}: {e.read().decode()[:300]}")
        me = api("/user")["login"]
        payload = {"name": name, "private": args.private, "description": args.description}
        api("/user/repos" if owner == me else f"/orgs/{owner}/repos", payload, "POST")
        print(f"[github] created {args.repo} (private={args.private})")

    # the token goes on the command line for this one push and is never written to .git/config
    url = f"https://github.com/{args.repo}.git"
    push_url = f"https://x-access-token:{token}@github.com/{args.repo}.git"
    try:
        run(["git", "remote", "set-url", "origin", url])
    except subprocess.CalledProcessError:
        run(["git", "remote", "add", "origin", url])
    try:
        subprocess.run(["git", "push", "-u", push_url, args.branch],
                       cwd=REPO_ROOT, check=True, text=True)
    except subprocess.CalledProcessError:
        sys.exit("push failed - check the token scope ('repo') and that the branch is not "
                 "protected")
    print(f"\ndone -> https://github.com/{args.repo}")
    print("The weights are not in this repo. Add a link to the Hugging Face model in README.md.")


if __name__ == "__main__":
    main()
