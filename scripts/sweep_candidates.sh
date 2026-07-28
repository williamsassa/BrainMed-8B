#!/usr/bin/env bash
# Evaluate every candidate model and let the report rank them.
#
#   EXP_NAME=brainmed-8b-v1 BASE_MODEL=FreedomIntelligence/HuatuoGPT-o1-8B \
#     bash scripts/sweep_candidates.sh
#
# Candidates:
#   - each saved checkpoint (best-by-val-loss, epoch-1..N, last)
#   - optional weight interpolations towards the backbone (SOUP_ALPHAS)
#
# Why sweep at all: val loss is a proxy for benchmark accuracy, not the same quantity. When
# the requirement is "beat the backbone and beat every published model", the checkpoint that
# ships has to be the one that demonstrably does it - which means measuring the candidates
# rather than assuming the lowest-val-loss one wins.
#
# Selection bias, stated plainly: picking a checkpoint by its score on the same benchmarks
# you then report is optimistic. Keep the val-loss checkpoint as the headline number, quote
# the swept winner as a separate, labelled result, and never present the maximum over
# candidates as if it were a single unbiased run.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP_NAME=${EXP_NAME:?set EXP_NAME}
BASE_MODEL=${BASE_MODEL:-FreedomIntelligence/HuatuoGPT-o1-8B}
CKPT_DIR=${CKPT_DIR:-./ckpts/$EXP_NAME}
SOUP_ALPHAS=${SOUP_ALPHAS:-}          # e.g. "0.3 0.5 0.7"; empty disables soups
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
QUICK=${QUICK:-0}                     # 1 = only the 6 Table-4 benchmarks (faster, cheaper)

ONLY_ARG=()
if [ "$QUICK" = "1" ]; then
  ONLY_ARG=(--only medqa_4opt medmcqa_val pubmedqa_test medbullets_op4 medbullets_op5 medxpertqa)
fi

[ -d "$CKPT_DIR" ] || { echo "ERROR: no checkpoints at $CKPT_DIR" >&2; exit 1; }

mapfile -t CANDIDATES < <(find "$CKPT_DIR" -maxdepth 1 -mindepth 1 -type d \
                            \( -name best -o -name 'epoch-*' -o -name last \) | sort)
[ ${#CANDIDATES[@]} -gt 0 ] || { echo "ERROR: no checkpoint subdirs in $CKPT_DIR" >&2; exit 1; }

echo "candidates found:"
printf '  %s\n' "${CANDIDATES[@]}"

# Soups are built from SOUP_FROM (default: the val-loss checkpoint). Validation loss on this
# corpus tracks conformity to the training style rather than medical accuracy, so it does not
# reliably pick the strongest checkpoint - once the sweep has scored the epochs, rebuild the
# soups from whichever one actually won:
#   SOUP_FROM=last SOUP_ALPHAS="0.3 0.5" bash scripts/sweep_candidates.sh
SOUP_FROM=${SOUP_FROM:-best}
[ -d "$CKPT_DIR/$SOUP_FROM" ] || { echo "ERROR: no checkpoint '$SOUP_FROM' in $CKPT_DIR" >&2; exit 1; }

for a in $SOUP_ALPHAS; do
  tag="soup-${SOUP_FROM}-a${a}"
  [ "$SOUP_FROM" = "best" ] && tag="soup-a${a}"        # keep earlier runs' names stable
  out="$CKPT_DIR/$tag"
  if [ ! -d "$out" ]; then
    echo "=== building $tag from $SOUP_FROM (alpha=$a) ==="
    python scripts/weight_soup.py --base "$BASE_MODEL" \
      --finetuned "$CKPT_DIR/$SOUP_FROM" --alpha "$a" --out "$out"
  fi
  CANDIDATES+=("$out")
done

for ckpt in "${CANDIDATES[@]}"; do
  name="${EXP_NAME}__$(basename "$ckpt")"
  if [ -f "./results/$name/summary_${name}.json" ]; then
    echo "== $name already evaluated, skipping"
    continue
  fi
  echo
  echo "############ evaluating $name ############"
  MODEL="$ckpt" RUN_NAME="$name" NUM_GPUS="$NUM_GPUS" \
    bash eval/run_eval.sh "${ONLY_ARG[@]}"
done

echo
echo "############ leaderboard ############"
python scripts/make_report.py --results_dir ./results --out ./results/REPORT.md
echo
echo "Ship the candidate that PASSES both win conditions. If several do, prefer the one"
echo "that also passes the no-regression gate against the backbone."
