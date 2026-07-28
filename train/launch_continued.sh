#!/usr/bin/env bash
# Continued fine-tuning of an ALREADY fine-tuned checkpoint (e.g. UCSC-VLAA/MedReason-8B).
#
#   bash train/launch_continued.sh
#
# Why this is not train/launch.sh with a different --model_path:
#
# 55.7% of our training corpus (24,529 of 44,007 rows, measured by 13-gram containment) was
# already used to train MedReason-8B. Running the paper's recipe - 3 epochs at 5e-6 - puts
# SIX cumulative epochs through that half. That recipe is calibrated for a backbone that has
# never seen the data; applied to one that has, it memorises rather than learns and erodes
# the behaviours the previous stages installed.
#
# So: one epoch, a fifth of the learning rate, shorter warmup, and validation often enough
# that the best checkpoint can land mid-epoch. The genuinely new signal here is the 13,336
# medical-o1 rows (94.7% unseen by this checkpoint), which is the subset the MedReason paper
# credits with MedQA recall - and MedQA is precisely where MedReason SFT cost its own
# backbone 0.8 points (72.6 -> 71.8, Table 4).
set -euo pipefail
cd "$(dirname "$0")/.."

export BASE_MODEL=${BASE_MODEL:-UCSC-VLAA/MedReason-8B}
export EXP_NAME=${EXP_NAME:-brainmed-mr8b-cont}
export EPOCHS=${EPOCHS:-1}
export LR=${LR:-1e-6}
export EVAL_EVERY=${EVAL_EVERY:-25}

echo "continued fine-tuning: $BASE_MODEL, ${EPOCHS} epoch(s) at lr ${LR}"
echo "(full recipe is in train/launch.sh; this preset only softens the schedule)"

exec bash train/launch.sh --warmup_rates 0.03 "$@"
