#!/usr/bin/env bash
# Full-parameter SFT, DeepSpeed ZeRO-3.
#
#   NUM_GPUS=4 BASE_MODEL=FreedomIntelligence/HuatuoGPT-o1-8B bash train/launch.sh
#
# Effective batch is pinned to 128 (the paper's) whatever NUM_GPUS is:
#   micro_bsz x grad_accum x NUM_GPUS = 128
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
BASE_MODEL=${BASE_MODEL:-FreedomIntelligence/HuatuoGPT-o1-8B}
EXP_NAME=${EXP_NAME:-brainmed-8b-v1}
DATA=${DATA:-./data/prepared/train.jsonl}
VAL=${VAL:-./data/prepared/val.jsonl}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-6}
MICRO_BSZ=${MICRO_BSZ:-4}
TARGET_BSZ=${TARGET_BSZ:-128}
DS_CONFIG=${DS_CONFIG:-./configs/deepspeed_zero3.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-./ckpts}

GRAD_ACC=$(( TARGET_BSZ / (MICRO_BSZ * NUM_GPUS) ))
if [ $(( GRAD_ACC * MICRO_BSZ * NUM_GPUS )) -ne "$TARGET_BSZ" ]; then
  echo "ERROR: ${MICRO_BSZ} x ${NUM_GPUS} does not divide ${TARGET_BSZ}; adjust MICRO_BSZ." >&2
  exit 1
fi

for f in "$DATA" "$VAL"; do
  [ -f "$f" ] || { echo "ERROR: missing $f - run data/prepare_data.py first" >&2; exit 1; }
done
: "${WANDB_API_KEY:?set WANDB_API_KEY (or export WANDB_MODE=offline)}"

echo "=========================================================="
echo " model        : $BASE_MODEL"
echo " gpus         : $NUM_GPUS"
echo " micro x acc  : $MICRO_BSZ x $GRAD_ACC  -> effective $TARGET_BSZ"
echo " epochs / lr  : $EPOCHS / $LR"
echo " train rows   : $(wc -l < "$DATA")"
echo " output       : $OUTPUT_DIR/$EXP_NAME"
echo "=========================================================="

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

accelerate launch \
  --config_file "$DS_CONFIG" \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 \
  --machine_rank 0 \
  --deepspeed_multinode_launcher standard \
  ./train/sft.py \
    --model_path "$BASE_MODEL" \
    --data_path "$DATA" \
    --val_path "$VAL" \
    --experiment_name "$EXP_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --n_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --train_bsz_per_gpu "$MICRO_BSZ" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --max_seq_len "${MAX_SEQ_LEN:-4096}" \
    --eval_every "${EVAL_EVERY:-50}" \
    --wandb_project "${WANDB_PROJECT:-brainmed-sft}" \
    "$@"

echo "training done -> $OUTPUT_DIR/$EXP_NAME  (best checkpoint in ./best)"
