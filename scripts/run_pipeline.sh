#!/usr/bin/env bash
# End-to-end run inside tmux: prepare -> baseline eval -> train -> eval -> report -> push.
#
#   bash scripts/run_pipeline.sh
#   tmux attach -t brainmed          # watch
#   tmux capture-pane -p -t brainmed:main -S -200   # last 200 lines without attaching
#
# Every stage writes a marker in ./state/, so re-running the script resumes where the pod
# died instead of redoing GPU work that is already paid for.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

SESSION=${SESSION:-brainmed}
EXP_NAME=${EXP_NAME:-brainmed-8b-v1}
BASE_MODEL=${BASE_MODEL:-FreedomIntelligence/HuatuoGPT-o1-8B}
HF_REPO=${HF_REPO:-BrainHealthAI/BrainMed-Reasoning-8B}
RUN_BASELINE=${RUN_BASELINE:-1}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

: "${HF_TOKEN:?export HF_TOKEN first}"
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

mkdir -p state logs results
done_marker() { [ -f "state/$1.done" ]; }
mark() { touch "state/$1.done"; }

run_stage() {
  local name=$1; shift
  if done_marker "$name"; then echo "== [$name] already done, skipping"; return; fi
  echo "== [$name] starting $(date -Is)"
  ( set -x; "$@" ) 2>&1 | tee -a "logs/${name}.log"
  mark "$name"
  echo "== [$name] done $(date -Is)"
}

pipeline() {
  run_stage benchmarks python eval/build_benchmarks.py
  run_stage prepare    python data/prepare_data.py --out_dir ./data/prepared

  # RUN_BASELINE=1 full backbone eval (~30 min)  -> gate + before/after figure
  #             =calib  3 benchmarks (~6 min)   -> harness calibration only
  #             =0      skipped                 -> published values only, no gate
  case "$RUN_BASELINE" in
    1)
      MODEL="$BASE_MODEL" RUN_NAME="base-$(basename "$BASE_MODEL")" NUM_GPUS="$NUM_GPUS" \
        run_stage eval_baseline bash eval/run_eval.sh
      ;;
    calib)
      MODEL="$BASE_MODEL" RUN_NAME="base-$(basename "$BASE_MODEL")" NUM_GPUS="$NUM_GPUS" \
        run_stage eval_baseline bash eval/run_eval.sh \
          --only ${CALIB_BENCHMARKS:-medqa_4opt medbullets_op4 medbullets_op5}
      ;;
    *)
      echo "== [eval_baseline] skipped (RUN_BASELINE=$RUN_BASELINE)"
      echo "   the no-regression gate and the before/after figure need a measured backbone;"
      echo "   comparisons will rest on published values from another harness."
      ;;
  esac

  NUM_GPUS="$NUM_GPUS" BASE_MODEL="$BASE_MODEL" EXP_NAME="$EXP_NAME" \
    run_stage train bash train/launch.sh

  MODEL="$ROOT/ckpts/$EXP_NAME/best" RUN_NAME="$EXP_NAME" NUM_GPUS="$NUM_GPUS" \
    run_stage eval_finetuned bash eval/run_eval.sh

  # SWEEP=1 evaluates every checkpoint and soup as well. Off by default: it multiplies the
  # evaluation cost, and selecting on the reported benchmarks is a bias you opt into
  # knowingly. Read the win conditions in REPORT.md first, then sweep if one failed.
  if [ "${SWEEP:-0}" = "1" ]; then
    EXP_NAME="$EXP_NAME" BASE_MODEL="$BASE_MODEL" NUM_GPUS="$NUM_GPUS" \
      SOUP_ALPHAS="${SOUP_ALPHAS:-0.3 0.5}" run_stage sweep bash scripts/sweep_candidates.sh
  fi

  run_stage report  python scripts/make_report.py --results_dir ./results --out ./results/REPORT.md
  run_stage figures python scripts/make_figures.py --results_dir ./results \
                      --metrics "./train_logs/$EXP_NAME/metrics.jsonl" \
                      --out_dir ./results/figures
  run_stage push    python scripts/push_to_hf.py --checkpoint "./ckpts/$EXP_NAME/best" \
                      --repo_id "$HF_REPO" --base_model "$BASE_MODEL" \
                      --results ./results --private

  echo
  echo "############ PIPELINE COMPLETE ############"
  cat results/REPORT.md
}

if [ "${INSIDE_TMUX:-0}" = "1" ]; then
  pipeline
  echo "pipeline finished; leaving the shell open for inspection."
  exec bash
fi

command -v tmux >/dev/null || { echo "tmux not installed: apt-get update && apt-get install -y tmux"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session '$SESSION' already exists - tmux attach -t $SESSION"; exit 1; }
# new-session -e needs tmux >= 3.2
tmux_ver=$(tmux -V | sed 's/[^0-9.]//g')
if [ "$(printf '%s\n3.2\n' "$tmux_ver" | sort -V | head -1)" != "3.2" ]; then
  echo "ERROR: tmux $tmux_ver is too old for 'new-session -e'; need >= 3.2" >&2
  exit 1
fi

# Secrets go in via `new-session -e`, never through send-keys: anything sent as keystrokes
# lands in the shell history and is visible in `ps` to every user on the box.
tmux new-session -d -s "$SESSION" -n main \
  -e "INSIDE_TMUX=1" \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "WANDB_API_KEY=$WANDB_API_KEY" \
  -e "EXP_NAME=$EXP_NAME" \
  -e "BASE_MODEL=$BASE_MODEL" \
  -e "HF_REPO=$HF_REPO" \
  -e "RUN_BASELINE=$RUN_BASELINE" \
  -e "NUM_GPUS=$NUM_GPUS" \
  -e "SWEEP=${SWEEP:-0}" \
  -e "HF_HUB_ENABLE_HF_TRANSFER=1"
tmux send-keys -t "$SESSION:main" "cd $ROOT && bash scripts/run_pipeline.sh" C-m

tmux new-window -t "$SESSION" -n gpu
tmux send-keys -t "$SESSION:gpu" "watch -n 5 nvidia-smi" C-m
tmux new-window -t "$SESSION" -n logs
# wait for the first log file: the window opens before any stage has written one, and a
# glob that matches nothing makes tail exit immediately with "cannot open logs/*.log"
tmux send-keys -t "$SESSION:logs" \
  "cd $ROOT && until ls logs/*.log >/dev/null 2>&1; do sleep 2; done && tail -n +1 -F logs/*.log" C-m

echo "tmux session '$SESSION' started."
echo "  attach : tmux attach -t $SESSION"
echo "  peek   : tmux capture-pane -p -t $SESSION:main -S -200"
echo "  detach : Ctrl-b then d"
