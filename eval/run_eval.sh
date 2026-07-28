#!/usr/bin/env bash
# Serve a checkpoint with vLLM, run the whole benchmark bundle, tear the server down.
#
#   MODEL=./ckpts/brainmed-8b-v1/best RUN_NAME=brainmed-8b-v1 bash eval/run_eval.sh
#   MODEL=FreedomIntelligence/HuatuoGPT-o1-8B RUN_NAME=base-huatuo bash eval/run_eval.sh
#
# The base-model run matters: published baselines were produced on another harness, so the
# only controlled before/after comparison is one measured here, same prompts, same scorer.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:?set MODEL}
RUN_NAME=${RUN_NAME:-$(basename "$MODEL")}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
PORT=${PORT:-30000}
OUT_DIR=${OUT_DIR:-./results/$RUN_NAME}
BENCH=${BENCH:-./eval/benchmarks}
ENGINE=${ENGINE:-vllm}
MAX_NEW=${MAX_NEW:-2000}
LOG=${LOG:-./logs/server_${RUN_NAME}.log}

mkdir -p "$(dirname "$LOG")" "$OUT_DIR"
[ -d "$BENCH" ] || { echo "ERROR: no benchmarks at $BENCH - run eval/build_benchmarks.py" >&2; exit 1; }

cleanup() {
  echo "stopping inference server..."
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  pkill -f "sglang.launch_server" 2>/dev/null || true
  pkill -f multiprocessing.spawn 2>/dev/null || true
  sleep 5
}
trap cleanup EXIT

echo "starting $ENGINE on $NUM_GPUS GPU(s), port $PORT  (log: $LOG)"
if [ "$ENGINE" = "sglang" ]; then
  python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" \
    --mem-fraction-static 0.85 --dp "$NUM_GPUS" --tp 1 --served-model-name default > "$LOG" 2>&1 &
else
  python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" \
    --served-model-name default --tensor-parallel-size "$NUM_GPUS" \
    --gpu-memory-utilization 0.90 --max-model-len 8192 --disable-log-requests > "$LOG" 2>&1 &
fi
SERVER_PID=$!

echo -n "waiting for the server"
for i in $(seq 1 180); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then echo " up (${i}s)"; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "server died - tail of $LOG:"; tail -40 "$LOG"; exit 1; }
  echo -n "."; sleep 1
  [ "$i" -eq 180 ] && { echo " timeout"; tail -40 "$LOG"; exit 1; }
done

python ./eval/eval.py \
  --model_path "$MODEL" \
  --port "$PORT" \
  --benchmarks "$BENCH" \
  --out_dir "$OUT_DIR" \
  --run_name "$RUN_NAME" \
  --max_new_tokens "$MAX_NEW" \
  ${WANDB_RUN_ID:+--wandb_run_id "$WANDB_RUN_ID"} \
  "$@"

echo "results -> $OUT_DIR/summary_${RUN_NAME}.json"
