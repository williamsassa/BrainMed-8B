#!/usr/bin/env bash
# Everything that can fail, failing before the GPU meter starts.
#
#   bash scripts/preflight.sh
#
# Checks credentials, disk, GPUs, the two reference repos, the prepared corpus, the
# benchmark bundle, and finally runs a 2-step training smoke test on the real model with
# the real DeepSpeed config. That last one is the point: it catches OOM, a missing
# flash-attn, a broken chat template and a bad ZeRO config in about five minutes instead of
# forty minutes into a paid run.
set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0; FAIL=0; WARN=0
ok()   { echo "  [ OK ] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN+1)); }

BASE_MODEL=${BASE_MODEL:-FreedomIntelligence/HuatuoGPT-o1-8B}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MICRO_BSZ=${MICRO_BSZ:-4}

echo "=== 1. credentials ==="
[ -n "${HF_TOKEN:-}" ] && ok "HF_TOKEN set" || bad "HF_TOKEN not set (export HF_TOKEN=hf_...)"
[ -n "${WANDB_API_KEY:-}" ] && ok "WANDB_API_KEY set" \
  || bad "WANDB_API_KEY not set (export it, or run training with --no_wandb)"
if [ -n "${HF_TOKEN:-}" ]; then
  python - <<'PY' && ok "HF token valid, write access confirmed" || bad "HF token rejected"
import os, sys
from huggingface_hub import HfApi
try:
    w = HfApi(token=os.environ["HF_TOKEN"]).whoami()
    role = w.get("auth", {}).get("accessToken", {}).get("role")
    print(f"        user={w['name']} role={role} orgs={[o['name'] for o in w.get('orgs', [])]}")
    sys.exit(0 if role == "write" else 1)
except Exception as e:
    print(f"        {e}"); sys.exit(1)
PY
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
  python -c "
import wandb, sys
try:
    wandb.Api(api_key='${WANDB_API_KEY}').viewer
except Exception as e:
    print(f'        {e}'); sys.exit(1)
" && ok "W&B key valid" || warn "W&B key could not be verified (offline is fine: --wandb_mode offline)"
fi

echo "=== 2. hardware ==="
[ "$NUM_GPUS" -gt 0 ] && ok "$NUM_GPUS GPU(s) visible" || bad "no GPU detected"
if [ "$NUM_GPUS" -gt 0 ]; then
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  [ "$mem" -ge 70000 ] && ok "${mem} MiB per GPU" \
    || warn "${mem} MiB per GPU: use configs/deepspeed_zero3_offload.yaml"
  acc=$(( 128 / (MICRO_BSZ * NUM_GPUS) ))
  [ $(( acc * MICRO_BSZ * NUM_GPUS )) -eq 128 ] \
    && ok "batch 128 = ${MICRO_BSZ} x ${acc} x ${NUM_GPUS}" \
    || bad "MICRO_BSZ=$MICRO_BSZ x NUM_GPUS=$NUM_GPUS does not divide 128"
fi
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 150 ] && ok "${free_gb} GB free" \
  || warn "${free_gb} GB free; ~120 GB needed (backbone + best + last + 3 epoch checkpoints)"
ram_gb=$(free -g | awk '/^Mem:/{print $2}')
ok "${ram_gb} GB host RAM, $(nproc) cpus"

echo "=== 3. python stack ==="
python - <<'PY'
import importlib, sys
need = {"torch": None, "transformers": None, "accelerate": None, "deepspeed": None,
        "datasets": None, "wandb": None, "vllm": None, "openai": None, "jinja2": None}
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"        {m:14s} {getattr(mod, '__version__', '?')}")
    except Exception:
        missing.append(m)
try:
    import flash_attn; print(f"        flash_attn     {flash_attn.__version__}")
except Exception:
    print("        flash_attn     MISSING -> pass --no_flash_attn")
if missing:
    print(f"        MISSING: {missing}"); sys.exit(1)
PY
[ $? -eq 0 ] && ok "all required packages importable" || bad "missing packages - rerun scripts/setup_runpod.sh"

echo "=== 4. inputs ==="
for d in external/MedReason external/HuatuoGPT-o1; do
  [ -d "$d" ] && ok "$d present" || bad "$d missing - rerun scripts/setup_runpod.sh"
done
if [ -f data/prepared/train.jsonl ]; then
  n=$(wc -l < data/prepared/train.jsonl)
  ok "corpus prepared: $n train rows"
  python - <<'PY' && ok "manifest consistent" || bad "manifest check failed"
import json, sys
m = json.load(open("data/prepared/manifest.json", encoding="utf-8"))["report"]
v = m["verbatim_check"]
print(f"        decontaminate={m['decontamination']['enabled']} "
      f"alignment={m['answer_alignment']['enabled']} "
      f"aligned={m['answer_alignment']['train_rows_aligned']}")
if v["question_altered"] or v["reasoning_altered"]:
    print("        question/reasoning were altered"); sys.exit(1)
if not v["answer_changes_are_suffix_only"]:
    print("        answers were rewritten, not suffixed"); sys.exit(1)
PY
else
  bad "data/prepared/train.jsonl missing - run: python data/prepare_data.py"
fi
if [ -d eval/benchmarks ]; then
  n=$(ls eval/benchmarks/*.jsonl 2>/dev/null | wc -l)
  [ "$n" -ge 8 ] && ok "$n benchmarks built" || bad "only $n benchmarks - run eval/build_benchmarks.py"
else
  bad "eval/benchmarks missing - run: python eval/build_benchmarks.py"
fi

echo "=== 5. backbone reachable ==="
python - <<PY && ok "backbone config + tokenizer + chat template OK" || bad "backbone unreachable or unusable"
import os, sys
from transformers import AutoConfig, AutoTokenizer
try:
    c = AutoConfig.from_pretrained("$BASE_MODEL", token=os.environ.get("HF_TOKEN"))
    t = AutoTokenizer.from_pretrained("$BASE_MODEL", token=os.environ.get("HF_TOKEN"))
    n = c.num_hidden_layers * (12 * c.hidden_size**2) / 1e9
    print(f"        {c.architectures} L={c.num_hidden_layers} d={c.hidden_size} ~{n:.1f}B core params")
    if not t.chat_template:
        print("        no chat_template on the tokenizer"); sys.exit(1)
    r = t.apply_chat_template([{"role": "system", "content": "s"},
                               {"role": "user", "content": "u"},
                               {"role": "assistant", "content": "a"}], tokenize=False)
    print(f"        chat template renders {len(r)} chars")
except Exception as e:
    print(f"        {e}"); sys.exit(1)
PY

echo "=== 6. training smoke test (2 optimizer steps on the real model) ==="
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  warn "skipped (SKIP_SMOKE=1)"
elif [ "$FAIL" -gt 0 ]; then
  warn "skipped: fix the failures above first"
else
  head -n 512 data/prepared/train.jsonl > /tmp/smoke_train.jsonl
  head -n 64  data/prepared/val.jsonl   > /tmp/smoke_val.jsonl
  acc=$(( 128 / (MICRO_BSZ * NUM_GPUS) ))
  if accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
      --num_processes "$NUM_GPUS" --num_machines 1 --machine_rank 0 \
      --deepspeed_multinode_launcher standard ./train/sft.py \
      --model_path "$BASE_MODEL" \
      --data_path /tmp/smoke_train.jsonl --val_path /tmp/smoke_val.jsonl \
      --experiment_name _preflight --output_dir /tmp/preflight_ckpt \
      --log_dir /tmp/preflight_logs --cache_dir /tmp/preflight_cache \
      --n_epochs 1 --train_bsz_per_gpu "$MICRO_BSZ" --gradient_accumulation_steps "$acc" \
      --eval_every 0 --no_eval_at_start --no_save_each_epoch --no_wandb \
      > /tmp/preflight.log 2>&1; then
    ok "2-step training ran; peak memory $(grep -o 'gpu_mem[^ ]*' /tmp/preflight.log | tail -1)"
    grep -E "train rows=|optimizer steps:|effective batch" /tmp/preflight.log | sed 's/^/        /'
    rm -rf /tmp/preflight_ckpt /tmp/preflight_logs /tmp/preflight_cache
  else
    bad "smoke test failed - last 25 lines:"
    tail -25 /tmp/preflight.log | sed 's/^/        /'
  fi
fi

echo
echo "=============================================="
echo " $PASS passed, $WARN warnings, $FAIL failures"
echo "=============================================="
[ "$FAIL" -eq 0 ] && echo "Ready. Launch: bash scripts/run_pipeline.sh" || echo "Fix the failures before launching."
exit $(( FAIL > 0 ))
