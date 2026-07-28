#!/usr/bin/env bash
# One-shot bootstrap for a RunPod pod.
#
# Recommended template : "RunPod PyTorch 2.5" (CUDA 12.4) or any nvidia/cuda:12.4 image
# Recommended pod      : 4 x H100 80GB SXM  (see README for the cost table)
# Container disk       : 60 GB      Volume: 400 GB mounted at /workspace
#
#   cd /workspace && git clone <this repo> brainmed-sft && cd brainmed-sft
#   bash scripts/setup_runpod.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

echo "=== system packages ==="
apt-get update -qq
apt-get install -y -qq tmux git curl jq ninja-build build-essential > /dev/null
echo "tmux $(tmux -V), $(nproc) cpus, $(free -g | awk '/^Mem:/{print $2}') GB RAM"

echo "=== python packages ==="
pip install -q --upgrade pip setuptools wheel packaging
# flash-attn is excluded here on purpose: from sdist it compiles for 20+ minutes and often
# runs the pod out of RAM. It is installed below from a prebuilt wheel instead.
grep -v '^flash-attn' requirements.txt > /tmp/req.txt
pip install -q -r /tmp/req.txt --extra-index-url https://download.pytorch.org/whl/cu124

echo "=== flash-attention (prebuilt wheel) ==="
install_flash_attn() {
  python -c "import flash_attn" 2>/dev/null && { echo "  already installed"; return 0; }
  local py cu abi url
  py=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  cu=$(python -c "import torch; print('cu12' if torch.version.cuda.startswith('12') else 'cu11')")
  abi=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+${cu}torch2.5cxx11abi${abi}-${py}-${py}-linux_x86_64.whl"
  echo "  trying $url"
  pip install -q "$url" && return 0
  echo "  prebuilt wheel unavailable; falling back to a source build (slow)"
  pip install -q flash-attn==2.7.4.post1 --no-build-isolation
}
install_flash_attn || echo "WARNING: flash-attn unavailable - launch training with --no_flash_attn"

echo "=== reference repositories (benchmark provenance) ==="
mkdir -p external
[ -d external/MedReason ]    || git clone --depth 1 https://github.com/UCSC-VLAA/MedReason.git external/MedReason
[ -d external/HuatuoGPT-o1 ] || git clone --depth 1 https://github.com/FreedomIntelligence/HuatuoGPT-o1.git external/HuatuoGPT-o1

echo "=== Hub cache on the volume, not the container disk ==="
# By default HF caches to ~/.cache/huggingface, which on RunPod lives on the small
# container disk alongside the ~15GB of torch/vllm wheels. One 8B checkpoint is 16GB and
# a second backbone would overflow it. Pin the cache to the persistent volume instead.
VOLUME=${VOLUME:-/workspace}
if [ -d "$VOLUME" ]; then
  export HF_HOME="$VOLUME/.hf"
  mkdir -p "$HF_HOME"
  echo "  HF_HOME=$HF_HOME"
else
  echo "  WARNING: $VOLUME not found, leaving HF_HOME at its default (container disk)"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
grep -q HF_HUB_ENABLE_HF_TRANSFER ~/.bashrc || {
  echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc
  echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
  [ -n "${HF_HOME:-}" ] && echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
}
df -h "$VOLUME" / 2>/dev/null | sed 's/^/  /'

echo "=== credentials check ==="
for v in HF_TOKEN WANDB_API_KEY; do
  if [ -z "${!v:-}" ]; then echo "  MISSING $v - export it before running the pipeline"; else echo "  $v set"; fi
done

echo "=== sanity ==="
python - <<'PY'
import torch, transformers, accelerate, deepspeed
print(f"  torch {torch.__version__} | cuda {torch.version.cuda} | gpus {torch.cuda.device_count()}")
print(f"  transformers {transformers.__version__} | accelerate {accelerate.__version__} | deepspeed {deepspeed.__version__}")
try:
    import flash_attn; print(f"  flash-attn {flash_attn.__version__}")
except ImportError:
    print("  flash-attn NOT available -> pass --no_flash_attn")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"    cuda:{i} {p.name} {p.total_memory/1e9:.0f} GB")
PY

echo
echo "setup complete. Next:"
echo "  export HF_TOKEN=...  WANDB_API_KEY=..."
echo "  bash scripts/run_pipeline.sh"
