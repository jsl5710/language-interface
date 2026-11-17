#!/bin/bash
#SBATCH --job-name=dlsd-llm-render
#SBATCH --account=ucb860_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# Serve Qwen3-32B with vLLM, run the renderer against it, then shut the server down.
#   sbatch slurm/serve_render.sbatch --language Spanish        # one language, 4 conditions, resumable
#   sbatch slurm/serve_render.sbatch --pilot                   # 20 items x 12 conditions
# Other GPU (needs >= 80 GB): sbatch --partition=aa100 --gres=gpu:a100_80gb:1 slurm/serve_render.sbatch ...
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME   # /home is at quota: keep any stray cache off it
export VLLM_USE_FLASHINFER_SAMPLER=0      # FlashInfer JIT needs nvcc, which the GPU nodes lack; use the built-in sampler
export PATH=$PATH:$DLSD_ROOT/env/vllm-venv/bin
export HF_HUB_OFFLINE=1            # weights load from local path; LaBSE from the pre-downloaded cache
PORT=$((8100 + SLURM_JOB_ID % 800))      # per-job port: two render jobs can share a node
export DLSD_RENDER_PORT=$PORT
MODEL_DIR=$DLSD_MODELS_DIR/Qwen__Qwen3-32B
LOG=$DLSD_ROOT/logs/vllm_qwen3_${SLURM_JOB_ID}.log

$DLSD_ROOT/env/vllm-venv/bin/vllm serve "$MODEL_DIR" --served-model-name Qwen/Qwen3-32B \
    --port $PORT --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.70 > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "waiting for vLLM on port $PORT"
for i in $(seq 1 180); do
    curl -sf localhost:$PORT/v1/models > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited early, see $LOG"; tail -20 "$LOG"; exit 1; }
    sleep 10
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "vLLM not ready after 30 min"; exit 1; }
echo "vLLM ready"

cd $DLSD_ROOT/dlsd-health-experiments
PYTHONPATH=src python -W ignore scripts/02b_render_llm.py "$@"
