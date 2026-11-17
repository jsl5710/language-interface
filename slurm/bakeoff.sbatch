#!/bin/bash
#SBATCH --job-name=dlsd-bakeoff
#SBATCH --account=ucb860_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# Translator bake-off: NLLB-200 3.3B, MADLAD-400 10B, Qwen3-32B on the same test set. Scoring runs afterwards on CPU (bakeoff_score.py).
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export PATH=$DLSD_ROOT/env/vllm-venv/bin:$PATH
cd $DLSD_ROOT/dlsd-health-experiments
export PYTHONPATH=src
PY=$DLSD_ROOT/env/vllm-venv/bin/python

for s in nllb madlad; do $PY -W ignore scripts/bakeoff_translate.py --system $s || echo "FAILED $s"; done

PORT=8030; LOG=$DLSD_ROOT/logs/vllm_bakeoff_${SLURM_JOB_ID}.log
$PY -m vllm.entrypoints.openai.api_server --model $DLSD_MODELS_DIR/Qwen__Qwen3-32B --served-model-name Qwen/Qwen3-32B \
    --port $PORT --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null' EXIT
for i in $(seq 1 180); do
    curl -sf localhost:$PORT/v1/models > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited early"; tail -20 "$LOG"; exit 1; }
    sleep 10
done
$PY -W ignore scripts/bakeoff_translate.py --system qwen
