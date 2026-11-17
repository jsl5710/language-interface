#!/bin/bash
#SBATCH --job-name=dlsd-score
#SBATCH --account=ucb860_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# Serve the Experiment 1 judge (Qwen3-32B, port 8020 as in configs/experiment.yaml) and label all generations.
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export PATH=$DLSD_ROOT/env/vllm-venv/bin:$PATH
PORT=$((8300 + SLURM_JOB_ID % 600)); export DLSD_JUDGE_PORT=$PORT; LOG=$DLSD_ROOT/logs/vllm_judge_exp1_${SLURM_JOB_ID}.log
$DLSD_ROOT/env/vllm-venv/bin/vllm serve $DLSD_MODELS_DIR/Qwen__Qwen3-32B --served-model-name Qwen/Qwen3-32B --port $PORT --max-model-len 8192 --dtype bfloat16 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null; sleep 5; pkill -9 -P $$ 2>/dev/null' EXIT
for i in $(seq 1 720); do
    curl -sf localhost:$PORT/v1/models > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited early"; tail -25 "$LOG"; exit 1; }
    sleep 10
done
cd $DLSD_ROOT/dlsd-health-experiments
PYTHONPATH=src python -W ignore scripts/04_score.py
