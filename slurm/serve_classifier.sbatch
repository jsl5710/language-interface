#!/bin/bash
#SBATCH --job-name=dlsd-clf
#SBATCH --account=ucb860_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# Serve ONE safety classifier and flag all rendered requests and generated responses with it.
#   sbatch slurm/serve_classifier.sbatch llama_guard    (llama_guard | wildguard | polyguard | qwen3guard)
NAME=$1
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export PATH=$DLSD_ROOT/env/vllm-venv/bin:$PATH
case $NAME in
  llama_guard) ID=meta-llama/Llama-Guard-4-12B;    DIR=meta-llama__Llama-Guard-4-12B;    PORT=8010;;
  wildguard)   ID=allenai/wildguard;               DIR=allenai__wildguard;               PORT=8011;;
  polyguard)   ID=ToxicityPrompts/PolyGuard-Qwen;  DIR=ToxicityPrompts__PolyGuard-Qwen;  PORT=8012;;
  qwen3guard)  ID=Qwen/Qwen3Guard-Gen-8B;          DIR=Qwen__Qwen3Guard-Gen-8B;          PORT=8013;;
  *) echo "unknown classifier $NAME"; exit 2;;
esac
PORT=$((8900 + SLURM_JOB_ID % 90)); export DLSD_CLF_PORT=$PORT      # per-job port
LOG=$DLSD_ROOT/logs/vllm_${NAME}_${SLURM_JOB_ID}.log
$DLSD_ROOT/env/vllm-venv/bin/vllm serve $DLSD_MODELS_DIR/$DIR --served-model-name $ID --port $PORT --max-model-len 8192 --dtype bfloat16 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null; sleep 5; pkill -9 -P $$ 2>/dev/null' EXIT
for i in $(seq 1 720); do
    curl -sf localhost:$PORT/v1/models > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited early"; tail -25 "$LOG"; exit 1; }
    sleep 10
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "vLLM not ready after 2 h"; exit 1; }
cd $DLSD_ROOT/dlsd-health-experiments
PYTHONPATH=src python -W ignore scripts/05_classify.py --classifier $NAME
