#!/bin/bash
#SBATCH --job-name=dlsd-annot
#SBATCH --account=ucb860_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# Serve ONE local LLM annotator and run the blind annotation tasks (stage 2m); harmful text stays on CURC.
#   sbatch slurm/serve_annotator.sbatch gptoss      (2 x L40: --partition=al40 --gres=gpu:l40:2 --export=ALL,DLSD_TP=2)
#   sbatch slurm/serve_annotator.sbatch aya         (2 x L40 as above)
#   sbatch --gres=gpu:h200:2 --export=ALL,DLSD_TP=2 slurm/serve_annotator.sbatch commandaplus   (2 x H200)
#   sbatch slurm/serve_annotator.sbatch apertus     (2 x L40 as above)
JUDGE=$1; shift
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME   # /home is at quota: keep any stray cache off it
export VLLM_USE_FLASHINFER_SAMPLER=0      # FlashInfer JIT needs nvcc, which the GPU nodes lack; use the built-in sampler
export PATH=$PATH:$DLSD_ROOT/env/vllm-venv/bin
export HF_HUB_OFFLINE=1
case $JUDGE in
  gptoss_unused)    MODEL_DIR=$DLSD_MODELS_DIR/openai__gpt-oss-120b;                 SERVED=openai/gpt-oss-120b;                 EXTRA="--max-model-len 8192";;
  aya)       MODEL_DIR=$DLSD_MODELS_DIR/CohereLabs__aya-expanse-32b;                SERVED=CohereLabs/aya-expanse-32b;                EXTRA="--max-model-len 8192 --dtype bfloat16";;
  commandaplus) MODEL_DIR=$DLSD_MODELS_DIR/CohereLabs__command-a-plus-05-2026-w4a4;  SERVED=CohereLabs/command-a-plus-05-2026-w4a4;  EXTRA="--max-model-len 8192";;
  apertus)   MODEL_DIR=$DLSD_MODELS_DIR/swiss-ai__Apertus-70B-Instruct-2509;  SERVED=swiss-ai/Apertus-70B-Instruct-2509;  EXTRA="--max-model-len 8192 --dtype bfloat16 --quantization fp8";;
  glm45air_unused)  MODEL_DIR=$DLSD_MODELS_DIR/zai-org__GLM-4.5-Air-FP8;             SERVED=zai-org/GLM-4.5-Air-FP8;             EXTRA="--max-model-len 8192";;
  *) echo "unknown annotator $JUDGE"; exit 2;;
esac
PORT=$((8900 + SLURM_JOB_ID % 800)); export DLSD_JUDGE_PORT=$PORT      # per-job port: two jobs can share a node
LOG=$DLSD_ROOT/logs/vllm_${JUDGE}_${SLURM_JOB_ID}.log
$DLSD_ROOT/env/vllm-venv/bin/vllm serve "$MODEL_DIR" --served-model-name $SERVED --port $PORT --gpu-memory-utilization 0.92 $EXTRA ${DLSD_TP:+--tensor-parallel-size $DLSD_TP} > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; sleep 5; pkill -9 -P $$ 2>/dev/null' EXIT
for i in $(seq 1 720); do
    curl -sf localhost:$PORT/v1/models > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited early"; tail -25 "$LOG"; exit 1; }
    sleep 10
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "vLLM not ready after 2 h"; exit 1; }
cd $DLSD_ROOT/dlsd-health-experiments
export PYTHONPATH=src
python -W ignore scripts/02m_llm_annotate.py --annotator $JUDGE "$@"
