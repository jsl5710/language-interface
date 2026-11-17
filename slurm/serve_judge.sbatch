#!/bin/bash
#SBATCH --job-name=dlsd-judge
#SBATCH --account=ucb860_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# Serve ONE judge of the code-switch panel and run it over the shards.
#   sbatch slurm/serve_judge.sbatch gpt_oss_120b --round 1
#   sbatch slurm/serve_judge.sbatch apertus_70b --round 1
#   sbatch slurm/serve_judge.sbatch glm_45_air --round 1
# 2 x L40 route (much shorter queue): sbatch --partition=al40 --gres=gpu:l40:2 --export=ALL,DLSD_TP=2 slurm/serve_judge.sbatch <judge> ...
# The first argument picks the judge; the rest goes to scripts/02c_judge_switch.py.
JUDGE=$1; shift
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME   # /home is at quota: keep any stray cache off it
export VLLM_USE_FLASHINFER_SAMPLER=0      # FlashInfer JIT needs nvcc, which the GPU nodes lack; use the built-in sampler
export PATH=$PATH:$DLSD_ROOT/env/vllm-venv/bin
export HF_HUB_OFFLINE=1
case $JUDGE in
  gpt_oss_120b)     MODEL_DIR=$DLSD_MODELS_DIR/openai__gpt-oss-120b;                 SERVED=openai/gpt-oss-120b;                 EXTRA="--max-model-len 8192";;
  apertus_70b)      MODEL_DIR=$DLSD_MODELS_DIR/swiss-ai__Apertus-70B-Instruct-2509;  SERVED=swiss-ai/Apertus-70B-Instruct-2509;  EXTRA="--max-model-len 8192 --dtype bfloat16 --quantization fp8";;   # bf16 weights (141 GB) do not fit one H200
  glm_45_air)       MODEL_DIR=$DLSD_MODELS_DIR/zai-org__GLM-4.5-Air-FP8;             SERVED=zai-org/GLM-4.5-Air-FP8;             EXTRA="--max-model-len 8192";;
  *) echo "unknown judge $JUDGE"; exit 2;;
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
# qualify on the held-out probe first; stop (exit 3) if the judge is below switch_judges.probe_min_accuracy
python -W ignore scripts/02i_probe_judge.py --judge $JUDGE || { echo "probe failed (exit $?): shards not labelled"; exit 3; }
[ -n "$DLSD_PROBE_ONLY" ] && { echo "probe only: shards not labelled"; exit 0; }
python -W ignore scripts/02c_judge_switch.py --judge $JUDGE --shots few "$@"
[ -n "$DLSD_SKIP_ZERO" ] || python -W ignore scripts/02c_judge_switch.py --judge $JUDGE --shots zero "$@"      # ablation: same server, no worked examples
