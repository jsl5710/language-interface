#!/bin/bash
#SBATCH --job-name=dlsd-gen
#SBATCH --account=ucb860_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# Serve ONE tested model with vLLM and run generation against it, then shut it down (one model at a time, as agreed).
#   sbatch slurm/serve_model.sbatch gemma exp1 [--pilot|--dry-run]
#   sbatch slurm/serve_model.sbatch gemma exp1,exp2 --axes baseline dialect resource     # both experiments, only these condition axes
#   Each experiment runs a 20-item pilot first and stops if more than 20% of multiple-choice answers cannot be parsed (docs/EXPERIMENTS.md 13, step 4).
# keys: gemma medgemma meditron mistral llama med42. Ports and model ids match configs/experiment.yaml.
KEY=$1; EXPS=$2; shift 2
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export PATH=$DLSD_ROOT/env/vllm-venv/bin:$PATH
EXTRA="--max-model-len 8192 --dtype bfloat16 --max-logprobs 20"
case $KEY in
  gemma)    ID=google/gemma-3-27b-it;                          DIR=google__gemma-3-27b-it;           PORT=8000;;
  medgemma) ID=google/medgemma-27b-text-it;                    DIR=google__medgemma-27b-text-it;     PORT=8001;;
  meditron) ID=epfl-llm/meditron-7b;                           DIR=epfl-llm__meditron-7b;            PORT=8002; EXTRA="--max-model-len 2048 --dtype bfloat16 --max-logprobs 20";;   # Llama-2 context: 2048 tokens
  mistral)  ID=mistralai/Mistral-Small-3.2-24B-Instruct-2506;  DIR=mistralai__Mistral-Small-3.2-24B-Instruct-2506; PORT=8003; EXTRA="$EXTRA --tokenizer-mode mistral --limit-mm-per-prompt {\"image\":0}";;   # the folder holds HF-format shards (no consolidated.safetensors), so the native loader finds no weights; images disabled because the folder has no HF image processor
  llama)    ID=meta-llama/Llama-3.1-8B-Instruct;               DIR=meta-llama__Llama-3.1-8B-Instruct; PORT=8004;;
  med42)    ID=m42-health/Llama3-Med42-8B;                     DIR=m42-health__Llama3-Med42-8B;      PORT=8005;;
  *) echo "unknown model key $KEY"; exit 2;;
esac
PORT=$((8200 + SLURM_JOB_ID % 700))      # per-job port so two jobs can share a node
export DLSD_MODEL_PORT=$PORT
export DLSD_SHARD=${KEY}_${SLURM_JOB_ID}      # own output files, merged afterwards (scripts/09_merge_shards.py)
LOG=$DLSD_ROOT/logs/vllm_${KEY}_${SLURM_JOB_ID}.log
$DLSD_ROOT/env/vllm-venv/bin/vllm serve $DLSD_MODELS_DIR/$DIR --served-model-name $ID --port $PORT --gpu-memory-utilization 0.90 $EXTRA > "$LOG" 2>&1 &
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null; sleep 5; pkill -9 -P $$ 2>/dev/null' EXIT      # jobs 35576894 and 35576921 finished their work and then hung until the time limit
for i in $(seq 1 720); do
    curl -sf localhost:$PORT/v1/models > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited early"; tail -25 "$LOG"; exit 1; }
    sleep 10
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "vLLM not ready after 2 h"; exit 1; }
# log-probability check (Experiment 2 needs first-token top log probabilities)
curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the letter A.\"}],\"max_tokens\":2,\"logprobs\":true,\"top_logprobs\":20}" | head -c 600 > $DLSD_ROOT/logs/logprob_check_${KEY}.txt
cd $DLSD_ROOT/dlsd-health-experiments
export PYTHONPATH=src
if [ -n "$DLSD_RETRY_EMPTY" ]; then      # re-ask empty multiple-choice replies (Meditron raw prompt; docs/DEVIATIONS.md 2026-11-15)
    python -W ignore scripts/03e_retry_empty.py --model $ID
    exit $?
fi
if [ -n "$DLSD_DETERMINISM" ]; then      # sanity check 10: re-run 20 temperature-0 rows
    python -W ignore scripts/03d_determinism.py --model $ID
    exit $?
fi
if [ -n "$DLSD_REGEN_VERBALIZED" ]; then      # only re-ask the verbalised-confidence question (revised parser, docs/DEVIATIONS.md 2026-11-14)
    python -W ignore scripts/03c_regen_verbalized.py --model $ID
    [ "$EXPS" != none ] || exit 0           # EXPS=none: regenerate only; otherwise continue with the generation below
fi
for EXP in ${EXPS//,/ }; do
    [ -n "$DLSD_SKIP_PILOT" ] || python -W ignore scripts/03_generate.py --experiment $EXP --model $ID --pilot "$@"
    [ -n "$DLSD_SKIP_PILOT" ] || python - "$ID" "results/pilot/generations_$EXP.$DLSD_SHARD.jsonl" <<'PY' || { echo "pilot parse check failed: stop and ask the author"; exit 3; }
import json, sys
mid, path = sys.argv[1:]
rows = [json.loads(l) for l in open(path) if l.strip()]
rows = [r for r in rows if r["model_id"] == mid and "parsed_letter" in r]
bad = sum(r["parsed_letter"] is None for r in rows)
print(f"pilot {mid}: {bad}/{len(rows)} unparseable")
sys.exit(1 if rows and bad / len(rows) > 0.20 else 0)
PY
    python -W ignore scripts/03_generate.py --experiment $EXP --model $ID "$@"
    # sensitivity run (DLSD_GATEFAILED=1): the same model on renderings the judge panel rejected, all conditions, separate output file
    [ -z "$DLSD_GATEFAILED" ] || python -W ignore scripts/03_generate.py --experiment $EXP --model $ID --gate-failed
done
