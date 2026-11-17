#!/bin/bash
#SBATCH --job-name=dlsd-backtr
#SBATCH --account=ucb860_asc1
#SBATCH --partition=al40
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# Back-translate the resource-level renderings to English with NLLB-200 3.3B (about 7 GB in bf16). Scoring runs afterwards on CPU.
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME
export HF_HUB_OFFLINE=1
cd $DLSD_ROOT/dlsd-health-experiments
PYTHONPATH=src $DLSD_ROOT/env/vllm-venv/bin/python -W ignore scripts/02q_backtranslate.py translate
