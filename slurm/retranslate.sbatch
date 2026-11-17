#!/bin/bash
#SBATCH --job-name=dlsd-retranslate
#SBATCH --account=ucb860_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# NLLB only (no vLLM): re-translate resource items that hold a value in parentheses.
source /projects/jalu7684/DivideHeathAI/env.sh
export HOME=$DLSD_SCRATCH/home; mkdir -p $HOME
export HF_HUB_OFFLINE=1
cd $DLSD_ROOT/dlsd-health-experiments
PYTHONPATH=src python -W ignore scripts/02h_retranslate_parens.py
