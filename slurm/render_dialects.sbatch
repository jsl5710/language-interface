#!/bin/bash
#SBATCH --job-name=dlsd-render-dialect
#SBATCH --account=ucb860_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%A_%a.out
# Usage: sbatch slurm/render_dialects.sbatch [--dry-run]     (one array task per dialect, resumable)
source /projects/jalu7684/DivideHeathAI/env.sh
cd /projects/jalu7684/DivideHeathAI/dlsd-health-experiments
DIALECTS=("AAVE" "Indian English" "Nigerian English" "Jamaican English" "White Zimbabwean English")
PYTHONPATH=src python -W ignore scripts/02a_render_dialects.py --dialect "${DIALECTS[$SLURM_ARRAY_TASK_ID]}" "$@"
