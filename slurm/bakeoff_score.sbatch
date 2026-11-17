#!/bin/bash
#SBATCH --job-name=dlsd-bakescore
#SBATCH --account=ucb860_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
source /projects/jalu7684/DivideHeathAI/env.sh
export HF_HUB_OFFLINE=1
cd /projects/jalu7684/DivideHeathAI/dlsd-health-experiments
PYTHONPATH=src python -W ignore scripts/bakeoff_score.py
