#!/bin/bash
#SBATCH --job-name=dlsd-dialfin
#SBATCH --account=ucb860_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
# After the dialect shards are complete: quality table, spot-check sample, determinism re-render, merge, sanity checks.
source /projects/jalu7684/DivideHeathAI/env.sh
cd /projects/jalu7684/DivideHeathAI/dlsd-health-experiments
export PYTHONPATH=src HF_HUB_OFFLINE=1
python -W ignore scripts/02a_render_dialects.py --summarize
python -W ignore scripts/02a_render_dialects.py --check-determinism
python -W ignore scripts/02e_merge_rendered.py
python -W ignore scripts/check_results.py
