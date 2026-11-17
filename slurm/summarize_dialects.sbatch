#!/bin/bash
#SBATCH --job-name=dlsd-dialsum
#SBATCH --account=ucb860_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
source /projects/jalu7684/DivideHeathAI/env.sh
cd /projects/jalu7684/DivideHeathAI/dlsd-health-experiments
PYTHONPATH=src python -W ignore scripts/02a_render_dialects.py --summarize
