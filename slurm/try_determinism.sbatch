#!/bin/bash
#SBATCH --job-name=dlsd-determ
#SBATCH --account=ucb860_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jalu7684/DivideHeathAI/logs/%x_%j.out
source /projects/jalu7684/DivideHeathAI/env.sh
cd /projects/jalu7684/DivideHeathAI/dlsd-health-experiments
PYTHONPATH=src python -W ignore /projects/jalu7684/DivideHeathAI/slurm/try_determinism.py
