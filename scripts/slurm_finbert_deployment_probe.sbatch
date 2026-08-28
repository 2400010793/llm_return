#!/bin/bash
#SBATCH --job-name=finbert-probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_finbert_probe/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_finbert_probe/%j.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_finbert_probe reports/finbert
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/team/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

.venv/bin/python scripts/probe_finbert_deployment.py \
  --models "${MODELS}" \
  --cache-dir "${HF_HOME}" \
  --max-length "${MAX_LENGTH:-256}" \
  --output "${OUTPUT}"
