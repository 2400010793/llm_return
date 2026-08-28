#!/bin/bash
#SBATCH --job-name=prompt-base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%j_baseline.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%j_baseline.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
cd "${REPO_ROOT}"
mkdir -p logs/prompt_mechanism "${STUDY_ROOT}/baselines"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/build_prompt_only_baselines.py \
  --embedding-root "${SINA_ROOT}/embeddings" \
  --output-root "${STUDY_ROOT}/baselines" --device cpu
