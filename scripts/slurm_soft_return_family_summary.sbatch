#!/bin/bash
#SBATCH --job-name=soft-family-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/soft_return_family/%j_summary.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/soft_return_family/%j_summary.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/soft_return_family_v1}
cd "${REPO_ROOT}"
.venv/bin/python scripts/summarize_soft_return_families.py \
  --fold-root "${STUDY_ROOT}/folds" --output-root "${STUDY_ROOT}/summary" --strict
