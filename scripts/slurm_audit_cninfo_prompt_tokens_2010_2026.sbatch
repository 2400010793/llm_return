#!/bin/bash
#SBATCH --job-name=cninfo-prompt-token-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_prompt_token_audit/%j.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_prompt_token_audit/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?Prompt-token audit must use scripts/task_tracker.py}"
PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
ROOT=${ROOT:-data/processed/prompt_token_embeddings_2010_2026_clean_v1}
OUTPUT=${OUTPUT:-reports/cninfo_prompt_token_embeddings_2010_2026_clean_v1_audit.json}
mkdir -p logs/slurm_cninfo_prompt_token_audit reports
"${PYTHON}" scripts/audit_cninfo_prompt_tokens_2010_2026.py \
  --root "${ROOT}" --rows 903665 --num-shards 64 --output "${OUTPUT}"
