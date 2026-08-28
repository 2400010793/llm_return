#!/bin/bash
#SBATCH --job-name=qwen-linear-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_qwen_linear/%j_audit.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_qwen_linear/%j_audit.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
mkdir -p logs/slurm_qwen_linear reports/regression/qwen_2018_2026_linear_v1
.venv/bin/python scripts/audit_qwen_linear_study.py \
  --source-root data/processed/embeddings/qwen3_embedding_8b_candidates_1000 \
  --panel data/processed/cninfo_full_o2o_residual_panel.parquet \
  --output reports/regression/qwen_2018_2026_linear_v1/embedding_alignment_audit.json
