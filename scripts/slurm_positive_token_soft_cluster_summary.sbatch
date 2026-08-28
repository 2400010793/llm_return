#!/bin/bash
#SBATCH --job-name=positive-token-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/positive_token_soft_cluster/%j_summary.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/positive_token_soft_cluster/%j_summary.err
set -euo pipefail
: "${OUTPUT_ROOT:?OUTPUT_ROOT required}"
cd /data/alpha_team2/shares/llm_return
.venv/bin/python scripts/summarize_positive_token_soft_cluster.py --root "${OUTPUT_ROOT}"
