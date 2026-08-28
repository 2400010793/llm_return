#!/bin/bash
#SBATCH --job-name=bge-embed-barrier
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_bge_embed_barrier/%j.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_bge_embed_barrier/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?BGE recovery barrier must use scripts/task_tracker.py}"

mkdir -p logs/slurm_bge_embed_barrier reports
"${REPO_ROOT}/.venv/bin/python" scripts/monitor_bge_embedding_completion.py \
  --job-variant 4330365:short \
  --job-variant 4330639:masked_short \
  --output-root data/processed/pooled_embeddings_increment_2010_2017_clean_v1 \
  --row-start 350578 \
  --row-end 903665 \
  --poll-seconds 30 \
  --max-requeues 3 \
  --report reports/bge_embedding_recovery_barrier_2010_2017.json
