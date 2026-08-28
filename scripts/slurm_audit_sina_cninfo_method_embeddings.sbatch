#!/bin/bash
#SBATCH --job-name=sina-cninfo-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/team/llm_return/logs/slurm_sina_cninfo_method/%j_audit.out
#SBATCH --error=/home/team/llm_return/logs/slurm_sina_cninfo_method/%j_audit.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p logs/slurm_sina_cninfo_method reports
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

.venv/bin/python scripts/audit_sina_cninfo_method_embeddings.py \
  --root /data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/embeddings \
  --expected-shards 4 --expected-rows 4928 \
  --output reports/sina_cninfo_method_embedding_audit_v1.json
