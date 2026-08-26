#!/bin/bash
#SBATCH --job-name=sina-cninfo-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_sina_cninfo_method/%j_audit.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_sina_cninfo_method/%j_audit.err

set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_sina_cninfo_method reports
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

.venv/bin/python scripts/audit_sina_cninfo_method_embeddings.py \
  --root /home/gaozh/news_content_quality_20260812/cninfo_method_v1/embeddings \
  --expected-shards 4 --expected-rows 4928 \
  --output reports/sina_cninfo_method_embedding_audit_v1.json
