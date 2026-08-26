#!/bin/bash
#SBATCH --job-name=sina1-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/sina_single_stock_cninfo_v1/%j_audit.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/sina_single_stock_cninfo_v1/%j_audit.err

set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
root=/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1

.venv/bin/python scripts/audit_sina_cninfo_method_embeddings.py \
  --root "${root}/embeddings" --expected-shards 64 --expected-rows 75894 \
  --output "${root}/embedding_audit.json"
