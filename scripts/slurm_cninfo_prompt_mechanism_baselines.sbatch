#!/bin/bash
#SBATCH --job-name=cninfo-prompt-base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/prompt_mechanism/%j_cninfo_base.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/prompt_mechanism/%j_cninfo_base.err
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
STUDY_ROOT=${STUDY_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_mechanism_v1}
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/build_prompt_only_baselines.py \
  --embedding-root data/processed/prompt_token_embeddings_2010_2026_clean_v1 \
  --output-root "${STUDY_ROOT}/cninfo_replication/baselines" \
  --model roberta --model bge_m3 --prompt-length short --device cpu
