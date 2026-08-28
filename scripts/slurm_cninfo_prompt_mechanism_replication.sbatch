#!/bin/bash
#SBATCH --job-name=cninfo-prompt-repl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%A_%a_cninfo.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%A_%a_cninfo.err
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
STUDY_ROOT=${STUDY_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_mechanism_v1}
cd "${REPO_ROOT}"
models=(roberta bge_m3)
model=${models[${SLURM_ARRAY_TASK_ID}]}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/run_cninfo_prompt_mechanism_replication.py \
  --embedding-root data/processed/prompt_token_embeddings_2010_2026_clean_v1 \
  --panel data/processed/cninfo_full_classification_panel_2010_2026.parquet \
  --baseline-root "${STUDY_ROOT}/cninfo_replication/baselines" \
  --output-root "${STUDY_ROOT}/cninfo_replication/results" \
  --model "${model}" --expected-shards 64 --expected-rows 903665 --batch-size 64
