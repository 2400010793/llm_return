#!/bin/bash
# One array element processes one matrix/year/mode and both return targets.
#SBATCH --job-name=all-emb-hdb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/all_embedding_hdbscan/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/all_embedding_hdbscan/%A_%a.err
set -euo pipefail
cd /data/alpha_team2/shares/llm_return
mkdir -p logs/all_embedding_hdbscan
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
MANIFEST=${ALL_EMB_MANIFEST:?set ALL_EMB_MANIFEST}
ARRAY_OFFSET=${ARRAY_OFFSET:-0}
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + ARRAY_OFFSET + 2))p" "$MANIFEST")
IFS=$'\t' read -r dataset model prompt variant feature test_year mode panel matrix metadata <<< "$ROW"
.venv/bin/python scripts/run_all_embedding_hdbscan_fold.py \
  --panel "$panel" --matrix "$matrix" --metadata "$metadata" \
  --output-root /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/results_all_embedding_cluster_regression \
  --dataset "$dataset" --model "$model" --prompt "$prompt" --variant "$variant" --feature "$feature" \
  --test-year "$test_year" --mode "$mode"
