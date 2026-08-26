#!/bin/bash
#SBATCH --job-name=qwen-new-hdb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_new_cluster/fold_%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/qwen_new_cluster/fold_%A_%a.err
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return
MANIFEST=${CORRECT_EMB_MANIFEST:?missing manifest}
IFS=$'\t' read -r source dataset model prompt variant feature test_year mode panel matrix metadata embedding_root pooled_feature <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")"
.venv/bin/python scripts/run_all_embedding_hdbscan_fold.py \
  --panel "${panel}" --matrix "${matrix}" --metadata "${metadata}" \
  --output-root "${QWEN_CLUSTER_OUTPUT:?missing output}" \
  --dataset "${dataset}" --model "${model}" --prompt "${prompt}" \
  --variant "${variant}" --feature "${feature}" --test-year "${test_year}" \
  --mode fixed --pca-sample-rows 20000 --cluster-sample-rows 20000
