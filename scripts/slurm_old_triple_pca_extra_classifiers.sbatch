#!/bin/bash
# PCA-128 only: 2 models x 4 variants x (Random Forest, exact KNN).
# NB-SVM/MultinomialNB/ComplementNB are intentionally excluded for dense PCA.
#SBATCH --job-name=triple-pca-extra
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-15%2
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_triple_pca_extra/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_triple_pca_extra/%A_%a.err
set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p logs/slurm_triple_pca_extra reports/classification/pooled_embeddings/triple_pca_extra
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
models=(roberta bge_m3)
variants=(short masked_short long masked_long)
classifiers=(random_forest knn)
model=${models[$((SLURM_ARRAY_TASK_ID / 8))]}
variant=${variants[$(((SLURM_ARRAY_TASK_ID / 2) % 4))]}
classifier=${classifiers[$((SLURM_ARRAY_TASK_ID % 2))]}
case "${variant}" in
  short|masked_short) root=data/processed/pooled_embeddings_v3 ;;
  long|masked_long) root=data/processed/pooled_long_embeddings_v4 ;;
esac
output="reports/classification/pooled_embeddings/triple_pca_extra/${model}_${variant}_title_body_full_concat_${classifier}_pca128.json"
/usr/bin/python3 scripts/run_pooled_embedding_classification.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --embedding-root "${root}" --model "${model}" --variant "${variant}" \
  --feature title_body_full_concat --classifier "${classifier}" \
  --reducer pca --reducer-components 128 --scaler auto \
  --train-target-column event_return_3d --evaluation-target-column next_day_return \
  --date-column entry_date --panel-row-index-column row_index \
  --search-stage coarse --seed 42 --output "${output}"