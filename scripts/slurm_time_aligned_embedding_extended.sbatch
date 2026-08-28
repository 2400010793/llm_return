#!/bin/bash
# Existing dense embeddings: three-day-event training / next-day evaluation.
# Array: embedding x classifier x reducer.
#SBATCH --job-name=embed-time-ext
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-29%6
#SBATCH --output=/home/team/llm_return/logs/slurm_embed_time_ext/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_embed_time_ext/%A_%a.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p reports/classification/time_aligned_embedding_extended logs/slurm_embed_time_ext
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

embeddings=(roberta bge_m3 qwen)
files=(
  data/processed/embeddings/chinese_roberta/chinese_roberta.npy
  data/processed/embeddings/bge_m3/bge_m3.npy
  data/processed/embeddings/qwen3_embedding_8b/embeddings.npy
)
classifiers=(logistic linear_svm random_forest mlp knn)
reducers=(none pca)
embedding=${embeddings[$((SLURM_ARRAY_TASK_ID / 10))]}
matrix=${files[$((SLURM_ARRAY_TASK_ID / 10))]}
classifier=${classifiers[$(((SLURM_ARRAY_TASK_ID / 2) % 5))]}
reducer=${reducers[$((SLURM_ARRAY_TASK_ID % 2))]}

case "${embedding}" in
  roberta) flag="--roberta-npy" ;;
  bge_m3) flag="--bge-m3-npy" ;;
  qwen) flag="--qwen-npy" ;;
esac

output="reports/classification/time_aligned_embedding_extended/${embedding}_${classifier}_${reducer}_seed42.json"
/usr/bin/python3 scripts/run_paper_rolling_classification.py \
  data/processed/cninfo_100_classification_panel_time_aligned.parquet \
  --train-target-column event_return_3d \
  --evaluation-target-column next_day_return \
  --representations "" \
  "${flag}" "${matrix}" \
  --classifiers "${classifier}" \
  --reducer "${reducer}" \
  --reducer-components 128 \
  --seed 42 \
  --search-stage coarse \
  --max-input-chars 0 \
  --output "${output}"