#!/bin/bash
#SBATCH --job-name=qwen-time-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-2%3
#SBATCH --output=/home/team/llm_return/logs/slurm_qwen_time_classification/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_qwen_time_classification/%A_%a.err

set -euo pipefail
cd /home/team/llm_return
mkdir -p reports/classification/qwen3_embedding_8b_time_aligned logs/slurm_qwen_time_classification
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

classifiers=(logistic random_forest mlp)
classifier=${classifiers[$SLURM_ARRAY_TASK_ID]}
output="reports/classification/qwen3_embedding_8b_time_aligned/${classifier}_seed42.json"

/usr/bin/python3 scripts/run_paper_rolling_classification.py \
  data/processed/cninfo_100_classification_panel_time_aligned.parquet \
  --train-target-column event_return_3d \
  --evaluation-target-column next_day_return \
  --representations "" \
  --qwen-npy data/processed/embeddings/qwen3_embedding_8b/embeddings.npy \
  --classifiers "${classifier}" \
  --seed 42 \
  --search-stage coarse \
  --max-input-chars 12000 \
  --output "${output}"