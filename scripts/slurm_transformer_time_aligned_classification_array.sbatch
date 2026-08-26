#!/bin/bash
#SBATCH --job-name=zh-time-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-5%3
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_zh_time_classification/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_zh_time_classification/%A_%a.err

set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p reports/classification/transformers_time_aligned logs/slurm_zh_time_classification
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

models=(roberta bge_m3)
files=(data/processed/embeddings/chinese_roberta/chinese_roberta.npy data/processed/embeddings/bge_m3/bge_m3.npy)
model=${models[$((SLURM_ARRAY_TASK_ID / 3))]}
embedding=${files[$((SLURM_ARRAY_TASK_ID / 3))]}
classifiers=(logistic random_forest mlp)
classifier=${classifiers[$((SLURM_ARRAY_TASK_ID % 3))]}
output="reports/classification/transformers_time_aligned/${model}_${classifier}_seed42.json"

extra_flag="--${model}-npy"
if [[ "${model}" == "bge_m3" ]]; then
  extra_flag="--bge-m3-npy"
fi
/usr/bin/python3 scripts/run_paper_rolling_classification.py \
  data/processed/cninfo_100_classification_panel_time_aligned.parquet \
  --train-target-column event_return_3d \
  --evaluation-target-column next_day_return \
  "${extra_flag}" "${embedding}" \
  --classifiers "${classifier}" \
  --seed 42 \
  --search-stage coarse \
  --max-input-chars 12000 \
  --output "${output}"