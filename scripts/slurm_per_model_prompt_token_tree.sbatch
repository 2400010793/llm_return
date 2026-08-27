#!/bin/bash
#SBATCH --job-name=prompt-token-tree
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/per_model_prompt_tree/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/per_model_prompt_tree/%A_%a.err
set -euo pipefail
: "${FAIR_ROOT:?}" "${TREE_ROOT:?}" "${TASK_RECORD_ID:?}"
cd /mnt/lustre3/home/gaozh/llm_return
datasets=(sina cninfo)
models=(roberta bge_m3 qwen3_embedding_8b)
index=${SLURM_ARRAY_TASK_ID}
dataset=${datasets[$((index / 3))]}
model=${models[$((index % 3))]}
if [[ "${dataset}" == "sina" ]]; then
  history=8
  validation=2
else
  history=3
  validation=1
fi
fair_root=${FAIR_ROOT}/${dataset}
output=${TREE_ROOT}/${dataset}/${model}
mkdir -p "$(dirname "${output}")" logs/per_model_prompt_tree
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
.venv/bin/python scripts/run_per_model_prompt_token_tree.py \
  --fair-root "${fair_root}" --dataset "${dataset}" --model "${model}" \
  --history-years "${history}" --validation-years "${validation}" \
  --output "${output}" --threads "${SLURM_CPUS_PER_TASK}"
