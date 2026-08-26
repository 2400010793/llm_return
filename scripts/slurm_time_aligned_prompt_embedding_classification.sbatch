#!/bin/bash
# Classify all newly generated prompt embeddings under the same rolling design.
# Inputs: plain, short, long, masked-short, masked-long; models: RoBERTa/BGE-M3.
# Array: 2 models x 5 input variants x 5 classifiers x 2 reducers.
#SBATCH --job-name=prompt-embed-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-99%6
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_prompt_embed_cls/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_prompt_embed_cls/%A_%a.err

set -euo pipefail
cd /home/gaozh/llm_return
mkdir -p reports/classification/time_aligned_prompt_embeddings logs/slurm_prompt_embed_cls
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

models=(roberta bge_m3)
inputs=(text_plain text_input_2_short text_input_3_long text_input_5_masked_short text_input_6_masked_long)
classifiers=(logistic linear_svm random_forest mlp knn)
reducers=(none pca)
model_index=$((SLURM_ARRAY_TASK_ID / 50))
input_index=$(((SLURM_ARRAY_TASK_ID / 10) % 5))
classifier=${classifiers[$(((SLURM_ARRAY_TASK_ID / 2) % 5))]}
reducer=${reducers[$((SLURM_ARRAY_TASK_ID % 2))]}
model=${models[$model_index]}
input=${inputs[$input_index]}
case "${model}" in
  roberta) name=chinese_roberta; flag=--roberta-npy; matrix="data/processed/embeddings/all_inputs/${name}/${input}/chinese_roberta.npy" ;;
  bge_m3) name=bge_m3; flag=--bge-m3-npy; matrix="data/processed/embeddings/all_inputs/${name}/${input}/bge_m3.npy" ;;
esac
output="reports/classification/time_aligned_prompt_embeddings/${model}_${input}_${classifier}_${reducer}_seed42.json"
/usr/bin/python3 scripts/run_paper_rolling_classification.py \
  data/processed/cninfo_100_classification_panel_time_aligned.parquet \
  --train-target-column event_return_3d \
  --evaluation-target-column next_day_return \
  --representations "" \
  "${flag}" "${matrix}" \
  --classifiers "${classifier}" \
  --reducer "${reducer}" --reducer-components 128 \
  --seed 42 --search-stage coarse --max-input-chars 0 \
  --output "${output}"