#!/bin/bash
# Gate the five document-level variants of one non-Qwen embedding model.
# Array mapping: 4 embedding models x 5 static/dynamic gate methods.
#SBATCH --job-name=sina-variant-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-19%16
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_sina_variant_gate/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_sina_variant_gate/%A_%a.err

set -euo pipefail

REPO_ROOT=/data/alpha_team2/shares/llm_return
DATA_ROOT=/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812
PANEL=${PANEL:-${DATA_ROOT}/classification/sina_single_stock_classification_panel.parquet}
EMBEDDING_ROOT=${EMBEDDING_ROOT:-${DATA_ROOT}/embeddings/single_stock}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DATA_ROOT}/classification/results_variant_gate_nonqwen}
CLASSIFIER=${CLASSIFIER:-logistic}

cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?gate array must be submitted through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
mkdir -p logs/slurm_sina_variant_gate "${OUTPUT_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK}
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

models=(chinese_bert chinese_roberta bge_m3 word2vec)
gates=(uniform variance fisher logistic_l1 dynamic)
model_index=$((SLURM_ARRAY_TASK_ID / 5))
gate_index=$((SLURM_ARRAY_TASK_ID % 5))
model=${models[$model_index]}
gate=${gates[$gate_index]}
output="${OUTPUT_ROOT}/${model}_${gate}_${CLASSIFIER}_none.json"

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} model=${model} gate=${gate} classifier=${CLASSIFIER}"

exec "${REPO_ROOT}/.venv/bin/python" scripts/run_sina_variant_gate_classification.py \
  "${PANEL}" \
  --embedding-root "${EMBEDDING_ROOT}" \
  --model "${model}" \
  --gate-mode "${gate}" \
  --classifier "${CLASSIFIER}" \
  --target-column next_day_return \
  --date-column entry_date \
  --stock-column stock_id \
  --panel-row-index-column article_id \
  --metadata-row-index-column article_id \
  --expected-rows 4928 \
  --search-stage coarse \
  --temperature 1.0 \
  --seed 42 \
  --output "${output}"
