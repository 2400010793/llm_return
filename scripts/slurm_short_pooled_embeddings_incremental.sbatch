#!/bin/bash
# Encode only the historical prompt shards; frozen 2018--2026 arrays are not
# passed through the transformer again.
#SBATCH --job-name=short-pool-increment
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-63%16
#SBATCH --output=/home/team/llm_return/logs/slurm_short_pool_increment/%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_short_pool_increment/%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?incremental embedding generation must use scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array}"

INPUT_ROOT=${INPUT_ROOT:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017/prompt_v3_fixed_parts}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/processed/pooled_embeddings_increment_2010_2017_v1}
NUM_SHARDS=${NUM_SHARDS:-32}
BATCH_SIZE=${BATCH_SIZE:-4}
VARIANTS=${VARIANTS:-short,masked_short}

shard=$((SLURM_ARRAY_TASK_ID / 2))
model_id=$((SLURM_ARRAY_TASK_ID % 2))
(( shard < NUM_SHARDS )) || { echo "invalid shard ${shard}" >&2; exit 2; }
case "${model_id}" in
  0) model=roberta; max_length=512 ;;
  1) model=bge_m3; max_length=1000 ;;
  *) echo "invalid model id ${model_id}" >&2; exit 2 ;;
esac

input="${INPUT_ROOT}/shard-${shard}"
base="${OUTPUT_ROOT}/shard-${shard}/${model}"
[[ -d "${input}" ]] || { echo "prompt shard is missing: ${input}" >&2; exit 2; }
mkdir -p logs/slurm_short_pool_increment "${base}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

"${REPO_ROOT}/.venv/bin/python" -m py_compile scripts/run_short_pooled_embeddings.py
IFS=',' read -r -a variants <<< "${VARIANTS}"
for variant in "${variants[@]}"; do
  case "${variant}" in
    short|masked_short) ;;
    *) echo "unsupported incremental variant: ${variant}" >&2; exit 2 ;;
  esac
  output="${base}/${variant}"
  if [[ -s "${output}/summary.json" && -s "${output}/metadata.jsonl" && -s "${output}/short_pooling.npz" ]]; then
    echo "complete output already exists: ${output}"
    continue
  fi
  mkdir -p "${output}"
  "${REPO_ROOT}/.venv/bin/python" scripts/run_short_pooled_embeddings.py \
    "${input}" --output-dir "${output}" --model "${model}" --variant "${variant}" \
    --batch-size "${BATCH_SIZE}" --max-length "${max_length}"
done
