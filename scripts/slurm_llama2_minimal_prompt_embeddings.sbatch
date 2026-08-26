#!/bin/bash
#SBATCH --job-name=llama2-minimal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/gaozh/llm_return/logs/llama2_minimal/%A_%a.out
#SBATCH --error=/home/gaozh/llm_return/logs/llama2_minimal/%A_%a.err
set -euo pipefail
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${DATASET:?DATASET must be sina or cninfo}"
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO_ROOT=${REPO_ROOT:-/home/gaozh/llm_return}
MODEL_PATH=${MODEL_PATH:-/mnt/lustre3/home/gaozh/models/Llama-2-13b-hf}
MODEL_NAME=${MODEL_NAME:-llama2_13b}
MODEL_DIRECTORY=${MODEL_DIRECTORY:-llama2_13b}
FORMAT_VERSION=${FORMAT_VERSION:-llama2_causal_readout_v2}
PROMPT_SPEC=${PROMPT_SPEC:-/home/gaozh/llm_return/configs/prompts/minimal_v2.json}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-32000}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-500}
case "${DATASET}" in
  sina|cninfo) ;;
  *) echo "invalid DATASET=${DATASET}" >&2; exit 2 ;;
esac
GPU_FREE_MIB=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits)
GPU_NAME=$(nvidia-smi --id="${GPU_ID}" --query-gpu=name --format=csv,noheader)
echo "host=$(hostname) gpu=${GPU_ID} name=${GPU_NAME} free_mib=${GPU_FREE_MIB}"
if (( GPU_FREE_MIB < MIN_FREE_GPU_MIB )); then
  echo "GPU ${GPU_ID} has ${GPU_FREE_MIB} MiB free; need ${MIN_FREE_GPU_MIB}" >&2
  exit 75
fi
CNINFO_EXISTING_INPUT=${CNINFO_EXISTING_INPUT:-${REPO_ROOT}/data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts}
CNINFO_INCREMENT_INPUT=${CNINFO_INCREMENT_INPUT:-${REPO_ROOT}/data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017_clean_v1/prompt_v3_fixed_parts}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
if [[ "${DATASET}" == sina ]]; then
  SHARD=$SLURM_ARRAY_TASK_ID
  INPUT=${SINA_ROOT}/inputs/shard-${SHARD}
  OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/prompt_minimal_v2/llama2_embeddings}
else
  SHARD=$SLURM_ARRAY_TASK_ID
  if (( SHARD < 32 )); then INPUT=${CNINFO_EXISTING_INPUT}/shard-${SHARD}; else INPUT=${CNINFO_INCREMENT_INPUT}/shard-$((SHARD-32)); fi
  OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/data/processed/cninfo_minimal_prompt_v2/llama2_embeddings}
fi
mkdir -p "${OUTPUT_ROOT}/shard-${SHARD}" logs/llama2_minimal
for VARIANT in short masked_short; do
  DEST=${OUTPUT_ROOT}/shard-${SHARD}/${MODEL_DIRECTORY}/${VARIANT}
  if [[ -s "${DEST}/COMPLETED" && -s "${DEST}/summary.json" ]]; then continue; fi
  [[ ! -e "${DEST}" ]] || { echo "incomplete destination exists: ${DEST}" >&2; exit 2; }
  STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/llama2-minimal.XXXXXX")
  trap 'rm -rf "${STAGE}"' EXIT
  ROW_ARG=()
  [[ -n "${ROWS:-}" ]] && ROW_ARG=(--rows "${ROWS}")
  /home/gaozh/llm_return/.venv/bin/python "${REPO_ROOT}/scripts/run_llama2_prompt_embeddings.py" "${INPUT}" --output-dir "${STAGE}/${VARIANT}" --model-path "${MODEL_PATH}" --model-name "${MODEL_NAME}" --format-version "${FORMAT_VERSION}" --variant "${VARIANT}" --prompt-spec "${PROMPT_SPEC}" --prompt-key short --separator $'\n\n' --batch-size "${BATCH_SIZE}" --max-length "${MAX_LENGTH}" "${ROW_ARG[@]}"
  mkdir -p "$(dirname "${DEST}")"
  mv "${STAGE}/${VARIANT}" "${DEST}.incoming.${SLURM_JOB_ID}"
  printf 'ok\n' > "${DEST}.incoming.${SLURM_JOB_ID}/COMPLETED"
  mv "${DEST}.incoming.${SLURM_JOB_ID}" "${DEST}"
  rm -rf "${STAGE}"
done
