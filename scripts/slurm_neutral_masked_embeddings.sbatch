#!/bin/bash
# One model load per 256-way shard; all six neutral prompts are encoded together.
#SBATCH --job-name=neutral-ms-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
# BGE-M3 and RoBERTa each load one model per task; 32G is sufficient and
# leaves more healthy CPU capacity available for the 256-way arrays.
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-255%256
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/neutral_masked_short/%x_%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/neutral_masked_short/%x_%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array task required}"
: "${DATASET:?DATASET must be sina or cninfo}"
: "${MODEL:?MODEL must be roberta or bge_m3}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
INPUT_ROOT=${INPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/neutral_masked_short_v1/inputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/neutral_masked_short_v1/embeddings}
CONFIG=${CONFIG:-${REPO_ROOT}/configs/prompts/neutral_masked_short_v1.json}
SHARD=${SLURM_ARRAY_TASK_ID}
INPUT=${INPUT_ROOT}/${DATASET}/shard-${SHARD}
DEST=${OUTPUT_ROOT}/${DATASET}/${MODEL}/shard-${SHARD}

[[ -d "${INPUT}" ]] || { echo "missing input ${INPUT}" >&2; exit 2; }
if [[ -s "${DEST}/COMPLETED" && -s "${DEST}/manifest.json" ]]; then
  echo "reusing complete ${DEST}"
  exit 0
fi
[[ ! -e "${DEST}" ]] || { echo "refusing incomplete destination ${DEST}" >&2; exit 2; }

mkdir -p "$(dirname "${DEST}")" "${REPO_ROOT}/logs/neutral_masked_short"
stage_parent=${SLURM_TMPDIR:-/tmp}
stage=$(mktemp -d "${stage_parent}/neutral-ms.XXXXXX")
incoming=${DEST}.incoming.${SLURM_JOB_ID}
cleanup() { rm -rf "${stage}"; [[ -e "${DEST}" ]] || rm -rf "${incoming}"; }
trap cleanup EXIT

cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ "${MODEL}" == "bge_m3" ]]; then
  max_length=1000
  batch_size=4
else
  max_length=512
  batch_size=8
fi
.venv/bin/python scripts/run_neutral_masked_embeddings.py "${INPUT}" \
  --output-dir "${stage}/output" --model "${MODEL}" --config "${CONFIG}" \
  --batch-size "${batch_size}" --max-length "${max_length}"
mkdir -p "$(dirname "${DEST}")"
cp -a "${stage}/output" "${incoming}"
printf 'ok\n' > "${incoming}/COMPLETED"
mv "${incoming}" "${DEST}"
echo "completed dataset=${DATASET} model=${MODEL} shard=${SHARD}"
