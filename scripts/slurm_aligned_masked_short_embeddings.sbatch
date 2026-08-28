#!/bin/bash
# One RoBERTa load per shard/group; emits one clean masked-short prompt group.
#SBATCH --job-name=aligned-ms-roberta
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/aligned_masked_short/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/aligned_masked_short/%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
INPUT_ROOT=${INPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/inputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/aligned_masked_short_v1}
CONFIG=${CONFIG:-${REPO_ROOT}/configs/prompts/aligned_masked_short_v1.json}
TASK_ID=${SLURM_ARRAY_TASK_ID}
SHARD=$((TASK_ID % 64))
GROUP_ID=$((TASK_ID / 64))
INPUT=${INPUT_ROOT}/shard-${SHARD}
DEST=${OUTPUT_ROOT}/roberta/shard-${SHARD}/group-${GROUP_ID}

[[ -d "${INPUT}" ]] || { echo "missing input ${INPUT}" >&2; exit 2; }
if [[ -s "${DEST}/COMPLETED" && -s "${DEST}/manifest.json" ]]; then
  echo "reusing ${DEST}"
  exit 0
fi
[[ ! -e "${DEST}" ]] || { echo "refusing incomplete destination ${DEST}" >&2; exit 2; }

mkdir -p "$(dirname "${DEST}")" /data/alpha_team2/shares/llm_return/logs/aligned_masked_short
stage_parent=${SLURM_TMPDIR:-/tmp}
stage=$(mktemp -d "${stage_parent}/aligned-ms.XXXXXX")
incoming=${DEST}.incoming.${SLURM_JOB_ID}
cleanup() {
  rm -rf "${stage}"
  [[ -e "${DEST}" ]] || rm -rf "${incoming}"
}
trap cleanup EXIT

cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
.venv/bin/python scripts/run_aligned_masked_short_embeddings.py "${INPUT}" \
  --output-dir "${stage}/output" --model roberta --config "${CONFIG}" \
  --group-id "${GROUP_ID}" --batch-size 8 --max-length 512

mkdir -p "$(dirname "${DEST}")"
cp -a "${stage}/output" "${incoming}"
printf 'ok\n' > "${incoming}/COMPLETED"
mv "${incoming}" "${DEST}"
echo "completed aligned masked-short shard=${SHARD} group=${GROUP_ID} destination=${DEST}"
