#!/bin/bash
#SBATCH --job-name=minimal-prompt-v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/minimal_prompt_v2/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/minimal_prompt_v2/%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as an array}"
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
DATASET=${DATASET:-sina}
CNINFO_EXISTING_INPUT=${CNINFO_EXISTING_INPUT:-${REPO_ROOT}/data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts}
CNINFO_INCREMENT_INPUT=${CNINFO_INCREMENT_INPUT:-${REPO_ROOT}/data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017_clean_v1/prompt_v3_fixed_parts}
if [[ "${DATASET}" == "cninfo" ]]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/data/processed/cninfo_minimal_prompt_v2/embeddings}
else
  OUTPUT_ROOT=${OUTPUT_ROOT:-${SINA_ROOT}/prompt_minimal_v2/embeddings}
fi
PROMPT_SPEC=${PROMPT_SPEC:-${REPO_ROOT}/configs/prompts/minimal_v2.json}

models=(roberta bge_m3)
case "${DATASET}" in
  sina)
    variants=(short masked_short long masked_long)
    shard=$((SLURM_ARRAY_TASK_ID % 64))
    variant_index=$(((SLURM_ARRAY_TASK_ID / 64) % 4))
    model_index=$((SLURM_ARRAY_TASK_ID / 256))
    input=${SINA_ROOT}/inputs/shard-${shard}
    ;;
  cninfo)
    variants=(short masked_short)
    if [[ "${SPLIT_VARIANTS:-0}" == "1" ]]; then
      # One BGE variant per task.  The previous layout ran both variants in
      # one task and could exceed the eight-hour walltime.
      shard=$((SLURM_ARRAY_TASK_ID / 2))
      variant_index=$((SLURM_ARRAY_TASK_ID % 2))
      model_index=1
    else
      shard=$((SLURM_ARRAY_TASK_ID % 64))
      variant_index=$(((SLURM_ARRAY_TASK_ID / 64) % 2))
      model_index=$((SLURM_ARRAY_TASK_ID / 128))
    fi
    if (( shard < 32 )); then
      input=${CNINFO_EXISTING_INPUT}/shard-${shard}
    else
      input=${CNINFO_INCREMENT_INPUT}/shard-$((shard - 32))
    fi
    ;;
  *) echo "unsupported DATASET=${DATASET}" >&2; exit 2 ;;
esac
model=${models[${model_index}]}
variant=${variants[${variant_index}]}
prompt_key=short
[[ "${variant}" == *long ]] && prompt_key=long

[[ -d "${input}" ]] || { echo "missing input shard: ${input}" >&2; exit 2; }
destination=${OUTPUT_ROOT}/shard-${shard}/${model}/${variant}
if [[ -s "${destination}/COMPLETED" && -s "${destination}/summary.json" ]]; then
  echo "reusing ${destination}"
  exit 0
fi
if [[ -e "${destination}" ]]; then
  echo "refusing to overwrite incomplete destination: ${destination}" >&2
  exit 2
fi

scratch_parent=${SLURM_TMPDIR:-/tmp}
stage=$(mktemp -d "${scratch_parent}/minimal-prompt-v2.XXXXXX")
incoming=${destination}.incoming.${SLURM_JOB_ID}
cleanup() {
  rm -rf "${stage}"
  if [[ ! -e "${destination}" ]]; then
    rm -rf "${incoming}"
  fi
  return 0
}
trap cleanup EXIT

case "${model}" in
  roberta) batch_size=8; max_length=512 ;;
  bge_m3) batch_size=4; max_length=1000 ;;
esac
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/run_prompt_token_embeddings_v5.py "${input}" \
  --output-dir "${stage}/output" --model "${model}" --variant "${variant}" \
  --prompt-spec "${PROMPT_SPEC}" --prompt-key "${prompt_key}" \
  --batch-size "${batch_size}" --max-length "${max_length}"

mkdir -p "$(dirname "${destination}")"
cp -a "${stage}/output" "${incoming}"
printf 'ok\n' > "${incoming}/COMPLETED"
mv "${incoming}" "${destination}"
echo "completed model=${model} variant=${variant} shard=${shard} destination=${destination}"
exit 0
