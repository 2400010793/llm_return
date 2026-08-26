#!/bin/bash
# Generate corrected, per-token Prompt embeddings from audited clean_v1 inputs.
# Array layout: 2 periods x 32 source shards x 2 embedding models.
# Each task processes short and masked_short sequentially to bound memory.
#SBATCH --job-name=cninfo-clean-prompt-tokens
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-127%16
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_cninfo_clean_prompt_tokens/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_cninfo_clean_prompt_tokens/%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?clean Prompt-token generation must use scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/processed/prompt_token_embeddings_2010_2026_clean_v1}
EXISTING_INPUT=${EXISTING_INPUT:-data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts}
INCREMENT_INPUT=${INCREMENT_INPUT:-data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017_clean_v1/prompt_v3_fixed_parts}
BATCH_SIZE=${BATCH_SIZE:-4}

period_id=$((SLURM_ARRAY_TASK_ID / 64))
within_period=$((SLURM_ARRAY_TASK_ID % 64))
source_shard=$((within_period / 2))
model_id=$((within_period % 2))
case "${period_id}" in
  0) input_root=${EXISTING_INPUT}; global_shard=${source_shard} ;;
  1) input_root=${INCREMENT_INPUT}; global_shard=$((source_shard + 32)) ;;
  *) echo "invalid period id ${period_id}" >&2; exit 2 ;;
esac
case "${model_id}" in
  0) model=roberta; max_length=512 ;;
  1) model=bge_m3; max_length=1000 ;;
  *) echo "invalid model id ${model_id}" >&2; exit 2 ;;
esac

input=${input_root}/shard-${source_shard}
[[ -d "${input}" ]] || { echo "missing clean Prompt input: ${input}" >&2; exit 2; }
mkdir -p logs/slurm_cninfo_clean_prompt_tokens "${OUTPUT_ROOT}/shard-${global_shard}/${model}"

scratch_parent=${SLURM_TMPDIR:-/tmp}
stage_root=$(mktemp -d "${scratch_parent%/}/cninfo-prompt-token-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.XXXXXX")
cleanup() {
  [[ -n "${stage_root:-}" && -d "${stage_root}" ]] && rm -rf -- "${stage_root}"
}
trap cleanup EXIT

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

publish_file() {
  local source=$1
  local destination=$2
  local temporary="${destination}.tmp.${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
  local attempt
  for attempt in 1 2 3; do
    rm -f -- "${temporary}"
    if cp --reflink=auto "${source}" "${temporary}" \
      && [[ $(stat -c '%s' "${source}") == $(stat -c '%s' "${temporary}") ]]; then
      mv -f -- "${temporary}" "${destination}"
      return 0
    fi
    rm -f -- "${temporary}"
    sleep "${attempt}"
  done
  echo "failed to publish ${source} to ${destination}" >&2
  return 1
}

for variant in short masked_short; do
  target=${OUTPUT_ROOT}/shard-${global_shard}/${model}/${variant}
  if [[ -s "${target}/COMPLETED" ]]; then
    echo "already complete: ${target}"
    continue
  fi
  stage=${stage_root}/${variant}
  mkdir -p "${stage}" "${target}"
  "${PYTHON}" scripts/run_prompt_token_embeddings_v5.py \
    "${input}" --output-dir "${stage}" --model "${model}" --variant "${variant}" \
    --batch-size "${BATCH_SIZE}" --max-length "${max_length}"

  "${PYTHON}" - "${stage}" "${target}" "${model}" "${variant}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

directory = Path(sys.argv[1])
target = Path(sys.argv[2])
model, variant = sys.argv[3:]
expected = {"roberta": (28, 768), "bge_m3": (18, 1024)}[model]
summary_path = directory / "summary.json"
summary = json.loads(summary_path.read_text())
matrix = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
assert summary["model"] == model and summary["variant"] == variant
assert matrix.shape == (summary["rows"], *expected) and matrix.dtype == np.float32
assert np.isfinite(matrix[0]).all() and np.isfinite(matrix[-1]).all()
assert np.load(directory / "prompt_input_ids.npy", mmap_mode="r").shape == (expected[0],)
summary["output_directory"] = str(target)
summary["embedding_file"] = str(target / "prompt_token_embeddings.npy")
summary["prompt_input_ids_file"] = str(target / "prompt_input_ids.npy")
summary["metadata_file"] = str(target / "metadata.jsonl")
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
PY

  for name in prompt_token_embeddings.npy prompt_input_ids.npy prompt_tokens.json metadata.jsonl summary.json; do
    publish_file "${stage}/${name}" "${target}/${name}"
  done
  marker_tmp=${target}/.COMPLETED.tmp.${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}
  printf '%s\n' "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} input=${input}" > "${marker_tmp}"
  mv -f -- "${marker_tmp}" "${target}/COMPLETED"
  echo "published ${target}"
done
