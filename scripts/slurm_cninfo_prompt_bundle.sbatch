#!/bin/bash
# Build the historical CNINFO prompt/mask inputs after text cleaning.
# No embeddings are generated here. The output is isolated from the existing
# 2018--2026 prompt assets and is audited against the frozen v3/v4 builder.
#SBATCH --job-name=cninfo-prompt-bundle
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_prompt_bundle/%j.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_cninfo_prompt_bundle/%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?CNINFO prompt construction must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
INPUT=${INPUT:-data/processed/cleaned/cninfo_announcements_2010_2026_final.jsonl}
BUNDLE_DIR=${BUNDLE_DIR:-data/processed/cleaned/cninfo_prompt_bundle_2010_2026}
SHORT_DIR=${SHORT_DIR:-${BUNDLE_DIR}/prompt_v3_fixed_parts}
LONG_DIR=${LONG_DIR:-${BUNDLE_DIR}/prompt_v4_fixed_long_parts}
MANIFEST=${MANIFEST:-${BUNDLE_DIR}/manifest.json}
SUMMARY=${SUMMARY:-reports/cninfo_prompt_bundle_2010_2026_audit.json}
PART_SIZE=${PART_SIZE:-5000}
NUM_SHARDS=${NUM_SHARDS:-32}

mkdir -p logs/slurm_cninfo_prompt_bundle "${BUNDLE_DIR}" "$(dirname "${SUMMARY}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" -m py_compile \
  scripts/build_cninfo_prompt_bundle.py \
  scripts/audit_cninfo_prompt_bundle.py

"${PYTHON}" scripts/build_cninfo_prompt_bundle.py \
  --input "${INPUT}" \
  --short-output-dir "${SHORT_DIR}" \
  --long-output-dir "${LONG_DIR}" \
  --manifest "${MANIFEST}" \
  --part-size "${PART_SIZE}" \
  --num-shards "${NUM_SHARDS}"

"${PYTHON}" scripts/audit_cninfo_prompt_bundle.py \
  --input "${INPUT}" \
  --short-dir "${SHORT_DIR}" \
  --long-dir "${LONG_DIR}" \
  --manifest "${MANIFEST}" \
  --summary "${SUMMARY}" \
  --num-shards "${NUM_SHARDS}"

echo "CNINFO prompt/mask bundle audited: ${MANIFEST}"