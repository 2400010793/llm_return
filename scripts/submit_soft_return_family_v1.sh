#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/soft_return_family_v1}
MANIFEST=${FAMILY_MANIFEST:-${REPO_ROOT}/configs/generated/soft_return_family_v1.tsv}
CONCURRENCY=${FAMILY_CONCURRENCY:-18}
cd "${REPO_ROOT}"
mkdir -p logs/soft_return_family "${STUDY_ROOT}"
.venv/bin/python -m py_compile scripts/run_soft_return_family_fold.py \
  scripts/summarize_soft_return_families.py scripts/build_soft_return_family_manifest.py
bash -n scripts/slurm_soft_return_family_fold.sbatch scripts/slurm_soft_return_family_summary.sbatch
.venv/bin/python scripts/build_soft_return_family_manifest.py --output "${MANIFEST}"
fold_job=$(sbatch --parsable --array="0-53%${CONCURRENCY}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT},FAMILY_MANIFEST=${MANIFEST}" \
  scripts/slurm_soft_return_family_fold.sbatch)
summary_job=$(sbatch --parsable --dependency="afterok:${fold_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT}" \
  scripts/slurm_soft_return_family_summary.sbatch)
echo "folds=${fold_job} summary=${summary_job}"
