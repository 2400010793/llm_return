#!/bin/bash
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
SUMMARY_ROOT=${SUMMARY_ROOT:-${SINA_ROOT}/prompt_minimal_v2/cross_model_span_v3/summary}
cd "${REPO_ROOT}"
test -s "${SUMMARY_ROOT}/simple_states_manifest.tsv"
tasks=$(($(wc -l < "${SUMMARY_ROOT}/simple_states_manifest.tsv") - 1))
sbatch --parsable --array="0-$((tasks - 1))%18" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},SUMMARY_ROOT=${SUMMARY_ROOT}" \
  scripts/slurm_cross_model_simple_states.sbatch
