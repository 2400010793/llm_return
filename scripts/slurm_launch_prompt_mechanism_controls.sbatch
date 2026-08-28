#!/bin/bash
#SBATCH --job-name=prompt-control-launch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%j_control_launch.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/prompt_mechanism/%j_control_launch.err
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
SINA_ROOT=${SINA_ROOT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
cd "${REPO_ROOT}"
mkdir -p "${STUDY_ROOT}/controls" logs/prompt_mechanism
selection=${STUDY_ROOT}/controls/selection.json
embed_manifest=${STUDY_ROOT}/controls/embedding_manifest.tsv
class_manifest=${STUDY_ROOT}/controls/classification_manifest.tsv
.venv/bin/python scripts/select_prompt_mechanism_controls.py \
  --summary-root "${STUDY_ROOT}/summary" --cache-root "${STUDY_ROOT}/cache" \
  --output "${selection}" --manifest "${embed_manifest}" \
  --classification-manifest "${class_manifest}" --shards 64
embed_tasks=$(($(wc -l < "${embed_manifest}") - 1))
class_tasks=$(($(wc -l < "${class_manifest}") - 1))
common="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT},CONTROL_SELECTION=${selection}"
embed_job=$(sbatch --parsable --array="0-$((embed_tasks - 1))%64" \
  --export="${common},CONTROL_EMBED_MANIFEST=${embed_manifest}" \
  scripts/slurm_prompt_mechanism_control_embeddings.sbatch)
class_job=$(sbatch --parsable --dependency="afterok:${embed_job}" \
  --array="0-$((class_tasks - 1))%32" \
  --export="${common},CONTROL_CLASS_MANIFEST=${class_manifest}" \
  scripts/slurm_prompt_mechanism_control_classification.sbatch)
summary_job=$(sbatch --parsable --dependency="afterok:${class_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},SINA_ROOT=${SINA_ROOT},STUDY_ROOT=${STUDY_ROOT}" \
  scripts/slurm_prompt_mechanism_control_summary.sbatch)
echo "control_embeddings=${embed_job} tasks=${embed_tasks} classification=${class_job} tasks=${class_tasks} summary=${summary_job}"
