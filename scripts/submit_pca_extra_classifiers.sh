#!/bin/bash
# Submit only mathematically compatible missing PCA classifiers.
set -euo pipefail
cd /home/team/llm_return
if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name pca-extra-classifiers \
    --purpose "Run mathematically compatible missing PCA classifiers" \
    --output reports/classification \
    --related-file scripts/slurm_old_triple_pca_extra_classifiers.sbatch \
    --related-file scripts/slurm_prompt_token_pca_extra_classifiers_v5.sbatch \
    -- bash "$0" "$@"
fi
tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}
SHORT_EMBED_JOB=${SHORT_EMBED_JOB:-3847458}
LONG_EMBED_JOB=${LONG_EMBED_JOB:-3847506}
PROMPT_PCA_JOB=${PROMPT_PCA_JOB:-3847792}
mkdir -p logs/slurm_triple_pca_extra logs/slurm_prompt_pca_extra_v5

/home/team/llm_return/.venv/bin/python -m py_compile \
  scripts/run_pooled_embedding_classification.py scripts/run_reduced_prompt_classification.py
bash -n scripts/slurm_old_triple_pca_extra_classifiers.sbatch scripts/slurm_prompt_token_pca_extra_classifiers_v5.sbatch

old_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${SHORT_EMBED_JOB}:${LONG_EMBED_JOB}" \
  scripts/slurm_old_triple_pca_extra_classifiers.sbatch)
prompt_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${PROMPT_PCA_JOB}" \
  scripts/slurm_prompt_token_pca_extra_classifiers_v5.sbatch)
echo "old_triple_pca_extra_job=${old_job} dependencies=${SHORT_EMBED_JOB},${LONG_EMBED_JOB}"
echo "prompt_pca_extra_job=${prompt_job} dependency=${PROMPT_PCA_JOB}"