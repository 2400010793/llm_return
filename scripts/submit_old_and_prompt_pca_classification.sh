#!/bin/bash
# Attach additional old triple-concat classifiers and v5 prompt-token PCA jobs.
set -euo pipefail
cd /home/team/llm_return
if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name old-and-prompt-pca-classification \
    --purpose "Run old triple-concat classifiers and corrected prompt-token PCA chain" \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --output reports/classification \
    --related-file scripts/slurm_prompt_token_pca_v5.sbatch \
    --related-file scripts/slurm_prompt_token_pca_classification_v5.sbatch \
    -- bash "$0" "$@"
fi
tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}
SHORT_EMBED_JOB=${SHORT_EMBED_JOB:-3847458}
LONG_EMBED_JOB=${LONG_EMBED_JOB:-3847506}
PROMPT_V5_AUDIT_JOB=${PROMPT_V5_AUDIT_JOB:-3847787}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
mkdir -p logs/slurm_prompt_pca_v5 logs/slurm_prompt_pca_cls_v5

/home/team/llm_return/.venv/bin/python -m py_compile \
  scripts/run_prompt_token_pca_v5.py scripts/run_reduced_prompt_classification.py
bash -n scripts/slurm_prompt_token_pca_v5.sbatch scripts/slurm_prompt_token_pca_classification_v5.sbatch

# Old title/body/full concatenation: add SVM, SGD and MLP; Logistic was already attached.
PHASES=triple_models PANEL="${PANEL}" bash scripts/submit_classification_after_embeddings.sh

# Corrected v5 prompt tokens: PCA array after audit, then four classifiers.
pca_job=$(tracked_sbatch --parsable --dependency="afterok:${PROMPT_V5_AUDIT_JOB}" scripts/slurm_prompt_token_pca_v5.sbatch)
classification_job=$(tracked_sbatch --parsable --dependency="afterok:${pca_job}" scripts/slurm_prompt_token_pca_classification_v5.sbatch)
echo "prompt_pca_job=${pca_job} dependency=${PROMPT_V5_AUDIT_JOB}"
echo "prompt_pca_classification_job=${classification_job} dependency=${pca_job}"