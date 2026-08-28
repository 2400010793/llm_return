#!/bin/bash
# Submit corrected prompt-token extraction after current pooled jobs, without
# changing or interrupting those jobs.
set -euo pipefail
cd /home/team/llm_return
if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name prompt-token-embeddings-v5 \
    --purpose "Extract every corrected prompt token after pooled embeddings" \
    --output data/processed/prompt_token_embeddings_v5 \
    --related-file scripts/slurm_prompt_token_embeddings_v5.sbatch \
    -- bash "$0" "$@"
fi
tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}
SHORT_EMBED_JOB=${SHORT_EMBED_JOB:-3847458}
LONG_EMBED_JOB=${LONG_EMBED_JOB:-3847506}
mkdir -p logs/slurm_prompt_token_v5 data/processed/prompt_token_embeddings_v5

/home/team/llm_return/.venv/bin/python -m py_compile scripts/run_prompt_token_embeddings_v5.py
bash -n scripts/slurm_prompt_token_embeddings_v5.sbatch

short_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${SHORT_EMBED_JOB}" \
  --export=ALL,LENGTH_GROUP=short \
  scripts/slurm_prompt_token_embeddings_v5.sbatch)
long_job=$(tracked_sbatch --parsable \
  --dependency="afterok:${LONG_EMBED_JOB}" \
  --export=ALL,LENGTH_GROUP=long \
  scripts/slurm_prompt_token_embeddings_v5.sbatch)
echo "short_prompt_token_job=${short_job} dependency=${SHORT_EMBED_JOB}"
echo "long_prompt_token_job=${long_job} dependency=${LONG_EMBED_JOB}"