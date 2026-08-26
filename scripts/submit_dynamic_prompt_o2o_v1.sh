#!/bin/bash
set -euo pipefail
cd /home/gaozh/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name dynamic-prompt-o2o-v1 \
    --purpose "Compare O2O prompt-token uniform/static/dynamic gates and hard keep-4 gates" \
    --input data/processed/cninfo_full_o2o_panel.parquet \
    --input data/processed/prompt_token_embeddings_v5 \
    --output reports/dynamic_prompt/o2o_seed42 \
    --related-file scripts/run_dynamic_prompt_token_gating.py \
    --related-file src/data/prompt_token_embeddings.py \
    --related-file src/models/dynamic_token_gating.py \
    --related-file scripts/slurm_dynamic_prompt_o2o_v1.sbatch \
    --tag o2o --tag prompt-token-gating --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p logs/slurm_prompt_o2o_gate reports/dynamic_prompt/o2o_seed42
.venv/bin/python -m py_compile \
  scripts/run_dynamic_prompt_token_gating.py \
  src/data/prompt_token_embeddings.py \
  src/models/dynamic_token_gating.py
bash -n scripts/slurm_dynamic_prompt_o2o_v1.sbatch

job_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable scripts/slurm_dynamic_prompt_o2o_v1.sbatch
)
echo "submitted dynamic_prompt_o2o=${job_id}"
