#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name overnight-qwen-topk-20260811 \
    --purpose "Safely run full-panel Qwen validation screens and non-duplicated O2O prompt-token top-k controls after Qwen embedding completion" \
    --input data/processed/embeddings/qwen3_embedding_8b_candidates_1000 \
    --input data/processed/prompt_token_embeddings_v5 \
    --input data/processed/cninfo_full_classification_panel.parquet \
    --input data/processed/cninfo_full_o2o_residual_panel.parquet \
    --output data/processed/pooled_embeddings_qwen_candidates_1000 \
    --output reports/classification/qwen_fullpanel_screen_20260811 \
    --output reports/regression/qwen_fullpanel_screen_20260811 \
    --output reports/dynamic_prompt/o2o_topk_overnight_20260811 \
    --related-file scripts/prepare_qwen_pooled_candidates.py \
    --related-file scripts/run_pooled_embedding_classification.py \
    --related-file scripts/run_pooled_embedding_regression.py \
    --related-file scripts/run_dynamic_prompt_token_gating.py \
    --related-file scripts/slurm_prepare_qwen_pooled_20260811.sbatch \
    --related-file scripts/slurm_qwen_classification_screen_20260811.sbatch \
    --related-file scripts/slurm_qwen_regression_screen_20260811.sbatch \
    --related-file scripts/slurm_prompt_topk_overnight_20260811.sbatch \
    --related-file configs/generated/qwen_overnight_classification_20260811.tsv \
    --related-file configs/generated/qwen_overnight_regression_20260811.tsv \
    --related-file configs/generated/prompt_topk_overnight_20260811.tsv \
    --tag qwen --tag next1 --tag o2o --tag prompt-token-topk --seed 42 \
    -- bash "$0" "$@"
fi

QWEN_JOB=3995435
EXISTING_GATE_JOB=4001529
CLASSIFICATION_MANIFEST=configs/generated/qwen_overnight_classification_20260811.tsv
REGRESSION_MANIFEST=configs/generated/qwen_overnight_regression_20260811.tsv
PROMPT_MANIFEST=configs/generated/prompt_topk_overnight_20260811.tsv

mkdir -p logs/slurm_qwen_overnight_20260811 logs/slurm_prompt_topk_20260811 \
  reports/classification/qwen_fullpanel_screen_20260811 \
  reports/regression/qwen_fullpanel_screen_20260811 \
  reports/dynamic_prompt/o2o_topk_overnight_20260811

.venv/bin/python -m py_compile \
  scripts/prepare_qwen_pooled_candidates.py \
  scripts/run_pooled_embedding_classification.py \
  scripts/run_pooled_embedding_regression.py \
  scripts/run_dynamic_prompt_token_gating.py
for script in \
  scripts/slurm_prepare_qwen_pooled_20260811.sbatch \
  scripts/slurm_qwen_classification_screen_20260811.sbatch \
  scripts/slurm_qwen_regression_screen_20260811.sbatch \
  scripts/slurm_prompt_topk_overnight_20260811.sbatch; do
  bash -n "${script}"
done

# The existing array hard-codes GPU 1. Add a dependency rather than cancelling
# it, so it cannot collide with the Qwen/Ollama workers currently using the
# same physical node and possible GPU.
existing_state=$(squeue -h -j "${EXISTING_GATE_JOB}" -o '%T' | head -n 1 || true)
if [[ "${existing_state}" == "PENDING" ]]; then
  scontrol update JobId="${EXISTING_GATE_JOB}" Dependency=afterok:${QWEN_JOB}
  echo "protected existing prompt job ${EXISTING_GATE_JOB} with afterok:${QWEN_JOB}"
elif [[ -n "${existing_state}" ]]; then
  echo "existing prompt job ${EXISTING_GATE_JOB} is ${existing_state}; left unchanged"
else
  echo "existing prompt job ${EXISTING_GATE_JOB} is no longer queued; left unchanged"
fi

classification_tasks=$(( $(wc -l < "${CLASSIFICATION_MANIFEST}") - 1 ))
regression_tasks=$(( $(wc -l < "${REGRESSION_MANIFEST}") - 1 ))
prompt_tasks=$(( $(wc -l < "${PROMPT_MANIFEST}") - 1 ))

prepare_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency=afterok:${QWEN_JOB} \
      scripts/slurm_prepare_qwen_pooled_20260811.sbatch
)
classification_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency=afterok:${prepare_id} \
      --array="0-$((classification_tasks - 1))%3" \
      --export=ALL,MANIFEST="${CLASSIFICATION_MANIFEST}" \
      scripts/slurm_qwen_classification_screen_20260811.sbatch
)
regression_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency=afterok:${prepare_id} \
      --array="0-$((regression_tasks - 1))%3" \
      --export=ALL,MANIFEST="${REGRESSION_MANIFEST}" \
      scripts/slurm_qwen_regression_screen_20260811.sbatch
)
prompt_id=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --dependency=afterok:${QWEN_JOB} \
      --array="0-$((prompt_tasks - 1))%2" \
      --export=ALL,MANIFEST="${PROMPT_MANIFEST}" \
      scripts/slurm_prompt_topk_overnight_20260811.sbatch
)

echo "submitted qwen_prepare=${prepare_id} dependency=afterok:${QWEN_JOB}"
echo "submitted qwen_classification=${classification_id} tasks=${classification_tasks} throttle=3 dependency=afterok:${prepare_id}"
echo "submitted qwen_regression=${regression_id} tasks=${regression_tasks} throttle=3 dependency=afterok:${prepare_id}"
echo "submitted prompt_topk=${prompt_id} tasks=${prompt_tasks} throttle=2 dependency=afterok:${QWEN_JOB}"
