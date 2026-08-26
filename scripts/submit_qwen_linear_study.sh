#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name qwen-linear-return-2018-2026 \
    --purpose "Strict linear-only Qwen return prediction, cluster diagnostics, fair baselines, and cost-aware backtests" \
    --input data/processed/embeddings/qwen3_embedding_8b_candidates_1000 \
    --input data/processed/pooled_embeddings_qwen_candidates_1000 \
    --input data/processed/pooled_embeddings_v3 \
    --input data/processed/cninfo_full_o2o_residual_panel.parquet \
    --output reports/regression/qwen_2018_2026_linear_v1 \
    --related-file scripts/run_pooled_embedding_regression.py \
    --related-file scripts/audit_qwen_linear_study.py \
    --related-file scripts/build_qwen_linear_manifests.py \
    --related-file scripts/select_qwen_linear_candidates.py \
    --related-file scripts/run_qwen_cluster_linear.py \
    --related-file scripts/summarize_qwen_linear_study.py \
    --related-file scripts/slurm_qwen_linear_regression.sbatch \
    --related-file scripts/slurm_qwen_linear_select.sbatch \
    --snapshot-from-task 20260820T101129Z-cross-model-simple-states-v3-4041913 \
    --tag qwen --tag regression --tag linear-only --tag rolling-oos --tag simple-states --seed 42 \
    -- bash "$0" "$@"
fi

ROOT=reports/regression/qwen_2018_2026_linear_v1
SCREEN=configs/generated/qwen_linear_screen.tsv
mkdir -p "${ROOT}" configs/generated logs/slurm_qwen_linear
.venv/bin/python scripts/build_qwen_linear_manifests.py --output "${SCREEN}" --result-root "${ROOT}"
for script in scripts/slurm_qwen_linear_audit.sbatch scripts/slurm_qwen_linear_regression.sbatch \
              scripts/slurm_qwen_linear_select.sbatch scripts/slurm_qwen_cluster_linear.sbatch \
              scripts/slurm_qwen_linear_finalize.sbatch; do
  bash -n "${script}"
done
audit_id=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable scripts/slurm_qwen_linear_audit.sbatch)
tasks=$(( $(wc -l < "${SCREEN}") - 1 ))
screen_id=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable --dependency=afterok:${audit_id} --array="0-$((tasks - 1))%3" \
  --export=ALL,MANIFEST="${SCREEN}" scripts/slurm_qwen_linear_regression.sbatch)
select_id=$(.venv/bin/python scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- \
  sbatch --parsable --dependency=afterok:${screen_id} --export=ALL,SCREEN_MANIFEST="${SCREEN}" \
  scripts/slurm_qwen_linear_select.sbatch)
echo "audit=${audit_id} screen=${screen_id} tasks=${tasks} select=${select_id}"
