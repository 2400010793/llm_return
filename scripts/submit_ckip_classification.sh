#!/bin/bash
set -euo pipefail
cd /mnt/lustre3/home/gaozh/llm_return

# CKIP-BERT is an archived negative baseline. Prevent accidental future compute
# while keeping an explicit reproduction path for audits.
if [[ "${ALLOW_RETIRED_CKIP:-0}" != "1" ]]; then
  echo "CKIP-BERT is retired after the 2026 OOS audit; set ALLOW_RETIRED_CKIP=1 only to reproduce the archived baseline." >&2
  exit 2
fi

PAPER_MANIFEST=${PAPER_MANIFEST:-configs/generated/pooled_ckip_bert_plain_paper_hk.tsv}
O2O_MANIFEST=${O2O_MANIFEST:-configs/generated/pooled_ckip_bert_plain_paper_hk_o2o.tsv}
CLASSIFICATION_PANEL=${CLASSIFICATION_PANEL:-data/processed/cninfo_full_classification_panel.parquet}
O2O_PANEL=${O2O_PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
MAX_PARALLEL=${MAX_PARALLEL:-4}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  exec .venv/bin/python scripts/task_tracker.py run \
    --name ckip-classification-relaunch \
    --purpose "Run feasible CKIP-BERT paper-label and O2O classification cells after complete embeddings" \
    --input "${CLASSIFICATION_PANEL}" \
    --input "${O2O_PANEL}" \
    --input data/processed/pooled_paper_hk_embeddings_v1 \
    --output reports/classification/pooled_embeddings/paper_hk \
    --output reports/classification/pooled_embeddings/paper_hk_o2o \
    --related-file scripts/build_pooled_classification_manifest.py \
    --related-file scripts/build_o2o_classification_manifest.py \
    --related-file scripts/run_pooled_embedding_classification.py \
    --related-file scripts/slurm_pooled_embedding_classification.sbatch \
    --related-file scripts/slurm_o2o_classification.sbatch \
    --seed 42 \
    -- bash "$0" "$@"
fi

mkdir -p configs/generated logs/slurm_pooled_cls logs/slurm_o2o_classification
test -s "${CLASSIFICATION_PANEL}"
test -s "${O2O_PANEL}"

.venv/bin/python scripts/build_pooled_classification_manifest.py \
  --phase paper_hk \
  --models ckip_bert \
  --variants plain \
  --expected-rows 350577 \
  --output "${PAPER_MANIFEST}"

.venv/bin/python scripts/build_o2o_classification_manifest.py \
  --embedding-root data/processed/pooled_paper_hk_embeddings_v1 \
  --models ckip_bert \
  --variants plain \
  --features full_mean \
  --classifiers logistic,mlp \
  --reducers none,pca:128 \
  --phase paper_hk_o2o \
  --output-root reports/classification/pooled_embeddings/paper_hk_o2o \
  --output "${O2O_MANIFEST}"

.venv/bin/python -m py_compile \
  scripts/build_pooled_classification_manifest.py \
  scripts/build_o2o_classification_manifest.py \
  scripts/run_pooled_embedding_classification.py
bash -n \
  scripts/slurm_pooled_embedding_classification.sbatch \
  scripts/slurm_o2o_classification.sbatch \
  "$0"

paper_tasks=$(( $(wc -l < "${PAPER_MANIFEST}") - 1 ))
o2o_tasks=$(( $(wc -l < "${O2O_MANIFEST}") - 1 ))
if [[ "${paper_tasks}" -ne 4 || "${o2o_tasks}" -ne 4 ]]; then
  echo "Expected 4 paper and 4 O2O tasks, found paper=${paper_tasks} O2O=${o2o_tasks}" >&2
  exit 2
fi

paper_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((paper_tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,MANIFEST=${PAPER_MANIFEST},PANEL=${CLASSIFICATION_PANEL}" \
      scripts/slurm_pooled_embedding_classification.sbatch
)
o2o_job=$(
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- \
    sbatch --parsable --array="0-$((o2o_tasks - 1))%${MAX_PARALLEL}" \
      --export="ALL,MANIFEST=${O2O_MANIFEST},PANEL=${O2O_PANEL},FILTER_COLUMN=o2o_target_eligible" \
      scripts/slurm_o2o_classification.sbatch
)

echo "submitted CKIP classification paper_job=${paper_job} tasks=${paper_tasks} o2o_job=${o2o_job} tasks=${o2o_tasks} max_parallel=${MAX_PARALLEL}"
