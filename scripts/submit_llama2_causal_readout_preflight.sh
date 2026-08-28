#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

PREFLIGHT_ROOT=${PREFLIGHT_ROOT:-/data/alpha_team2/shares/llm_return/data/interim/llama2_causal_readout_preflight_v2}
# Leave the node unconstrained by default so Slurm can place the probe on any
# available compatible GPU node.  Set GPU_NODE only when a node-specific probe
# is intentional.
GPU_NODE=${GPU_NODE:-}
GPU_ID=${GPU_ID:-0}
ROWS=${ROWS:-32}
BATCH_SIZE=${BATCH_SIZE:-1}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  SNAPSHOT_ARGS=()
  [[ -z "${SNAPSHOT_FROM_TASK}" ]] || SNAPSHOT_ARGS=(--snapshot-from-task "${SNAPSHOT_FROM_TASK}")
  exec .venv/bin/python scripts/task_tracker.py run \
    --name llama2-causal-readout-preflight-v2 \
    --purpose "Validate article-then-readout Llama-2 embeddings, mask pairing, shapes, and contextual variation before full submission" \
    --input /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/inputs \
    --input data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts \
    --input /mnt/lustre3/home/team/models/Llama-2-13b-hf \
    --output "${PREFLIGHT_ROOT}" \
    --related-file scripts/run_llama2_prompt_embeddings.py \
    --related-file scripts/audit_llama2_prompt_embeddings.py \
    --related-file scripts/slurm_llama2_minimal_prompt_embeddings.sbatch \
    --related-file scripts/slurm_audit_llama2_prompt_embeddings.sbatch \
    --related-file scripts/submit_llama2_causal_readout_preflight.sh \
    "${SNAPSHOT_ARGS[@]}" \
    --tag llama2 --tag gpu --tag causal-readout --tag preflight --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

mkdir -p logs/llama2_minimal "${PREFLIGHT_ROOT}"
sina_args=(--parsable --partition=gpu)
[[ -z "${GPU_NODE}" ]] || sina_args+=(--nodelist="${GPU_NODE}")
sina_job=$(tracked_sbatch "${sina_args[@]}" \
  --array=0 --export="ALL,DATASET=sina,ROWS=${ROWS},BATCH_SIZE=${BATCH_SIZE},GPU_ID=${GPU_ID},OUTPUT_ROOT=${PREFLIGHT_ROOT}/sina" \
  scripts/slurm_llama2_minimal_prompt_embeddings.sbatch)
cninfo_args=(--parsable --partition=gpu)
[[ -z "${GPU_NODE}" ]] || cninfo_args+=(--nodelist="${GPU_NODE}")
cninfo_job=$(tracked_sbatch "${cninfo_args[@]}" \
  --dependency="afterok:${sina_job}" --array=0 \
  --export="ALL,DATASET=cninfo,ROWS=${ROWS},BATCH_SIZE=${BATCH_SIZE},GPU_ID=${GPU_ID},OUTPUT_ROOT=${PREFLIGHT_ROOT}/cninfo" \
  scripts/slurm_llama2_minimal_prompt_embeddings.sbatch)
sina_audit=$(tracked_sbatch --parsable --dependency="afterok:${sina_job}" \
  --export="ALL,OUTPUT_ROOT=${PREFLIGHT_ROOT}/sina,AUDIT_OUTPUT=${PREFLIGHT_ROOT}/sina_audit.json,EXPECTED_SHARDS=1,EXPECTED_ROWS=${ROWS},ALLOW_SUBSET_ARG=--allow-subset" \
  scripts/slurm_audit_llama2_prompt_embeddings.sbatch)
cninfo_audit=$(tracked_sbatch --parsable --dependency="afterok:${cninfo_job}" \
  --export="ALL,OUTPUT_ROOT=${PREFLIGHT_ROOT}/cninfo,AUDIT_OUTPUT=${PREFLIGHT_ROOT}/cninfo_audit.json,EXPECTED_SHARDS=1,EXPECTED_ROWS=${ROWS},ALLOW_SUBSET_ARG=--allow-subset" \
  scripts/slurm_audit_llama2_prompt_embeddings.sbatch)

echo "sina=${sina_job} sina_audit=${sina_audit}"
echo "cninfo=${cninfo_job} cninfo_audit=${cninfo_audit}"
echo "gpu_node=${GPU_NODE} gpu_id=${GPU_ID} rows_per_dataset=${ROWS} batch_size=${BATCH_SIZE}"
