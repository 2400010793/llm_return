#!/bin/bash
set -euo pipefail
cd /data/alpha_team2/shares/llm_return

# Leave the node unconstrained by default.  The 10-GPU nodes are shared and
# pinning to c019 can strand the job when its dependency chain is dead or the
# node is occupied.  Set GPU_NODE explicitly only for a verified node-specific
# probe or a controlled rerun.
GPU_NODE=${GPU_NODE:-}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7,8,9}
BATCH_SIZE=${BATCH_SIZE:-2}
EXCLUSIVE_USER=${EXCLUSIVE_USER:-1}
SINA_OUTPUT=${SINA_OUTPUT:-/data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/prompt_minimal_v2/llama2_embeddings}
CNINFO_OUTPUT=${CNINFO_OUTPUT:-/data/alpha_team2/shares/llm_return/data/processed/cninfo_minimal_prompt_v2/llama2_embeddings}
AUDIT_ROOT=${AUDIT_ROOT:-/data/alpha_team2/shares/llm_return/reports/llama2_causal_readout_v2}
SNAPSHOT_FROM_TASK=${SNAPSHOT_FROM_TASK:-}

if [[ -z "${TASK_RECORD_ID:-}" ]]; then
  SNAPSHOT_ARGS=()
  [[ -z "${SNAPSHOT_FROM_TASK}" ]] || SNAPSHOT_ARGS=(--snapshot-from-task "${SNAPSHOT_FROM_TASK}")
  exec .venv/bin/python scripts/task_tracker.py run \
    --name llama2-causal-readout-full-v2 \
    --purpose "Generate audited Llama-2 article-then-readout embeddings for Sina and CNINFO with one serial worker per physical GPU" \
    --input /data/alpha_team2/shares/llm_return/datasets/sina_cninfo_full_2010_2026/single_stock_cninfo_v1/inputs \
    --input data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts \
    --input data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017_clean_v1/prompt_v3_fixed_parts \
    --input /mnt/lustre3/home/team/models/Llama-2-13b-hf \
    --output "${SINA_OUTPUT}" \
    --output "${CNINFO_OUTPUT}" \
    --output "${AUDIT_ROOT}" \
    --related-file scripts/run_llama2_prompt_embeddings.py \
    --related-file scripts/audit_llama2_prompt_embeddings.py \
    --related-file scripts/slurm_llama2_minimal_prompt_embeddings.sbatch \
    --related-file scripts/slurm_audit_llama2_prompt_embeddings.sbatch \
    --related-file scripts/submit_llama2_causal_readout_full.sh \
    "${SNAPSHOT_ARGS[@]}" \
    --tag llama2 --tag gpu --tag causal-readout --tag full --tag memory-safe --seed 42 \
    -- bash "$0" "$@"
fi

tracked_sbatch() {
  .venv/bin/python scripts/task_tracker.py child-submit \
    --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}

IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
worker_count=${#gpu_ids[@]}
if (( worker_count < 1 )); then
  echo "GPU_IDS must contain at least one device" >&2
  exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
    echo "invalid GPU id: ${gpu_id}" >&2
    exit 2
  }
done
mkdir -p logs/llama2_minimal "${SINA_OUTPUT}" "${CNINFO_OUTPUT}" "${AUDIT_ROOT}"

submit_workers() {
  local dataset=$1
  local output_root=$2
  local dependency=${3:-}
  local slot gpu_id array_spec job_id
  local -a jobs=()
  for ((slot=0; slot<worker_count; slot++)); do
    gpu_id=${gpu_ids[${slot}]}
    array_spec="${slot}-63:${worker_count}%1"
    args=(
      --parsable --partition=gpu
      --array="${array_spec}"
      --export="ALL,DATASET=${dataset},GPU_ID=${gpu_id},BATCH_SIZE=${BATCH_SIZE},OUTPUT_ROOT=${output_root}"
    )
    [[ -z "${GPU_NODE}" ]] || args+=(--nodelist="${GPU_NODE}")
    if [[ "${EXCLUSIVE_USER}" == 1 ]]; then
      args+=(--exclusive=user)
    fi
    [[ -z "${dependency}" ]] || args+=(--dependency="afterok:${dependency}")
    job_id=$(tracked_sbatch "${args[@]}" scripts/slurm_llama2_minimal_prompt_embeddings.sbatch)
    jobs+=("${job_id}")
  done
  local joined
  joined=$(IFS=:; echo "${jobs[*]}")
  echo "${joined}"
}

sina_jobs=$(submit_workers sina "${SINA_OUTPUT}")
cninfo_jobs=$(submit_workers cninfo "${CNINFO_OUTPUT}" "${sina_jobs}")
sina_audit=$(tracked_sbatch --parsable --dependency="afterok:${sina_jobs}" \
  --export="ALL,OUTPUT_ROOT=${SINA_OUTPUT},AUDIT_OUTPUT=${AUDIT_ROOT}/sina_audit.json,EXPECTED_SHARDS=64,EXPECTED_ROWS=75894" \
  scripts/slurm_audit_llama2_prompt_embeddings.sbatch)
cninfo_audit=$(tracked_sbatch --parsable --dependency="afterok:${cninfo_jobs}" \
  --export="ALL,OUTPUT_ROOT=${CNINFO_OUTPUT},AUDIT_OUTPUT=${AUDIT_ROOT}/cninfo_audit.json,EXPECTED_SHARDS=64,EXPECTED_ROWS=903665" \
  scripts/slurm_audit_llama2_prompt_embeddings.sbatch)

echo "sina_workers=${sina_jobs}"
echo "sina_audit=${sina_audit}"
echo "cninfo_workers=${cninfo_jobs}"
echo "cninfo_audit=${cninfo_audit}"
echo "node=${GPU_NODE} gpu_ids=${GPU_IDS} batch_size=${BATCH_SIZE} exclusive=user"
