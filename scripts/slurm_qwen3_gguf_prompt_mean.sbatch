#!/bin/bash
#SBATCH --job-name=qwen3-gguf-prompt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/qwen3_gguf_prompt/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/qwen3_gguf_prompt/%A_%a.err
set -euo pipefail
: "${TASK_RECORD_ID:?submit through task_tracker.py}"
: "${DATASET:?}"
: "${VARIANT:?}"
: "${PROMPT_ID:?}"
: "${PROMPT_TEXT:?}"
: "${SHARDS:?}"
: "${INPUT_ROOT:?}"
: "${OUTPUT_ROOT:?}"
MODEL=${MODEL:-/home/team/.ollama/models/blobs/sha256-3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85}
GPU_SLOTS_PER_DEVICE=${GPU_SLOTS_PER_DEVICE:-3}
GPU_CANDIDATES=${GPU_CANDIDATES:-0,2,3}
GPU_BIND_MODE=${GPU_BIND_MODE:-dynamic}
shard=${SLURM_ARRAY_TASK_ID}
(( shard < SHARDS )) || exit 2
if [[ "${DATASET}" == sina ]]; then
  input="${INPUT_ROOT}/shard-${shard}"
else
  input="${INPUT_ROOT}/part-$(printf '%05d' "${shard}").jsonl"
fi
output="${OUTPUT_ROOT}/${DATASET}/${PROMPT_ID}/${VARIANT}/shard-${shard}"
[[ -f "${output}/COMPLETED" ]] && exit 0
mkdir -p /data/alpha_team2/shares/llm_return/logs/qwen3_gguf_prompt
gpu=-1
gpu_slot=-1
# Colons are safe inside Slurm's comma-delimited --export argument. Keep
# accepting commas for existing jobs submitted before this convention.
GPU_CANDIDATES=${GPU_CANDIDATES//:/,}
IFS=',' read -r -a candidate_ids <<< "${GPU_CANDIDATES}"
# In sequence mode, workers walk the explicitly supplied per-node GPU list.
# The lock and memory check remain mandatory, so a busy external GPU is never
# claimed merely because it is next in the sequence.
if [[ "${GPU_BIND_MODE}" == sequence ]]; then
  seq_index=$(( SLURM_ARRAY_TASK_ID % ${#candidate_ids[@]} ))
  candidate_ids=("${candidate_ids[$seq_index]}" "${candidate_ids[@]:0:$seq_index}" "${candidate_ids[@]:$((seq_index + 1))}")
fi
for attempt in $(seq 1 240); do
  for candidate in "${candidate_ids[@]}"; do
    for slot in $(seq 0 $((GPU_SLOTS_PER_DEVICE - 1))); do
      exec {lock_fd}>"/tmp/team-qwen3-gpu-${candidate}-slot-${slot}.lock"
      if flock -n "${lock_fd}"; then
        free=$(nvidia-smi --id="${candidate}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)
        [[ "${free}" =~ ^[0-9]+$ ]] || free=0
        if (( free > 12000 )); then gpu=${candidate}; gpu_slot=${slot}; break 3; fi
        flock -u "${lock_fd}"; eval "exec ${lock_fd}>&-"
      fi
    done
  done
  sleep 15
done
(( gpu >= 0 )) || { echo 'no GPU slot with >12GiB free' >&2; exit 75; }
export CUDA_VISIBLE_DEVICES=${gpu}
export PYTHONPATH=/home/team/llm_return
echo "host=$(hostname) gpu=${gpu} gpu_slot=${gpu_slot} dataset=${DATASET} prompt=${PROMPT_ID} variant=${VARIANT} shard=${shard}"
cd /home/team/llm_return
.venv/bin/python scripts/run_qwen3_gguf_prompt_embeddings.py "${input}" \
  --model "${MODEL}" --prompt "${PROMPT_TEXT}" --variant "${VARIANT}" \
  --max-length 500 --output-dir "${output}"
