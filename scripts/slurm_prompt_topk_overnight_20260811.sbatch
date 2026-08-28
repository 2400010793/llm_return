#!/bin/bash
#SBATCH --job-name=prompt-topk-night
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_prompt_topk_20260811/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_prompt_topk_20260811/%A_%a.err

set -euo pipefail
REPO_ROOT=/data/alpha_team2/shares/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MANIFEST:?export the prompt top-k manifest}"
: "${SLURM_ARRAY_TASK_ID:?run as an array job}"

# GPU 1 is reserved for the already-submitted O2O gate array. These locks are
# intentionally disjoint from it and prevent new tasks colliding with each
# other on a node where Slurm does not expose GPU GRES.
GPU_ID=
for candidate_gpu in 2 4 5 6; do
  exec {lock_fd}>"/tmp/llm_return_prompt_topk_gpu_${candidate_gpu}.lock"
  if flock -n "${lock_fd}"; then
    GPU_ID=${candidate_gpu}
    GPU_LOCK_FD=${lock_fd}
    break
  fi
  eval "exec ${lock_fd}>&-"
done
if [[ -z "${GPU_ID}" ]]; then
  echo "No configured prompt top-k GPU lock is available" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p logs/slurm_prompt_topk_20260811 \
  reports/dynamic_prompt/o2o_topk_overnight_20260811

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id task gate_method gate_keep seed output <<< "${line}"
if [[ -z "${line}" || "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Manifest mismatch for array task ${SLURM_ARRAY_TASK_ID}: ${line}" >&2
  exit 2
fi

"${REPO_ROOT}/.venv/bin/python" scripts/run_dynamic_prompt_token_gating.py \
  data/processed/cninfo_full_o2o_panel.parquet \
  --input-root data/processed/prompt_token_embeddings_v5 \
  --model roberta --variant masked_short --task "${task}" \
  --gate-mode uniform --hard-gate-method "${gate_method}" \
  --hard-gate-keep "${gate_keep}" --run-mode screen \
  --return-column next_day_open_to_open_return \
  --classification-target-column next_day_open_to_open_label \
  --batch-size 256 --prediction-batch-size 512 \
  --max-epochs 50 --patience 7 --seed "${seed}" --device cuda \
  --output "${output}"
