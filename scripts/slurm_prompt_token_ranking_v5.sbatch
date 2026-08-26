#!/bin/bash
#SBATCH --job-name=prompt-rank
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_prompt_token_ranking_v5/%A_%a.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/slurm_prompt_token_ranking_v5/%A_%a.err

set -euo pipefail
REPO_ROOT=/mnt/lustre3/home/gaozh/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?prompt-token ranking must be submitted through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST to a prompt-token ranking TSV}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
mkdir -p logs/slurm_prompt_token_ranking_v5 reports/prompt_token_rankings_v5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id model variant target method keep_tokens output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || {
  echo "manifest task mismatch: ${task_id} != ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
}

/usr/bin/python3 scripts/rank_prompt_tokens_v5.py \
  "${PANEL}" \
  --input-root data/processed/prompt_token_embeddings_v5 \
  --model "${model}" --variant "${variant}" \
  --target-column "${target}" --method "${method}" \
  --keep-tokens "${keep_tokens}" --batch-size 512 --expected-shards 32 \
  --output "${output}"
