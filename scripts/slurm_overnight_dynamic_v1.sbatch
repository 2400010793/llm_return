#!/bin/bash
#SBATCH --job-name=overnight-dynamic-v1
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/team/llm_return/logs/slurm_overnight_v1/dynamic_%A_%a.out
#SBATCH --error=/home/team/llm_return/logs/slurm_overnight_v1/dynamic_%A_%a.err

set -euo pipefail
REPO_ROOT=/home/team/llm_return
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MANIFEST:?export MANIFEST}"
: "${CUDA_DEVICE:?export CUDA_DEVICE}"
: "${SLURM_ARRAY_TASK_ID:?run as an array}"

mkdir -p logs/slurm_overnight_v1 reports/dynamic_prompt/overnight_v1
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
line=${line%$'\r'}
IFS=$'\t' read -r task_id model variant token_count hidden_size window config_tag gate_mode representation_size gate_hidden_size dropout learning_rate weight_decay fit_years validation_years seed output <<< "${line}"
[[ "${task_id}" == "${SLURM_ARRAY_TASK_ID}" ]] || { echo "manifest task mismatch" >&2; exit 2; }

"${REPO_ROOT}/.venv/bin/python" scripts/run_dynamic_prompt_token_gating.py \
  data/processed/cninfo_full_classification_panel.parquet \
  --input-root data/processed/prompt_token_embeddings_v5 \
  --model "${model}" --variant "${variant}" \
  --expected-token-count "${token_count}" --expected-hidden-size "${hidden_size}" \
  --task regression --gate-mode "${gate_mode}" --run-mode screen \
  --representation-size "${representation_size}" --gate-hidden-size "${gate_hidden_size}" \
  --dropout "${dropout}" --learning-rate "${learning_rate}" --weight-decay "${weight_decay}" \
  --fit-years "${fit_years}" --validation-years "${validation_years}" \
  --test-years 2026 --batch-size 256 --prediction-batch-size 512 \
  --max-epochs 50 --patience 7 --seed "${seed}" --device cuda --output "${output}"

