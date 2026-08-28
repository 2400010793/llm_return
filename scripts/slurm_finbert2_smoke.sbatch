#!/bin/bash
#SBATCH --job-name=finbert2-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_finbert2/%j_smoke.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_finbert2/%j_smoke.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
input=data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts/shard-0
root=reports/finbert2/smoke
mkdir -p logs/slurm_finbert2 "${root}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FINBERT_MODEL_ROOT=/data/alpha_team2/shares/llm_return/models
export PYTHONPATH=/data/alpha_team2/shares/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

for variant in plain short masked_short; do
  extra=()
  if [[ "${variant}" == plain ]]; then
    extra+=(--prompt-condition no_prompt_natural)
  fi
  /usr/bin/time -v /usr/bin/python3 scripts/run_short_pooled_embeddings.py \
    "${input}" --output-dir "${root}/${variant}" \
    --model finbert2_base --variant "${variant}" --rows 100 \
    --batch-size 4 --max-length 512 "${extra[@]}"
done
