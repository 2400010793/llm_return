#!/bin/bash
#SBATCH --job-name=finbert2-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_finbert2/%A_%a_embed.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_finbert2/%A_%a_embed.err

set -euo pipefail
cd /data/alpha_team2/shares/llm_return
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${SLURM_ARRAY_TASK_ID:?run as a Slurm array}"
: "${VARIANT_GROUP:?export VARIANT_GROUP=plain or prompted}"

case "${VARIANT_GROUP}" in
  plain)
    logical_shard=${SLURM_ARRAY_TASK_ID}
    variant=plain
    ;;
  prompted)
    logical_shard=$((SLURM_ARRAY_TASK_ID / 2))
    if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then variant=short; else variant=masked_short; fi
    ;;
  *) echo "invalid VARIANT_GROUP=${VARIANT_GROUP}" >&2; exit 2 ;;
esac
if (( logical_shard < 0 || logical_shard >= 64 )); then
  echo "logical shard out of range: ${logical_shard}" >&2
  exit 2
fi
if (( logical_shard < 32 )); then
  source_root=data/processed/cleaned/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts
  source_shard=${logical_shard}
else
  source_root=data/processed/cleaned/cninfo_prompt_bundle_increment_2010_2017_clean_v1/prompt_v3_fixed_parts
  source_shard=$((logical_shard - 32))
fi

input="${source_root}/shard-${source_shard}"
output="data/processed/pooled_finbert2_embeddings_2010_2026_v1/shard-${logical_shard}/finbert2_base/${variant}"
[[ -d "${input}" ]] || { echo "missing input shard: ${input}" >&2; exit 2; }
mkdir -p logs/slurm_finbert2 "${output}"
if [[ -s "${output}/summary.json" && -s "${output}/metadata.jsonl" && -s "${output}/short_pooling.npz" ]]; then
  echo "complete output already exists: ${output}"
  exit 0
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FINBERT_MODEL_ROOT=/data/alpha_team2/shares/llm_return/models
export PYTHONPATH=/data/alpha_team2/shares/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}
extra=()
if [[ "${variant}" == plain ]]; then
  extra+=(--prompt-condition no_prompt_natural)
fi
/usr/bin/time -v /usr/bin/python3 scripts/run_short_pooled_embeddings.py \
  "${input}" --output-dir "${output}" \
  --model finbert2_base --variant "${variant}" \
  --batch-size "${BATCH_SIZE:-4}" --max-length 512 "${extra[@]}"
