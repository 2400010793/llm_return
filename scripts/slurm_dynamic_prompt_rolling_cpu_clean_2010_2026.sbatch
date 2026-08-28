#!/bin/bash
# CPU-only rolling prompt-token gates: 3 gate modes x 9 test years.
#SBATCH --job-name=prompt-dynamic-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-26%6
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_dynamic_prompt_cpu_clean/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_dynamic_prompt_cpu_clean/%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/alpha_team2/shares/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?submit through scripts/task_tracker.py}"
: "${MODEL:?export MODEL=roberta or bge_m3}"
: "${SLURM_ARRAY_TASK_ID:?run as a Slurm array}"

PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}
INPUT_ROOT=${INPUT_ROOT:-data/processed/prompt_token_embeddings_2010_2026_clean_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/dynamic_prompt/rolling_cpu_clean_2010_2026}
gates=(uniform static dynamic)
gate=${gates[$((SLURM_ARRAY_TASK_ID / 9))]}
fold=$((SLURM_ARRAY_TASK_ID % 9))
test_year=$((2018 + fold))
fit_start=$((test_year - 8))
fit_end=$((test_year - 3))
validation_start=$((test_year - 2))
validation_end=$((test_year - 1))
fit_years=$(seq -s, "${fit_start}" "${fit_end}")
validation_years=$(seq -s, "${validation_start}" "${validation_end}")

case "${MODEL}" in
  roberta) token_count=28; hidden_size=768 ;;
  bge_m3) token_count=18; hidden_size=1024 ;;
  *) echo "invalid MODEL=${MODEL}" >&2; exit 2 ;;
esac

output=${OUTPUT_ROOT}/${MODEL}_masked_short_${gate}_test${test_year}_seed42.json
mkdir -p logs/slurm_dynamic_prompt_cpu_clean "${OUTPUT_ROOT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/run_dynamic_prompt_token_gating.py "${PANEL}" \
  --input-root "${INPUT_ROOT}" --model "${MODEL}" --variant masked_short \
  --task classification --gate-mode "${gate}" --run-mode final-test \
  --fit-years "${fit_years}" --validation-years "${validation_years}" \
  --test-years "${test_year}" --return-column next_day_return \
  --classification-target-column next_day_label \
  --expected-shards 64 --expected-rows 903665 \
  --expected-token-count "${token_count}" --expected-hidden-size "${hidden_size}" \
  --batch-size 256 --prediction-batch-size 512 \
  --max-epochs 50 --patience 7 --seed 42 --device cpu --output "${output}"
