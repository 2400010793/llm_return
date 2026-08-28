#!/bin/bash
# One model load per shard; the Python runner emits all four prompts and both
# short variants.  The 24h limit protects the large CNINFO array from timeout.
#SBATCH --job-name=positive-prompt-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/positive_prompt_embeddings/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/positive_prompt_embeddings/%A_%a.err
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?array task required}"
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
DATASET=${DATASET:?DATASET must be sina or cninfo}
MODEL=${MODEL:?MODEL must be roberta or bge_m3}
SHARDS=${SHARDS:?SHARDS required}
INPUT_ROOT=${INPUT_ROOT:?INPUT_ROOT required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT required}
PROMPT_CONFIG=${PROMPT_CONFIG:-${REPO_ROOT}/configs/generated/positive_prompt_templates_v1.json}
shard=${SLURM_ARRAY_TASK_ID}
(( shard < SHARDS )) || exit 2
if [[ "${DATASET}" == "sina" ]]; then
  input="${INPUT_ROOT}/shard-${shard}"
else
  input="${INPUT_ROOT}/part-$(printf '%05d' "${shard}").jsonl"
fi
destination="${OUTPUT_ROOT}/${MODEL}/shard-${shard}"
if [[ -s "${destination}/COMPLETED" ]]; then exit 0; fi
[[ -e "${destination}" ]] && { echo "refusing incomplete destination: ${destination}" >&2; exit 1; }
mkdir -p "$(dirname "${destination}")" /data/alpha_team2/shares/llm_return/logs/positive_prompt_embeddings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "${REPO_ROOT}"
batch_size=4
max_length=512
if [[ "${MODEL}" == "bge_m3" ]]; then max_length=1000; batch_size=4; fi
.venv/bin/python scripts/run_prompt_template_embeddings.py "${input}" \
  --output-dir "${destination}" --model "${MODEL}" \
  --prompt-config "${PROMPT_CONFIG}" --batch-size "${batch_size}" \
  --max-length "${max_length}"