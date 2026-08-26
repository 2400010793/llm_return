#!/bin/bash
#SBATCH --job-name=prompt-control-embed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/lustre3/home/gaozh/llm_return/logs/prompt_mechanism/%A_%a_control_embed.out
#SBATCH --error=/mnt/lustre3/home/gaozh/llm_return/logs/prompt_mechanism/%A_%a_control_embed.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?run as array}"
REPO_ROOT=${REPO_ROOT:-/mnt/lustre3/home/gaozh/llm_return}
SINA_ROOT=${SINA_ROOT:-/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1}
STUDY_ROOT=${STUDY_ROOT:-${SINA_ROOT}/prompt_mechanism_v1}
MANIFEST=${CONTROL_EMBED_MANIFEST:-${STUDY_ROOT}/controls/embedding_manifest.tsv}
SELECTION=${CONTROL_SELECTION:-${STUDY_ROOT}/controls/selection.json}
cd "${REPO_ROOT}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r task_id spec_index shard model source_variant control_id output_variant <<< "${line%$'\r'}"
destination=${STUDY_ROOT}/controls/embeddings/shard-${shard}/${model}/${output_variant}
if [[ -s "${destination}/summary.json" && -s "${destination}/metadata.jsonl" && -s "${destination}/short_pooling.npz" ]]; then
  echo "reusing ${destination}"; exit 0
fi
case "${model}" in
  bge_m3) max_length=1000; batch_size=4 ;;
  xlm_roberta_large) max_length=512; batch_size=2 ;;
  *) max_length=512; batch_size=4 ;;
esac
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/run_prompt_position_control_embeddings.py \
  "${SINA_ROOT}/inputs/shard-${shard}" --output-dir "${destination}" \
  --control-selection "${SELECTION}" --spec-index "${spec_index}" \
  --model "${model}" --source-variant "${source_variant}" \
  --output-variant "${output_variant}" --batch-size "${batch_size}" \
  --max-length "${max_length}" --device cpu
