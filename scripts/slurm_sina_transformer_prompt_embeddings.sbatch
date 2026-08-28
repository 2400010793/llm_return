#!/bin/bash
# Encode the five Sina prompt/mask variants with existing local Transformer models.
# Array mapping: 2 models x 5 variants. Qwen is intentionally excluded.
#SBATCH --job-name=sina-transformer-prompt
#SBATCH --partition=gpu
#SBATCH --nodelist=c018-10gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-9%1
#SBATCH --output=/data/alpha_team2/shares/llm_return/logs/slurm_sina_transformer_prompt/%A_%a.out
#SBATCH --error=/data/alpha_team2/shares/llm_return/logs/slurm_sina_transformer_prompt/%A_%a.err

set -euo pipefail

REPO_ROOT=/data/alpha_team2/shares/llm_return
DATA_ROOT=/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812
INPUT=${DATA_ROOT}/prompts/sina_single_stock_prompt_variants.parquet
RELEASE_ROOT=${DATA_ROOT}/embeddings/single_stock
STAGE=/tmp/sina-transformer-prompt-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}

cd "${REPO_ROOT}"
mkdir -p logs/slurm_sina_transformer_prompt

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=${HF_HOME:-/home/team/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

models=(chinese_roberta bge_m3)
variants=(text_plain text_prompt_short text_prompt_long text_masked_short text_masked_long)
names=(plain prompt_short prompt_long masked_short masked_long)

model_index=$((SLURM_ARRAY_TASK_ID / 5))
variant_index=$((SLURM_ARRAY_TASK_ID % 5))
model=${models[$model_index]}
variant=${variants[$variant_index]}
name=${names[$variant_index]}

case "${model}" in
  chinese_roberta)
    model_flag=--roberta
    model_arg=--roberta-model
    model_path=/home/team/llm_return/models/chinese-roberta-wwm-ext
    matrix_name=chinese_roberta.npy
    expected_dimensions=768
    ;;
  bge_m3)
    model_flag=--bge-m3
    model_arg=--bge-m3-model
    model_path=/home/team/llm_return/models/bge-m3
    matrix_name=bge_m3.npy
    expected_dimensions=1024
    ;;
  *)
    echo "unsupported model: ${model}" >&2
    exit 1
    ;;
esac

OUTPUT=${STAGE}/${model}/${name}
DEST=${RELEASE_ROOT}/${model}_prompt/${name}
INCOMING=${RELEASE_ROOT}/.${model}_prompt.${name}.${SLURM_JOB_ID}.release

if [[ -e "${DEST}" || -e "${INCOMING}" ]]; then
  echo "refusing to overwrite existing embedding directory: ${DEST}" >&2
  exit 1
fi

cleanup() {
  rm -rf "${STAGE}"
}
trap cleanup EXIT

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "model=${model} variant=${variant} output=${DEST}"

.venv/bin/python scripts/run_text_representations.py "${INPUT}" \
  --output-dir "${OUTPUT}" \
  --text-column "${variant}" \
  "${model_flag}" "${model_arg}" "${model_path}" \
  --transformer-batch-size 16 \
  --max-length 512 \
  --max-input-chars 12000

.venv/bin/python scripts/audit_embedding_matrix.py \
  --input "${INPUT}" \
  --text-column "${variant}" \
  --matrix "${OUTPUT}/${matrix_name}" \
  --metadata "${OUTPUT}/metadata.parquet" \
  --output "${OUTPUT}/audit.json" \
  --expected-dimensions "${expected_dimensions}" \
  --max-input-chars 12000 \
  --allow-duplicate-vectors

mkdir -p "${RELEASE_ROOT}/${model}_prompt"
cp -a "${OUTPUT}" "${INCOMING}"
mv "${INCOMING}" "${DEST}"
echo "promoted=${DEST}"