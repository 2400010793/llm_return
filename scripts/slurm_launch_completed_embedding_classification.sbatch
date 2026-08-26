#!/bin/bash
# Lightweight dependency target: once one model's 32 embedding shards and the
# label panel are complete, generate manifests and submit classification arrays.
# Required exports: MODEL, VARIANT_GROUP=short|long.
# Optional: PHASES joined with '+', INCLUDE_PROMPT_TOKENS.
#SBATCH --job-name=launch-embed-cls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/gaozh/llm_return/logs/slurm_pooled_cls_launcher/%j.out
#SBATCH --error=/home/gaozh/llm_return/logs/slurm_pooled_cls_launcher/%j.err

set -euo pipefail
cd /home/gaozh/llm_return
: "${MODEL:?export MODEL}"
: "${VARIANT_GROUP:?export VARIANT_GROUP=short, long, or paper_hk}"
: "${TASK_RECORD_ID:?launcher must be submitted through scripts/task_tracker.py}"
tracked_sbatch() {
  /usr/bin/python3 scripts/task_tracker.py child-submit --task-id "${TASK_RECORD_ID}" -- sbatch "$@"
}
case "${VARIANT_GROUP}" in
  short) VARIANTS="short,masked_short" ;;
  long) VARIANTS="long,masked_long" ;;
  paper_hk) VARIANTS="plain" ;;
  *) echo "invalid VARIANT_GROUP=${VARIANT_GROUP}" >&2; exit 2 ;;
esac
PHASES=${PHASES:-screening+aggregation}
INCLUDE_PROMPT_TOKENS=${INCLUDE_PROMPT_TOKENS:-0}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel.parquet}
CLASSIFICATION_CONCURRENCY=${CLASSIFICATION_CONCURRENCY:-4}
mkdir -p configs/generated logs/slurm_pooled_cls logs/slurm_pooled_cls_launcher
export PYTHONPATH=/home/gaozh/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

if [[ ! -s "${PANEL}" ]]; then
  echo "Classification panel is missing: ${PANEL}" >&2
  exit 2
fi

IFS='+' read -ra phase_list <<< "${PHASES}"
for phase in "${phase_list[@]}"; do
  phase=${phase//[[:space:]]/}
  [[ -n "${phase}" ]] || continue
  variant_tag=${VARIANTS//,/_}
  manifest="configs/generated/pooled_${MODEL}_${variant_tag}_${phase}.tsv"
  extra=()
  if [[ "${INCLUDE_PROMPT_TOKENS}" == "1" ]]; then
    extra+=(--include-prompt-tokens)
  fi
  /usr/bin/python3 scripts/build_pooled_classification_manifest.py \
    --phase "${phase}" \
    --models "${MODEL}" \
    --variants "${VARIANTS}" \
    --expected-rows 350577 \
    --output "${manifest}" \
    "${extra[@]}"
  tasks=$(($(wc -l < "${manifest}") - 1))
  if (( tasks < 1 )); then
    echo "No pending tasks in ${manifest}"
    continue
  fi
  job_id=$(tracked_sbatch --parsable \
    --array="0-$((tasks - 1))%${CLASSIFICATION_CONCURRENCY}" \
    --export="ALL,MANIFEST=${manifest},PANEL=${PANEL}" \
    scripts/slurm_pooled_embedding_classification.sbatch)
  echo "submitted model=${MODEL} variants=${VARIANTS} phase=${phase} tasks=${tasks} job=${job_id}"
done
