#!/bin/bash
#SBATCH --job-name=launch-embed-reg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/team/llm_return/logs/slurm_pooled_regression_launcher/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_pooled_regression_launcher/%j.err

set -euo pipefail
cd /home/team/llm_return
: "${MODEL:?export MODEL}"
: "${VARIANT_GROUP:?export VARIANT_GROUP}"
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
PANEL=${PANEL:-data/processed/cninfo_full_o2o_panel.parquet}
REGRESSION_CONCURRENCY=${REGRESSION_CONCURRENCY:-4}
FEATURE=${FEATURE:-title_body_full_concat}
TARGETS=${TARGETS:-next_day_open_to_open_return}
REGRESSORS=${REGRESSORS:-ridge}
REDUCERS=${REDUCERS:-none,pca:128}
manifest="configs/generated/regression_${MODEL}_${VARIANT_GROUP}_triple_concat.tsv"
mkdir -p configs/generated logs/slurm_pooled_regression logs/slurm_pooled_regression_launcher
export PYTHONPATH=/home/team/llm_return/.venv/lib/python3.9/site-packages${PYTHONPATH:+:${PYTHONPATH}}

/usr/bin/python3 scripts/build_pooled_regression_manifest.py \
  --model "${MODEL}" --variants "${VARIANTS}" \
  --feature "${FEATURE}" \
  --targets "${TARGETS}" \
  --regressors "${REGRESSORS}" \
  --reducers "${REDUCERS}" \
  --expected-rows 350577 --output "${manifest}"
tasks=$(($(wc -l < "${manifest}") - 1))
job_id=$(tracked_sbatch --parsable \
  --array="0-$((tasks - 1))%${REGRESSION_CONCURRENCY}" \
  --export="ALL,MANIFEST=${manifest},PANEL=${PANEL}" \
  scripts/slurm_pooled_embedding_regression.sbatch)
echo "submitted regression model=${MODEL} group=${VARIANT_GROUP} tasks=${tasks} job=${job_id}"
