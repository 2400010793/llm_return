#!/bin/bash
# Repartition already-cleaned, already-masked CNINFO bundles without re-running
# the expensive identity/time regex pass over the 2010--2026 source archive.
set -euo pipefail

OUTPUT=${1:?output directory required}
ROOT=/data/alpha_team2/shares/llm_return/data/processed/cleaned
PERIOD_ROOTS=(
  "${ROOT}/cninfo_prompt_bundle_existing_2018_2026_clean_v1/prompt_v3_fixed_parts"
  "${ROOT}/cninfo_prompt_bundle_increment_2010_2017_clean_v1/prompt_v3_fixed_parts"
)
mkdir -p "${OUTPUT}"
for root in "${PERIOD_ROOTS[@]}"; do
  [[ -d "${root}" ]] || { echo "missing bundle ${root}" >&2; exit 2; }
done

build_one() {
  local period=$1
  local source_root=$2
  local source_shard=$3
  local base=$((period * 128 + source_shard * 4))
  local scratch
  scratch=$(mktemp -d "${TMPDIR:-/tmp}/neutral-cninfo-split.XXXXXX")
  trap 'rm -rf "${scratch}"' RETURN
  for shard in 0 1 2 3; do mkdir -p "${OUTPUT}/shard-$((base + shard))"; done
  mapfile -t parts < <(find "${source_root}/shard-${source_shard}" -maxdepth 1 -type f -name 'part-*.jsonl' | sort)
  ((${#parts[@]} > 0)) || { echo "missing source parts for ${source_root}/shard-${source_shard}" >&2; exit 2; }
  # Round-robin mode works on stdin and preserves every record without first
  # scanning the multi-gigabyte source shard to determine its line count.
  cat "${parts[@]}" | split -n r/4 -d - "${scratch}/chunk-"
  for shard in 0 1 2 3; do
    local chunk
    chunk=$(printf '%s/chunk-%02d' "${scratch}" "${shard}")
    [[ -s "${chunk}" ]] || { echo "empty split ${chunk}" >&2; exit 2; }
    cp "${chunk}" "${OUTPUT}/shard-$((base + shard))/part-00000.jsonl"
  done
}

export OUTPUT
export -f build_one
for period in 0 1; do
  source_root=${PERIOD_ROOTS[$period]}
  for shard in $(seq 0 31); do
    printf '%s\t%s\t%s\n' "${period}" "${source_root}" "${shard}"
  done
done | xargs -P 8 -n 3 bash -c 'build_one "$@"' _

cat > "${OUTPUT}/manifest.json" <<EOF
{
  "format_version": "neutral_masked_short_input_v1",
  "dataset": "cninfo",
  "rows": 903665,
  "shards": 256,
  "source": "prebuilt_cninfo_prompt_v3_fixed_parts",
  "mask_policy": "identity_and_time_only",
  "prompt_fields": ["masked_short_prompt", "masked_short_title", "masked_short_body", "text_input_5_masked_short"],
  "row_index_range": [1, 903665],
  "repartition": "64_source_shards_to_256_output_shards"
}
EOF
echo "built CNINFO neutral inputs: ${OUTPUT}"
