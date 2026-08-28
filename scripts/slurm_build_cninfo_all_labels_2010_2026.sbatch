#!/bin/bash
# Build the all-article one-day classification label panel after the
# historical cleaning/merge job has completed. This job never collects data.
#SBATCH --job-name=cninfo-all-labels
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/home/team/llm_return/logs/slurm_cninfo_all_labels/%j.out
#SBATCH --error=/home/team/llm_return/logs/slurm_cninfo_all_labels/%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/home/team/llm_return}
cd "${REPO_ROOT}"
: "${TASK_RECORD_ID:?label-panel construction must use scripts/task_tracker.py}"

PYTHON=${PYTHON:-${REPO_ROOT}/.venv/bin/python}
INPUT=${INPUT:-data/processed/cleaned/cninfo_announcements_2010_2026_final.jsonl}
RETURNS=${RETURNS:-/home/team/ret.parquet}
PANEL=${PANEL:-data/processed/cninfo_full_classification_panel_2010_2026.parquet}

mkdir -p logs/slurm_cninfo_all_labels "$(dirname "${PANEL}")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO_ROOT}/.venv/lib/python3.9/site-packages"${PYTHONPATH:+:${PYTHONPATH}}

[[ -s "${INPUT}" ]] || { echo "cleaned input is missing: ${INPUT}" >&2; exit 2; }
[[ -s "${RETURNS}" ]] || { echo "return matrix is missing: ${RETURNS}" >&2; exit 2; }

"${PYTHON}" -m py_compile scripts/build_cninfo_full_classification_panel.py
"${PYTHON}" scripts/build_cninfo_full_classification_panel.py \
  --input "${INPUT}" \
  --returns "${RETURNS}" \
  --output "${PANEL}"

"${PYTHON}" - "${PANEL}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1])
frame = pd.read_parquet(path, columns=[
    "row_index", "entry_date", "next_day_return", "next_day_label",
])
expected = pd.Series(range(1, len(frame) + 1), dtype="int64")
actual = pd.to_numeric(frame["row_index"], errors="raise").astype("int64")
if not actual.reset_index(drop=True).equals(expected):
    raise SystemExit("label panel row_index is not the contiguous sequence 1..N")
years = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year.dropna().unique()
if len(years) < 9:
    raise SystemExit(f"rolling classification needs at least 9 years; found {sorted(years)}")
if frame["next_day_return"].notna().sum() == 0:
    raise SystemExit("next_day_return has no finite labels")
print(json.dumps({
    "panel": str(path),
    "rows": int(len(frame)),
    "years": sorted(int(year) for year in years),
    "finite_next_day_return": int(frame["next_day_return"].notna().sum()),
    "finite_next_day_label": int(frame["next_day_label"].notna().sum()),
    "primary_target": "next_day_return",
    "legacy_event_target_not_used_for_primary_classification": True,
}, ensure_ascii=False))
PY
