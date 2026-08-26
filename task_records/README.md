# Computational task records

This directory is the persistent provenance ledger for Slurm and other long-running computations. Each task receives one immutable task ID and one directory.

## Submit a top-level task

Use the tracker around a submission script or direct `sbatch` command:

    python scripts/task_tracker.py run \
      --name roberta-full-mean-pca-svm \
      --purpose "Aligned full-panel RoBERTa mean-pooling classification" \
      --input data/processed/cninfo_full_classification_panel.parquet \
      --output reports/classification/roberta_full_mean \
      --seed 42 \
      --related-file scripts/slurm_example.sbatch \
      -- sbatch --parsable scripts/slurm_example.sbatch

The record and code snapshot are written before the command runs. The command receives `TASK_RECORD_ID` and `TASK_RECORD_ROOT` in its environment.

## Submit child jobs

Launchers that create jobs use:

    python scripts/task_tracker.py child-submit --task-id "$TASK_RECORD_ID" -- sbatch --parsable ...

Every child submission is appended to the same manifest with its exact command and returned job ID.

## Update or close a task

    python scripts/task_tracker.py sync --task-id TASK_ID
    python scripts/task_tracker.py close --task-id TASK_ID --status completed_audited --note "Audit passed"

Use `close` only after outputs and audits have been checked. A retry remains under the original task ID and is recorded as another child submission/attempt.

## Record contents

- `manifest.json`: purpose, commands, resources, inputs/outputs, seed, environment, job lineage, status and audits.
- `code/git-status.txt`: repository state before submission.
- `code/tracked-dirty.patch`: complete tracked working-tree and staged patch.
- `code/untracked-files.txt`: exact untracked-file inventory.
- `code/untracked.tar.gz`: snapshot of untracked repository files, excluding this ledger.
- `code/file-hashes.json`: SHA-256 of command/relevant files and snapshots.
- `stdout.txt` and `stderr.txt`: top-level submission command output.

Secrets are redacted. Records are append/update-only operational evidence; do not reuse a task ID for a different scientific purpose.
