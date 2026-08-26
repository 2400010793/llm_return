# Project Guidelines

## Computational task provenance

- Before every new Slurm submission or other long-running computation in this repository, create a persistent task record with `scripts/task_tracker.py`. Do not call `sbatch` directly for a new task.
- Use `run` for a top-level submission command and `child-submit` for jobs created by a recorded launcher. Preserve `TASK_RECORD_ID` in exported Slurm environments.
- The record must exist before submission and contain Git HEAD, status, the complete tracked dirty patch, an archive of untracked repository files, relevant file hashes, the exact command, resources, dependencies, inputs, outputs, environment, and seed.
- Update the same record with job IDs, lifecycle states, retries, logs, outputs, hashes, and audit conclusions. Do not rely on chat history as the sole provenance record.
- Do not auto-commit code to create a snapshot. Never store secrets; sensitive environment values must be redacted.
- Pure status queries, read-only audits, syntax checks, and short local tests do not require a task record.
- Do not cancel or interrupt existing jobs unless the user explicitly asks to kill them.

See [task_records/README.md](../task_records/README.md) for the workflow.
