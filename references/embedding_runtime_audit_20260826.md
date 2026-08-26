# Embedding Runtime Audit (2026-08-26)

This note freezes the timing evidence used by the README, handoff, and comprehensive report.
It separates scheduler queue delay, task compute time, and observed array wall time.

## Production contract

- Inputs: Sina 796,553 rows; CNINFO 903,665 rows.
- Prompts per task: six neutral `masked_short` prompts.
- Array shape: 256 shards with a maximum concurrency of 256.
- Resources: 8 CPUs, 32 GB memory, and a 24-hour limit per task.
- RoBERTa: batch size 8, maximum length 512.
- BGE-M3: batch size 4, maximum length 1,000.
- Each task loads its model once and processes all six prompts for one shard.

The submission manifest is the task record whose directory starts with
`20260825T085657Z-neutral-masked-short-roberta-bge-full-2010-2026`. Base job IDs are:

| Job | Dataset | Model |
|---:|---|---|
| 4985394 | Sina | RoBERTa |
| 4985395 | Sina | BGE-M3 |
| 4985396 | CNINFO | RoBERTa |
| 4985397 | CNINFO | BGE-M3 |

## Completed-task timing

Only completed array elements are included in the elapsed distribution.

| Dataset/model | States | Min | Median | P90 | Max | Observed active wall window |
|---|---|---:|---:|---:|---:|---:|
| Sina/RoBERTa | 256 completed | 2,101s | 2,810s | 11,076s | 12,732s | 16,097s |
| Sina/BGE-M3 | 189 completed, 67 cancelled | 5,214s | 18,951s | 32,855s | 55,155s | 56,985s partial |
| CNINFO/RoBERTa | 255 completed, 1 failed | 2,327s | 4,270s | 7,307s | 16,421s | 33,417s partial |
| CNINFO/BGE-M3 | 78 completed, 178 cancelled | 15,169s | 16,604s | 17,647s | 24,125s | 30,481s partial |

The active wall window is measured from the earliest started completed task to the latest completed
end time. It excludes time spent queued before the first task started. A partial active window is not
a full-array completion time. CNINFO/RoBERTa shard 225 failed in the base array and required a
supplemental job; the final artifact must be checked through its manifest rather than inferred from
the base array state.

Completed retained output counts were 588,049 rows for Sina/BGE-M3 and 221,551 rows for
CNINFO/BGE-M3. These are useful outputs, but they are not full-dataset embeddings.

## Capacity formula

Use the following estimate before submitting a new array:

```text
compute_wall = ceil(shards / effective_concurrency)
               * representative_shard_elapsed
               * (1 + I/O_and_validation_buffer)
turnaround = scheduler_queue_delay + compute_wall
```

Use P90 rather than the minimum or median when the goal is a completion budget. Add at least 20%
for shared-filesystem I/O, manifest generation, and validation. With 256 healthy simultaneous CPU
slots, the current conditional BGE-M3 budgets are 10-16 hours for Sina and 5-7 hours for CNINFO.
These are planning ranges, not service-level guarantees.

## Qwen reference

Slurm job 3651805 is the only clean completed throughput reference available for Qwen. It used one
RTX 4090, 8 CPUs, and 64 GB memory. It encoded 57,741 rows with `qwen3-embedding:8b`, batch size
8 and a 12,000-character input cap in 18,715 seconds (5:11:55), or approximately 3.085 rows/s.
The output was one pooled 4,096-dimensional embedding per row.

A linear one-pass extrapolation at that measured throughput is 71.7 hours for 796,553 Sina rows and
81.4 hours for 903,665 CNINFO rows. This is not an estimate for the complete prompt-token contract:
token-level output, GGUF runner choice, token limit, cache behavior, disk writes, and GPU concurrency
all change runtime. A full representative shard benchmark is required before budgeting a six-prompt
Qwen token run.

## Reproduction

The source values were read from Slurm accounting (`ElapsedRaw`, `Start`, `End`, and `State`), the
frozen task manifest, shard manifests, and the completed Qwen log. Re-run accounting queries with
`sacct -X` and filter elapsed statistics to `State=COMPLETED`; report cancelled and failed task counts
separately. Never use submission-to-end wall time as model compute time.
