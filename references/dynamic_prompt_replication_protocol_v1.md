# Dynamic Prompt Replication Protocol v1

## Scope

This protocol evaluates whether frozen contextualized prompt-token embeddings
from Chinese RoBERTa predict A-share returns and whether announcement-specific
dynamic token weights improve on parameter-matched uniform and static gates.

The current source text is CNINFO corporate announcements. The experiment is a
method replication and China-market extension of the news-return design, not a
strict data replication of a financial-news paper.

## Frozen design

- Panel: `data/processed/cninfo_full_classification_panel.parquet`.
- Representation: RoBERTa `masked_short`, 28 contextualized prompt positions,
  768 dimensions per position, frozen Transformer.
- Fit: 2018--2023; validation: 2024--2025; test: available 2026 observations.
- Primary unit: one equal-weighted prediction per stock and entry date.
- Primary task: continuous next-day return; validation selection uses daily
  cross-sectional stock-day Rank IC.
- Auxiliary task: next-day direction; validation selection uses stock-day AUC.
- The 2026 period has already been inspected in earlier experiments and is
  incomplete. It is exploratory, not a pristine confirmatory holdout.
- No 2026 metric may select a model, epoch, threshold, token subset, or seed.

## Primary comparison

`uniform`, `static`, and `dynamic` gates use the same token projection,
representation size, prediction head, optimizer, batch size, and training
schedule. Gate mode is the only intended architectural difference.

Fisher Top-8 and random Top-8 are secondary hard-gate comparisons. They must
use the same projected representation and head rather than concatenating raw
vectors into a larger classifier.

## Required diagnostics

- Row and target alignment by unique `row_index`.
- Training-only preprocessing and target scaling.
- Stock-day validation after every epoch and validation-only early stopping.
- Per-announcement token weights, mean weight, standard deviation, entropy,
  and between-announcement variation.
- Date-clustered paired bootstrap for final model differences.
- Text/label permutation and metadata-only placebos before a semantic claim.

## Interpretation gates

Dynamic gating is supported only if it improves on both uniform and static
gates across repeated seeds, with date-clustered uncertainty, while producing
meaningfully announcement-varying and non-collapsed weights.

Existing segment-gate descriptions must be derived from persisted group order
and selected indices. For `title_body_full_concat`, positions map to title,
body, and full in that order; narrative summaries must not infer the selected
segments from model names alone.

