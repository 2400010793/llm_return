# Classification report trust audit

This is a read-only audit of existing result JSON files. Historical reports are not modified. `valid_legacy` and `valid_full_panel` are scientifically distinct test universes and must not be ranked as if they were one sample.

## Status counts

| Status | Result rows |
|---|---:|
| `valid_full_panel` | 72 |
| `valid_legacy` | 30 |

## Trust rules

| Directory | Status | Basis |
|---|---|---|
| `time_aligned_prompt_embeddings` | `invalid_alignment` | 350,577-row embeddings were indexed by positions from the 57,741-row legacy panel without metadata-key alignment. |
| `transformers_time_aligned` | `mislabelled_representation` | Transformer-named files contain non-transformer representation labels (for example word_tfidf); filenames are not reliable evidence. |
| `time_aligned_embedding_extended` | `valid_legacy` | Legacy 98-stock experiment is internally row-aligned; it is valid only for its 57,741-row panel and 3,050-row 2026 test set. |
| `pooled_embeddings` | `valid_full_panel` | Full-panel pooled pipeline uses row-index alignment with explicit expected-row audits; results are not directly comparable to legacy tests. |

## Trusted result rows by accuracy (top 25)

| Scope | Report | Representation | Classifier | Accuracy | Majority | N test |
|---|---|---|---|---:|---:|---:|
| `valid_legacy` | `time_aligned_embedding_extended/qwen_linear_svm_pca_seed42.json` | `npy:data/processed/embeddings/qwen3_embedding_8b/embeddings.npy` | `linear_svm` | 0.541967 | 0.535738 | 3050 |
| `valid_legacy` | `time_aligned_embedding_extended/roberta_linear_svm_pca_seed42.json` | `npy:data/processed/embeddings/chinese_roberta/chinese_roberta.npy` | `linear_svm` | 0.540000 | 0.535738 | 3050 |
| `valid_full_panel` | `pooled_embeddings/triple_models/bge_m3_masked_short_title_body_full_concat_mlp_none_0.json` | `bge_m3:masked_short:title_body_full_concat` | `mlp` | 0.526226 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/triple_models/bge_m3_short_title_body_full_concat_mlp_none_0.json` | `bge_m3:short:title_body_full_concat` | `mlp` | 0.522901 | 0.515375 | 21658.0 |
| `valid_legacy` | `time_aligned_embedding_extended/roberta_random_forest_none_seed42.json` | `npy:data/processed/embeddings/chinese_roberta/chinese_roberta.npy` | `random_forest` | 0.521967 | 0.535738 | 3050 |
| `valid_full_panel` | `pooled_embeddings/triple_models/roberta_masked_short_title_body_full_concat_mlp_none_0.json` | `roberta:masked_short:title_body_full_concat` | `mlp` | 0.521239 | 0.515375 | 21658.0 |
| `valid_legacy` | `time_aligned_embedding_extended/bge_m3_linear_svm_pca_seed42.json` | `npy:data/processed/embeddings/bge_m3/bge_m3.npy` | `linear_svm` | 0.520000 | 0.535738 | 3050 |
| `valid_legacy` | `time_aligned_embedding_extended/qwen_mlp_none_seed42.json` | `npy:data/processed/embeddings/qwen3_embedding_8b/embeddings.npy` | `mlp` | 0.519672 | 0.535738 | 3050 |
| `valid_legacy` | `time_aligned_embedding_extended/roberta_knn_none_seed42.json` | `npy:data/processed/embeddings/chinese_roberta/chinese_roberta.npy` | `knn` | 0.519672 | 0.535738 | 3050 |
| `valid_legacy` | `time_aligned_embedding_extended/roberta_mlp_none_seed42.json` | `npy:data/processed/embeddings/chinese_roberta/chinese_roberta.npy` | `mlp` | 0.519016 | 0.535738 | 3050 |
| `valid_full_panel` | `pooled_embeddings/screening/bge_m3_masked_short_title_mean_logistic_none_0.json` | `bge_m3:masked_short:title_mean` | `logistic` | 0.517222 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/aggregation/bge_m3_masked_short_title_body_concat_logistic_none_0.json` | `bge_m3:masked_short:title_body_concat` | `logistic` | 0.516206 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/screening/roberta_masked_short_title_mean_logistic_none_0.json` | `roberta:masked_short:title_mean` | `logistic` | 0.516160 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/triple_models/bge_m3_masked_short_title_body_full_concat_linear_svm_pca_128.json` | `bge_m3:masked_short:title_body_full_concat` | `linear_svm` | 0.515560 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/triple_models/roberta_short_title_body_full_concat_mlp_none_0.json` | `roberta:short:title_body_full_concat` | `mlp` | 0.515468 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/triple_models/roberta_masked_short_title_body_full_concat_linear_svm_none_0.json` | `roberta:masked_short:title_body_full_concat` | `linear_svm` | 0.515422 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/triple_models/roberta_short_title_body_full_concat_linear_svm_pca_128.json` | `roberta:short:title_body_full_concat` | `linear_svm` | 0.515375 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/screening/bge_m3_masked_short_full_mean_logistic_none_0.json` | `bge_m3:masked_short:full_mean` | `logistic` | 0.514683 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/aggregation/roberta_short_title_body_concat_logistic_none_0.json` | `roberta:short:title_body_concat` | `logistic` | 0.514221 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/screening/bge_m3_masked_short_title_mean_logistic_pca_128.json` | `bge_m3:masked_short:title_mean` | `logistic` | 0.514129 | 0.515375 | 21658.0 |
| `valid_legacy` | `time_aligned_embedding_extended/roberta_linear_svm_none_seed42.json` | `npy:data/processed/embeddings/chinese_roberta/chinese_roberta.npy` | `linear_svm` | 0.514098 | 0.535738 | 3050 |
| `valid_legacy` | `time_aligned_embedding_extended/roberta_logistic_none_seed42.json` | `npy:data/processed/embeddings/chinese_roberta/chinese_roberta.npy` | `logistic` | 0.513770 | 0.535738 | 3050 |
| `valid_full_panel` | `pooled_embeddings/screening/bge_m3_masked_short_body_mean_logistic_none_0.json` | `bge_m3:masked_short:body_mean` | `logistic` | 0.513759 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/triple_concat/bge_m3_masked_short_title_body_full_concat_logistic_none_0.json` | `bge_m3:masked_short:title_body_full_concat` | `logistic` | 0.513667 | 0.515375 | 21658.0 |
| `valid_full_panel` | `pooled_embeddings/aggregation/bge_m3_masked_short_title_body_mean_logistic_none_0.json` | `bge_m3:masked_short:title_body_mean` | `logistic` | 0.513575 | 0.515375 | 21658.0 |

## Interpretation

- The historical RoBERTa approximately 54% result remains valid for the legacy 98-stock panel.
- Reports marked `invalid_alignment` must not be cited, ranked, or used for model selection.
- Reports marked `mislabelled_representation` require reconstruction from commands and artifacts before use.
- Future precomputed embedding reports must include the `embedding_alignment` audit emitted by the repaired rolling classifier.
