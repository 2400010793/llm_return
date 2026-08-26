# Historical pooled-classification runtime audit

Slurm `ElapsedRaw` is mapped to the exact array cell through its persisted TSV manifest. These measurements describe completed historical jobs; they are not scheduler limits.

| Classifier | Reducer | Jobs | Median minutes | P90 minutes | Max hours | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|
| `linear_svm` | `none` | 4 | 1017.98 | 2107.04 | 40.56 | 35.33 |
| `linear_svm` | `pca` | 4 | 0.90 | 1.16 | 0.02 | 23.97 |
| `logistic` | `none` | 24 | 1.04 | 7.66 | 0.16 | 39.02 |
| `logistic` | `pca` | 24 | 0.33 | 0.65 | 0.02 | 25.04 |
| `mlp` | `none` | 4 | 29.79 | 35.39 | 0.59 | 24.30 |
| `mlp` | `pca` | 4 | 7.63 | 12.05 | 0.22 | 25.04 |
| `sgd` | `none` | 4 | 24.85 | 43.00 | 0.83 | 23.53 |
| `sgd` | `pca` | 4 | 1.39 | 1.96 | 0.04 | 24.98 |

## Interpretation

- Calibrated Linear SVM without PCA is the dominant long-running cell; PCA-128 makes it much cheaper.
- PCA cells in these manifests include embedding loading, two training-window PCA fits, tuning and final fitting.
- Future reports record phase-level runtime directly, so subsequent estimates need not rely only on Slurm wall time.
