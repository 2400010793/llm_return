"""Compare raw Qwen article and token representations on their row intersection."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load(root: Path, representation: str) -> tuple[pd.DataFrame, np.ndarray]:
    base = root / representation
    meta = pd.read_parquet(base / "metadata.parquet", columns=["row_index"])
    matrix = np.load(base / "matrix.npy", mmap_mode="r")
    if len(meta) != len(matrix):
        raise ValueError(f"metadata/matrix mismatch: {base}")
    meta = meta.copy()
    meta["row_index"] = pd.to_numeric(meta["row_index"], errors="raise").astype(np.int64)
    if meta.row_index.duplicated().any():
        raise ValueError(f"duplicate row_index: {base}")
    return meta, matrix


def pair_stats(left: np.ndarray, right: np.ndarray, name_left: str, name_right: str) -> dict[str, float | str]:
    x = np.asarray(left, dtype=np.float32)
    y = np.asarray(right, dtype=np.float32)
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    cosine = np.sum(x * y, axis=1) / np.maximum(nx * ny, 1e-12)
    return {
        "left": name_left, "right": name_right, "rows": len(x),
        "cosine_mean": float(np.mean(cosine)), "cosine_median": float(np.median(cosine)),
        "cosine_p05": float(np.quantile(cosine, .05)), "cosine_p95": float(np.quantile(cosine, .95)),
        "l2_mean": float(np.mean(np.linalg.norm(x - y, axis=1))),
        "norm_left_mean": float(np.mean(nx)), "norm_right_mean": float(np.mean(ny)),
        "component_pearson": float(pearsonr(x.ravel(), y.ravel())[0]),
        "row_cosine_spearman": float(spearmanr(cosine, np.linalg.norm(x - y, axis=1)).statistic),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--representations", default="article_mean,prompt_mean,return_token")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    reps = [x.strip() for x in args.representations.split(",") if x.strip()]
    loaded = {rep: load(args.root, rep) for rep in reps}
    common = set(loaded[reps[0]][0].row_index)
    for rep in reps[1:]:
        common &= set(loaded[rep][0].row_index)
    common = np.array(sorted(common), dtype=np.int64)
    aligned = {}
    for rep, (meta, matrix) in loaded.items():
        pos = pd.Series(np.arange(len(meta)), index=meta.row_index)
        aligned[rep] = np.asarray(matrix[pos.loc[common].to_numpy()], dtype=np.float32)
    rows = []
    for i, left in enumerate(reps):
        for right in reps[i + 1:]:
            rows.append(pair_stats(aligned[left], aligned[right], left, right))
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    pd.DataFrame([{"common_rows": len(common), "representations": ",".join(reps)}]).to_csv(args.output.with_name(args.output.stem + "_audit.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()