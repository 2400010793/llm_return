"""Create an interactive 2-D UMAP/HDBSCAN view of Qwen prompt embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import plotly.express as px


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--variant", choices=("short", "masked_short"), default="masked_short")
    p.add_argument("--representations", default="article_mean,target_token")
    p.add_argument("--sample-per-group", type=int, default=2500)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = args.input_root / args.variant
    meta = pd.read_parquet(base / "intersection_metadata.parquet", columns=["row_index"])
    common = meta.row_index.to_numpy()
    rows, matrices = [], []
    for path in sorted(base.glob("*_*.npy")):
        stem = path.stem
        representation = next((value for value in args.representations.split(",") if stem.endswith("_" + value)), None)
        if representation is None:
            continue
        prompt = stem[:-(len(representation) + 1)]
        if representation not in args.representations.split(","):
            continue
        matrix = np.load(path, mmap_mode="r")
        if len(matrix) != len(common):
            raise ValueError(f"matrix length mismatch: {path}")
        rng = np.random.default_rng(42 + len(rows))
        take = np.sort(rng.choice(len(matrix), min(args.sample_per_group, len(matrix)), replace=False))
        matrices.append(np.asarray(matrix[take], dtype=np.float32))
        rows.append(pd.DataFrame({"row_index": common[take], "prompt": prompt, "representation": representation}))
    if not matrices:
        raise ValueError("no matching matrices")
    x = np.vstack(matrices)
    labels_meta = pd.concat(rows, ignore_index=True)
    pca_n = min(args.pca_components, x.shape[1], len(x) - 1)
    z = PCA(n_components=pca_n, svd_solver="randomized", random_state=42).fit_transform(x)
    z = StandardScaler().fit_transform(z).astype(np.float32)
    embedding = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42, n_jobs=1).fit_transform(z)
    cluster = hdbscan.HDBSCAN(min_cluster_size=max(25, args.sample_per_group // 50), min_samples=10, prediction_data=True).fit(embedding)
    labels_meta["umap_1"], labels_meta["umap_2"] = embedding[:, 0], embedding[:, 1]
    labels_meta["cluster"] = cluster.labels_.astype(str)
    labels_meta["cluster_probability"] = cluster.probabilities_
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels_meta.to_parquet(args.output.with_suffix(".parquet"), index=False)
    fig = px.scatter(labels_meta, x="umap_1", y="umap_2", color="prompt", symbol="representation", hover_data=["row_index", "cluster", "cluster_probability"], title=f"Qwen prompt embeddings: {args.variant}")
    fig.write_html(args.output, include_plotlyjs=True)
    labels_meta.groupby(["prompt", "representation"], as_index=False).agg(rows=("row_index", "size"), clusters=("cluster", "nunique"), noise_rate=("cluster", lambda x: float((x == "-1").mean()))).to_csv(args.output.with_name(args.output.stem + "_summary.csv"), index=False)
    print(labels_meta.groupby(["prompt", "representation"]).size().to_string())
    print(f"output={args.output}")


if __name__ == "__main__":
    main()