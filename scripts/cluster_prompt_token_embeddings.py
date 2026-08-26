"""Cluster prompt target-span token embeddings and probe low/neutral/high separability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def _leaf(root: Path, shard: int, prompt_id: str) -> Path | None:
    candidates = sorted((root / f"shard-{shard}").glob(f"group-*/{prompt_id}/masked_short"))
    if len(candidates) != 1 or not (candidates[0].parents[1] / "COMPLETED").exists():
        return None
    return candidates[0]


def _load_target(leaf: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    hidden = np.load(leaf / "prompt_token_embeddings.npy", mmap_mode="r")
    metadata = pd.read_json(leaf / "metadata.jsonl", lines=True)
    audit = json.loads((leaf / "prompt_spec.json").read_text(encoding="utf-8"))["token_audit"]
    target = np.asarray(hidden[:, audit["target_token_indices"], :].mean(axis=1), dtype=np.float32)
    return target, metadata["row_index"].to_numpy(np.int64), audit


def _centroid(root: Path, prompt_id: str, shards: int) -> tuple[np.ndarray, int, dict]:
    total = None
    count = 0
    audit = None
    for shard in range(shards):
        leaf = _leaf(root, shard, prompt_id)
        if leaf is None:
            continue
        values, _, audit = _load_target(leaf)
        subtotal = values.sum(axis=0, dtype=np.float64)
        total = subtotal if total is None else total + subtotal
        count += len(values)
    if total is None or count == 0 or audit is None:
        raise FileNotFoundError(f"no completed shards for {prompt_id}")
    return (total / count).astype(np.float32), count, audit


def _cosine_matrix(values: np.ndarray) -> np.ndarray:
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return normalized @ normalized.T


def _agglomerative(distance: np.ndarray, k: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return model.fit_predict(distance)


def _centroid_clustering(records: list[dict], values: np.ndarray) -> dict:
    cosine = _cosine_matrix(values)
    distance = np.clip(1.0 - cosine, 0.0, 2.0)
    pca = PCA(n_components=min(3, len(records), values.shape[1]), random_state=42)
    coordinates = pca.fit_transform(values)
    for record, coord in zip(records, coordinates):
        record["pca"] = [float(x) for x in coord]
    clusters = {}
    for k in range(2, min(6, len(records) - 1) + 1):
        labels = _agglomerative(distance, k)
        clusters[str(k)] = {record["prompt_id"]: int(label) for record, label in zip(records, labels)}
    pairs = []
    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            pairs.append({"left": left["prompt_id"], "right": records[j]["prompt_id"], "cosine": float(cosine[i, j])})
    return {
        "records": records,
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "clusters": clusters,
        "pairwise_cosine": pairs,
    }


def _axis_probe(root: Path, prompt_ids: list[str], shards: int, sample_per_shard: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    observations = []
    labels = []
    prompt_labels = []
    shards_used = 0
    for shard in range(shards):
        leaves = [_leaf(root, shard, prompt_id) for prompt_id in prompt_ids]
        if any(leaf is None for leaf in leaves):
            continue
        loaded = [_load_target(leaf) for leaf in leaves if leaf is not None]
        row_ids = [item[1] for item in loaded]
        if not all(np.array_equal(row_ids[0], rows) for rows in row_ids[1:]):
            raise ValueError(f"row_index mismatch in shard {shard} for {prompt_ids}")
        n = len(loaded[0][0])
        selected = rng.choice(n, size=min(sample_per_shard, n), replace=False)
        for level, (values, _, _) in enumerate(loaded):
            observations.append(values[selected])
            labels.extend([level] * len(selected))
            prompt_labels.extend([prompt_ids[level]] * len(selected))
        shards_used += 1
    matrix = np.vstack(observations).astype(np.float32)
    n_components = min(20, matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(n_components=n_components, whiten=True, random_state=seed).fit_transform(matrix)
    km = KMeans(n_clusters=3, n_init=20, random_state=seed).fit(reduced)
    sample_size = min(10000, len(reduced))
    return {
        "prompt_ids": prompt_ids,
        "shards": shards_used,
        "rows": int(len(matrix)),
        "sample_per_shard": sample_per_shard,
        "pca_components": n_components,
        "kmeans_inertia": float(km.inertia_),
        "adjusted_rand_index": float(adjusted_rand_score(labels, km.labels_)),
        "normalized_mutual_information": float(normalized_mutual_info_score(labels, km.labels_)),
        "silhouette_pca": float(silhouette_score(reduced, km.labels_, sample_size=sample_size, random_state=seed)),
        "cluster_sizes": {str(k): int(v) for k, v in zip(*np.unique(km.labels_, return_counts=True))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--axes-json", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--anchor-root", type=Path)
    parser.add_argument("--anchor-json", type=Path)
    parser.add_argument("--sample-per-shard", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    axes = json.loads(args.axes_json.read_text(encoding="utf-8"))
    anchors = json.loads(args.anchor_json.read_text(encoding="utf-8")) if args.anchor_json else {}
    records = []
    values = []
    for axis, prompt_ids in axes.items():
        for level, prompt_id in enumerate(prompt_ids):
            vector, rows, audit = _centroid(args.root, prompt_id, args.shards)
            records.append({"prompt_id": prompt_id, "axis": axis, "level": level, "prompt_tokens": audit["tokens"], "rows": rows})
            values.append(vector)
    centroid = _centroid_clustering(records, np.vstack(values))
    anchor_records = []
    anchor_values = []
    if args.anchor_root:
        for axis, prompt_id in anchors.items():
            vector, rows, audit = _centroid(args.anchor_root, prompt_id, args.shards)
            anchor_records.append({"prompt_id": prompt_id, "axis": axis, "level": "anchor", "prompt_tokens": audit["tokens"], "rows": rows})
            anchor_values.append(vector)
        anchor = _centroid_clustering(anchor_records, np.vstack(anchor_values)) if anchor_values else None
    else:
        anchor = None
    probes = {axis: _axis_probe(args.root, prompt_ids, args.shards, args.sample_per_shard, 42 + index) for index, (axis, prompt_ids) in enumerate(axes.items())}
    result = {"format_version": "prompt_token_embedding_cluster_v1", "representation": "target_span_mean_token_embedding", "centroid_clustering": centroid, "anchor_centroid_clustering": anchor, "axis_level_probes": probes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "prompts": len(records), "probes": len(probes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
