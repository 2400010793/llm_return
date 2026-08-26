"""Leakage-safe fixed/tuned PCA+HDBSCAN linear regression for one matrix fold.

The runner is intentionally dataset/model/prompt agnostic.  One invocation
reads one precomputed matrix, evaluates both return targets, and writes base,
hard-cluster, and soft-cluster Ridge predictions.  Parameter selection, when
requested, uses only validation Top20% long-short stock-day return.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.pooled_embeddings import align_embeddings_to_panel, load_pooled_embeddings

TARGETS = ("next_day_return", "event_return_3d")
PCA_VALUES = (32, 64, 128)
MIN_CLUSTER_VALUES = (25, 50, 100, 250)
MIN_SAMPLES_VALUES = (3, 5, 10, 25)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def stock_day(frame: pd.DataFrame, pred: np.ndarray, target: str) -> pd.DataFrame:
    out = frame[["stock_id", "entry_date", target]].copy()
    out["prediction"] = pred
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["entry_date", target, "prediction"]).groupby(
        ["stock_id", "entry_date"], as_index=False
    ).agg(prediction=("prediction", "mean"), actual=(target, "mean"))


def top20_metrics(daily: pd.DataFrame) -> dict[str, float]:
    rows = []
    for _, group in daily.groupby("entry_date"):
        group = group.dropna(subset=["prediction", "actual"])
        if len(group) < 10:
            continue
        n = max(1, int(np.ceil(len(group) * 0.20)))
        ordered = group.sort_values("prediction")
        long_ret = float(ordered.tail(n)["actual"].mean())
        short_ret = float(ordered.head(n)["actual"].mean())
        rows.append((long_ret, short_ret, long_ret - short_ret))
    if not rows:
        return {"days": 0, "long_bp": np.nan, "short_bp": np.nan, "ls_bp": np.nan}
    values = np.asarray(rows, dtype=float)
    return {
        "days": int(len(values)),
        "long_bp": float(np.nanmean(values[:, 0]) * 1e4),
        "short_bp": float(np.nanmean(values[:, 1]) * 1e4),
        "ls_bp": float(np.nanmean(values[:, 2]) * 1e4),
    }


def rank_ic(daily: pd.DataFrame) -> float:
    values = []
    for _, group in daily.groupby("entry_date"):
        if len(group) >= 5 and group.prediction.nunique() > 1 and group.actual.nunique() > 1:
            values.append(group.prediction.corr(group.actual, method="spearman"))
    return float(np.nanmean(values)) if values else np.nan


def fit_projection(matrix, fit_rows: np.ndarray, components: int, sample_rows: int, seed: int = 42):
    chosen = np.asarray(fit_rows, dtype=int)
    if len(chosen) > sample_rows:
        chosen = np.sort(np.random.default_rng(seed).choice(chosen, sample_rows, replace=False))
    scaler = StandardScaler().fit(np.asarray(matrix[chosen], dtype=np.float32))
    scaled = scaler.transform(np.asarray(matrix[chosen], dtype=np.float32))
    n = min(int(components), scaled.shape[1], len(chosen) - 1)
    pca = PCA(n_components=n, svd_solver="randomized", random_state=seed, iterated_power=3).fit(scaled)

    def transform(rows: np.ndarray) -> np.ndarray:
        return pca.transform(scaler.transform(np.asarray(matrix[np.asarray(rows, dtype=int)], dtype=np.float32))).astype(np.float32)

    return scaler, pca, transform


def fit_cluster(z: np.ndarray, min_cluster_size: int, min_samples: int, sample_rows: int):
    chosen = np.arange(len(z))
    if sample_rows > 0 and len(chosen) > sample_rows:
        chosen = np.sort(np.random.default_rng(42).choice(chosen, sample_rows, replace=False))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size), min_samples=int(min_samples),
        metric="euclidean", prediction_data=True,
    ).fit(z[chosen])
    labels = clusterer.labels_[clusterer.labels_ >= 0]
    if len(np.unique(labels)) < 2:
        return None
    centers = np.vstack([z[chosen][clusterer.labels_ == label].mean(axis=0) for label in np.unique(labels)])
    distances = ((z[chosen][:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) ** 0.5
    scale = float(np.nanmedian(np.min(distances, axis=1)))
    return {"hdbscan": clusterer, "centers": centers.astype(np.float32), "scale": max(scale, 1e-6), "sample_rows": int(len(chosen))}


def cluster_features(clusterer, z: np.ndarray, mode: str):
    centers = clusterer["centers"]
    distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) ** 0.5
    labels = distances.argmin(axis=1).astype(int)
    strengths = np.exp(-distances[np.arange(len(z)), labels] / clusterer["scale"]).astype(np.float32)
    n_clusters = int(len(centers))
    noise = (labels < 0).astype(np.float32)[:, None]
    safe_labels = np.clip(labels, 0, max(n_clusters - 1, 0))
    hard = np.zeros((len(z), n_clusters), dtype=np.float32)
    if n_clusters:
        hard[np.arange(len(z)), safe_labels] = 1.0
    if mode == "hard":
        extra = np.hstack([hard, strengths[:, None].astype(np.float32), noise])
    elif mode == "soft":
        # Weighting the nearest HDBSCAN prototype by normalized distance is a
        # scalable soft assignment for full pooled runs.
        probs = hard * strengths[:, None].astype(np.float32)
        extra = np.hstack([probs, strengths[:, None].astype(np.float32), noise])
    else:
        raise ValueError(mode)
    return np.hstack([z, extra]), labels, strengths


def windows(test_year: int):
    return (
        np.arange(test_year - 8, test_year - 2),
        np.arange(test_year - 2, test_year),
        np.arange(test_year - 8, test_year),
    )


def candidate_search(frame, matrix, fit_rows, val_rows, target, mode, args):
    if args.mode == "fixed":
        # Fixed experiments deliberately do not fit/select on validation.
        if mode == "base":
            return {"score": np.nan, "components": 64, "alpha": 1.0, "cluster": None}, []
        return {"score": np.nan, "components": 64, "alpha": 1.0,
                "cluster": None, "min_cluster_size": 50, "min_samples": 10}, []
    candidates = []
    for components in (PCA_VALUES if args.mode == "tuned" else (64,)):
        scaler, pca, transform = fit_projection(matrix, fit_rows, components, args.pca_sample_rows)
        zfit, zval = transform(fit_rows), transform(val_rows)
        if mode == "base":
            for alpha in (ALPHAS if args.mode == "tuned" else (1.0,)):
                model = Ridge(alpha=alpha).fit(zfit, pd.to_numeric(frame.iloc[fit_rows][target], errors="coerce"))
                score = top20_metrics(stock_day(frame.iloc[val_rows], model.predict(zval), target))["ls_bp"]
                candidates.append({"score": score, "components": components, "alpha": alpha, "cluster": None})
            continue
        for mcs in (MIN_CLUSTER_VALUES if args.mode == "tuned" else (50,)):
            for ms in (MIN_SAMPLES_VALUES if args.mode == "tuned" else (10,)):
                clusterer = fit_cluster(zfit, mcs, ms, args.cluster_sample_rows)
                if clusterer is None:
                    continue
                xfit, _, _ = cluster_features(clusterer, zfit, mode)
                xval, _, _ = cluster_features(clusterer, zval, mode)
                for alpha in (ALPHAS if args.mode == "tuned" else (1.0,)):
                    model = Ridge(alpha=alpha).fit(xfit, pd.to_numeric(frame.iloc[fit_rows][target], errors="coerce"))
                    score = top20_metrics(stock_day(frame.iloc[val_rows], model.predict(xval), target))["ls_bp"]
                    candidates.append({"score": score, "components": components, "alpha": alpha, "cluster": clusterer, "min_cluster_size": mcs, "min_samples": ms})
    valid = [x for x in candidates if np.isfinite(x["score"])]
    if not valid:
        return {"score": np.nan, "components": 64, "alpha": 1.0, "cluster": None}, candidates
    return max(valid, key=lambda x: (x["score"], -abs(float(x["alpha"])))), candidates


def run(args):
    panel = pd.read_parquet(args.panel)
    if args.embedding_root:
        pooled = load_pooled_embeddings(
            args.embedding_root, model=args.model, variant=args.variant,
            feature=args.pooled_feature, require_complete_rows=args.expected_rows,
            max_matrix_gib=args.max_matrix_gib,
        )
        frame, matrix = align_embeddings_to_panel(panel, pooled, panel_row_index_column="row_index")
        if len(frame) != len(panel):
            raise ValueError(f"only {len(frame)}/{len(panel)} panel rows have pooled embeddings")
        matrix = np.asarray(matrix, dtype=np.float32)
    else:
        if not args.matrix or not args.metadata:
            raise ValueError("matrix/metadata are required when embedding_root is not set")
        metadata = pd.read_parquet(args.metadata)
        matrix = np.load(args.matrix, mmap_mode="r")
        if len(metadata) != matrix.shape[0] or "row_index" not in metadata:
            raise ValueError("matrix and metadata row count/row_index mismatch")
        lookup = pd.Series(np.arange(len(metadata), dtype=np.int64), index=metadata.row_index.astype(int))
        positions = panel.row_index.map(lookup)
        keep = positions.notna().to_numpy()
        frame = panel.loc[keep].reset_index(drop=True)
        matrix = matrix[positions[keep].astype(int).to_numpy()]
    years = pd.to_datetime(frame.entry_date, errors="coerce").dt.year.to_numpy()
    test_year = int(args.test_year)
    fit_years, val_years, all_years = windows(test_year)
    results, candidates_all = [], []
    for target in TARGETS:
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
        fit = np.flatnonzero(np.isfinite(y) & np.isin(years, fit_years))
        val = np.flatnonzero(np.isfinite(y) & np.isin(years, val_years))
        all_train = np.flatnonzero(np.isfinite(y) & np.isin(years, all_years))
        test = np.flatnonzero(np.isfinite(y) & (years == test_year))
        if min(len(fit), len(val), len(all_train), len(test)) == 0:
            raise ValueError(f"empty rolling split for {target} {test_year}")
        target_dir = args.output_root / args.dataset / args.model / args.prompt / args.variant / args.feature / target / args.mode / str(test_year)
        target_dir.mkdir(parents=True, exist_ok=True)
        selections = {}
        for mode in ("base", "hard", "soft"):
            selected, candidates = candidate_search(frame, matrix, fit, val, target, mode, args)
            selections[mode] = selected
            for row in candidates:
                candidates_all.append({"target": target, "model_mode": mode, "test_year": test_year, **{k: v for k, v in row.items() if k != "cluster"}})
        out = frame.iloc[test][["row_index", "stock_id", "entry_date", target]].copy()
        projection_cache = {}
        cluster_cache = {}
        for model_mode, selected in selections.items():
            components = int(selected["components"])
            if components not in projection_cache:
                _, _, transform = fit_projection(matrix, all_train, components, args.pca_sample_rows)
                projection_cache[components] = (transform(all_train), transform(test))
            zall, ztest = projection_cache[components]
            clusterer = None
            if model_mode in ("hard", "soft"):
                key = (components, int(selected.get("min_cluster_size", 50)), int(selected.get("min_samples", 10)))
                if key not in cluster_cache:
                    cluster_cache[key] = fit_cluster(zall, key[1], key[2], args.cluster_sample_rows)
                clusterer = cluster_cache[key]
            if clusterer is None:
                xall, xtest = zall, ztest
                labels = np.full(len(test), -1, dtype=np.int32)
                strengths = np.zeros(len(test), dtype=np.float32)
            else:
                xall, _, _ = cluster_features(clusterer, zall, model_mode)
                xtest, labels, strengths = cluster_features(clusterer, ztest, model_mode)
            model = Ridge(alpha=float(selected["alpha"])).fit(xall, y[all_train])
            prediction = model.predict(xtest)
            name = {"base": "prediction_pca_ridge", "hard": "prediction_hdbscan_hard", "soft": "prediction_hdbscan_soft"}[model_mode]
            out[name] = prediction
            daily = stock_day(frame.iloc[test], prediction, target)
            metrics = top20_metrics(daily)
            results.append({"dataset": args.dataset, "model": args.model, "prompt": args.prompt, "variant": args.variant, "feature": args.feature, "target": target, "test_year": test_year, "mode": args.mode, "model_mode": model_mode, "components": components, "alpha": float(selected["alpha"]), "min_cluster_size": selected.get("min_cluster_size"), "min_samples": selected.get("min_samples"), "validation_ls_bp": float(selected["score"]), "test_rank_ic": rank_ic(daily), **{f"test_{k}": v for k, v in metrics.items()}, "cluster_count": int(len(np.unique(labels[labels >= 0]))), "noise_fraction": float(np.mean(labels < 0)), "rows_fit": len(fit), "rows_validation": len(val), "rows_all_train": len(all_train), "rows_test": len(test)})
            pd.DataFrame({"row_index": frame.iloc[test].row_index.to_numpy(), "stock_id": frame.iloc[test].stock_id.to_numpy(), "entry_date": frame.iloc[test].entry_date.to_numpy(), target: y[test], "prediction": prediction, "cluster": labels, "membership_strength": strengths}).to_parquet(target_dir / f"{model_mode}_predictions.parquet", index=False)
        out.to_parquet(target_dir / "predictions.parquet", index=False)
    candidate_dir = args.output_root / args.dataset / args.model / args.prompt / args.variant / args.feature / args.mode
    candidate_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates_all).to_parquet(candidate_dir / f"candidates_{test_year}.parquet", index=False)
    summary_path = args.output_root / args.dataset / args.model / args.prompt / args.variant / args.feature / args.mode / f"metrics_{test_year}.parquet"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(summary_path, index=False)
    audit = {"matrix": str(args.matrix) if args.matrix else None, "metadata": str(args.metadata) if args.metadata else None, "embedding_root": str(args.embedding_root) if args.embedding_root else None, "pooled_feature": args.pooled_feature, "panel": str(args.panel), "shape": list(matrix.shape), "test_year": test_year, "mode": args.mode, "targets": list(TARGETS), "row_alignment_rows": int(len(frame)), "prompt": args.prompt, "model": args.model, "variant": args.variant, "feature": args.feature, "expected_rows": args.expected_rows}
    (summary_path.parent / f"audit_{test_year}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test_year": test_year, "rows": len(frame), "results": len(results), "output": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--matrix", type=Path)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--embedding-root", type=Path)
    p.add_argument("--pooled-feature", default=None)
    p.add_argument("--expected-rows", type=int, default=None)
    p.add_argument("--max-matrix-gib", type=float, default=64.0)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--test-year", type=int, required=True)
    p.add_argument("--mode", choices=("fixed", "tuned"), required=True)
    p.add_argument("--pca-sample-rows", type=int, default=20000)
    p.add_argument("--cluster-sample-rows", type=int, default=0, help="0 fits HDBSCAN on all training rows")
    run(p.parse_args())
