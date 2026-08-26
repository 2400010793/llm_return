"""Leakage-safe Qwen cluster diagnostics with linear prediction heads only."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import align_embeddings_to_panel, load_pooled_embeddings
from src.models.return_prediction import (
    evaluate_stock_day_predictions,
    finite_target,
    fit_preprocessor,
    make_return_regressor,
    return_regressor_candidates,
    select_return_regressor,
    select_ridge_alpha,
)


SEEDS = (42, 43, 44, 45, 46)
K_VALUES = (4, 8, 12)
RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 50.0, 100.0)


def metric_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    def finite(name: str, fallback: float) -> float:
        value = float(metrics.get(name, np.nan))
        return value if np.isfinite(value) else fallback
    return finite("rank_ic_mean", -np.inf), finite("oos_r2_vs_historical_mean", -np.inf), -finite("mae", np.inf)


def cluster_features(values: np.ndarray, model: MiniBatchKMeans, mode: str) -> np.ndarray:
    if mode == "base":
        return values
    distances = model.transform(values).astype(np.float32)
    if mode == "distance":
        extra = distances
    elif mode == "probability":
        squared = np.square(distances, dtype=np.float32)
        nearest = squared.min(axis=1)
        scale = float(np.median(nearest[nearest > 0])) if np.any(nearest > 0) else 1.0
        extra = softmax(-squared / max(scale, 1e-12), axis=1).astype(np.float32)
    else:
        raise ValueError(f"unsupported cluster feature mode: {mode}")
    return np.concatenate((values, extra), axis=1)


def select_linear(xfit: np.ndarray, yfit: np.ndarray, xval: np.ndarray,
                  validation: pd.DataFrame, target: str, regressor: str):
    if regressor == "ridge":
        selected = select_ridge_alpha(
            xfit, yfit, xval, validation, RIDGE_ALPHAS,
            stock_column="stock_id", date_column="entry_date", target_column=target,
            min_stocks_per_day=5, max_expansions=0,
        )
        return {"alpha": selected.alpha, "seed": 42}, selected.metrics, selected.audit
    selected = select_return_regressor(
        "huber_sgd", xfit, yfit, xval, validation,
        stock_column="stock_id", date_column="entry_date", target_column=target,
        min_stocks_per_day=5, seed=42,
        candidates=return_regressor_candidates("huber_sgd", seed=42, stage="fine"),
    )
    return selected.params, selected.metrics, selected.audit


def stability(xfit: np.ndarray, k: int) -> float:
    rng = np.random.default_rng(42)
    sample = np.arange(len(xfit))
    if len(sample) > 50000:
        sample = np.sort(rng.choice(sample, 50000, replace=False))
    values = xfit[sample]
    labels = []
    for seed in SEEDS:
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048,
                                n_init=1, max_iter=200).fit(values)
        labels.append(model.labels_)
    return float(np.mean([adjusted_rand_score(labels[a], labels[b])
                          for a, b in combinations(range(len(labels)), 2)]))


def stock_days(frame: pd.DataFrame, positions: np.ndarray, predictions: np.ndarray,
               target: str, test_year: int) -> pd.DataFrame:
    selected = frame.iloc[positions][["stock_id", "entry_date", target]].copy()
    selected["prediction"] = predictions
    result = selected.groupby(["stock_id", "entry_date"], as_index=False).agg(
        actual_return=(target, "first"), prediction=("prediction", "mean"),
        n_announcements=("prediction", "size"),
    )
    result["test_year"] = test_year
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", default="next_day_open_to_open_winsor_residual")
    parser.add_argument("--fit-window-years", type=int, default=6)
    parser.add_argument("--validation-window-years", type=int, default=2)
    args = parser.parse_args()

    frozen = json.loads(args.selected.read_text(encoding="utf-8"))["selected"]
    qwen = [row for row in frozen if row["model"] == "qwen3_embedding_8b"]
    chosen = max(qwen, key=lambda row: metric_key(row))
    variant = str(chosen["variant"])
    panel = pd.read_parquet(args.panel)
    embeddings = load_pooled_embeddings(
        args.embedding_root, model="qwen3_embedding_8b", variant=variant,
        feature="full_mean", require_complete_rows=350577, max_matrix_gib=16,
    )
    frame, matrix = align_embeddings_to_panel(panel, embeddings, panel_row_index_column="row_index")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="raise")
    years = sorted(int(year) for year in frame["entry_date"].dt.year.unique())
    history = args.fit_window_years + args.validation_window_years
    returns, finite = finite_target(frame[args.target])
    all_predictions, all_base_predictions = [], []
    diagnostic_rows, increment_rows, selection_rows = [], [], []

    for position in range(history, len(years)):
        test_year = years[position]
        in_years = years[position-history:position]
        fit_years = in_years[:args.fit_window_years]
        validation_years = in_years[args.fit_window_years:]
        year_values = frame["entry_date"].dt.year.to_numpy()
        fit = np.flatnonzero(finite & np.isin(year_values, fit_years))
        validation = np.flatnonzero(finite & np.isin(year_values, validation_years))
        all_train = np.flatnonzero(finite & np.isin(year_values, in_years))
        test = np.flatnonzero(finite & (year_values == test_year))
        xfit, xval, _ = fit_preprocessor(matrix[fit], matrix[validation], reducer="pca", components=128, seed=42)
        candidates = []
        base_cache = {}
        for regressor in ("ridge", "huber_sgd"):
            scaler = StandardScaler().fit(xfit)
            xf, xv = scaler.transform(xfit).astype(np.float32), scaler.transform(xval).astype(np.float32)
            params, metrics, audit = select_linear(xf, returns[fit], xv, frame.iloc[validation], args.target, regressor)
            row = {"mode": "base", "k": 0, "regressor": regressor, "params": params,
                   "metrics": metrics, "audit": audit}
            candidates.append(row); base_cache[regressor] = row
        for k in K_VALUES:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048,
                                 n_init=3, max_iter=200).fit(xfit)
            ari = stability(xfit, k)
            fit_labels = km.labels_
            for cluster in range(k):
                values = returns[fit][fit_labels == cluster]
                diagnostic_rows.append({
                    "test_year": test_year, "scope": "fit", "k": k, "cluster": cluster,
                    "rows": len(values), "mean_return": float(np.mean(values)),
                    "median_return": float(np.median(values)),
                    "positive_rate": float(np.mean(values > 0)), "ari_mean": ari,
                    "industry_available": False,
                })
            for mode in ("distance", "probability"):
                raw_fit, raw_val = cluster_features(xfit, km, mode), cluster_features(xval, km, mode)
                scaler = StandardScaler().fit(raw_fit)
                xf = scaler.transform(raw_fit).astype(np.float32)
                xv = scaler.transform(raw_val).astype(np.float32)
                for regressor in ("ridge", "huber_sgd"):
                    params, metrics, audit = select_linear(
                        xf, returns[fit], xv, frame.iloc[validation], args.target, regressor,
                    )
                    candidates.append({"mode": mode, "k": k, "regressor": regressor,
                                       "params": params, "metrics": metrics, "audit": audit,
                                       "ari_mean": ari})
        cluster_winner = max((row for row in candidates if row["mode"] != "base"),
                             key=lambda row: metric_key(row["metrics"]))
        base_winner = max(base_cache.values(), key=lambda row: metric_key(row["metrics"]))
        for kind, selected in (("cluster", cluster_winner), ("base", base_winner)):
            selection_rows.append({"test_year": test_year, "kind": kind, "variant": variant,
                                   "mode": selected["mode"], "k": selected["k"],
                                   "regressor": selected["regressor"],
                                   "params": json.dumps(selected["params"], sort_keys=True),
                                   **{f"validation_{key}": value for key, value in selected["metrics"].items()}})

        xall, xtest, _ = fit_preprocessor(matrix[all_train], matrix[test], reducer="pca", components=128, seed=42)
        fold_outputs = {}
        for kind, selected in (("cluster", cluster_winner), ("base", base_winner)):
            if selected["mode"] == "base":
                raw_all, raw_test = xall, xtest
            else:
                km = MiniBatchKMeans(n_clusters=int(selected["k"]), random_state=42,
                                     batch_size=2048, n_init=3, max_iter=200).fit(xall)
                raw_all = cluster_features(xall, km, str(selected["mode"]))
                raw_test = cluster_features(xtest, km, str(selected["mode"]))
            scaler = StandardScaler().fit(raw_all)
            xa, xt = scaler.transform(raw_all).astype(np.float32), scaler.transform(raw_test).astype(np.float32)
            model = make_return_regressor(str(selected["regressor"]), selected["params"])
            model.fit(xa, returns[all_train])
            predictions = np.asarray(model.predict(xt), dtype=float)
            metrics, _ = evaluate_stock_day_predictions(
                frame.iloc[test], predictions, historical_mean=float(np.mean(returns[all_train])),
                stock_column="stock_id", date_column="entry_date", target_column=args.target,
                min_stocks_per_day=5,
            )
            fold_outputs[kind] = (predictions, metrics)
        cluster_prediction, cluster_metrics = fold_outputs["cluster"]
        base_prediction, base_metrics = fold_outputs["base"]
        all_predictions.append(stock_days(frame, test, cluster_prediction, args.target, test_year))
        all_base_predictions.append(stock_days(frame, test, base_prediction, args.target, test_year))
        increment_rows.append({"test_year": test_year, "variant": variant,
                               **{f"cluster_{key}": value for key, value in cluster_metrics.items()},
                               **{f"base_{key}": value for key, value in base_metrics.items()},
                               "rank_ic_increment": cluster_metrics["rank_ic_mean"] - base_metrics["rank_ic_mean"]})

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.concat(all_predictions, ignore_index=True).to_parquet(args.output_root / "stock_day_predictions.parquet", index=False)
    pd.concat(all_base_predictions, ignore_index=True).to_parquet(args.output_root / "base_stock_day_predictions.parquet", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(args.output_root / "cluster_diagnostics.csv", index=False)
    pd.DataFrame(increment_rows).to_csv(args.output_root / "cluster_linear_increment.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.output_root / "cluster_validation_selection.csv", index=False)
    (args.output_root / "summary.json").write_text(json.dumps({
        "variant": variant, "fit_window_years": args.fit_window_years,
        "validation_window_years": args.validation_window_years,
        "test_years": [int(year) for year in years[history:]],
        "industry_diagnostics": "unavailable: the frozen O2O panel has no industry column",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
