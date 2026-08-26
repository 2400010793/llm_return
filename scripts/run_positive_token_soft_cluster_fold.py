"""Run one leakage-safe soft-cluster fold for a prompt direction token.

The four supported matrices contain the contextualized embeddings of 盈利,
超额收益, 收益, or 亏损.  Cluster geometry is learned without labels.  Cluster
return means, PCA dimensions, k, shrinkage, temperature, and response target are
selected on the historical validation window only.  Two predictions are kept:
one selected by validation RankIC and one by validation long-short Sharpe.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


SEEDS = (17, 29, 42, 71, 113)
COMPONENTS = (32, 64)
K_VALUES = (2, 4, 6, 8)
SHRINKAGES = (25.0, 100.0, 500.0)
TEMPERATURES = (0.0, 0.5, 1.0, 2.0)
TARGET_MODES = ("raw", "demeaned", "rank")


def response_targets(frame: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    """Build article-aligned targets from unique stock-day observations."""
    base = frame[["stock_id", "entry_date", target]].copy()
    base["entry_date"] = pd.to_datetime(base["entry_date"]).dt.normalize()
    stock_day = base.groupby(["stock_id", "entry_date"], as_index=False)[target].mean()
    stock_day["raw"] = pd.to_numeric(stock_day[target], errors="coerce")
    stock_day["demeaned"] = stock_day["raw"] - stock_day.groupby("entry_date")["raw"].transform("mean")
    stock_day["rank"] = stock_day.groupby("entry_date")["raw"].rank(pct=True) - 0.5
    aligned = base[["stock_id", "entry_date"]].merge(
        stock_day[["stock_id", "entry_date", *TARGET_MODES]],
        on=["stock_id", "entry_date"], how="left", validate="many_to_one",
    )
    return {name: aligned[name].to_numpy(dtype=float) for name in TARGET_MODES}


def shrunk_means(labels: np.ndarray, values: np.ndarray, k: int, shrinkage: float) -> np.ndarray:
    global_mean = float(np.mean(values))
    result = np.full(k, global_mean, dtype=np.float64)
    for cluster in range(k):
        selected = labels == cluster
        count = int(selected.sum())
        if count:
            result[cluster] = (float(values[selected].sum()) + shrinkage * global_mean) / (count + shrinkage)
    return result


def soft_predict(distances: np.ndarray, means: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 0:
        return means[np.argmin(distances, axis=1)]
    squared = np.square(np.asarray(distances, dtype=np.float64))
    # Scaling by only the nearest-centroid distance makes all non-nearest
    # posterior weights underflow in moderate dimensions.  The robust scale
    # across every candidate centroid preserves a genuinely soft assignment.
    positive = squared[squared > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    logits = -squared / max(temperature * scale, 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ means


def stock_days(frame: pd.DataFrame, positions: np.ndarray, prediction: np.ndarray, target: str) -> pd.DataFrame:
    selected = frame.iloc[positions][["stock_id", "entry_date", target]].copy()
    selected["prediction"] = prediction
    selected["entry_date"] = pd.to_datetime(selected["entry_date"]).dt.normalize()
    return selected.groupby(["stock_id", "entry_date"], as_index=False).agg(
        actual_return=(target, "mean"), prediction=("prediction", "mean"), news=("prediction", "size"),
    )


def validation_metrics(days: pd.DataFrame) -> dict[str, float]:
    ics, spreads, long_excess = [], [], []
    for _, group in days.groupby("entry_date"):
        if len(group) < 10 or group["prediction"].nunique() < 2:
            continue
        ic = spearmanr(group["prediction"], group["actual_return"]).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
        ranks = group["prediction"].rank(method="first")
        buckets = pd.qcut(ranks, 5, labels=False, duplicates="drop")
        if buckets.nunique() == 5:
            top_return = float(group.loc[buckets.eq(4), "actual_return"].mean())
            spreads.append(top_return - float(group.loc[buckets.eq(0), "actual_return"].mean()))
            long_excess.append(top_return - float(group["actual_return"].mean()))
    spread = np.asarray(spreads, dtype=float)
    long_values = np.asarray(long_excess, dtype=float)
    spread_sharpe = float(spread.mean() / spread.std(ddof=1) * np.sqrt(252)) if len(spread) > 1 and spread.std(ddof=1) > 0 else float("nan")
    long_sharpe = float(long_values.mean() / long_values.std(ddof=1) * np.sqrt(252)) if len(long_values) > 1 and long_values.std(ddof=1) > 0 else float("nan")
    return {
        "daily_rank_ic": float(np.mean(ics)) if ics else float("nan"),
        "spread_mean": float(spread.mean()) if len(spread) else float("nan"),
        "spread_sharpe": spread_sharpe,
        "spread_positive_days": float(np.mean(spread > 0)) if len(spread) else float("nan"),
        "long_excess_mean": float(long_values.mean()) if len(long_values) else float("nan"),
        "long_excess_sharpe": long_sharpe,
        "long_excess_positive_days": float(np.mean(long_values > 0)) if len(long_values) else float("nan"),
    }


def fit_projection(
    matrix: np.ndarray, positions: np.ndarray, sample_rows: int, projection_mode: str,
) -> tuple[PCA | None, StandardScaler]:
    selected = positions
    if len(selected) > sample_rows:
        selected = np.sort(np.random.default_rng(42).choice(selected, sample_rows, replace=False))
    values = np.asarray(matrix[selected], dtype=np.float32)
    if projection_mode == "raw":
        return None, StandardScaler().fit(values)
    count = min(max(COMPONENTS), values.shape[1], len(values) - 1)
    pca = PCA(n_components=count, svd_solver="randomized", random_state=42, iterated_power=3).fit(values)
    scaler = StandardScaler().fit(pca.transform(values))
    return pca, scaler


def project(matrix: np.ndarray, positions: np.ndarray, pca: PCA | None, scaler: StandardScaler, components: int) -> np.ndarray:
    values = np.asarray(matrix[positions], dtype=np.float32)
    if pca is None:
        return scaler.transform(values).astype(np.float32)
    return scaler.transform(pca.transform(values))[:, :components].astype(np.float32)


def select_candidates(matrix: np.ndarray, frame: pd.DataFrame, responses: dict[str, np.ndarray], target: str,
                      fit_idx: np.ndarray, val_idx: np.ndarray, sample_rows: int,
                      projection_mode: str) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    pca, scaler = fit_projection(matrix, fit_idx, sample_rows, projection_mode)
    rows: list[dict[str, object]] = []
    component_values = COMPONENTS if pca is not None else (matrix.shape[1],)
    for components in component_values:
        x_fit = project(matrix, fit_idx, pca, scaler, components)
        x_val = project(matrix, val_idx, pca, scaler, components)
        for k in K_VALUES:
            models = [MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=3, max_iter=200).fit(x_fit) for seed in SEEDS]
            fit_labels = [model.predict(x_fit) for model in models]
            val_labels = [model.predict(x_val) for model in models]
            ari = np.mean([adjusted_rand_score(val_labels[a], val_labels[b]) for a in range(len(SEEDS)) for b in range(a + 1, len(SEEDS))])
            distances = [model.transform(x_val) for model in models]
            for target_mode, shrinkage, temperature in product(TARGET_MODES, SHRINKAGES, TEMPERATURES):
                prediction = np.zeros(len(val_idx), dtype=np.float64)
                for labels, distance in zip(fit_labels, distances):
                    means = shrunk_means(labels, responses[target_mode][fit_idx], k, shrinkage)
                    prediction += soft_predict(distance, means, temperature)
                prediction /= len(SEEDS)
                metrics = validation_metrics(stock_days(frame, val_idx, prediction, target))
                rows.append({
                    "components": components, "k": k, "target_mode": target_mode,
                    "shrinkage": shrinkage, "temperature": temperature,
                    "ari": float(ari), **metrics,
                })
    table = pd.DataFrame(rows)
    valid_ic = table.dropna(subset=["daily_rank_ic"])
    valid_spread = table.dropna(subset=["spread_sharpe"])
    valid_long = table.dropna(subset=["long_excess_sharpe"])
    if valid_ic.empty or valid_spread.empty or valid_long.empty:
        raise ValueError("validation window produced no selectable soft-cluster candidate")
    ic_winner = valid_ic.sort_values(["daily_rank_ic", "ari", "k"], ascending=[False, False, True]).iloc[0]
    spread_winner = valid_spread.sort_values(["spread_sharpe", "daily_rank_ic", "ari", "k"], ascending=[False, False, False, True]).iloc[0]
    long_winner = valid_long.sort_values(["long_excess_sharpe", "daily_rank_ic", "ari", "k"], ascending=[False, False, False, True]).iloc[0]
    fields = ("components", "k", "target_mode", "shrinkage", "temperature", "ari", "daily_rank_ic", "spread_mean", "spread_sharpe", "spread_positive_days", "long_excess_mean", "long_excess_sharpe", "long_excess_positive_days")
    return {
        "rank_ic": {field: (int(ic_winner[field]) if field in {"components", "k"} else ic_winner[field].item() if hasattr(ic_winner[field], "item") else ic_winner[field]) for field in fields},
        "long_short": {field: (int(spread_winner[field]) if field in {"components", "k"} else spread_winner[field].item() if hasattr(spread_winner[field], "item") else spread_winner[field]) for field in fields},
        "long_only": {field: (int(long_winner[field]) if field in {"components", "k"} else long_winner[field].item() if hasattr(long_winner[field], "item") else long_winner[field]) for field in fields},
    }, table


def final_prediction(matrix: np.ndarray, responses: dict[str, np.ndarray], train_idx: np.ndarray,
                     test_idx: np.ndarray, selected: dict[str, object], sample_rows: int,
                     projection_mode: str) -> np.ndarray:
    pca, scaler = fit_projection(matrix, train_idx, sample_rows, projection_mode)
    components, k = int(selected["components"]), int(selected["k"])
    x_train = project(matrix, train_idx, pca, scaler, components)
    x_test = project(matrix, test_idx, pca, scaler, components)
    result = np.zeros(len(test_idx), dtype=np.float64)
    values = responses[str(selected["target_mode"])][train_idx]
    for seed in SEEDS:
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=3, max_iter=200).fit(x_train)
        means = shrunk_means(model.predict(x_train), values, k, float(selected["shrinkage"]))
        result += soft_predict(model.transform(x_test), means, float(selected["temperature"]))
    return result / len(SEEDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", choices=("profit", "excess_return", "return", "loss"), required=True)
    parser.add_argument(
        "--model", choices=("roberta", "bge_m3", "qwen3_embedding_8b"), required=True
    )
    parser.add_argument("--variant", choices=("short", "masked_short"), required=True)
    parser.add_argument("--representation", default="")
    parser.add_argument("--projection-mode", choices=("pca", "raw"), default="pca")
    parser.add_argument("--test-year", type=int, choices=range(2018, 2027), required=True)
    parser.add_argument("--target", default="next_day_return")
    parser.add_argument("--pca-sample-rows", type=int, default=20000)
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    metadata = pd.read_parquet(args.metadata)
    matrix = np.load(args.matrix, mmap_mode="r")
    if len(frame) != len(matrix) or len(metadata) != len(frame):
        raise ValueError("panel, metadata, and matrix row counts differ")
    if not np.array_equal(frame["row_index"].to_numpy(), metadata["row_index"].to_numpy()):
        raise ValueError("matrix metadata is not aligned to panel row_index")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"]).dt.normalize()
    years = frame["entry_date"].dt.year.to_numpy()
    response = response_targets(frame, args.target)
    finite = np.isfinite(response["raw"])
    fit_years = list(range(args.test_year - 8, args.test_year - 2))
    validation_years = list(range(args.test_year - 2, args.test_year))
    all_train_years = list(range(args.test_year - 8, args.test_year))
    fit_idx = np.flatnonzero(finite & np.isin(years, fit_years))
    val_idx = np.flatnonzero(finite & np.isin(years, validation_years))
    train_idx = np.flatnonzero(finite & np.isin(years, all_train_years))
    test_idx = np.flatnonzero(finite & (years == args.test_year))
    selected, search = select_candidates(
        matrix, frame, response, args.target, fit_idx, val_idx,
        args.pca_sample_rows, args.projection_mode,
    )

    output = args.output_root / args.prompt / args.model / args.variant
    if args.representation:
        output = output / args.representation
    if args.projection_mode == "raw":
        output = output / "raw"
    output = output / str(args.test_year)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed fold: {output}")
    stage = output.with_name(f".{output.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    try:
        predictions = frame.iloc[test_idx][["row_index", "stock_id", "entry_date", args.target]].copy()
        for objective, winner in selected.items():
            values = final_prediction(
                matrix, response, train_idx, test_idx, winner,
                args.pca_sample_rows, args.projection_mode,
            )
            predictions[f"prediction_{objective}"] = values
        aggregated = predictions.groupby(["stock_id", "entry_date"], as_index=False).agg(
            actual_return=(args.target, "mean"),
            prediction_rank_ic=("prediction_rank_ic", "mean"),
            prediction_long_short=("prediction_long_short", "mean"),
            prediction_long_only=("prediction_long_only", "mean"),
            news=("row_index", "size"),
        )
        search.to_parquet(stage / "validation_search.parquet", index=False)
        predictions.to_parquet(stage / "news_predictions.parquet", index=False)
        aggregated.to_parquet(stage / "stock_day_predictions.parquet", index=False)
        metrics = {
            "format_version": "positive_token_soft_cluster_fold_v1",
            "prompt": args.prompt, "model": args.model, "variant": args.variant,
            "representation": args.representation or None,
            "projection_mode": args.projection_mode,
            "test_year": args.test_year, "target": args.target,
            "fit_years": fit_years, "validation_years": validation_years,
            "all_train_years": all_train_years, "selected": selected,
            "rows": {"fit": len(fit_idx), "validation": len(val_idx), "all_train": len(train_idx), "test": len(test_idx), "stock_days": len(aggregated)},
        }
        (stage / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (stage / "COMPLETED").write_text("positive_token_soft_cluster_fold_v1\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "selected": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
