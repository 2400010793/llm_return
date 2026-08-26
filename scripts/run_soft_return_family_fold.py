"""Fit one rolling soft-cluster return-family predictor from `收益` embeddings."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.prompt_token_mechanisms import SEEDS
from src.evaluation.artifacts import atomic_json


K_VALUES = (2, 4, 6, 8)
SHRINKAGES = (25.0, 100.0, 500.0)
TEMPERATURES = (0.0, 0.5, 1.0, 2.0)
TARGETS = ("event_return_3d", "next_day_return")
FEATURE_MODES = ("unmasked", "masked", "fusion")


def _positions(frame: pd.DataFrame, window: dict[str, object], target: str) -> dict[str, np.ndarray]:
    years = pd.to_datetime(frame["entry_date"], errors="coerce").dt.year.to_numpy()
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    return {
        "fit": np.flatnonzero(finite & np.isin(years, window["fit_years"])),
        "validation": np.flatnonzero(finite & np.isin(years, window["validation_years"])),
        "all_train": np.flatnonzero(finite & np.isin(years, window["all_train_years"])),
        "test": np.flatnonzero(finite & (years == int(window["test_year"]))),
    }


def _features(
    unmasked: np.ndarray, masked: np.ndarray, positions: np.ndarray, mode: str,
) -> np.ndarray:
    base = np.asarray(unmasked[positions], dtype=np.float32)
    if mode == "unmasked":
        return base
    other = np.asarray(masked[positions], dtype=np.float32)
    if mode == "masked":
        return other
    return np.concatenate((base, other, other - base), axis=1)


def _fit_projection(
    unmasked: np.ndarray, masked: np.ndarray, positions: np.ndarray, mode: str,
    *, components: int, sample_rows: int,
) -> tuple[PCA, StandardScaler]:
    selected = positions
    if len(selected) > sample_rows:
        selected = np.sort(np.random.default_rng(42).choice(selected, sample_rows, replace=False))
    values = _features(unmasked, masked, selected, mode)
    count = min(components, values.shape[1], len(values) - 1)
    pca = PCA(
        n_components=count, svd_solver="randomized", random_state=42, iterated_power=3,
    ).fit(values)
    scaler = StandardScaler().fit(pca.transform(values))
    return pca, scaler


def _project(
    unmasked: np.ndarray, masked: np.ndarray, positions: np.ndarray, mode: str,
    pca: PCA, scaler: StandardScaler,
) -> np.ndarray:
    values = _features(unmasked, masked, positions, mode)
    return scaler.transform(pca.transform(values)).astype(np.float32)


def _shrunk_means(labels: np.ndarray, returns: np.ndarray, k: int, shrinkage: float) -> np.ndarray:
    global_mean = float(np.mean(returns))
    means = np.full(k, global_mean, dtype=np.float64)
    for cluster in range(k):
        selected = labels == cluster
        count = int(selected.sum())
        if count:
            means[cluster] = (
                float(returns[selected].sum()) + shrinkage * global_mean
            ) / (count + shrinkage)
    return means


def _soft_predict(distances: np.ndarray, means: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    if temperature == 0:
        return means[np.argmin(values, axis=1)]
    squared = np.square(values)
    nearest = np.min(squared, axis=1)
    scale = float(np.median(nearest[nearest > 0])) if np.any(nearest > 0) else 1.0
    logits = -squared / max(temperature * scale, 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ means


def _stock_days(
    frame: pd.DataFrame, positions: np.ndarray, predictions: np.ndarray, target: str,
) -> pd.DataFrame:
    selected = frame.iloc[positions][["stock_id", "entry_date", target]].copy()
    selected["prediction"] = predictions
    selected["entry_date"] = pd.to_datetime(selected["entry_date"])
    return selected.groupby(["stock_id", "entry_date"], as_index=False).agg(
        prediction=("prediction", "mean"), actual_return=(target, "mean"),
        news=("prediction", "size"),
    )


def _daily_metrics(stock_days: pd.DataFrame) -> dict[str, float | int]:
    correlations = []
    spreads = []
    for _, day in stock_days.groupby("entry_date"):
        if len(day) < 5 or day["prediction"].nunique() < 2:
            continue
        correlation = spearmanr(day["prediction"], day["actual_return"]).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        ranks = day["prediction"].rank(method="first")
        buckets = pd.qcut(ranks, q=min(3, len(day)), labels=False, duplicates="drop")
        if buckets.nunique() >= 2:
            spreads.append(float(
                day.loc[buckets == buckets.max(), "actual_return"].mean()
                - day.loc[buckets == buckets.min(), "actual_return"].mean()
            ))
    global_ic = spearmanr(stock_days["prediction"], stock_days["actual_return"]).statistic
    return {
        "stock_days": int(len(stock_days)), "days": int(stock_days["entry_date"].nunique()),
        "daily_ic": float(np.mean(correlations)) if correlations else float("nan"),
        "global_ic": float(global_ic) if np.isfinite(global_ic) else float("nan"),
        "daily_tercile_spread": float(np.mean(spreads)) if spreads else float("nan"),
        "direction_accuracy": float(np.mean(
            (stock_days["prediction"].to_numpy() > 0)
            == (stock_days["actual_return"].to_numpy() > 0)
        )),
    }


def _long_threshold_metrics(stock_days: pd.DataFrame, fraction: float) -> tuple[float, float, int]:
    values = []
    counts = []
    for _, day in stock_days.groupby("entry_date"):
        if len(day) < 5:
            continue
        cutoff = day["prediction"].quantile(1.0 - fraction)
        selected = day[day["prediction"] >= cutoff]
        if len(selected):
            values.append(float(selected["actual_return"].mean()))
            counts.append(int(len(selected)))
    return (float(np.mean(values)) if values else float("nan"),
            float(np.mean(np.asarray(values) > 0)) if values else float("nan"),
            int(np.mean(counts)) if counts else 0)


def _mean_ari(assignments: list[np.ndarray]) -> float:
    values = [
        adjusted_rand_score(assignments[left], assignments[right])
        for left in range(len(assignments)) for right in range(left + 1, len(assignments))
    ]
    return float(np.mean(values)) if values else 1.0


def _candidate_predictions(
    fit: np.ndarray, fit_returns: np.ndarray, other: np.ndarray,
    *, k: int,
) -> tuple[dict[tuple[float, float], np.ndarray], float]:
    accumulated = {
        (shrinkage, temperature): np.zeros(len(other), dtype=np.float64)
        for shrinkage, temperature in product(SHRINKAGES, TEMPERATURES)
    }
    assignments = []
    for seed in SEEDS:
        model = MiniBatchKMeans(
            n_clusters=k, random_state=int(seed), batch_size=2048,
            n_init=3, max_iter=200,
        ).fit(fit)
        fit_labels = model.predict(fit)
        assignments.append(model.predict(other))
        distances = model.transform(other)
        for shrinkage in SHRINKAGES:
            means = _shrunk_means(fit_labels, fit_returns, k, shrinkage)
            for temperature in TEMPERATURES:
                accumulated[(shrinkage, temperature)] += _soft_predict(
                    distances, means, temperature,
                )
    for key in accumulated:
        accumulated[key] /= len(SEEDS)
    return accumulated, _mean_ari(assignments)


def _select_candidate(
    fit: np.ndarray, fit_returns: np.ndarray, validation: np.ndarray,
    frame: pd.DataFrame, validation_positions: np.ndarray, target: str,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows = []
    for k in K_VALUES:
        predictions, ari = _candidate_predictions(fit, fit_returns, validation, k=k)
        for (shrinkage, temperature), values in predictions.items():
            metrics = _daily_metrics(_stock_days(
                frame, validation_positions, values, target,
            ))
            daily_ic = float(metrics["daily_ic"])
            selection_ic = daily_ic if np.isfinite(daily_ic) else float(metrics["global_ic"])
            score = selection_ic + 0.02 * ari - 0.001 * k
            for fraction in (0.50, 0.30, 0.20, 0.15, 0.10):
                long_return, positive_days, long_count = _long_threshold_metrics(
                    _stock_days(frame, validation_positions, values, target), fraction,
                )
                rows.append({
                "k": k, "shrinkage": shrinkage, "temperature": temperature,
                "long_fraction": fraction, "validation_long_return": long_return,
                "validation_long_positive_days": positive_days,
                "validation_long_count": long_count,
                "ari_mean": ari, "selection_score": score, **metrics,
                })
    table = pd.DataFrame(rows)
    base_winner = table.sort_values(
        ["selection_score", "daily_tercile_spread", "direction_accuracy", "k"],
        ascending=[False, False, False, True],
    ).drop_duplicates(["k", "shrinkage", "temperature"]).iloc[0]
    winner = table[
        (table["k"] == base_winner["k"])
        & (table["shrinkage"] == base_winner["shrinkage"])
        & (table["temperature"] == base_winner["temperature"])
    ].sort_values(["validation_long_return", "validation_long_positive_days", "long_fraction"], ascending=[False, False, True]).iloc[0]
    return {
        "k": int(winner["k"]), "shrinkage": float(winner["shrinkage"]),
        "temperature": float(winner["temperature"]),
        "validation_daily_ic": float(winner["daily_ic"]),
        "validation_global_ic": float(winner["global_ic"]),
        "validation_tercile_spread": float(winner["daily_tercile_spread"]),
        "validation_accuracy": float(winner["direction_accuracy"]),
        "ari_mean": float(winner["ari_mean"]),
        "selection_score": float(winner["selection_score"]),
        "long_fraction": float(winner["long_fraction"]),
        "validation_long_return": float(winner["validation_long_return"]),
        "validation_long_positive_days": float(winner["validation_long_positive_days"]),
    }, table


def _final_predictions(
    train: np.ndarray, train_returns: np.ndarray, others: list[np.ndarray],
    selected: dict[str, float | int],
) -> list[np.ndarray]:
    outputs = [np.zeros(len(values), dtype=np.float64) for values in others]
    k = int(selected["k"])
    for seed in SEEDS:
        model = MiniBatchKMeans(
            n_clusters=k, random_state=int(seed), batch_size=2048,
            n_init=3, max_iter=200,
        ).fit(train)
        labels = model.predict(train)
        means = _shrunk_means(
            labels, train_returns, k, float(selected["shrinkage"]),
        )
        for index, values in enumerate(others):
            outputs[index] += _soft_predict(
                model.transform(values), means, float(selected["temperature"]),
            )
    return [values / len(SEEDS) for values in outputs]


def run(args: argparse.Namespace) -> dict[str, object]:
    mechanism_cache = args.mechanism_cache_root / args.model / "short"
    manifest = json.loads((mechanism_cache / "manifest.json").read_text(encoding="utf-8"))
    window = next(
        row for row in manifest["windows"] if int(row["test_year"]) == args.test_year
    )
    frame = pd.read_parquet(mechanism_cache / "rows.parquet")
    scopes = _positions(frame, window, args.target)
    span_cache = args.span_cache_root / args.model / "short" / "return_span"
    unmasked = np.load(span_cache / "short_return_span.npy", mmap_mode="r")
    masked = np.load(span_cache / "masked_short_return_span.npy", mmap_mode="r")
    returns = pd.to_numeric(frame[args.target], errors="coerce").to_numpy(dtype=float)

    fit_pca, fit_scaler = _fit_projection(
        unmasked, masked, scopes["fit"], args.feature_mode,
        components=args.components, sample_rows=args.pca_sample_rows,
    )
    fit = _project(
        unmasked, masked, scopes["fit"], args.feature_mode, fit_pca, fit_scaler,
    )
    validation = _project(
        unmasked, masked, scopes["validation"], args.feature_mode, fit_pca, fit_scaler,
    )
    selected, validation_table = _select_candidate(
        fit, returns[scopes["fit"]], validation,
        frame, scopes["validation"], args.target,
    )

    final_pca, final_scaler = _fit_projection(
        unmasked, masked, scopes["all_train"], args.feature_mode,
        components=args.components, sample_rows=args.pca_sample_rows,
    )
    all_train = _project(
        unmasked, masked, scopes["all_train"], args.feature_mode, final_pca, final_scaler,
    )
    test = _project(
        unmasked, masked, scopes["test"], args.feature_mode, final_pca, final_scaler,
    )
    train_predictions, test_predictions = _final_predictions(
        all_train, returns[scopes["all_train"]], [all_train, test], selected,
    )
    train_stock_days = _stock_days(
        frame, scopes["all_train"], train_predictions, args.target,
    )
    test_stock_days = _stock_days(frame, scopes["test"], test_predictions, args.target)
    thresholds = np.quantile(train_stock_days["prediction"], [1 / 3, 2 / 3])
    test_stock_days["family"] = pd.cut(
        test_stock_days["prediction"],
        bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
        labels=["negative", "neutral", "positive"], include_lowest=True,
    ).astype(str)
    family = test_stock_days.groupby("family").agg(
        stock_days=("actual_return", "size"),
        mean_return=("actual_return", "mean"),
        median_return=("actual_return", "median"),
        positive_rate=("actual_return", lambda values: float((values > 0).mean())),
        mean_prediction=("prediction", "mean"),
    ).reset_index()
    family_lookup = family.set_index("family")
    negative = float(family_lookup.at["negative", "mean_return"])
    positive = float(family_lookup.at["positive", "mean_return"])
    round_trip_cost = args.one_way_cost_bps * 2 / 10_000.0
    test_metrics = _daily_metrics(test_stock_days)
    long_fraction = float(selected["long_fraction"])
    long_return, long_positive_days, long_count = _long_threshold_metrics(test_stock_days, long_fraction)
    test_metrics.update({
        "article_accuracy": float(np.mean(
            (test_predictions > 0) == (returns[scopes["test"]] > 0)
        )),
        "long_gross": positive, "long_net": positive - round_trip_cost,
        "short_gross": -negative, "short_net": -negative - round_trip_cost,
        "long_short_gross_on_gross": (positive - negative) / 2,
        "long_short_net_on_gross": (positive - negative) / 2 - round_trip_cost,
        "selected_long_fraction": long_fraction,
        "long_threshold_gross": long_return,
        "long_threshold_net": long_return - round_trip_cost,
        "long_threshold_positive_days": long_positive_days,
        "long_threshold_average_count": long_count,
    })

    destination = args.output_root / args.feature_mode / args.target / str(args.test_year)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite soft-family fold: {destination}")
    stage = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    try:
        validation_table.to_csv(stage / "validation_search.csv", index=False)
        prediction = frame.iloc[scopes["test"]][
            ["row_index", "stock_id", "entry_date", args.target]
        ].copy()
        prediction["predicted_return"] = test_predictions
        prediction.to_parquet(stage / "news_predictions.parquet", index=False)
        test_stock_days.to_parquet(stage / "stock_day_predictions.parquet", index=False)
        family.to_csv(stage / "family_returns.csv", index=False)
        report = {
            "format_version": "soft_return_family_fold_v2",
            "model": args.model,
            "feature_mode": args.feature_mode, "target": args.target,
            "test_year": args.test_year, "window": window,
            "selected": selected, "test_metrics": test_metrics,
            "family_thresholds": thresholds.tolist(),
            "one_way_cost_bps": args.one_way_cost_bps,
            "round_trip_cost_bps": args.one_way_cost_bps * 2,
            "rows": {name: int(len(values)) for name, values in scopes.items()},
        }
        atomic_json(stage / "metrics.json", report)
        (stage / "COMPLETED").write_text("soft_return_family_fold_v1\n", encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(destination)
        return {**report, "output": str(destination)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span-cache-root", type=Path, required=True)
    parser.add_argument("--mechanism-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), default="bge_m3")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--test-year", type=int, choices=range(2018, 2027), required=True)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--pca-sample-rows", type=int, default=10000)
    parser.add_argument("--one-way-cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
