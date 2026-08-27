#!/usr/bin/env python3
"""Aggregate one model's four prompt-token OOS factors with tree models.

The first-level factors are produced by the fixed PCA32/Ridge fair pipeline.
This stage never loads raw embeddings and never mixes models: one invocation
uses exactly one model and the four prompt-token factors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from xgboost import XGBRegressor


PROMPTS = ("profit", "return", "excess_return", "loss")
KEYS = ["stock_id", "entry_date"]


def load_year(root: Path, model: str, year: int) -> pd.DataFrame:
    merged = None
    for prompt in PROMPTS:
        path = root / "rolling" / model / prompt / "token" / str(year) / "test_stock_day_predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame = frame[frame["method"].eq("ridge_pca32")].copy()
        if frame.empty:
            raise ValueError(f"no ridge_pca32 rows in {path}")
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="raise").dt.normalize()
        frame["stock_id"] = frame["stock_id"].astype(str)
        feature = f"prompt_{prompt}"
        frame = frame[KEYS + ["actual_return", "prediction"]].rename(columns={"prediction": feature})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_new"))
            if not np.allclose(merged["actual_return"], merged["actual_return_new"], equal_nan=True, rtol=0.0, atol=0.0):
                raise ValueError(f"actual return mismatch for {model}/{year}/{prompt}")
            merged = merged.drop(columns="actual_return_new")
    assert merged is not None
    features = [f"prompt_{prompt}" for prompt in PROMPTS]
    if merged[features].isna().any().any() or not np.isfinite(merged[features].to_numpy(float)).all():
        raise ValueError(f"non-finite factor for {model}/{year}")
    merged["year"] = int(year)
    return merged.sort_values(KEYS, kind="stable").reset_index(drop=True)


def rank_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for feature in features:
        result[feature] = result.groupby("entry_date", sort=False)[feature].rank(method="average", pct=True)
    return result


def rank_ic(frame: pd.DataFrame, score: str) -> float:
    values = []
    for _, group in frame.groupby("entry_date", sort=False):
        if len(group) >= 5 and group[score].nunique() > 1 and group["actual_return"].nunique() > 1:
            value = spearmanr(group[score], group["actual_return"]).statistic
            if np.isfinite(value):
                values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def metrics(frame: pd.DataFrame, score: str) -> dict[str, float]:
    daily = []
    top, bottom, spreads = [], [], []
    for _, group in frame.groupby("entry_date", sort=False):
        group = group.dropna(subset=[score, "actual_return"])
        if len(group) < 10 or group[score].nunique() < 2:
            continue
        value = spearmanr(group[score], group["actual_return"]).statistic
        if np.isfinite(value):
            daily.append(float(value))
        n = max(1, int(np.ceil(len(group) * 0.2)))
        ordered = group.sort_values([score, "stock_id"], kind="mergesort")
        lo = float(ordered.head(n)["actual_return"].mean())
        hi = float(ordered.tail(n)["actual_return"].mean())
        bottom.append(lo); top.append(hi); spreads.append(hi - lo)
    values = np.asarray(daily, dtype=float)
    return {
        "rank_ic": float(values.mean()) if len(values) else float("nan"),
        "rank_ic_ir": float(values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) else float("nan"),
        "rank_ic_days": int(len(values)),
        "positive_rank_ic_day_fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
        "top20_mean_bp": float(np.mean(top) * 10000) if top else float("nan"),
        "bottom20_mean_bp": float(np.mean(bottom) * 10000) if bottom else float("nan"),
        "long_short_mean_bp": float(np.mean(spreads) * 10000) if spreads else float("nan"),
    }


def estimators(threads: int) -> dict[str, object]:
    return {
        "xgboost": XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=10.0, objective="reg:squarederror",
            tree_method="hist", max_bin=127, random_state=42, n_jobs=threads,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300, num_leaves=15, max_depth=5, learning_rate=0.03,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=10.0, random_state=42, n_jobs=threads,
            verbosity=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=300, depth=5, learning_rate=0.03, l2_leaf_reg=10.0,
            random_seed=42, loss_function="RMSE", verbose=False,
            thread_count=threads, allow_writing_files=False,
        ),
    }


def run_outer(all_rows: pd.DataFrame, model: str, test_year: int, history: int, validation_years: int, threads: int):
    train_years = list(range(test_year - history, test_year))
    val_years = train_years[-validation_years:]
    fit_years = train_years[:-validation_years]
    fit = rank_features(all_rows[all_rows.year.isin(fit_years)], list(f"prompt_{p}" for p in PROMPTS))
    validation = rank_features(all_rows[all_rows.year.isin(val_years)], list(f"prompt_{p}" for p in PROMPTS))
    test = rank_features(all_rows[all_rows.year.eq(test_year)], list(f"prompt_{p}" for p in PROMPTS))
    features = [f"prompt_{p}" for p in PROMPTS]
    if min(len(fit), len(validation), len(test)) == 0:
        raise ValueError(f"empty split {model}/{test_year}: fit={len(fit)} val={len(validation)} test={len(test)}")

    val_scores = {name: rank_ic(validation, name) for name in features}
    best_single = max(features, key=lambda name: val_scores[name])
    outputs, selections = [], []

    def add_output(name: str, prediction: np.ndarray, family: str, validation_rank_ic):
        frame = test[KEYS + ["actual_return"]].copy()
        frame["prediction"] = prediction
        frame["method"] = name
        outputs.append(frame)
        selections.append({"method": name, "family": family, "validation_rank_ic": validation_rank_ic})

    add_output("equal_weight", test[features].mean(axis=1).to_numpy(float), "equal_weight", float(np.mean(list(val_scores.values()))))
    add_output("best_single", test[best_single].to_numpy(float), "identity", val_scores[best_single])

    validation_tree_scores = {}
    for family, template in estimators(threads).items():
        estimator = template
        estimator.fit(fit[features], fit["actual_return"])
        validation_prediction = estimator.predict(validation[features])
        validation_tree_scores[family] = rank_ic(
            validation.assign(prediction=validation_prediction), "prediction"
        )
    best_family = max(validation_tree_scores, key=validation_tree_scores.get)
    refit = pd.concat([fit, validation], ignore_index=True)
    for family, template in estimators(threads).items():
        template.fit(refit[features], refit["actual_return"])
        add_output(family, template.predict(test[features]), family, validation_tree_scores[family])
    add_output("tree_selected", next(x["prediction"].to_numpy(float) for x in outputs if x["method"].eq(best_family).all()), best_family, validation_tree_scores[best_family])
    return pd.concat(outputs, ignore_index=True), {
        "model": model, "test_year": int(test_year), "fit_years": fit_years,
        "validation_years": val_years, "test_years": [test_year],
        "best_single": best_single, "validation_tree_rank_ic": validation_tree_scores,
        "selected_tree_family": best_family, "factor_count": len(features),
        "fit_rows": len(fit), "validation_rows": len(validation), "test_rows": len(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--history-years", type=int, required=True)
    parser.add_argument("--validation-years", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if (args.output / "COMPLETED").is_file():
        print(json.dumps({"resumed": True, "output": str(args.output)})); return
    root_manifest = json.loads((args.fair_root / "intersection" / "manifest.json").read_text(encoding="utf-8"))
    factor_years = [int(year) for year in root_manifest["test_years"]]
    all_rows = pd.concat([load_year(args.fair_root, args.model, year) for year in factor_years], ignore_index=True)
    available_years = set(factor_years)
    years = [
        year for year in factor_years
        if set(range(year - args.history_years, year)).issubset(available_years)
    ]
    if not years:
        raise ValueError(
            f"no test year has {args.history_years} complete prior OOS factor years; "
            f"available={factor_years}"
        )
    outputs, audits, metric_rows = [], [], []
    for year in years:
        predictions, audit = run_outer(all_rows, args.model, year, args.history_years, args.validation_years, args.threads)
        predictions["test_year"] = year
        outputs.append(predictions); audits.append(audit)
        for method, group in predictions.groupby("method", sort=False):
            metric_rows.append({"dataset": args.dataset, "model": args.model, "test_year": year, "method": method, **metrics(group, "prediction")})
    stage = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    pd.concat(outputs, ignore_index=True).to_parquet(stage / "test_stock_day_predictions.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(stage / "metrics.csv", index=False)
    (stage / "selection.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    manifest = {
        "format_version": "per_model_prompt_token_tree_v1", "dataset": args.dataset,
        "model": args.model, "prompts": list(PROMPTS), "representation": "token",
        "first_level": "fair rolling ridge_pca32 OOS factors", "raw_embedding_regression": False,
        "history_years": args.history_years, "validation_years": args.validation_years,
        "tree_parameters": "same fixed XGBoost/LightGBM/CatBoost settings across models",
        "factor_years": factor_years, "years": years, "rows": len(all_rows),
        "software": {"xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__},
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage / "COMPLETED").write_text("per_model_prompt_token_tree_v1\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    stage.replace(args.output)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
