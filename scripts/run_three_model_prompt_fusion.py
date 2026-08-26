#!/usr/bin/env python3
"""Fuse strict OOS PCA32 factors for one outer test year."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge


MODELS = ("roberta", "bge_m3", "qwen3_embedding_8b")
PROMPTS = ("profit", "return", "excess_return", "loss")
REPRESENTATIONS = ("token", "body")


def rank_ic(frame: pd.DataFrame, score: str) -> float:
    values = []
    for _, group in frame.groupby("entry_date", sort=False):
        if len(group) >= 5 and group[score].nunique() > 1:
            value = spearmanr(group[score], group.actual_return).statistic
            if np.isfinite(value): values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def normalize_daily(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        mean = result.groupby("entry_date")[column].transform("mean")
        std = result.groupby("entry_date")[column].transform("std").replace(0, np.nan)
        result[column] = ((result[column] - mean) / std).fillna(0.0)
    return result


def load_factor(root: Path, model: str, prompt: str, representation: str, year: int, split: str):
    path = root / "rolling" / model / prompt / representation / str(year) / f"{split}_stock_day_predictions.parquet"
    frame = pd.read_parquet(path)
    frame = frame[frame.method.eq("ridge_pca32")][["stock_id", "entry_date", "actual_return", "prediction"]].copy()
    frame["entry_date"] = pd.to_datetime(frame.entry_date).dt.normalize()
    name = f"{model}__{prompt}__{representation}"
    return frame.rename(columns={"prediction": name}), name


def merge_factors(root: Path, year: int, split: str):
    merged, names = None, []
    for model in MODELS:
        for prompt in PROMPTS:
            for representation in REPRESENTATIONS:
                frame, name = load_factor(root, model, prompt, representation, year, split)
                names.append(name)
                if merged is None:
                    merged = frame
                else:
                    merged = merged.merge(frame, on=["stock_id", "entry_date"], how="inner", validate="one_to_one", suffixes=("", "_new"))
                    if not np.allclose(merged.actual_return, merged.actual_return_new, equal_nan=True):
                        raise ValueError(f"actual return mismatch: {name}")
                    merged = merged.drop(columns="actual_return_new")
    assert merged is not None
    return normalize_daily(merged.dropna(subset=["actual_return", *names]), names), names


def groups(names: list[str]) -> dict[str, list[str]]:
    result = {}
    for model in MODELS:
        result[f"model_{model}_tokens"] = [name for name in names if name.startswith(model + "__") and name.endswith("__token")]
    for prompt in PROMPTS:
        result[f"prompt_{prompt}_models"] = [name for name in names if f"__{prompt}__token" in name]
    result["all_12_tokens"] = [name for name in names if name.endswith("__token")]
    result["all_12_bodies"] = [name for name in names if name.endswith("__body")]
    result["all_24_token_body"] = list(names)
    return result


def fixed_meta(kind: str):
    if kind == "ridge":
        return Ridge(alpha=100.0), {"alpha": 100.0}
    if kind == "elasticnet":
        return ElasticNet(
            alpha=1e-4, l1_ratio=0.5, max_iter=10000, random_state=42,
        ), {"alpha": 1e-4, "l1_ratio": 0.5}
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
            l2_regularization=10.0, random_state=42,
        ), {"learning_rate": 0.05, "max_iter": 200,
             "max_leaf_nodes": 15, "l2_regularization": 10.0}
    raise ValueError(kind)


def fit_meta(kind: str, validation: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    model, parameters = fixed_meta(kind)
    model.fit(validation[features], validation["actual_return"])
    validation_prediction = model.predict(validation[features])
    score_frame = validation[["entry_date", "actual_return"]].copy()
    score_frame["score"] = validation_prediction
    return model.predict(test[features]), {
        "fixed_parameters": parameters,
        "training_years": sorted(validation.entry_date.dt.year.unique().astype(int).tolist()),
        "in_sample_training_rank_ic": rank_ic(score_frame, "score"),
        "hyperparameter_search": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "COMPLETED").is_file():
        print(json.dumps({"resumed": True, "output": str(args.output)})); return
    validation, names = merge_factors(args.root, args.test_year, "validation")
    test, test_names = merge_factors(args.root, args.test_year, "test")
    if names != test_names: raise ValueError("validation/test factor names differ")
    outputs, audit = [], []
    for group, features in groups(names).items():
        factor_ic = {feature: rank_ic(validation, feature) for feature in features}
        best = max(features, key=lambda feature: factor_ic[feature])
        methods = {"equal_weight": test[features].mean(axis=1).to_numpy(), "best_single": test[best].to_numpy()}
        selections = {"equal_weight": {"features": features}, "best_single": {"factor": best, "validation_rank_ic": factor_ic[best]}}
        for kind in ("ridge", "elasticnet", "hist_gradient_boosting"):
            methods[kind], selections[kind] = fit_meta(kind, validation, test, features)
        for method, prediction in methods.items():
            frame = test[["stock_id", "entry_date", "actual_return"]].copy()
            frame["prediction"] = prediction; frame["group"] = group; frame["method"] = method
            outputs.append(frame)
            audit.append({"group": group, "method": method, **selections[method]})
    stage = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    pd.concat(outputs, ignore_index=True).to_parquet(stage / "test_stock_day_predictions.parquet", index=False)
    (stage / "selection.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    manifest = {"format_version": "three_model_prompt_fusion_fixed_v2", "test_year": args.test_year,
                "fusion_training_years": sorted(validation.entry_date.dt.year.unique().astype(int).tolist()),
                "hyperparameter_search": False, "factor_count": len(names),
                "groups": {key: value for key, value in groups(names).items()}, "validation_rows": len(validation), "test_rows": len(test)}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (stage / "COMPLETED").write_text("three_model_prompt_fusion_fixed_v2\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    stage.replace(args.output)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
