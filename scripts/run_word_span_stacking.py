"""Evaluate leakage-safe stacking of the existing word-span predictions.

The base predictions are already out-of-sample predictions. This script fits
only a second-level model on historical prediction/return pairs. For every
test year it uses six training years, two validation years, and the following
test year. Features are daily cross-sectional percentile ranks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

PROMPTS = ("profit", "excess_return", "return", "loss")
WORD_SPANS = {"profit": "profit_span", "excess_return": "excess_span", "return": "plain_return_span", "loss": "loss_span"}


def cross_sectional_rank(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result.groupby("entry_date")[column].rank(pct=True)
    return result


def rank_ic(frame: pd.DataFrame, score: str) -> float:
    values = []
    for _, group in frame.groupby("entry_date"):
        if len(group) >= 3 and group[score].nunique() > 1 and group.actual_return.nunique() > 1:
            values.append(group[score].corr(group.actual_return, method="spearman"))
    return float(np.nanmean(values)) if values else np.nan


def portfolio(frame: pd.DataFrame, score: str, fraction: float = 0.20) -> dict[str, float]:
    previous: dict[str, float] = {}
    gross_values, net_values, costs = [], [], []
    for _, group in frame.groupby("entry_date", sort=True):
        group = group.sort_values([score, "stock_id"], kind="mergesort")
        n = min(max(1, int(np.floor(len(group) * fraction))), len(group) // 2)
        if n < 1:
            continue
        weights = {str(x): 1.0 / n for x in group.tail(n).stock_id}
        outcomes = dict(zip(group.stock_id.astype(str), group.actual_return.astype(float)))
        names = set(previous) | set(weights)
        bought = sum(max(weights.get(x, 0) - previous.get(x, 0), 0) for x in names)
        sold = sum(max(previous.get(x, 0) - weights.get(x, 0), 0) for x in names)
        cost = 0.0005 * bought + 0.001 * sold
        gross_values.append(sum(w * outcomes.get(x, 0) for x, w in weights.items()))
        net_values.append(gross_values[-1] - cost)
        costs.append(cost)
        previous = weights
    if not net_values:
        return {"long_gross": np.nan, "long_net": np.nan, "long_cost": np.nan}
    liquidation = 0.001 * sum(previous.values())
    return {"long_gross": float(np.sum(gross_values)), "long_net": float(np.sum(net_values) - liquidation), "long_cost": float(np.sum(costs) + liquidation)}


def load_features(root: Path, variant: str, models: tuple[str, ...]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model in models:
        for prompt in PROMPTS:
            path = root / "results" / "regression" / f"{prompt}_{model}_{variant}_{WORD_SPANS[prompt]}" / "stock_day_predictions.stock_day_predictions.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            data = pd.read_parquet(path, columns=["stock_id", "entry_date", "actual_return", "prediction"]).rename(columns={"prediction": f"{model}_{prompt}"})
            data["entry_date"] = pd.to_datetime(data["entry_date"])
            if merged is None:
                merged = data
            else:
                merged = merged.merge(data, on=["stock_id", "entry_date"], how="inner", validate="one_to_one", suffixes=("", "_other"))
                if not np.allclose(merged.actual_return, merged.actual_return_other, equal_nan=True):
                    raise ValueError("actual_return differs between prediction files")
                merged = merged.drop(columns="actual_return_other")
    assert merged is not None
    columns = [f"{model}_{prompt}" for model in models for prompt in PROMPTS]
    return cross_sectional_rank(merged, columns)


def candidates(families: set[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    if "linear" in families:
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            result[f"ridge_a{alpha:g}"] = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        for epsilon in (1.1, 1.35, 1.75):
            result[f"huber_e{epsilon:g}"] = make_pipeline(StandardScaler(), HuberRegressor(epsilon=epsilon, alpha=0.0001, max_iter=500))
        for alpha in (0.001, 0.01, 0.1):
            result[f"elastic_a{alpha:g}"] = make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=3000))
    if "tree" in families:
        result["catboost"] = CatBoostRegressor(
            iterations=300, depth=5, learning_rate=0.03, loss_function="RMSE",
            random_seed=42, verbose=False, thread_count=4,
        )
        result["lightgbm"] = LGBMRegressor(
            n_estimators=300, num_leaves=15, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=4, verbosity=-1,
        )
        result["xgboost"] = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            objective="reg:squarederror", random_state=42, n_jobs=4,
        )
    return result


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    models = tuple(args.models.split(","))
    if not models or any(model not in ("roberta", "bge_m3") for model in models):
        raise ValueError("--models must contain roberta and/or bge_m3")
    frame = load_features(args.root, args.variant, models)
    years = sorted(frame.entry_date.dt.year.unique())
    rows = []
    for test_year in years:
        train_years = [y for y in years if test_year - args.train_years - args.validation_years <= y < test_year - args.validation_years]
        validation_years = [y for y in years if test_year - args.validation_years <= y < test_year]
        if len(train_years) < args.train_years or len(validation_years) < args.validation_years:
            continue
        train = frame[frame.entry_date.dt.year.isin(train_years)]
        validation = frame[frame.entry_date.dt.year.isin(validation_years)]
        test = frame[frame.entry_date.dt.year == test_year].copy()
        columns = [f"{model}_{prompt}" for model in models for prompt in PROMPTS]
        single_scores = {column: rank_ic(validation.assign(score=validation[column]), "score") for column in columns}
        best_single = max(single_scores, key=single_scores.get)
        candidate_scores = {}
        for name, estimator in candidates(args.families).items():
            estimator.fit(train[columns], train.actual_return)
            candidate_scores[name] = rank_ic(validation.assign(score=estimator.predict(validation[columns])), "score")
        best_model = max(candidate_scores, key=candidate_scores.get)
        specs = {"best_single": best_single, "rank_equal": "rank_equal", "stacking": best_model}
        for method, choice in specs.items():
            if method == "best_single":
                test["score"] = test[choice]
            elif method == "rank_equal":
                test["score"] = test[columns].mean(axis=1)
            else:
                estimator = candidates(args.families)[choice]
                estimator.fit(train[columns], train.actual_return)
                test["score"] = estimator.predict(test[columns])
            rows.append({"test_year": int(test_year), "train_years": ",".join(map(str, train_years)), "validation_years": ",".join(map(str, validation_years)), "method": method, "selected_model": choice, "rows": len(test), "rankic": rank_ic(test, "score"), **portfolio(test, "score")})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", choices=("short", "masked_short"), default="masked_short")
    parser.add_argument("--models", default="roberta,bge_m3")
    parser.add_argument("--train-years", type=int, default=6)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--families", default="linear,tree", help="Candidate families: linear, tree")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.families = {family.strip() for family in args.families.split(",") if family.strip()}
    if not args.families or not args.families <= {"linear", "tree"}:
        parser.error("--families must contain linear and/or tree")
    output = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    overall = output.groupby("method", as_index=False).agg(test_years=("test_year", "nunique"), rankic=("rankic", "mean"), long_gross=("long_gross", "mean"), long_net=("long_net", "mean"), long_cost=("long_cost", "mean"), positive_years=("long_net", lambda x: int((x > 0).sum())))
    overall.to_csv(args.output.with_name(args.output.stem + "_overall.csv"), index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()