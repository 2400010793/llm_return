"""Stack existing CNINFO RoBERTa/BGE-M3 OOS factors with tree models.

The level-one predictions must already be out of sample.  This runner uses
2018-2023 for meta-model fitting, 2024-2025 for model/feature-count selection,
and 2026 once for the final test.  Raw embedding matrices are never loaded.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


KEYS = ["stock_id", "entry_date"]
FIT_YEARS = tuple(range(2018, 2024))
VALIDATION_YEARS = (2024, 2025)
TEST_YEAR = 2026


def load_oos_factors(root: Path) -> tuple[pd.DataFrame, list[str], list[dict]]:
    paths = sorted(root.glob("*.stock_day_predictions.parquet"))
    if not paths:
        raise FileNotFoundError(f"no stock-day predictions found in {root}")
    merged: pd.DataFrame | None = None
    inputs = []
    for path in paths:
        feature = path.name.removesuffix(".stock_day_predictions.parquet")
        frame = pd.read_parquet(
            path, columns=[*KEYS, "actual_return", "prediction", "test_year"]
        ).rename(columns={"prediction": feature})
        frame["stock_id"] = frame["stock_id"].astype(str)
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="raise")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"duplicate stock-day keys: {path}")
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(
                frame, on=[*KEYS, "test_year"], how="inner", validate="one_to_one",
                suffixes=("", "_other"),
            )
            if not np.allclose(
                merged["actual_return"], merged["actual_return_other"],
                equal_nan=True, rtol=0.0, atol=0.0,
            ):
                raise ValueError(f"actual returns differ in {path}")
            merged = merged.drop(columns="actual_return_other")
        inputs.append({"feature": feature, "path": str(path), "rows": len(frame)})
    assert merged is not None
    features = [item["feature"] for item in inputs]
    if len(features) != 72:
        raise ValueError(f"expected 72 OOS factors, found {len(features)}")
    if len(merged) != inputs[0]["rows"]:
        raise ValueError(
            f"factor key sets are not identical: intersection={len(merged)}, "
            f"reference={inputs[0]['rows']}"
        )
    years = tuple(sorted(merged["test_year"].unique().tolist()))
    expected_years = tuple(range(2018, 2027))
    if years != expected_years:
        raise ValueError(f"expected years {expected_years}, found {years}")
    if merged[features].isna().any().any() or not np.isfinite(
        merged[features].to_numpy(dtype=np.float32)
    ).all():
        raise ValueError("factor predictions contain missing or non-finite values")
    merged = merged.sort_values(["entry_date", "stock_id"], kind="stable").reset_index(drop=True)
    for feature in features:
        merged[feature] = merged.groupby("entry_date", sort=False)[feature].rank(
            method="average", pct=True
        )
    return merged, features, inputs


def daily_rank_ic_vector(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    sums = np.zeros(len(features), dtype=np.float64)
    counts = np.zeros(len(features), dtype=np.int64)
    for _, group in frame.groupby("entry_date", sort=False):
        if len(group) < 5:
            continue
        y = group["actual_return"].rank(method="average", pct=True).to_numpy(dtype=np.float64)
        y -= y.mean()
        y_norm = np.linalg.norm(y)
        if y_norm <= 0:
            continue
        x = group[features].to_numpy(dtype=np.float64)
        x -= x.mean(axis=0, keepdims=True)
        denominator = np.linalg.norm(x, axis=0) * y_norm
        valid = denominator > 0
        values = np.zeros(len(features), dtype=np.float64)
        values[valid] = (x[:, valid].T @ y) / denominator[valid]
        sums[valid] += values[valid]
        counts[valid] += 1
    return {
        feature: float(sums[index] / counts[index]) if counts[index] else float("nan")
        for index, feature in enumerate(features)
    }


def score_metrics(frame: pd.DataFrame, score: str) -> dict[str, float]:
    daily = []
    for _, group in frame.groupby("entry_date", sort=False):
        if len(group) < 5 or group[score].nunique() < 2 or group["actual_return"].nunique() < 2:
            continue
        daily.append(group[score].corr(group["actual_return"], method="spearman"))
    values = np.asarray(daily, dtype=float)
    return {
        "rank_ic_mean": float(np.nanmean(values)) if len(values) else float("nan"),
        "rank_ic_median": float(np.nanmedian(values)) if len(values) else float("nan"),
        "rank_ic_std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan"),
        "rank_ic_days": int(np.isfinite(values).sum()),
        "positive_rank_ic_day_fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
    }


def portfolio_metrics(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, float]:
    rows = []
    for _, group in frame.groupby("entry_date", sort=False):
        group = group.dropna(subset=[score, "actual_return"])
        n = int(np.floor(len(group) * fraction))
        if n < 1 or len(group) < 2 * n:
            continue
        ordered = group.sort_values([score, "stock_id"], kind="stable")
        low = float(ordered.head(n)["actual_return"].mean())
        high = float(ordered.tail(n)["actual_return"].mean())
        rows.append((high, low, high - low))
    values = np.asarray(rows, dtype=float)
    return {
        "portfolio_days": int(len(values)),
        "top20_mean_bp": float(values[:, 0].mean() * 10_000) if len(values) else float("nan"),
        "bottom20_mean_bp": float(values[:, 1].mean() * 10_000) if len(values) else float("nan"),
        "long_short_mean_bp": float(values[:, 2].mean() * 10_000) if len(values) else float("nan"),
    }


def top_overlap(frame: pd.DataFrame, left: str, right: str, fraction: float) -> dict[str, float]:
    overlap, jaccard = [], []
    for _, group in frame.groupby("entry_date", sort=False):
        n = int(np.floor(len(group) * fraction))
        if n < 1:
            continue
        left_ids = set(group.nlargest(n, left)["stock_id"])
        right_ids = set(group.nlargest(n, right)["stock_id"])
        common = len(left_ids & right_ids)
        overlap.append(common / n)
        jaccard.append(common / max(len(left_ids | right_ids), 1))
    return {
        "top20_overlap_fraction": float(np.mean(overlap)) if overlap else float("nan"),
        "top20_jaccard": float(np.mean(jaccard)) if jaccard else float("nan"),
        "top20_overlap_days": len(overlap),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--feature-counts", default="4,8,16,32,72")
    parser.add_argument("--top-fraction", type=float, default=0.20)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, features, inputs = load_oos_factors(args.input_root)
    fit = frame[frame["test_year"].isin(FIT_YEARS)].copy()
    validation = frame[frame["test_year"].isin(VALIDATION_YEARS)].copy()
    test = frame[frame["test_year"].eq(TEST_YEAR)].copy()
    if min(len(fit), len(validation), len(test)) == 0:
        raise ValueError("one or more strict 6+2+1 windows are empty")

    train_ic = daily_rank_ic_vector(fit, features)
    feature_ranking = sorted(features, key=lambda value: train_ic[value], reverse=True)
    pd.DataFrame({
        "feature": feature_ranking,
        "fit_rank_ic": [train_ic[value] for value in feature_ranking],
        "fit_rank": range(1, len(feature_ranking) + 1),
    }).to_csv(args.output_dir / "fit_feature_ranking.csv", index=False)

    counts = sorted({
        min(int(value), len(features))
        for value in args.feature_counts.split(",") if value.strip()
    })
    if not counts or counts[0] < 1:
        raise ValueError("feature-counts must contain positive integers")
    x_fit_all = fit[features]
    x_validation_all = validation[features]
    y_fit = fit["actual_return"].to_numpy(dtype=float)
    candidate_rows = []
    selected_specs: dict[str, dict] = {}

    validation_single_ic = daily_rank_ic_vector(validation, features)
    best_single = max(features, key=lambda value: validation_single_ic[value])
    selected_specs["best_single"] = {"features": [best_single], "model": "identity"}

    for count in counts:
        selected = feature_ranking[:count]
        validation_score = validation[selected].mean(axis=1).to_numpy(dtype=float)
        score_frame = validation[[*KEYS, "actual_return"]].copy()
        score_frame["score"] = validation_score
        metrics = score_metrics(score_frame, "score")
        candidate_rows.append({
            "family": "rank_equal", "candidate": f"top_{count}",
            "feature_count": count, **metrics,
        })

    for count in counts:
        selected = feature_ranking[:count]
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_fit_all[selected], y_fit)
            score_frame = validation[[*KEYS, "actual_return"]].copy()
            score_frame["score"] = model.predict(x_validation_all[selected])
            candidate_rows.append({
                "family": "ridge", "candidate": f"top_{count}_alpha_{alpha:g}",
                "feature_count": count, "alpha": alpha,
                **score_metrics(score_frame, "score"),
            })

    tree_templates = estimators(args.threads)
    for family, template in tree_templates.items():
        for count in counts:
            selected = feature_ranking[:count]
            model = clone(template)
            model.fit(x_fit_all[selected], y_fit)
            score_frame = validation[[*KEYS, "actual_return"]].copy()
            score_frame["score"] = model.predict(x_validation_all[selected])
            candidate_rows.append({
                "family": family, "candidate": f"top_{count}",
                "feature_count": count, **score_metrics(score_frame, "score"),
            })

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["family", "rank_ic_mean"], ascending=[True, False]
    )
    candidates.to_csv(args.output_dir / "validation_search.csv", index=False)
    for family in ("rank_equal", "ridge", *tree_templates):
        row = candidates[candidates["family"].eq(family)].sort_values(
            "rank_ic_mean", ascending=False
        ).iloc[0]
        selected_specs[family] = {
            "features": feature_ranking[:int(row["feature_count"])],
            "model": family,
            "validation_rank_ic": float(row["rank_ic_mean"]),
            "alpha": float(row["alpha"]) if family == "ridge" else None,
        }
    best_tree = max(
        tree_templates, key=lambda family: selected_specs[family]["validation_rank_ic"]
    )
    selected_specs["tree_selected"] = {**selected_specs[best_tree], "selected_family": best_tree}

    all_train = frame[frame["test_year"].isin((*FIT_YEARS, *VALIDATION_YEARS))].copy()
    y_all = all_train["actual_return"].to_numpy(dtype=float)
    predictions = test[[*KEYS, "actual_return", "test_year"]].copy()
    importance_rows = []
    for method, spec in selected_specs.items():
        selected = spec["features"]
        if method == "best_single":
            values = test[selected[0]].to_numpy(dtype=float)
        elif method == "rank_equal":
            values = test[selected].mean(axis=1).to_numpy(dtype=float)
        else:
            family = spec.get("selected_family", spec["model"])
            if family == "ridge":
                model = make_pipeline(
                    StandardScaler(), Ridge(alpha=float(spec["alpha"]))
                )
            else:
                model = clone(tree_templates[family])
            model.fit(all_train[selected], y_all)
            values = np.asarray(model.predict(test[selected]), dtype=float)
            if family == "ridge":
                importance = np.abs(model[-1].coef_)
            else:
                importance = np.asarray(model.feature_importances_, dtype=float)
            for feature, value in zip(selected, importance):
                importance_rows.append({
                    "method": method, "fitted_family": family,
                    "feature": feature, "importance": float(value),
                })
        predictions[method] = values
    predictions.to_parquet(args.output_dir / "test_2026_predictions.parquet", index=False)
    pd.DataFrame(importance_rows).sort_values(
        ["method", "importance"], ascending=[True, False]
    ).to_csv(args.output_dir / "feature_importance.csv", index=False)

    metric_rows = []
    for method in selected_specs:
        metric_rows.append({
            "method": method,
            "fitted_family": selected_specs[method].get(
                "selected_family", selected_specs[method]["model"]
            ),
            "feature_count": len(selected_specs[method]["features"]),
            "validation_rank_ic": selected_specs[method].get(
                "validation_rank_ic",
                validation_single_ic[best_single] if method == "best_single" else None,
            ),
            **score_metrics(predictions, method),
            **portfolio_metrics(predictions, method, args.top_fraction),
            **top_overlap(predictions, method, "best_single", args.top_fraction),
        })
    metrics = pd.DataFrame(metric_rows).sort_values("rank_ic_mean", ascending=False)
    metrics.to_csv(args.output_dir / "test_2026_metrics.csv", index=False)
    factor_columns = list(selected_specs)
    predictions[factor_columns].corr(method="spearman").to_csv(
        args.output_dir / "test_2026_factor_spearman.csv"
    )

    spec = {
        "format_version": "cninfo_oos_tree_stacking_v1",
        "input_root": str(args.input_root), "input_factor_count": len(features),
        "common_stock_days": len(frame),
        "fit_years": list(FIT_YEARS), "validation_years": list(VALIDATION_YEARS),
        "test_year": TEST_YEAR, "feature_selection": "fit-period mean daily RankIC",
        "model_selection": "validation-period mean daily RankIC",
        "features": "daily cross-sectional percentile ranks of level-one OOS predictions",
        "selected_specs": selected_specs,
        "software": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        },
        "upstream_inputs": inputs,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 巨潮 RoBERTa/BGE-M3 OOS 树模型聚合", "",
        "## 协议", "",
        f"- 一级输入：{len(features)} 个已经滚动生成的 OOS 预测因子，共同股票日 {len(frame):,} 条。",
        "- 二级训练：2018--2023；验证选择：2024--2025；封存测试：2026。",
        "- 特征为每日横截面百分位排名；特征排序只看训练期，模型和特征数只看验证期。",
        "- XGBoost、LightGBM、CatBoost 均在训练+验证期重拟合后一次性预测 2026。", "",
        "## 2026 样本外结果", "", metrics.to_markdown(index=False, floatfmt=".6f"), "",
        "## 解释", "",
        "- `best_single` 是验证期选出的单一一级因子。",
        "- `rank_equal` 和 `ridge` 是非树聚合基准。",
        "- `tree_selected` 只依据验证期在三种树模型中选择，不依据 2026 测试结果。",
        "- Top20 收益为同一巨潮公告股票池内的毛收益，尚未扣交易成本。", "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
