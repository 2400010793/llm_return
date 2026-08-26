import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.run_pooled_embedding_regression import PAPER_RIDGE_ALPHAS, main
from src.evaluation.prediction_metrics import daily_rank_ic
from src.models.return_prediction import (
    aggregate_stock_day_predictions,
    make_return_regressor,
    return_regressor_candidates,
    select_return_regressor,
    select_ridge_alpha,
)


def test_fine_regression_grid_expands_without_changing_coarse_default() -> None:
    coarse_huber = return_regressor_candidates("huber_sgd", seed=42)
    fine_huber = return_regressor_candidates("huber_sgd", seed=42, stage="fine")
    coarse_mlp = return_regressor_candidates("small_mlp", seed=42)
    fine_mlp = return_regressor_candidates("small_mlp", seed=42, stage="fine")
    assert len(coarse_huber) == 6
    assert len(fine_huber) == 25
    assert len(coarse_mlp) == 8
    assert len(fine_mlp) == 16


def test_paper_ridge_alpha_grid_is_fixed_and_explicit() -> None:
    assert PAPER_RIDGE_ALPHAS == (
        1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 50.0, 100.0,
    )


def test_stock_day_aggregation_is_order_invariant() -> None:
    frame = pd.DataFrame({
        "stock_id": ["A", "A", "B"],
        "entry_date": pd.to_datetime(["2026-01-02"] * 3),
        "next_day_return": [0.01, 0.01, -0.02],
    })
    predictions = np.array([0.03, 0.01, -0.01])
    first = aggregate_stock_day_predictions(
        frame, predictions, stock_column="stock_id", date_column="entry_date",
        target_column="next_day_return",
    )
    second = aggregate_stock_day_predictions(
        frame.iloc[::-1], predictions[::-1], stock_column="stock_id",
        date_column="entry_date", target_column="next_day_return",
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[first.stock_id.eq("A"), "prediction"].item() == pytest.approx(0.02)
    assert first.loc[first.stock_id.eq("A"), "n_announcements"].item() == 2


def test_stock_day_aggregation_rejects_inconsistent_targets() -> None:
    frame = pd.DataFrame({
        "stock_id": ["A", "A"],
        "entry_date": pd.to_datetime(["2026-01-02"] * 2),
        "next_day_return": [0.01, 0.02],
    })
    with pytest.raises(ValueError, match="inconsistent targets"):
        aggregate_stock_day_predictions(
            frame, np.array([0.0, 0.0]), stock_column="stock_id",
            date_column="entry_date", target_column="next_day_return",
        )


def test_daily_ic_supports_platform_pearson_without_changing_spearman_default() -> None:
    frame = pd.DataFrame({
        "entry_date": pd.to_datetime(["2026-01-02"] * 4),
        "actual_return": [0.0, 1.0, 2.0, 3.0],
        "prediction": [0.0, 1.0, 2.0, 10.0],
    })
    default = daily_rank_ic(frame, min_stocks=4)
    pearson = daily_rank_ic(frame, min_stocks=4, method="pearson")
    assert default["rank_ic_mean"] == pytest.approx(1.0)
    assert pearson["rank_ic_mean"] < default["rank_ic_mean"]
    with pytest.raises(ValueError, match="spearman or pearson"):
        daily_rank_ic(frame, min_stocks=4, method="kendall")


def test_alpha_expansion_keeps_prior_candidates(monkeypatch) -> None:
    frame = pd.DataFrame({
        "stock_id": ["A", "B", "C", "D", "E"],
        "entry_date": pd.to_datetime(["2025-01-02"] * 5),
        "next_day_return": [-2.0, -1.0, 0.0, 1.0, 2.0],
    })
    calls = []

    def fake_evaluate(_frame, _predictions, **_kwargs):
        alpha = calls[-1]
        # Initial upper bound 10 wins, but expanded 100 is worse. Selection
        # must return 10 rather than expanding repeatedly to 1000.
        score = {1.0: 0.1, 10.0: 0.3, 100.0: 0.2}[alpha]
        return {"rank_ic_mean": score, "oos_r2_vs_historical_mean": 0.0, "mse": 1.0}, frame

    class FakeRidge:
        def __init__(self, alpha):
            self.alpha = float(alpha)

        def fit(self, _x, _y):
            calls.append(self.alpha)
            return self

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr("src.models.return_prediction.Ridge", FakeRidge)
    monkeypatch.setattr(
        "src.models.return_prediction.evaluate_stock_day_predictions", fake_evaluate,
    )
    selection = select_ridge_alpha(
        np.ones((5, 2)), np.arange(5.0), np.ones((5, 2)), frame, [1.0, 10.0],
        stock_column="stock_id", date_column="entry_date",
        target_column="next_day_return", min_stocks_per_day=5, max_expansions=2,
    )
    assert selection.alpha == 10.0
    assert calls == [1.0, 10.0, 100.0]


@pytest.mark.parametrize(
    ("regressor", "params"),
    [
        ("ridge", {"alpha": 1.0, "seed": 42}),
        ("elasticnet_sgd", {"alpha": 1e-5, "l1_ratio": 0.2, "seed": 42}),
        ("huber_sgd", {"alpha": 1e-5, "epsilon": 0.05, "seed": 42}),
        ("small_mlp", {
            "hidden_layer_sizes": (8,), "alpha": 1e-4,
            "max_iter": 20, "seed": 42,
        }),
    ],
)
def test_return_regressor_factory_predicts_finite_values(regressor, params) -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 4))
    y = x[:, 0] * 0.02 + rng.normal(scale=0.001, size=len(x))
    model = make_return_regressor(regressor, params)
    model.fit(x, y)
    predictions = model.predict(x[:5])
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()


def test_nonridge_selection_uses_stock_day_validation() -> None:
    rng = np.random.default_rng(7)
    x_fit = rng.normal(size=(30, 3))
    y_fit = x_fit[:, 0] * 0.01
    x_validation = rng.normal(size=(10, 3))
    frame = pd.DataFrame({
        "stock_id": [f"S{i}" for i in range(10)],
        "entry_date": pd.to_datetime(["2025-01-02"] * 10),
        "next_day_return": x_validation[:, 0] * 0.01,
    })
    selection = select_return_regressor(
        "elasticnet_sgd", x_fit, y_fit, x_validation, frame,
        stock_column="stock_id", date_column="entry_date",
        target_column="next_day_return", min_stocks_per_day=5,
        candidates=[{"alpha": 1e-5, "l1_ratio": 0.2, "seed": 42}],
    )
    assert selection.regressor == "elasticnet_sgd"
    assert selection.metrics["stock_days"] == 10
    assert len(selection.audit) == 1


def test_small_mlp_selection_accepts_mapping_candidates() -> None:
    rng = np.random.default_rng(11)
    x_fit = rng.normal(size=(30, 3))
    y_fit = x_fit[:, 0] * 0.01
    x_validation = rng.normal(size=(10, 3))
    frame = pd.DataFrame({
        "stock_id": [f"S{i}" for i in range(10)],
        "entry_date": pd.to_datetime(["2025-01-02"] * 10),
        "next_day_return": x_validation[:, 0] * 0.01,
    })
    selection = select_return_regressor(
        "small_mlp", x_fit, y_fit, x_validation, frame,
        stock_column="stock_id", date_column="entry_date",
        target_column="next_day_return", min_stocks_per_day=5,
        candidates=[{
            "hidden_layer_sizes": (4,), "alpha": 1e-4,
            "max_iter": 2, "early_stopping_patience": 1, "seed": 42,
        }],
    )
    assert selection.regressor == "small_mlp"
    assert selection.params["selected_epoch"] >= 1
    assert len(selection.audit[0]["epoch_history"]) >= 1


def test_small_stock_day_run_persists_models_predictions_and_portfolio(tmp_path, monkeypatch):
    stocks = [f"S{i}" for i in range(6)]
    records = []
    embeddings = []
    row_index = 1
    for year in range(2018, 2027):
        for stock_number, stock in enumerate(stocks):
            target = (stock_number - 2.5) * 0.01 + (year - 2022) * 0.0001
            for announcement in range(2):
                records.append({
                    "row_index": row_index,
                    "stock_id": stock,
                    "entry_date": f"{year}-06-03",
                    "next_day_return": target,
                })
                embeddings.append([stock_number, year - 2018, announcement])
                row_index += 1
    matrix = np.asarray(embeddings, dtype=np.float32)
    rows = len(records)
    embedding_root = tmp_path / "embeddings"
    part = embedding_root / "shard-0" / "roberta" / "short"
    part.mkdir(parents=True)
    np.savez_compressed(
        part / "short_pooling.npz",
        prompt_mean=matrix,
        title_mean=matrix,
        body_mean=matrix,
        title_body_mean=matrix,
        full_mean=matrix,
    )
    (part / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": index}) + "\n" for index in range(1, rows + 1)),
        encoding="utf-8",
    )
    (part / "summary.json").write_text(json.dumps({
        "model": "roberta", "variant": "short", "rows": rows,
        "outputs": {
            key: [rows, 3]
            for key in ("prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean")
        },
    }), encoding="utf-8")
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame(records).to_parquet(panel_path, index=False)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "run_pooled_embedding_regression.py", str(panel_path),
        "--embedding-root", str(embedding_root), "--model", "roberta",
        "--variant", "short", "--feature", "full_mean",
        "--target-column", "next_day_return", "--alphas", "0.1,1",
        "--max-alpha-expansions", "0", "--expected-embedding-rows", str(rows),
        "--output", str(output),
    ])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["design"]["prediction_unit"] == "stock_day_mean_of_announcement_predictions"
    assert report["results"][0]["stock_days"] == 6
    assert report["results"][0]["rank_ic_days"] == 1
    assert report["results"][0]["portfolio"]["n_days"] == 1
    stock_days = pd.read_parquet(output.with_suffix(".stock_day_predictions.parquet"))
    announcements = pd.read_parquet(output.with_suffix(".announcement_predictions.parquet"))
    assert len(stock_days) == 6
    assert len(announcements) == 12
    assert stock_days["n_announcements"].eq(2).all()
    bundle = output.with_suffix(".artifacts")
    assert (bundle / "COMPLETED").is_file()
    assert (bundle / "test_year_2026" / "test_quintile_portfolio.parquet").is_file()
    model = joblib.load(bundle / "test_year_2026" / "final_model.joblib")
    assert model.alpha in {0.1, 1.0}

    model_mtime = (bundle / "test_year_2026" / "final_model.joblib").stat().st_mtime_ns
    main()
    assert (bundle / "test_year_2026" / "final_model.joblib").stat().st_mtime_ns == model_mtime


def test_screen_mode_needs_only_eight_years_and_writes_no_test_predictions(
    tmp_path, monkeypatch,
):
    records, vectors = [], []
    row_index = 1
    for year in range(2018, 2026):
        for stock_number in range(6):
            records.append({
                "row_index": row_index, "stock_id": f"S{stock_number}",
                "entry_date": f"{year}-06-03",
                "next_day_return": (stock_number - 2.5) * 0.01,
            })
            vectors.append([stock_number, year - 2018, 1.0])
            row_index += 1
    matrix = np.asarray(vectors, dtype=np.float32)
    root = tmp_path / "embeddings"
    part = root / "shard-0" / "roberta" / "short"
    part.mkdir(parents=True)
    np.savez_compressed(
        part / "short_pooling.npz",
        **{name: matrix for name in (
            "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
        )},
    )
    (part / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": i}) + "\n" for i in range(1, row_index)),
        encoding="utf-8",
    )
    (part / "summary.json").write_text(json.dumps({
        "model": "roberta", "variant": "short", "rows": len(records),
        "outputs": {name: [len(records), 3] for name in (
            "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
        )},
    }), encoding="utf-8")
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(records).to_parquet(panel, index=False)
    output = tmp_path / "screen.json"
    monkeypatch.setattr(sys, "argv", [
        "run_pooled_embedding_regression.py", str(panel),
        "--embedding-root", str(root), "--model", "roberta",
        "--variant", "short", "--feature", "full_mean",
        "--run-mode", "screen", "--regressor", "huber_sgd",
        "--target-column", "next_day_return",
        "--expected-embedding-rows", str(len(records)), "--output", str(output),
    ])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["design"]["run_mode"] == "screen"
    assert report["results"][0]["validation_years"] == [2024, 2025]
    assert report["results"][0]["regressor"] == "huber_sgd"
    assert report["announcement_predictions"] is None
    assert report["stock_day_predictions"] is None
    assert not output.with_suffix(".announcement_predictions.parquet").exists()
    assert (output.with_suffix(".artifacts") / "screen" / "validation_model.joblib").is_file()


def test_configurable_three_one_one_windows_produce_five_folds(tmp_path, monkeypatch):
    records, vectors = [], []
    row_index = 1
    for year in range(2018, 2027):
        for stock_number in range(6):
            records.append({
                "row_index": row_index, "stock_id": f"S{stock_number}",
                "entry_date": f"{year}-06-03",
                "next_day_return": (stock_number - 2.5) * 0.01,
            })
            vectors.append([stock_number, year - 2018, 1.0])
            row_index += 1
    matrix = np.asarray(vectors, dtype=np.float32)
    root = tmp_path / "embeddings"
    part = root / "shard-0" / "roberta" / "short"
    part.mkdir(parents=True)
    np.savez_compressed(part / "short_pooling.npz", full_mean=matrix)
    (part / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": i}) + "\n" for i in range(1, row_index)),
        encoding="utf-8",
    )
    (part / "summary.json").write_text(json.dumps({
        "model": "roberta", "variant": "short", "rows": len(records),
        "outputs": {"full_mean": [len(records), 3]},
    }), encoding="utf-8")
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(records).to_parquet(panel, index=False)
    output = tmp_path / "rolling.json"
    monkeypatch.setattr(sys, "argv", [
        "run_pooled_embedding_regression.py", str(panel),
        "--embedding-root", str(root), "--model", "roberta",
        "--variant", "short", "--feature", "full_mean",
        "--target-column", "next_day_return", "--alphas", "1",
        "--fit-window-years", "3", "--validation-window-years", "1",
        "--expected-embedding-rows", str(len(records)), "--output", str(output),
    ])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["design"]["fit_years"] == 3
    assert report["design"]["validation_years"] == 1
    assert [row["test_year"] for row in report["results"]] == [2022, 2023, 2024, 2025, 2026]
    assert report["results"][0]["fit_years"] == [2018, 2019, 2020]
    assert report["results"][0]["validation_years"] == [2021]
