from pathlib import Path

from scripts.build_next1_strong_manifest import build_rows


def test_strong_screen_manifest_is_bounded_and_direct_next1(tmp_path: Path) -> None:
    (tmp_path / "data/processed/pooled_embeddings_v3").mkdir(parents=True)
    rows = build_rows(tmp_path, phase="screen")
    assert len(rows) == 20
    assert {row["classifier"] for row in rows} == {
        "hist_gradient_boosting", "extra_trees", "xgboost", "lightgbm", "catboost",
    }
    assert {row["search_stage"] for row in rows} == {"coarse"}
    assert {row["train_target"] for row in rows} == {"next_day_return"}
    assert {row["evaluation_target"] for row in rows} == {"next_day_return"}


def test_strong_fine_manifest_accepts_selected_models_and_seeds(tmp_path: Path) -> None:
    (tmp_path / "data/processed/pooled_embeddings_v3").mkdir(parents=True)
    rows = build_rows(
        tmp_path, phase="fine", classifiers=("lightgbm", "xgboost"),
        seeds=(13, 42),
    )
    assert len(rows) == 16
    assert {row["search_stage"] for row in rows} == {"fine"}
    assert {row["seed"] for row in rows} == {13, 42}
