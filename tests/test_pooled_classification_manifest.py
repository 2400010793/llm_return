from pathlib import Path

import pytest

from scripts.build_pooled_classification_manifest import build_rows


@pytest.fixture
def complete_embedding_tree(tmp_path: Path) -> Path:
    root = tmp_path
    for relative in (
        "data/processed/pooled_embeddings_v3/shard-0/roberta/short",
        "data/processed/pooled_embeddings_v3/shard-0/roberta/masked_short",
    ):
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            '{"model":"roberta","variant":"' + directory.name + '","rows":4}',
            encoding="utf-8",
        )
        (directory / "metadata.jsonl").write_text("\n", encoding="utf-8")
        (directory / "short_pooling.npz").write_bytes(b"probe")
    return root


def test_screening_excludes_prompt_tokens_by_default(complete_embedding_tree):
    rows = build_rows(
        complete_embedding_tree, "screening", expected_rows=4,
        models=("roberta",), variants=("short", "masked_short"),
    )
    assert len(rows) == 12
    assert not any(str(row["feature"]).startswith("prompt_tokens") for row in rows)


def test_prompt_tokens_require_explicit_opt_in(complete_embedding_tree):
    rows = build_rows(
        complete_embedding_tree, "screening", expected_rows=4,
        models=("roberta",), variants=("short",), include_prompt_tokens=True,
    )
    assert len(rows) == 7
    assert sum(row["feature"] == "prompt_tokens_flat" for row in rows) == 1


@pytest.mark.parametrize("phase", ["linear", "triple_models"])
def test_unreduced_linear_svm_is_disabled_by_default(complete_embedding_tree, phase):
    rows = build_rows(
        complete_embedding_tree, phase, expected_rows=4,
        models=("roberta",), variants=("short",), include_existing=True,
    )
    assert not any(
        row["classifier"] == "linear_svm" and row["reducer"] == "none"
        for row in rows
    )
    assert any(
        row["classifier"] == "linear_svm" and row["reducer"] == "pca"
        for row in rows
    )


def test_unreduced_linear_svm_requires_explicit_opt_in(complete_embedding_tree):
    rows = build_rows(
        complete_embedding_tree, "triple_models", expected_rows=4,
        models=("roberta",), variants=("short",), include_existing=True,
        include_unreduced_linear_svm=True,
    )
    assert any(
        row["classifier"] == "linear_svm" and row["reducer"] == "none"
        for row in rows
    )


def test_next1_core_contains_only_unreduced_logistic_and_mlp(complete_embedding_tree):
    rows = build_rows(
        complete_embedding_tree, "next1_core", expected_rows=4,
        models=("roberta",), variants=("short", "masked_short"),
        include_existing=True,
    )
    assert len(rows) == 12
    assert {row["feature"] for row in rows} == {
        "body_mean", "full_mean", "title_body_full_concat",
    }
    assert {row["classifier"] for row in rows} == {"logistic", "mlp"}
    assert {row["reducer"] for row in rows} == {"none"}
    assert {row["train_target"] for row in rows} == {"next_day_return"}
    assert {row["evaluation_target"] for row in rows} == {"next_day_return"}


def test_all_pooling_phase_covers_every_declared_pooling(complete_embedding_tree):
    rows = build_rows(
        complete_embedding_tree, "all_pooling", expected_rows=4,
        models=("roberta",), variants=("short", "masked_short"),
        include_existing=True,
    )
    assert len(rows) == 7 * 13 * 2
    assert {row["feature"] for row in rows} == {
        "prompt_mean", "title_mean", "body_mean", "title_body_mean", "full_mean",
        "title_body_concat", "title_body_full_concat",
    }
    assert {row["reducer"] for row in rows} == {"none"}
    assert {row["classifier"] for row in rows} == {
        "logistic", "linear_svm", "sgd", "mlp", "simple_mlp", "lstm",
        "random_forest", "extra_trees", "hist_gradient_boosting",
        "xgboost", "lightgbm", "catboost", "knn",
    }
    assert {row["train_target"] for row in rows} == {"next_day_return"}
    assert {row["evaluation_target"] for row in rows} == {"next_day_return"}


def test_manifest_can_use_a_dedicated_embedding_root(complete_embedding_tree):
    custom_root = complete_embedding_tree / "data/processed/custom_embeddings"
    for variant in ("short", "masked_short"):
        source = complete_embedding_tree / (
            "data/processed/pooled_embeddings_v3/shard-0/roberta/" + variant
        )
        target = custom_root / "shard-0/roberta" / variant
        target.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "metadata.jsonl", "short_pooling.npz"):
            target.joinpath(name).write_bytes(source.joinpath(name).read_bytes())
    rows = build_rows(
        complete_embedding_tree, "all_pooling", expected_rows=4,
        models=("roberta",), variants=("short", "masked_short"),
        include_existing=True, embedding_root_override=custom_root,
    )
    assert rows
    assert {row["embedding_root"] for row in rows} == {str(custom_root)}

