from pathlib import Path

from scripts.build_pooled_classification_manifest import ROOTS, phase_cells
from scripts.build_o2o_classification_manifest import build_rows as build_o2o_rows
from scripts.prefetch_paper_hk_models import MODELS
from scripts.run_short_pooled_embeddings import DEFAULT_MAX_LENGTH, MODEL_PATHS, components


def test_plain_variant_uses_raw_text_without_prompt_segments():
    row = {
        "text_plain": "原始中文公告正文",
        "short_prompt": "不应进入论文基准",
        "short_title": "不应进入论文基准",
        "short_body": "不应进入论文基准",
    }

    assert components(row, "plain") == ("", "", "原始中文公告正文")


def test_paper_hk_models_and_root_match_prefetch_contract():
    assert ROOTS["plain"] == Path("data/processed/pooled_paper_hk_embeddings_v1")
    assert set(MODELS) == {"ckip_bert", "xlm_roberta_large"}
    for alias, (_, directory_name) in MODELS.items():
        assert Path(MODEL_PATHS[alias]).name == directory_name
        assert DEFAULT_MAX_LENGTH[alias] == 512


def test_paper_hk_phase_keeps_paper_baseline_and_declared_extensions():
    cells = set(phase_cells("paper_hk"))
    assert ("full_mean", "logistic", "none", 0) in cells
    assert ("full_mean", "mlp", "none", 0) in cells
    assert ("full_mean", "logistic", "pca", 128) in cells
    assert ("full_mean", "mlp", "pca", 128) in cells
    assert len(cells) == 4


def test_ckip_o2o_manifest_is_bounded_to_four_feasible_cells(tmp_path):
    embedding_root = tmp_path / "embeddings"
    for shard, rows in ((0, 2), (1, 1)):
        part = embedding_root / f"shard-{shard}" / "ckip_bert" / "plain"
        part.mkdir(parents=True)
        (part / "summary.json").write_text(f'{{"rows": {rows}}}', encoding="utf-8")
        (part / "metadata.jsonl").write_text("{}\n" * rows, encoding="utf-8")
        (part / "short_pooling.npz").write_bytes(b"placeholder")

    rows = build_o2o_rows(
        tmp_path,
        expected_rows=3,
        embedding_root=Path("embeddings"),
        models=("ckip_bert",),
        variants=("plain",),
        features=("full_mean",),
        classifiers=("logistic", "mlp"),
        reducers=(("none", 0), ("pca", 128)),
        phase="paper_hk_o2o",
        output_root=Path("reports/paper_hk_o2o"),
    )

    assert len(rows) == 4
    assert {row["phase"] for row in rows} == {"paper_hk_o2o"}
    assert {row["model"] for row in rows} == {"ckip_bert"}
    assert {row["train_target"] for row in rows} == {
        "next_day_open_to_open_return"
    }
