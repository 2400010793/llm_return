from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.prompt_token_mechanisms import (
    YearlyTokenMoments,
    aggregate_moments,
    build_semantic_token_map,
    cluster_token_responses,
    rolling_windows,
    select_stable_kmeans,
    token_metrics_from_moments,
)
from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore
from scripts.select_prompt_mechanism_controls import select_controls


LONG_PROMPT = (
    "任务：判断目标股票在新闻发布后的下一交易日更可能上涨还是下跌。"
    "请识别可能改变投资者预期的信息，包括业绩与现金流、订单与投资、"
    "融资与股权变动、监管与诉讼、经营风险、行业变化及公司治理；"
    "忽略广告、栏目导航和无关模板。不得使用新闻发布后的价格、收益或其他未来信息。"
)


def test_semantic_mapping_uses_character_spans_and_partitions_tokens() -> None:
    tokens = list(LONG_PROMPT)
    offsets = [(index, index + 1) for index in range(len(tokens))]
    mapping = build_semantic_token_map(LONG_PROMPT, tokens, offsets)
    direction = mapping.group_positions["direction"]
    assert "".join(tokens[position] for position in direction) == "上涨还是下跌"
    performance = mapping.group_positions["performance_cashflow"]
    assert "".join(tokens[position] for position in performance) == "业绩与现金流"
    assigned = sorted(
        position for positions in mapping.group_positions.values() for position in positions
    )
    assert assigned == list(range(len(tokens)))
    assert mapping.group_positions["generic_instruction"]


def test_semantic_mapping_keeps_standalone_sentencepiece_boundaries_generic() -> None:
    mapping = build_semantic_token_map(
        "分析股票", ["▁", "分析", "股票"], [(0, 1), (0, 2), (2, 4)],
        semantic_phrases=(("analysis", "分析"), ("target_stock", "股票")),
        require_generic=False,
    )
    assert mapping.group_positions["analysis"] == (1,)
    assert mapping.group_positions["target_stock"] == (2,)
    assert mapping.group_positions["generic_instruction"] == (0,)


def test_exact_return_span_excludes_future_tokens() -> None:
    prompt = "分析股票未来收益"
    mapping = build_semantic_token_map(
        prompt,
        tuple(prompt),
        [(index, index + 1) for index in range(len(prompt))],
        semantic_phrases=(("return_span", "收益"),),
        require_generic=False,
    )
    assert mapping.group_positions["return_span"] == (6, 7)


def test_yearly_moments_rank_known_label_signal_first(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    years = np.repeat([2010, 2011], 100)
    target = np.tile(np.r_[-np.ones(50), np.ones(50)], 2)
    values = rng.normal(0, 0.2, size=(200, 3, 4)).astype(np.float32)
    values[:, 1, 0] += target * 3.0
    moments = YearlyTokenMoments([2010, 2011], 3, 4)
    moments.update(
        values, years,
        {"event_return_3d": target, "next_day_return": target},
    )
    path = tmp_path / "moments.npz"
    moments.save(path)
    with np.load(path) as archive:
        aggregate = aggregate_moments(
            archive, [2010, 2011], target="next_day_return",
        )
    metrics = token_metrics_from_moments(
        aggregate, prompt_only=np.zeros((3, 4)),
        mask_delta_mean=np.zeros((3, 4)),
    )
    assert int(np.argmax(metrics["fisher"])) == 1
    assert np.isfinite(metrics["context_standardized_l2"]).all()


def test_rolling_windows_never_include_test_year_in_fit_or_validation() -> None:
    windows = rolling_windows(range(2010, 2027))
    assert windows[0] == {
        "test_year": 2018,
        "fit_years": (2010, 2011, 2012, 2013, 2014, 2015),
        "validation_years": (2016, 2017),
        "all_train_years": (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017),
    }
    assert windows[-1]["test_year"] == 2026
    for window in windows:
        assert max(window["all_train_years"]) < window["test_year"]


def test_token_clustering_and_k_selection_are_reproducible() -> None:
    rng = np.random.default_rng(4)
    first = np.r_[rng.normal(-2, 0.1, (60, 5)), rng.normal(2, 0.1, (60, 5))]
    validation = np.r_[rng.normal(-2, 0.1, (30, 5)), rng.normal(2, 0.1, (30, 5))]
    k_first, rows_first = select_stable_kmeans(
        first, validation, k_values=[2, 3], seeds=[13, 42], sample_size=60,
    )
    k_second, rows_second = select_stable_kmeans(
        first, validation, k_values=[2, 3], seeds=[13, 42], sample_size=60,
    )
    assert k_first == k_second == 2
    assert rows_first == rows_second
    labels, tree = cluster_token_responses(
        [np.r_[np.ones((3, 4)), -np.ones((3, 4))]], cluster_count=2,
    )
    assert len(np.unique(labels)) == 2
    assert tree.shape == (5, 4)


def _write_shard(root: Path, shard: int, rows: list[int], values: np.ndarray) -> None:
    directory = root / f"shard-{shard}" / "roberta" / "long"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps({"rows": len(rows)}), encoding="utf-8",
    )
    pd.DataFrame({"row_index": rows}).to_json(
        directory / "metadata.jsonl", orient="records", lines=True,
    )
    np.save(directory / "prompt_token_embeddings.npy", values)
    (directory / "prompt_tokens.json").write_text(
        json.dumps({"text": "甲乙", "tokens": ["甲", "乙"]}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_prompt_store_streams_every_unlabelled_row_once(tmp_path: Path) -> None:
    _write_shard(tmp_path, 0, [1, 3], np.ones((2, 2, 3), dtype=np.float32))
    _write_shard(tmp_path, 1, [2], np.full((1, 2, 3), 2, dtype=np.float32))
    store = PromptTokenEmbeddingStore(
        tmp_path, model="roberta", variant="long",
        expected_shards=2, expected_rows=3,
    )
    batches = list(store.iter_value_batches(batch_size=1))
    assert [batch.row_indexes.tolist() for batch in batches] == [[1], [3], [2]]
    assert all(batch.values.shape == (1, 2, 3) for batch in batches)


def test_control_selection_uses_validation_only_and_equal_length_placebos(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary"
    cache = tmp_path / "cache" / "roberta" / "long"
    summary.mkdir(); cache.mkdir(parents=True)
    (summary / "audit.json").write_text(json.dumps({
        "completed_folds": 288, "expected_full_folds": 288,
    }), encoding="utf-8")
    rows = []
    for target in ("event_return_3d", "next_day_return"):
        rows.append({
            "model": "roberta", "prompt_length": "long", "variant": "masked_long",
            "target": target, "test_year": 2026, "classifier": "logistic",
            "representation": "full_prompt_mean", "validation_accuracy": 0.54,
        })
        for group, accuracy in (
            ("direction", 0.51), ("orders_investment", 0.50),
            ("operating_risk", 0.52), ("generic_instruction", 0.49),
        ):
            rows.append({
                "model": "roberta", "prompt_length": "long", "variant": "masked_long",
                "target": target, "test_year": 2026, "classifier": "logistic",
                "representation": f"leaveout::{group}", "validation_accuracy": accuracy,
            })
    pd.DataFrame(rows).to_parquet(summary / "classification_by_year.parquet", index=False)
    (cache / "manifest.json").write_text(json.dumps({
        "group_positions": {
            "direction": [3, 4], "orders_investment": [5, 6, 7],
            "operating_risk": [8, 9], "generic_instruction": list(range(20, 40)),
        },
    }), encoding="utf-8")
    selected = select_controls(summary, tmp_path / "cache")
    specifications = selected["control_specs"]
    assert selected["selection_scope"] == "validation_only"
    assert any(row["control_id"] == "no_prompt_position_matched" for row in specifications)
    for semantic in ("direction", "orders_investment", "operating_risk"):
        ablation = next(row for row in specifications if row["control_id"] == f"ablate_{semantic}")
        placebo = next(row for row in specifications if row["control_id"] == f"placebo_{semantic}")
        assert len(ablation["positions"]) == len(placebo["positions"])
        assert placebo["is_placebo"] is True
