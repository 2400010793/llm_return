import json
from pathlib import Path

import numpy as np
import pytest

from src.data.aligned_prompts import audit_aligned_prompts, load_aligned_prompt_config, three_point_factors


class CharTokenizer:
    name_or_path = "test-char-tokenizer"

    def __init__(self):
        self.backend_tokenizer = None

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": list(range(1, len(text) + 1))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]

    def get_vocab(self):
        return {"x": 1}


def config():
    return {
        "variant": "masked_short",
        "model": "roberta",
        "axes": {
            "axis": {
                "levels": {
                    "low": {"prompt_id": "low", "target": "低冲击", "text": "分析股票低冲击"},
                    "neutral": {"prompt_id": "neutral", "target": "中冲击", "text": "分析股票中冲击"},
                    "high": {"prompt_id": "high", "target": "高冲击", "text": "分析股票高冲击"},
                }
            }
        },
    }


def test_aligned_prompt_audit_requires_equal_axis_slots():
    result = audit_aligned_prompts(config(), CharTokenizer())
    axis = result["axes"]["axis"]
    assert axis["aligned"] is True
    assert axis["token_budget"] == 7
    assert axis["target_token_count"] == 3
    assert result["prompts"]["neutral"]["target_token_indices"] == [4, 5, 6]


def test_aligned_prompt_audit_rejects_unequal_target_tokens():
    bad = config()
    bad["axes"]["axis"]["levels"]["high"]["target"] = "高冲击程度"
    bad["axes"]["axis"]["levels"]["high"]["text"] = "分析股票高冲击程度"
    with pytest.raises(ValueError, match="unaligned"):
        audit_aligned_prompts(bad, CharTokenizer())


def test_three_point_factors_have_expected_shapes_and_values():
    low = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    neutral = np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    high = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    factors = three_point_factors(low, neutral, high)
    assert factors["direction"].shape == (2, 2)
    assert factors["magnitude"].shape == (2, 1)
    np.testing.assert_allclose(factors["direction"], [[2.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(factors["clarity"][0], [1.0])
    assert np.isfinite(factors["neutral_curvature"]).all()


def test_config_file_contains_only_masked_short_axes():
    path = Path("configs/prompts/aligned_masked_short_v1.json")
    loaded = load_aligned_prompt_config(path)
    assert loaded["variant"] == "masked_short"
    assert len(loaded["axes"]) == 6
    assert sum(len(axis["levels"]) for axis in loaded["axes"].values()) == 18
