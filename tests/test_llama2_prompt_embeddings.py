from argparse import Namespace
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_llama2_prompt_embeddings import POOLING_NAMES, audit
from scripts.run_llama2_prompt_embeddings import compose_causal_readout_sequence


def test_causal_readout_is_last_and_survives_truncation() -> None:
    ids, segments, report = compose_causal_readout_sequence(
        [10, 11], [20, 21, 22], [], [90, 91],
        bos_id=1, eos_id=2, max_length=7,
    )
    assert ids == [1, 10, 11, 20, 90, 91, 2]
    assert segments == [-1, 1, 1, 2, 0, 0, -1]
    assert report["sequence_truncated"] is True
    assert report["prompt_start_zero_based"] == 4
    assert report["prompt_end_zero_based"] == 5


def _write_variant(
    root: Path, variant: str, values: np.ndarray, offset: float,
) -> None:
    destination = root / "shard-0" / "llama2_13b" / variant
    destination.mkdir(parents=True)
    np.save(destination / "prompt_token_embeddings.npy", values + offset)
    np.save(
        destination / "prompt_input_ids.npy",
        np.array([7, 8], dtype=np.int32),
    )
    (destination / "prompt_tokens.json").write_text("{}", encoding="utf-8")
    arrays = {
        name: np.ones((3, 4), dtype=np.float16) for name in POOLING_NAMES
    }
    np.savez_compressed(destination / "short_pooling.npz", **arrays)
    with (destination / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in range(3):
            handle.write(json.dumps({
                "row_index": row + 1,
                "sequence_layout": "article_then_readout_prompt",
                "prompt_start_zero_based": 5,
                "prompt_end_zero_based": 6,
                "saved_token_count": 8,
            }) + "\n")
    prompt = "分析股票未来收益"
    (destination / "summary.json").write_text(json.dumps({
        "format_version": "llama2_causal_readout_v2",
        "sequence_layout": "article_then_readout_prompt",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "rows": 3,
        "prompt_token_count": 2,
        "hidden_size": 4,
        "storage_dtype": "float16",
        "cls_pooling": "last_readout_token",
    }), encoding="utf-8")
    (destination / "COMPLETED").write_text("ok\n", encoding="utf-8")


def test_audit_requires_context_and_mask_variation(tmp_path: Path) -> None:
    prompt_spec = tmp_path / "prompt.json"
    prompt_spec.write_text(json.dumps({
        "prompts": {"short": {"text": "分析股票未来收益"}},
    }), encoding="utf-8")
    values = np.arange(24, dtype=np.float16).reshape(3, 2, 4)
    _write_variant(tmp_path / "output", "short", values, 0.0)
    _write_variant(tmp_path / "output", "masked_short", values, 0.5)
    result = audit(Namespace(
        root=tmp_path / "output",
        prompt_spec=prompt_spec,
        output=tmp_path / "audit.json",
        expected_shards=1,
        expected_rows=3,
        minimum_delta=1e-4,
        strict=True,
    ))
    assert result["valid"] is True
    assert result["maximum_prompt_sample_delta"] > 0
    assert result["mean_mask_sample_delta"] > 0
