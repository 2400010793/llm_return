"""Validation and factor construction for short, aligned prompt axes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LEVELS = ("low", "neutral", "high")


def load_aligned_prompt_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("variant") != "masked_short":
        raise ValueError("aligned prompt config must use masked_short")
    axes = config.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("aligned prompt config must contain axes")
    seen: set[str] = set()
    for axis, spec in axes.items():
        levels = spec.get("levels") if isinstance(spec, dict) else None
        if not isinstance(levels, dict) or set(levels) != set(LEVELS):
            raise ValueError(f"axis {axis} must contain low, neutral, high")
        for level in LEVELS:
            item = levels[level]
            if not isinstance(item, dict) or not item.get("text") or not item.get("target"):
                raise ValueError(f"missing prompt text/target for {axis}/{level}")
            prompt_id = str(item.get("prompt_id", ""))
            if not prompt_id or prompt_id in seen:
                raise ValueError(f"duplicate or empty prompt_id: {prompt_id}")
            seen.add(prompt_id)
            text = str(item["text"])
            target = str(item["target"])
            if text.count(target) != 1:
                raise ValueError(f"target must occur exactly once in {axis}/{level}")
    return config


def tokenizer_digest(tokenizer: Any) -> str:
    """Hash the serialized fast-tokenizer backend when available."""
    try:
        serialized = tokenizer.backend_tokenizer.to_str()
    except AttributeError:
        serialized = json.dumps(
            {"name": tokenizer.name_or_path, "vocab": tokenizer.get_vocab()},
            ensure_ascii=False, sort_keys=True,
        )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _prompt_audit(text: str, target: str, tokenizer: Any) -> dict[str, Any]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    tokens = [str(value) for value in tokenizer.convert_ids_to_tokens(ids)]
    offsets = [list(map(int, pair)) for pair in encoded["offset_mapping"]]
    start_char = text.index(target)
    end_char = start_char + len(target)
    target_positions = [
        index for index, (start, end) in enumerate(offsets)
        if start < end_char and end > start_char
    ]
    if not target_positions:
        raise ValueError(f"target span does not map to tokens: {text!r} / {target!r}")
    prefix_count = len([offset for offset in offsets if offset[1] <= start_char])
    suffix_count = len([offset for offset in offsets if offset[0] >= end_char])
    return {
        "text": text,
        "target": target,
        "input_ids": ids,
        "tokens": tokens,
        "offset_mapping": offsets,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "total_prompt_tokens": len(ids),
        "target_start_char": start_char,
        "target_end_char": end_char,
        "target_token_indices": target_positions,
        "target_token_count": len(target_positions),
        "target_start_token": target_positions[0],
        "target_end_token_exclusive": target_positions[-1] + 1,
        "prefix_token_count": prefix_count,
        "suffix_token_count": suffix_count,
    }


def audit_aligned_prompts(config: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Audit every prompt and require exact alignment within each axis."""
    tokenizer_hash = tokenizer_digest(tokenizer)
    axes: dict[str, Any] = {}
    all_audits: dict[str, Any] = {}
    for axis, spec in config["axes"].items():
        level_audits: dict[str, Any] = {}
        for level in LEVELS:
            item = spec["levels"][level]
            audit = _prompt_audit(str(item["text"]), str(item["target"]), tokenizer)
            audit.update({"axis": axis, "level": level, "prompt_id": item["prompt_id"]})
            level_audits[level] = audit
            all_audits[str(item["prompt_id"])] = audit
        reference = level_audits["neutral"]
        fields = (
            "total_prompt_tokens", "target_token_count", "target_start_token",
            "target_end_token_exclusive", "prefix_token_count", "suffix_token_count",
        )
        mismatches = {
            field: {level: level_audits[level][field] for level in LEVELS}
            for field in fields
            if len({level_audits[level][field] for level in LEVELS}) != 1
        }
        if mismatches:
            raise ValueError(f"unaligned tokens in axis {axis}: {mismatches}")
        axes[axis] = {
            "label": spec.get("label", axis),
            "levels": level_audits,
            "aligned": True,
            "token_budget": reference["total_prompt_tokens"],
            "target_token_count": reference["target_token_count"],
        }
    return {
        "format_version": "aligned_prompt_preflight_v1",
        "model": config.get("model"),
        "variant": "masked_short",
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_sha256": tokenizer_hash,
        "alignment_scope": "within_axis",
        "axes": axes,
        "prompts": all_audits,
    }


def three_point_factors(
    low: np.ndarray,
    neutral: np.ndarray,
    high: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Construct vector and scalar factors from three aligned representations."""
    low = np.asarray(low, dtype=np.float32)
    neutral = np.asarray(neutral, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if low.shape != neutral.shape or low.shape != high.shape:
        raise ValueError("low, neutral, and high matrices must have identical shapes")
    if low.ndim != 2 or not np.isfinite(low).all() or not np.isfinite(neutral).all() or not np.isfinite(high).all():
        raise ValueError("three-point matrices must be finite two-dimensional arrays")
    high_deviation = high - neutral
    low_deviation = low - neutral
    direction = high - low
    high_distance = np.linalg.norm(high_deviation, axis=1)
    low_distance = np.linalg.norm(low_deviation, axis=1)
    magnitude = high_distance + low_distance
    direction_norm = np.linalg.norm(direction, axis=1)
    midpoint = (low + high) / 2.0
    return {
        "direction": direction.astype(np.float32),
        "high_deviation": high_deviation.astype(np.float32),
        "low_deviation": low_deviation.astype(np.float32),
        "magnitude": magnitude[:, None].astype(np.float32),
        "clarity": (direction_norm / (magnitude + float(epsilon)))[:, None].astype(np.float32),
        "neutral_curvature": np.linalg.norm(neutral - midpoint, axis=1)[:, None].astype(np.float32),
    }
