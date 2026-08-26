"""Load sharded pooled transformer embeddings with explicit row alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


POOLED_KEYS = (
    "prompt_mean",
    "title_mean",
    "body_mean",
    "title_body_mean",
    "full_mean",
    "cls",
    "title_max",
    "body_max",
    "title_body_max",
    "full_max",
)
FEATURE_GROUPS = {
    # Existing token-weighted pooling results.
    "prompt_mean": ("prompt_mean",),
    **{key: (key,) for key in POOLED_KEYS if key != "prompt_mean"},
    # Segment-preserving early fusion within the same transformer model.
    "title_body_concat": ("title_mean", "body_mean"),
    # Requested ablation: full_mean overlaps with title/body by construction,
    # but concatenating all three lets the classifier test whether retaining
    # both segment-specific and global coordinates adds predictive value.
    "title_body_full_concat": ("title_mean", "body_mean", "full_mean"),
    # Legacy combined archives may preserve every contextualized prompt token.
    # New large runs store this tensor separately as a memory-mapped token
    # store; pooled loaders support both layouts without duplicating the
    # external store before the final row-alignment concatenation.
    "prompt_tokens_flat": ("prompt_token_embeddings",),
    "prompt_tokens_title_body_concat": (
        "prompt_token_embeddings", "title_mean", "body_mean"
    ),
}


@dataclass(frozen=True)
class PooledEmbeddingSet:
    matrix: np.ndarray
    metadata: pd.DataFrame
    parts: tuple[Path, ...]
    model: str
    variant: str
    prompt_condition: str
    feature: str
    component_shapes: tuple[tuple[int, ...], ...]


def discover_pooled_parts(root: Path, model: str, variant: str) -> list[Path]:
    """Return complete shard output directories in numeric shard order."""
    candidates = list(root.glob(f"shard-*/{model}/{variant}"))

    def shard_number(path: Path) -> int:
        name = path.parents[1].name
        try:
            return int(name.split("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid shard directory: {path}") from exc

    required = ("summary.json", "metadata.jsonl", "short_pooling.npz")
    incomplete = {
        path: [name for name in required if not (path / name).is_file()]
        for path in candidates
    }
    incomplete = {path: missing for path, missing in incomplete.items() if missing}
    if incomplete:
        details = "; ".join(f"{path}: {','.join(missing)}" for path, missing in incomplete.items())
        raise ValueError(f"incomplete embedding shard outputs: {details}")
    complete = sorted(candidates, key=shard_number)
    if complete:
        shard_ids = [shard_number(path) for path in complete]
        expected = list(range(shard_ids[-1] + 1))
        if shard_ids != expected:
            missing = sorted(set(expected).difference(shard_ids))
            raise ValueError(f"non-contiguous embedding shards; missing shard IDs: {missing}")
    return complete


def load_pooled_embeddings(
    root: Path,
    *,
    model: str,
    variant: str,
    feature: str,
    require_complete_rows: int | None = None,
    max_matrix_gib: float | None = None,
) -> PooledEmbeddingSet:
    """Load one pooled feature and verify metadata/row-index integrity.

    Alignment is based on the persisted ``row_index`` rather than shard or
    file order. This is required because source rows were distributed modulo
    the number of embedding shards.
    """
    if feature not in FEATURE_GROUPS:
        raise ValueError(f"feature must be one of: {', '.join(FEATURE_GROUPS)}")
    parts = discover_pooled_parts(root, model, variant)
    if not parts:
        raise ValueError(f"no complete embedding parts under {root} for {model}/{variant}")

    summaries = [
        json.loads((part / "summary.json").read_text(encoding="utf-8"))
        for part in parts
    ]
    prompt_conditions = {
        str(summary.get("prompt_condition", "task_prompt")) for summary in summaries
    }
    if len(prompt_conditions) != 1:
        raise ValueError(f"mixed prompt conditions across shards: {sorted(prompt_conditions)}")
    prompt_condition = prompt_conditions.pop()
    if max_matrix_gib is not None:
        feature_keys = FEATURE_GROUPS[feature]
        estimated_elements = 0
        for part, summary in zip(parts, summaries):
            outputs = summary.get("outputs", {})
            missing = [key for key in feature_keys if key not in outputs]
            if missing:
                raise ValueError(f"cannot estimate {missing} from summary in {part}")
            estimated_elements += sum(int(np.prod(outputs[key])) for key in feature_keys)
        estimated_gib = estimated_elements * np.dtype(np.float32).itemsize / 1024 ** 3
        if estimated_gib > max_matrix_gib:
            raise ValueError(
                f"estimated {feature} matrix is {estimated_gib:.2f} GiB, exceeding "
                f"the {max_matrix_gib:.2f} GiB safety limit"
            )

    matrices: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []
    expected_dimension: int | None = None
    expected_component_shapes: tuple[tuple[int, ...], ...] | None = None
    for part, summary in zip(parts, summaries):
        if summary.get("model") != model or summary.get("variant") != variant:
            raise ValueError(
                f"summary identity mismatch in {part}: "
                f"{summary.get('model')}/{summary.get('variant')} != {model}/{variant}"
            )
        expected_rows = int(summary["rows"])
        with np.load(part / "short_pooling.npz") as archive:
            feature_keys = FEATURE_GROUPS[feature]
            arrays: list[np.ndarray] = []
            missing: list[str] = []
            for key in feature_keys:
                if key in archive:
                    source = archive[key]
                elif key == "prompt_token_embeddings" and (
                    part / "prompt_token_embeddings.npy"
                ).is_file():
                    # The CNINFO-method generator writes prompt tokens as a
                    # separate .npy memmap because putting them in the
                    # compressed pooled archive would require a second dense
                    # copy during embedding generation.
                    source = np.load(
                        part / "prompt_token_embeddings.npy", mmap_mode="r"
                    )
                else:
                    missing.append(key)
                    continue
                arrays.append(np.asarray(source, dtype=np.float32))
            if missing:
                raise ValueError(
                    f"{missing} missing from {part / 'short_pooling.npz'} "
                    "and supported external prompt-token files"
                )
            component_shapes = tuple(tuple(int(value) for value in array.shape[1:]) for array in arrays)
            if expected_component_shapes is None:
                expected_component_shapes = component_shapes
            elif component_shapes != expected_component_shapes:
                raise ValueError(
                    f"component shape mismatch in {part}: "
                    f"{component_shapes} != {expected_component_shapes}"
                )
            declared_outputs = summary.get("outputs", {})
            for key, array in zip(feature_keys, arrays):
                declared_shape = declared_outputs.get(key)
                if declared_shape is not None and list(array.shape) != declared_shape:
                    raise ValueError(
                        f"summary/array shape mismatch for {key} in {part}: "
                        f"{declared_shape} != {list(array.shape)}"
                    )
            row_counts = {len(array) for array in arrays}
            if len(row_counts) != 1:
                raise ValueError(f"component row mismatch for {feature} in {part}")
            arrays = [array.reshape(len(array), -1) for array in arrays]
            matrix = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)
        if matrix.ndim != 2:
            raise ValueError(f"{feature} must be 2-D in {part}; found {matrix.shape}")
        if len(matrix) != expected_rows:
            raise ValueError(f"summary/matrix row mismatch in {part}: {expected_rows} != {len(matrix)}")
        if expected_dimension is None:
            expected_dimension = int(matrix.shape[1])
        elif matrix.shape[1] != expected_dimension:
            raise ValueError(f"embedding dimension mismatch in {part}: {matrix.shape[1]} != {expected_dimension}")

        metadata = pd.read_json(part / "metadata.jsonl", lines=True)
        if len(metadata) != len(matrix):
            raise ValueError(f"metadata/matrix row mismatch in {part}: {len(metadata)} != {len(matrix)}")
        if "row_index" not in metadata:
            raise ValueError(f"row_index missing from {part / 'metadata.jsonl'}")
        metadata = metadata.copy()
        metadata["embedding_part"] = str(part)
        metadata["embedding_part_row"] = np.arange(len(metadata), dtype=np.int64)
        matrices.append(matrix)
        metadata_frames.append(metadata)

    matrix = np.concatenate(matrices, axis=0)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    metadata["row_index"] = pd.to_numeric(metadata["row_index"], errors="raise").astype(np.int64)
    if metadata["row_index"].duplicated().any():
        duplicates = metadata.loc[metadata["row_index"].duplicated(), "row_index"].head().tolist()
        raise ValueError(f"duplicate embedding row_index values: {duplicates}")
    if require_complete_rows is not None and len(metadata) != require_complete_rows:
        raise ValueError(f"expected {require_complete_rows} embedding rows; found {len(metadata)}")
    if not np.isfinite(matrix).all():
        bad = int(matrix.size - np.isfinite(matrix).sum())
        raise ValueError(f"embedding matrix contains {bad} non-finite values")

    order = np.argsort(metadata["row_index"].to_numpy(), kind="stable")
    matrix = matrix[order]
    metadata = metadata.iloc[order].reset_index(drop=True)
    if require_complete_rows is not None:
        expected_indices = np.arange(1, require_complete_rows + 1, dtype=np.int64)
        actual_indices = metadata["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_indices, expected_indices):
            missing = np.setdiff1d(expected_indices, actual_indices, assume_unique=True)[:10].tolist()
            extra = np.setdiff1d(actual_indices, expected_indices, assume_unique=True)[:10].tolist()
            raise ValueError(
                "embedding row_index must exactly cover 1.."
                f"{require_complete_rows}; missing={missing} extra={extra}"
            )
    return PooledEmbeddingSet(
        matrix=matrix,
        metadata=metadata,
        parts=tuple(parts),
        model=model,
        variant=variant,
        prompt_condition=prompt_condition,
        feature=feature,
        component_shapes=expected_component_shapes or (),
    )


def align_embeddings_to_panel(
    panel: pd.DataFrame,
    embeddings: PooledEmbeddingSet,
    *,
    panel_row_index_column: str = "row_index",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Inner-align a panel and embedding matrix by unique source row index."""
    if panel_row_index_column not in panel:
        raise ValueError(f"panel is missing {panel_row_index_column}")
    frame = panel.copy()
    frame[panel_row_index_column] = pd.to_numeric(
        frame[panel_row_index_column], errors="raise"
    ).astype(np.int64)
    if frame[panel_row_index_column].duplicated().any():
        raise ValueError(f"panel {panel_row_index_column} must be unique")

    lookup = pd.Series(
        np.arange(len(embeddings.metadata), dtype=np.int64),
        index=embeddings.metadata["row_index"].to_numpy(),
    )
    positions = frame[panel_row_index_column].map(lookup)
    matched = positions.notna()
    frame = frame.loc[matched].copy()
    matrix = embeddings.matrix[positions.loc[matched].astype(np.int64).to_numpy()]
    frame["embedding_row_index"] = frame[panel_row_index_column].to_numpy()
    return frame.reset_index(drop=True), matrix
