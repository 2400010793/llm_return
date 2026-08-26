"""Streaming access to sharded frozen prompt-token embedding tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def prompt_shard_number(path: Path) -> int:
    """Return the numeric ID from ``shard-N/model/variant``."""
    try:
        return int(path.parents[1].name.split("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid prompt-token shard path: {path}") from exc


@dataclass(frozen=True)
class PromptTokenShard:
    directory: Path
    row_indexes: np.ndarray
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class PromptTokenBatch:
    values: np.ndarray
    targets: np.ndarray
    row_indexes: np.ndarray


@dataclass(frozen=True)
class PromptTokenValuesBatch:
    """Unlabelled prompt-token values in their source-shard row order."""

    values: np.ndarray
    row_indexes: np.ndarray


class PromptTokenEmbeddingStore:
    """Validate prompt-token shards and stream selected rows without concatenation."""

    def __init__(
        self,
        root: Path,
        *,
        model: str,
        variant: str,
        expected_shards: int | None = None,
        expected_rows: int | None = None,
        expected_token_count: int | None = None,
        expected_hidden_size: int | None = None,
    ) -> None:
        directories = sorted(
            root.glob(f"shard-*/{model}/{variant}"), key=prompt_shard_number,
        )
        if not directories:
            raise ValueError(f"no prompt-token shards for {model}/{variant} under {root}")
        if expected_shards is not None and len(directories) != expected_shards:
            raise ValueError(
                f"expected {expected_shards} prompt-token shards; found {len(directories)}"
            )
        shard_ids = [prompt_shard_number(directory) for directory in directories]
        if shard_ids != list(range(len(shard_ids))):
            raise ValueError(f"prompt-token shards are not contiguous: {shard_ids}")

        required = (
            "summary.json", "metadata.jsonl", "prompt_token_embeddings.npy",
            "prompt_tokens.json",
        )
        shards: list[PromptTokenShard] = []
        all_rows: list[np.ndarray] = []
        token_shape: tuple[int, int] | None = None
        prompt_tokens: list[str] | None = None
        prompt_text: str | None = None
        for directory in directories:
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                raise ValueError(f"incomplete prompt-token shard {directory}: {missing}")
            metadata = pd.read_json(directory / "metadata.jsonl", lines=True)
            if "row_index" not in metadata:
                raise ValueError(f"row_index missing from {directory / 'metadata.jsonl'}")
            row_indexes = pd.to_numeric(
                metadata["row_index"], errors="raise"
            ).to_numpy(dtype=np.int64)
            if len(np.unique(row_indexes)) != len(row_indexes):
                raise ValueError(f"duplicate row_index inside {directory}")
            matrix = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
            if matrix.ndim != 3 or len(matrix) != len(row_indexes):
                raise ValueError(
                    f"prompt matrix/metadata mismatch in {directory}: "
                    f"{matrix.shape} vs {len(row_indexes)} rows"
                )
            current_shape = (int(matrix.shape[1]), int(matrix.shape[2]))
            if token_shape is None:
                token_shape = current_shape
            elif current_shape != token_shape:
                raise ValueError(
                    f"prompt token shape mismatch in {directory}: "
                    f"{current_shape} != {token_shape}"
                )
            prompt = json.loads(
                (directory / "prompt_tokens.json").read_text(encoding="utf-8")
            )
            current_tokens = [str(value) for value in prompt["tokens"]]
            if prompt_tokens is None:
                prompt_tokens = current_tokens
                prompt_text = prompt.get("text")
            elif current_tokens != prompt_tokens:
                raise ValueError(f"prompt tokens differ in {directory}")
            elif prompt.get("text") != prompt_text:
                raise ValueError(f"prompt text differs in {directory}")
            shards.append(PromptTokenShard(directory, row_indexes, tuple(matrix.shape)))
            all_rows.append(row_indexes)

        assert token_shape is not None
        if expected_token_count is not None and token_shape[0] != expected_token_count:
            raise ValueError(
                f"expected {expected_token_count} prompt positions; found {token_shape[0]}"
            )
        if expected_hidden_size is not None and token_shape[1] != expected_hidden_size:
            raise ValueError(
                f"expected hidden size {expected_hidden_size}; found {token_shape[1]}"
            )
        combined_rows = np.concatenate(all_rows)
        if len(np.unique(combined_rows)) != len(combined_rows):
            raise ValueError("duplicate row_index across prompt-token shards")
        if expected_rows is not None and len(combined_rows) != expected_rows:
            raise ValueError(f"expected {expected_rows} prompt rows; found {len(combined_rows)}")
        if (combined_rows < 1).any():
            raise ValueError("prompt row_index values must be positive")

        self.root = root
        self.model = model
        self.variant = variant
        self.shards = tuple(shards)
        self.token_count, self.hidden_size = token_shape
        self.prompt_tokens = tuple(prompt_tokens or ())
        self.prompt_text = prompt_text
        self.rows = int(len(combined_rows))
        self.max_row_index = int(combined_rows.max())
        self.row_indexes = np.sort(combined_rows)

    def iter_batches(
        self,
        selected_by_row: np.ndarray,
        targets_by_row: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[PromptTokenBatch]:
        """Yield selected finite-target rows, indexing both arrays by row_index."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if selected_by_row.ndim != 1 or targets_by_row.ndim != 1:
            raise ValueError("selection and target lookup arrays must be one-dimensional")
        if len(selected_by_row) <= self.max_row_index or len(targets_by_row) <= self.max_row_index:
            raise ValueError("lookup arrays do not cover every prompt row_index")
        rng = np.random.default_rng(seed)
        shard_order = np.arange(len(self.shards))
        if shuffle:
            rng.shuffle(shard_order)
        for shard_position in shard_order:
            shard = self.shards[int(shard_position)]
            eligible = selected_by_row[shard.row_indexes]
            eligible &= np.isfinite(targets_by_row[shard.row_indexes])
            local_positions = np.flatnonzero(eligible)
            if shuffle:
                rng.shuffle(local_positions)
            matrix = np.load(
                shard.directory / "prompt_token_embeddings.npy", mmap_mode="r"
            )
            for start in range(0, len(local_positions), batch_size):
                positions = local_positions[start:start + batch_size]
                rows = shard.row_indexes[positions]
                yield PromptTokenBatch(
                    values=np.asarray(matrix[positions], dtype=np.float32),
                    targets=np.asarray(targets_by_row[rows], dtype=np.float32),
                    row_indexes=rows.copy(),
                )

    def count_selected(
        self, selected_by_row: np.ndarray, targets_by_row: np.ndarray,
    ) -> int:
        """Count selected rows with finite targets."""
        if len(selected_by_row) <= self.max_row_index or len(targets_by_row) <= self.max_row_index:
            raise ValueError("lookup arrays do not cover every prompt row_index")
        return int(sum(
            np.sum(
                selected_by_row[shard.row_indexes]
                & np.isfinite(targets_by_row[shard.row_indexes])
            )
            for shard in self.shards
        ))

    def iter_value_batches(self, *, batch_size: int) -> Iterator[PromptTokenValuesBatch]:
        """Stream every stored row once without requiring a panel target lookup."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for shard in self.shards:
            matrix = np.load(
                shard.directory / "prompt_token_embeddings.npy", mmap_mode="r"
            )
            for start in range(0, len(shard.row_indexes), batch_size):
                stop = min(start + batch_size, len(shard.row_indexes))
                yield PromptTokenValuesBatch(
                    values=np.asarray(matrix[start:stop], dtype=np.float32),
                    row_indexes=shard.row_indexes[start:stop].copy(),
                )

    def select_positions(
        self, positions: np.ndarray | list[int] | tuple[int, ...],
    ) -> "SelectedPromptTokenEmbeddingStore":
        """Return a streaming view that retains only specified token positions."""
        return SelectedPromptTokenEmbeddingStore(self, positions)


class SelectedPromptTokenEmbeddingStore:
    """Read-only token-position view over a validated prompt embedding store.

    The underlying shard matrices stay memory mapped.  Position selection is
    applied to each streamed batch, so hard-gate experiments do not materialize
    another full embedding dataset.
    """

    def __init__(
        self,
        base: PromptTokenEmbeddingStore,
        positions: np.ndarray | list[int] | tuple[int, ...],
    ) -> None:
        selected = np.asarray(positions, dtype=np.int64)
        if selected.ndim != 1 or not len(selected):
            raise ValueError("selected token positions must be a non-empty vector")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("selected token positions must be unique")
        if (selected < 0).any() or (selected >= base.token_count).any():
            raise ValueError(
                f"selected token positions must lie in 0..{base.token_count - 1}"
            )
        self.base = base
        self.selected_positions = selected.copy()
        self.root = base.root
        self.model = base.model
        self.variant = base.variant
        self.shards = base.shards
        self.token_count = int(len(selected))
        self.hidden_size = base.hidden_size
        self.prompt_tokens = tuple(base.prompt_tokens[position] for position in selected)
        self.prompt_text = base.prompt_text
        self.rows = base.rows
        self.max_row_index = base.max_row_index
        self.row_indexes = base.row_indexes

    def iter_batches(
        self,
        selected_by_row: np.ndarray,
        targets_by_row: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[PromptTokenBatch]:
        for batch in self.base.iter_batches(
            selected_by_row, targets_by_row,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
        ):
            yield PromptTokenBatch(
                values=np.ascontiguousarray(
                    batch.values[:, self.selected_positions, :], dtype=np.float32,
                ),
                targets=batch.targets,
                row_indexes=batch.row_indexes,
            )

    def count_selected(
        self, selected_by_row: np.ndarray, targets_by_row: np.ndarray,
    ) -> int:
        return self.base.count_selected(selected_by_row, targets_by_row)
