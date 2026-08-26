"""Rolling within-stock semantic novelty without future-date comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_prior_cosine(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    *,
    trading_dates: pd.Series | pd.Index,
    eligible: np.ndarray | pd.Series | None = None,
    stock_column: str = "stock_id",
    date_column: str = "entry_date",
    timestamp_column: str = "published_at",
    row_column: str = "row_index",
    lookback_trading_days: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum prior cosine and its source row for every document.

    Comparisons are restricted to earlier documents for the same stock whose
    mapped trading date is no more than ``lookback_trading_days`` before the
    current trading date. Documents earlier on the same mapped trading date are
    included. All eligible earlier documents remain in the comparison window,
    including documents that would themselves later be marked redundant.
    """
    if lookback_trading_days < 1:
        raise ValueError("lookback_trading_days must be positive")
    if len(frame) != len(matrix):
        raise ValueError("frame and matrix must have equal rows")
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    required = {stock_column, date_column, timestamp_column, row_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {', '.join(sorted(missing))}")
    if frame[row_column].duplicated().any():
        raise ValueError(f"{row_column} must be unique")

    valid = np.ones(len(frame), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if valid.shape != (len(frame),):
        raise ValueError("eligible must contain one boolean per row")
    values = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("matrix contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    normalized = np.zeros_like(values, dtype=np.float32)
    nonzero = norms > 0
    normalized[nonzero] = values[nonzero] / norms[nonzero, None]

    calendar = pd.DatetimeIndex(pd.to_datetime(trading_dates, errors="coerce"))
    calendar = calendar[calendar.notna()].normalize().drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("trading_dates contains no valid dates")
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    date_rank = calendar.get_indexer(dates)
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    timestamp_ns = timestamps.astype("int64", copy=False).to_numpy()
    # Invalid timestamps sort after valid timestamps on their mapped date.
    timestamp_ns[timestamps.isna().to_numpy()] = np.iinfo(np.int64).max
    row_ids = pd.to_numeric(frame[row_column], errors="raise").to_numpy(dtype=np.int64)

    maximum = np.full(len(frame), np.nan, dtype=np.float32)
    source_row = np.full(len(frame), -1, dtype=np.int64)
    groups = frame.groupby(stock_column, sort=False, observed=True).indices
    for raw_indices in groups.values():
        indices = np.asarray(raw_indices, dtype=np.int64)
        order = np.lexsort((row_ids[indices], timestamp_ns[indices], date_rank[indices]))
        ordered = indices[order]
        history: list[int] = []
        for current in ordered:
            current_rank = int(date_rank[current])
            if current_rank < 0 or not valid[current]:
                continue
            history = [
                prior for prior in history
                if 0 <= current_rank - int(date_rank[prior]) <= lookback_trading_days
            ]
            if history and nonzero[current]:
                prior_indices = np.asarray(history, dtype=np.int64)
                scores = normalized[prior_indices] @ normalized[current]
                best_position = int(np.argmax(scores))
                maximum[current] = float(scores[best_position])
                source_row[current] = int(row_ids[prior_indices[best_position]])
            history.append(int(current))
    return maximum, source_row


__all__ = ["rolling_prior_cosine"]
