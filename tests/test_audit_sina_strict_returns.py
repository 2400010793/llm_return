from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_sina_strict_returns import (
    form_portfolio,
    select_per_test_year,
    summarize_series,
)


def _report(name: str, validation_rank_days: float = 3.0) -> dict:
    return {
        "_name": name,
        "_path": f"{name}.json",
        "results": [{
            "test_year": 2026,
            "validation_metrics": {
                "rank_ic_mean": 0.1 if name == "a" else 0.2,
                "rank_ic_days": validation_rank_days,
                "n": 120,
                "oos_r2_vs_historical_mean": 0.0,
                "mse": 1.0,
            },
        }],
    }


def test_regression_selection_rejects_sparse_validation_rank_ic() -> None:
    selected = select_per_test_year(
        [_report("a", 1.0), _report("b", 1.0)],
        kind="regression",
        min_rank_ic_days=3,
    )

    assert selected[0]["status"] == "insufficient_validation_data"
    assert selected[0]["selected_model"] is None


def test_regression_selection_uses_validation_not_test_metrics() -> None:
    selected = select_per_test_year(
        [_report("a"), _report("b")],
        kind="regression",
        min_rank_ic_days=1,
    )

    assert selected[0]["status"] == "selected"
    assert selected[0]["selected_model"] == "b"


def test_form_portfolio_respects_minimum_cross_section() -> None:
    frame = pd.DataFrame({
        "stock_id": ["a", "b", "c", "d", "e", "f"],
        "entry_date": pd.to_datetime(["2026-01-01"] * 6),
        "actual_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "score": [0, 1, 2, 3, 4, 5],
        "test_year": [2026] * 6,
    })

    assert len(form_portfolio(frame, min_stocks=5)) == 1
    assert form_portfolio(frame, min_stocks=7).empty


def test_summary_does_not_annualize_sparse_event_days() -> None:
    summary = summarize_series(
        [0.1, -0.05], rng=np.random.default_rng(42), n_bootstrap=100
    )

    assert np.isclose(summary["cumulative_return"], 0.045)
    assert "annualized_return" not in summary