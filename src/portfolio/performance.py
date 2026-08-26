"""Performance summaries for daily portfolio return series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_metrics(
    portfolio: pd.DataFrame,
    *,
    annualization: int = 252,
    return_column: str = "long_short",
) -> dict[str, float]:
    """Summarize any daily portfolio return column.

    The default remains ``long_short`` for backward compatibility.  Holdings-
    level strategy results can instead pass ``gross_return`` or ``net_return``.
    """
    if annualization < 1:
        raise ValueError("annualization must be positive")
    if portfolio.empty:
        return {
            "n_days": 0.0, "mean": float("nan"),
            "annualized_return": float("nan"), "volatility": float("nan"),
            "annualized_volatility": float("nan"), "sharpe": float("nan"),
            "mean_t_stat": float("nan"),
            "geometric_annualized_return": float("nan"),
            "sortino": float("nan"), "calmar": float("nan"),
            "cumulative_return": float("nan"), "max_drawdown": float("nan"),
            "positive_day_rate": float("nan"), "best_day": float("nan"),
            "worst_day": float("nan"),
        }
    if return_column not in portfolio:
        raise ValueError(f"portfolio must contain {return_column}")
    values = pd.to_numeric(portfolio[return_column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"portfolio contains no finite {return_column} returns")
    mean = float(np.mean(values))
    volatility = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    cumulative = np.cumprod(1.0 + values)
    # Include the initial wealth of one; otherwise a loss on the first day is
    # incorrectly assigned a zero drawdown.
    running_peak = np.maximum.accumulate(np.maximum(cumulative, 1.0))
    drawdowns = cumulative / running_peak - 1.0
    cumulative_return = float(cumulative[-1] - 1.0)
    geometric_annualized = (
        float(cumulative[-1] ** (annualization / len(values)) - 1.0)
        if cumulative[-1] > 0 else float("nan")
    )
    downside_deviation = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    max_drawdown = float(drawdowns.min())
    return {
        "n_days": float(len(values)),
        "mean": mean,
        "annualized_return": mean * annualization,
        "volatility": volatility,
        "annualized_volatility": (
            volatility * np.sqrt(annualization) if np.isfinite(volatility) else float("nan")
        ),
        "sharpe": (
            mean / volatility * np.sqrt(annualization)
            if volatility > 0 and np.isfinite(volatility) else float("nan")
        ),
        "mean_t_stat": (
            mean / volatility * np.sqrt(len(values))
            if volatility > 0 and np.isfinite(volatility) else float("nan")
        ),
        "geometric_annualized_return": geometric_annualized,
        "sortino": (
            mean / downside_deviation * np.sqrt(annualization)
            if downside_deviation > 0 else float("nan")
        ),
        "calmar": (
            geometric_annualized / abs(max_drawdown)
            if max_drawdown < 0 and np.isfinite(geometric_annualized) else float("nan")
        ),
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "positive_day_rate": float((values > 0).mean()),
        "best_day": float(values.max()),
        "worst_day": float(values.min()),
    }


def strategy_metrics(portfolio: pd.DataFrame, *, annualization: int = 252) -> dict[str, object]:
    """Summarize gross/net performance and execution diagnostics."""
    required = {"gross_return", "net_return", "transaction_cost", "turnover"}
    missing = required.difference(portfolio.columns)
    if missing:
        raise ValueError(f"strategy result missing columns: {', '.join(sorted(missing))}")
    gross = portfolio_metrics(
        portfolio, annualization=annualization, return_column="gross_return"
    )
    net = portfolio_metrics(
        portfolio, annualization=annualization, return_column="net_return"
    )
    costs = pd.to_numeric(portfolio["transaction_cost"], errors="coerce")
    borrow = pd.to_numeric(
        portfolio.get("borrow_cost", pd.Series(0.0, index=portfolio.index)),
        errors="coerce",
    )
    turnover = pd.to_numeric(portfolio["turnover"], errors="coerce")
    blocked = pd.to_numeric(
        portfolio.get("blocked_notional", pd.Series(0.0, index=portfolio.index)),
        errors="coerce",
    )
    return {
        "gross": gross,
        "net": net,
        "execution": {
            "mean_daily_turnover": float(turnover.mean()),
            "annualized_turnover": float(turnover.mean() * annualization),
            "mean_daily_transaction_cost": float(costs.mean()),
            "annualized_transaction_cost": float(costs.mean() * annualization),
            "mean_daily_borrow_cost": float(borrow.mean()),
            "annualized_borrow_cost": float(borrow.mean() * annualization),
            "total_cost_drag": float((costs.fillna(0.0) + borrow.fillna(0.0)).sum()),
            "mean_blocked_notional": float(blocked.mean()),
        },
    }


__all__ = ["portfolio_metrics", "strategy_metrics"]
