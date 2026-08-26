from pathlib import Path

import pandas as pd

from scripts.summarize_simple_states_results import collect_results


def test_collect_results_ranks_without_flipping_direction(tmp_path: Path):
    for factor, ic, sharpe in (("a", 0.01, 0.2), ("b", -0.02, 0.5)):
        folder = tmp_path / factor
        folder.mkdir()
        pd.DataFrame([{
            "factor": factor, "IC": ic, "ICIR": ic / 2,
            "Ret": sharpe / 10, "Sharpe": sharpe,
            "ICTN": 0.0, "SharpeTN": 0.0,
        }]).to_csv(folder / f"{factor}.stats.csv", index=False)
    result = collect_results(tmp_path)
    assert result["factor"].tolist() == ["a", "b"]
    assert result["rank_IC"].tolist() == [1, 2]
    assert result.loc[result["factor"].eq("b"), "rank_Sharpe"].item() == 1


def test_collect_results_keeps_missing_metric_rank_nullable(tmp_path: Path):
    folder = tmp_path / "factor"
    folder.mkdir()
    pd.DataFrame([{
        "factor": "factor", "IC": 0.01, "ICIR": 0.1,
        "Ret": 0.02, "Sharpe": 0.2,
        "ICTN": 0.0, "SharpeTN": float("nan"),
    }]).to_csv(folder / "factor.stats.csv", index=False)

    result = collect_results(tmp_path)

    assert str(result["rank_SharpeTN"].dtype) == "Int64"
    assert pd.isna(result.loc[0, "rank_SharpeTN"])
