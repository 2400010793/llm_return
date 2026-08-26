"""Freeze linear candidates from validation reports and build test manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_qwen_linear_manifests import write


def score(row: dict[str, object]) -> tuple[float, float, float]:
    def number(name: str, fallback: float) -> float:
        value = float(row.get(name, np.nan))
        return value if np.isfinite(value) else fallback
    return (
        number("rank_ic_mean", -np.inf),
        number("oos_r2_vs_historical_mean", -np.inf),
        -number("mae", np.inf),
    )


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--rolling-manifest", type=Path, required=True)
    args = parser.parse_args()
    ranking: list[dict[str, object]] = []
    missing: list[str] = []
    for task in load_manifest(args.screen_manifest):
        output = Path(task["output"])
        if not output.is_file():
            missing.append(str(output))
            continue
        report = json.loads(output.read_text(encoding="utf-8"))
        result = report["results"][0]
        ranking.append({**task, **result["validation_metrics"],
                        "best_params": json.dumps(result["best_params"], sort_keys=True)})
    if missing:
        raise SystemExit(f"linear screen incomplete: {len(missing)} missing outputs")

    selected: list[dict[str, object]] = []
    qwen = [row for row in ranking if row["model"] == "qwen3_embedding_8b"]
    for variant in ("plain", "short", "masked_short"):
        selected.append(max((row for row in qwen if row["variant"] == variant), key=score))
    for model in ("roberta", "bge_m3"):
        selected.append(max((row for row in ranking if row["model"] == model), key=score))

    ranking.sort(key=score, reverse=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranking).to_csv(args.output_root / "validation_linear_ranking.csv", index=False)
    frozen = {
        "selection_data": "2018-2023 fit, 2024-2025 validation; no test metrics",
        "selection_rule": ["rank_ic_mean", "oos_r2_vs_historical_mean", "negative_mae"],
        "selected": selected,
    }
    (args.output_root / "selected_candidates.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def tasks(phase: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for candidate in selected:
            seeds = [42]
            label = "best"
            for seed in seeds:
                stem = (
                    f"{candidate['model']}_{candidate['variant']}_{candidate['regressor']}_"
                    f"{candidate['reducer']}{candidate['components']}_seed{seed}"
                )
                rows.append({
                    key: candidate[key] for key in (
                        "model", "embedding_root", "variant", "target", "regressor",
                        "reducer", "components",
                    )
                } | {"run_mode": "final-test", "seed": seed,
                     "output": str(args.output_root / phase / f"{label}_{stem}.json")})
        # Huber stability is independent of whether Huber wins the overall screen.
        for variant in ("plain", "short", "masked_short"):
            huber = max((row for row in qwen if row["variant"] == variant
                         and row["regressor"] == "huber_sgd"), key=score)
            for seed in (42, 43, 44, 45, 46):
                stem = f"qwen3_embedding_8b_{variant}_huber_{huber['reducer']}{huber['components']}_seed{seed}"
                rows.append({
                    key: huber[key] for key in (
                        "model", "embedding_root", "variant", "target", "regressor",
                        "reducer", "components",
                    )
                } | {"run_mode": "final-test", "seed": seed,
                     "output": str(args.output_root / phase / "huber_stability" / f"{stem}.json")})
        return rows

    write(args.strict_manifest, tasks("strict_6_2_1"))
    write(args.rolling_manifest, tasks("rolling_3_1_1"))
    print(json.dumps({"screen": len(ranking), "selected": len(selected),
                      "strict_tasks": len(load_manifest(args.strict_manifest)),
                      "rolling_tasks": len(load_manifest(args.rolling_manifest))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
