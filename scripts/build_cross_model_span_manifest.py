"""Build bounded cross-model span regression jobs."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-raw", action="store_true")
    args = p.parse_args()
    rows = []
    semantics = ("return_span", "stock_span", "return_span,stock_span")
    modes = [("pca", 64), ("pca", 128)]
    if args.include_raw:
        modes.append(("raw", 0))
    for values in product(semantics, ("short", "masked_short"), ("next_day_return", "event_return_3d"), modes, (False, True)):
        semantic, variant, target, (label, components), cluster = values
        rows.append({
            "task_id": len(rows), "semantics": semantic, "variant": variant,
            "target": target, "components": components, "cluster": int(cluster),
            "mode": label,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"tasks={len(rows)}")


if __name__ == "__main__":
    main()
