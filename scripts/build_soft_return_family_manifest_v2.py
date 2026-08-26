from itertools import product
from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for i, (model, mode, target, year) in enumerate(product(
        ("roberta", "bge_m3"), ("unmasked", "masked", "fusion"),
        ("event_return_3d", "next_day_return"), range(2018, 2027),
    )):
        rows.append({"task_id": i, "model": model, "feature_mode": mode,
                     "target": target, "test_year": year})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"tasks={len(rows)}")

if __name__ == "__main__":
    main()
