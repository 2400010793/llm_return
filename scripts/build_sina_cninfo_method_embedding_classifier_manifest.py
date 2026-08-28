"""Build the fair 4-model x 5-prompt x 5-representation classifier matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import discover_pooled_parts


MODELS = ("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large")
VARIANTS = ("plain", "short", "masked_short", "long", "masked_long")
# These five representations are present for every model/variant, including
# natural plain input. Prompt-only pooling is intentionally excluded because it
# does not exist for plain input and would break the balanced 4 x 5 matrix.
# The plain assets do not contain title_mean; these are the five features
# actually present for all 20 model/variant cells.
FEATURES = ("body_mean", "title_body_mean", "full_mean", "cls", "full_max")
CLASSIFIERS = (
    "logistic",
    "linear_svm",
    "sgd",
    "simple_mlp",
    "hist_gradient_boosting",
)
TARGET_CONFIG = {
    "event3_to_event3": ("event_return_3d", "event_return_3d"),
    "next1_to_next1": ("next_day_return", "next_day_return"),
    # Preserve the old Sina experiment only as an explicit legacy ablation.
    "event3_to_next1": ("event_return_3d", "next_day_return"),
}


def _rows_in_parts(parts: list[Path]) -> int:
    return sum(
        int(json.loads((part / "summary.json").read_text(encoding="utf-8"))["rows"])
        for part in parts
    )


def _validate_features(parts: list[Path], features: tuple[str, ...]) -> None:
    for part in parts:
        summary = json.loads((part / "summary.json").read_text(encoding="utf-8"))
        outputs = summary.get("outputs", {})
        missing = [feature for feature in features if feature not in outputs]
        if missing:
            raise ValueError(f"{part}: missing pooled features {missing}")


def _report_is_reusable(
    path: Path,
    *,
    panel: Path,
    embedding_root: Path,
    model: str,
    variant: str,
    feature: str,
    classifier: str,
    train_target: str,
    evaluation_target: str,
    expected_rows: int,
) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        experiment = report["experiment"]
        design = report["design"]
        input_info = report["input"]
        alignment = input_info.get("embedding_alignment", input_info.get("alignment", {}))
        artifact_bundle = Path(report["artifact_bundle"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        experiment.get("model") == model
        and experiment.get("variant") == variant
        and experiment.get("feature") == feature
        and experiment.get("classifier") == classifier
        and experiment.get("reducer") == "none"
        and experiment.get("reducer_components") in (None, 0)
        and design.get("run_mode") == "final-test"
        and design.get("train_target") == train_target
        and design.get("evaluation_target") == evaluation_target
        and design.get("seed") == 42
        and Path(input_info.get("panel", "")).resolve() == panel
        and Path(input_info.get("embedding_root", "")).resolve() == embedding_root
        and input_info.get("embedding_rows") == expected_rows
        and alignment.get("panel_rows") == expected_rows
        and alignment.get("embedding_rows") == expected_rows
        and alignment.get("matched_rows") == expected_rows
        and alignment.get("unmatched_panel_rows") == 0
        and artifact_bundle.is_dir()
        and (artifact_bundle / "COMPLETED").is_file()
    )


def build_rows(
    embedding_root: Path,
    output_root: Path,
    *,
    expected_shards: int = 4,
    expected_rows: int = 4928,
    panel: Path,
    target_modes: tuple[str, ...] = ("event3_to_next1",),
    reuse_output_roots: tuple[Path, ...] = (),
) -> list[dict[str, object]]:
    embedding_root = embedding_root.resolve()
    output_root = output_root.resolve()
    panel = panel.resolve()
    reuse_output_roots = tuple(root.resolve() for root in reuse_output_roots)
    rows: list[dict[str, object]] = []
    unknown_modes = sorted(set(target_modes).difference(TARGET_CONFIG))
    if unknown_modes:
        raise ValueError(f"unknown target modes: {', '.join(unknown_modes)}")
    for target_mode in target_modes:
        train_target, evaluation_target = TARGET_CONFIG[target_mode]
        for model in MODELS:
            for variant in VARIANTS:
                parts = discover_pooled_parts(embedding_root, model, variant)
                if len(parts) != expected_shards:
                    raise ValueError(
                        f"unexpected shard count for {model}/{variant}: "
                        f"{len(parts)} != {expected_shards}"
                    )
                observed_rows = _rows_in_parts(parts)
                if observed_rows != expected_rows:
                    raise ValueError(
                        f"unexpected row count for {model}/{variant}: "
                        f"{observed_rows} != {expected_rows}"
                    )
                _validate_features(parts, FEATURES)
                for feature in FEATURES:
                    for classifier in CLASSIFIERS:
                        output = output_root / target_mode / "classification" / (
                            f"{model}_{variant}_{feature}_{classifier}_none_0.json"
                        )
                        if any(
                            _report_is_reusable(
                                root / target_mode / "classification" / output.name,
                                panel=panel,
                                embedding_root=embedding_root,
                                model=model,
                                variant=variant,
                                feature=feature,
                                classifier=classifier,
                                train_target=train_target,
                                evaluation_target=evaluation_target,
                                expected_rows=expected_rows,
                            )
                            for root in (output_root, *reuse_output_roots)
                        ):
                            continue
                        rows.append({
                            "task_id": len(rows),
                            "embedding_root": str(embedding_root),
                            "model": model,
                            "variant": variant,
                            "feature": feature,
                            "classifier": classifier,
                            "reducer": "none",
                            "components": 0,
                            "output": str(output),
                            "train_target": train_target,
                            "evaluation_target": evaluation_target,
                        })
    expected_tasks = (
        len(target_modes) * len(MODELS) * len(VARIANTS)
        * len(FEATURES) * len(CLASSIFIERS)
    )
    if len(rows) > expected_tasks:
        raise AssertionError(f"unexpected task count: {len(rows)} > {expected_tasks}")
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(
            "/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/"
            "cninfo_method_v1/embeddings"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/"
            "cninfo_method_v1"
        ),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(
            "/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/"
            "classification/sina_single_stock_classification_panel.parquet"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "configs/generated/"
            "sina_cninfo_method_embedding_classifier_v1.tsv"
        ),
    )
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-rows", type=int, default=4928)
    parser.add_argument(
        "--target-mode",
        choices=tuple(TARGET_CONFIG),
        action="append",
        default=None,
        help=(
            "Target protocol to schedule; repeat for multiple protocols. "
            "The default preserves the legacy event3-to-next1 experiment."
        ),
    )
    parser.add_argument(
        "--reuse-output-root",
        type=Path,
        action="append",
        default=[],
        help="Existing exact report roots to reuse instead of scheduling again",
    )
    args = parser.parse_args()
    if args.expected_shards < 1 or args.expected_rows < 1:
        raise ValueError("expected-shards and expected-rows must be positive")
    target_modes = tuple(args.target_mode or ("event3_to_next1",))

    rows = build_rows(
        args.embedding_root,
        args.output_root,
        expected_shards=args.expected_shards,
        expected_rows=args.expected_rows,
        panel=args.panel,
        target_modes=target_modes,
        reuse_output_roots=tuple(args.reuse_output_root),
    )
    write_manifest(args.manifest, rows)
    summary = {
        "tasks": len(rows),
        "candidate_tasks": (
            len(target_modes) * len(MODELS) * len(VARIANTS)
            * len(FEATURES) * len(CLASSIFIERS)
        ),
        "reused_outputs": (
            len(target_modes) * len(MODELS) * len(VARIANTS)
            * len(FEATURES) * len(CLASSIFIERS)
            - len(rows)
        ),
        "models": list(MODELS),
        "variants": list(VARIANTS),
        "features": list(FEATURES),
        "classifiers": list(CLASSIFIERS),
        "reducer": "none",
        "target_modes": {
            mode: {
                "train_target": TARGET_CONFIG[mode][0],
                "evaluation_target": TARGET_CONFIG[mode][1],
            }
            for mode in target_modes
        },
        "protocol": "6y fit / 2y validation / 1y rolling OOS",
        "embedding_root": str(args.embedding_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "manifest": str(args.manifest.resolve()),
    }
    args.manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
