"""Load and validate the project's YAML configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration used by data and modeling code."""

    root: Path
    paths: dict[str, Path]
    sample: dict[str, Any]
    models: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return value


def load_config(root: str | Path = PROJECT_ROOT) -> ProjectConfig:
    """Load config files and resolve all project-relative paths."""
    project_root = Path(root).expanduser().resolve()
    paths_raw = _read_yaml(project_root / "configs" / "paths.yaml")
    sample = _read_yaml(project_root / "configs" / "sample.yaml")
    models = _read_yaml(project_root / "configs" / "models.yaml")

    required_paths = {"raw_data", "interim_data", "processed_data", "reports", "logs"}
    missing = required_paths.difference(paths_raw)
    if missing:
        raise ValueError(f"Missing path settings: {', '.join(sorted(missing))}")
    paths = {
        key: (value if Path(value).is_absolute() else project_root / value)
        for key, value in paths_raw.items()
    }
    return ProjectConfig(root=project_root, paths=paths, sample=sample, models=models)


def prepare_directories(config: ProjectConfig) -> None:
    """Create configured runtime directories without touching raw data files."""
    for path in config.paths.values():
        path.mkdir(parents=True, exist_ok=True)
