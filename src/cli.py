"""Small command-line entry point for checking the project configuration."""

from __future__ import annotations

import argparse

from src.config import load_config, prepare_directories


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare project configuration")
    parser.add_argument("--root", default=None, help="Project root; defaults to the repository root")
    parser.add_argument("--prepare-dirs", action="store_true", help="Create configured runtime directories")
    args = parser.parse_args()

    config = load_config(args.root) if args.root else load_config()
    if args.prepare_dirs:
        prepare_directories(config)
    print(f"project_root={config.root}")
    for name, path in config.paths.items():
        print(f"{name}={path}")
    print(f"text_models={','.join(config.models.get('text_models', []))}")


if __name__ == "__main__":
    main()
