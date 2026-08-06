from pathlib import Path

from src.config import load_config


def test_load_config_resolves_project_paths() -> None:
    config = load_config(Path(__file__).resolve().parents[1])

    assert config.root.name == "llm_return"
    assert config.paths["raw_data"] == config.root / "data" / "raw"
    assert config.sample["market"] == "China_A_share"
    assert "tfidf_ridge" in config.models["text_models"]
