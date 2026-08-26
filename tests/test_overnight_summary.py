from scripts.summarize_overnight_v1 import select_dynamic_confirmation


def test_dynamic_configuration_is_selected_on_2024_then_read_on_2025() -> None:
    rows = [
        {"model": "r", "variant": "m", "gate_mode": "dynamic", "config_tag": "a", "window": "tune2024", "rank_ic_mean": 0.1, "output": "a24"},
        {"model": "r", "variant": "m", "gate_mode": "dynamic", "config_tag": "b", "window": "tune2024", "rank_ic_mean": 0.2, "output": "b24"},
        {"model": "r", "variant": "m", "gate_mode": "dynamic", "config_tag": "a", "window": "confirm2025", "rank_ic_mean": 0.5, "output": "a25"},
        {"model": "r", "variant": "m", "gate_mode": "dynamic", "config_tag": "b", "window": "confirm2025", "rank_ic_mean": 0.3, "output": "b25"},
    ]
    selected = select_dynamic_confirmation(rows)[0]
    assert selected["selected_config_tag"] == "b"
    assert selected["confirm2025_rank_ic_mean"] == 0.3

