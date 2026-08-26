from pathlib import Path

from scripts import build_pooled_gating_manifest as manifest


def test_next1_gating_manifest_is_bounded(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(manifest, "discover_pooled_parts", lambda *_args: [tmp_path] * 32)
    monkeypatch.setattr(manifest, "completed_rows", lambda _parts: 350577)
    rows = manifest.build_rows(tmp_path)
    assert len(rows) == 8
    assert {row["gate_method"] for row in rows} == {"variance", "logistic_l1"}
    assert {row["gate_keep"] for row in rows} == {2}
    assert {row["classifier"] for row in rows} == {"simple_mlp"}
    assert {row["train_target"] for row in rows} == {"next_day_return"}
    assert {row["evaluation_target"] for row in rows} == {"next_day_return"}
    assert all("next1_segment_gating" in str(row["output"]) for row in rows)