import numpy as np

from scripts.run_short_pooled_embeddings import (
    conditioned_prompt_ids,
    save_npz_atomic,
    sequence_audit,
)


def test_save_npz_atomic_stages_and_publishes(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = tmp_path / "output" / "short_pooling.npz"
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))

    save_npz_atomic(target, {"full_mean": np.arange(12).reshape(3, 4)})

    with np.load(target, allow_pickle=False) as archive:
        assert archive["full_mean"].tolist() == np.arange(12).reshape(3, 4).tolist()
    assert list(scratch.iterdir()) == []


def test_task_prompt_keeps_fixed_slots_and_masks_padding():
    ids, attention = conditioned_prompt_ids(
        [11, 12], prompt_slots=4, pad_id=0, condition="task_prompt"
    )
    assert ids == [11, 12, 0, 0]
    assert attention == [1, 1, 0, 0]


def test_position_matched_no_prompt_preserves_slots_but_masks_all():
    ids, attention = conditioned_prompt_ids(
        [11, 12], prompt_slots=4, pad_id=0,
        condition="no_prompt_position_matched",
    )
    assert ids == [0, 0, 0, 0]
    assert attention == [0, 0, 0, 0]


def test_natural_no_prompt_removes_slots_and_releases_budget():
    ids, attention = conditioned_prompt_ids(
        [11, 12], prompt_slots=4, pad_id=0, condition="no_prompt_natural"
    )
    assert ids == []
    assert attention == []


def test_sequence_audit_records_visible_positions_and_hash():
    audit = sequence_audit(
        [101, 0, 0, 21, 22, 31, 102],
        [-1, 0, 0, 1, 1, 2, -1],
        [1, 0, 0, 1, 1, 1, 1],
    )
    assert audit["visible_prompt_tokens"] == 0
    assert audit["visible_title_tokens"] == 2
    assert audit["visible_body_tokens"] == 1
    assert audit["title_start_zero_based"] == 3
    assert audit["title_end_zero_based"] == 4
    assert audit["body_start_zero_based"] == 5
    assert len(str(audit["input_sha256"])) == 64
