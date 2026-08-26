import json

import numpy as np

from scripts.monitor_bge_embedding_completion import (
    output_complete,
    output_directory,
    parse_job_variant,
)


def test_output_directory_maps_odd_bge_task_to_shard(tmp_path):
    assert output_directory(tmp_path, 17, "short") == (
        tmp_path / "shard-8" / "bge_m3" / "short"
    )


def test_output_complete_requires_summary_outputs_and_files(tmp_path):
    directory = tmp_path / "part"
    directory.mkdir()
    (directory / "metadata.jsonl").write_text('{"row_index": 1}\n')
    np.savez_compressed(
        directory / "short_pooling.npz",
        **{
            name: np.ones((1, 2)) for name in (
                "prompt_token_embeddings", "prompt_mean", "title_mean", "body_mean",
                "title_body_mean", "full_mean",
            )
        },
    )
    (directory / "summary.json").write_text(json.dumps({
        "rows": 1,
        "outputs": {
            name: [1, 2] for name in (
                "prompt_token_embeddings", "prompt_mean", "title_mean", "body_mean",
                "title_body_mean", "full_mean",
            )
        },
    }))

    assert output_complete(directory)
    (directory / "metadata.jsonl").unlink()
    assert not output_complete(directory)


def test_parse_job_variant():
    assert parse_job_variant("4330365:short") == ("4330365", "short")
