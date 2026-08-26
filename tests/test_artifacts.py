import joblib
import numpy as np

from src.evaluation.artifacts import atomic_joblib


def test_atomic_joblib_stages_locally_and_publishes(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = tmp_path / "output" / "model.joblib"
    value = {"weights": np.arange(12).reshape(3, 4)}
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))

    atomic_joblib(target, value)

    loaded = joblib.load(target)
    assert loaded["weights"].tolist() == value["weights"].tolist()
    assert list(scratch.iterdir()) == []
