import argparse
import json
import subprocess
from pathlib import Path

from scripts import task_tracker


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_task_record_precedes_fake_submission_and_captures_dirty_state(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    (repo / "tracked.txt").write_text("before\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-qm", "initial")
    (repo / "tracked.txt").write_text("after\n", encoding="utf-8")
    (repo / "untracked.py").write_text("print('snapshot')\n", encoding="utf-8")

    fake_sbatch = repo / "fake_sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        "test -f \"$TASK_RECORD_ROOT/$TASK_RECORD_ID/manifest.json\" || exit 9\n"
        "printf '987654\\n'\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    record_root = repo / "task_records"
    monkeypatch.setattr(task_tracker, "ROOT", repo)
    args = argparse.Namespace(
        record_root=str(record_root),
        name="fake submit",
        purpose="test pre-submission provenance",
        related_file=[],
        input=["input"],
        output=["output"],
        seed=42,
        tag=["test"],
    )

    task_id, task_dir, initial = task_tracker.create_task(args, [str(fake_sbatch)])
    assert initial["status"] == "recorded"
    assert "tracked.txt" in (task_dir / "code" / "tracked-dirty.patch").read_text(encoding="utf-8")
    assert "untracked.py" in (task_dir / "code" / "untracked-files.txt").read_text(encoding="utf-8")

    returncode = task_tracker.run_recorded_command(task_id, task_dir, [str(fake_sbatch)])
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    assert returncode == 0
    assert manifest["status"] == "submitted"
    assert manifest["top_level_job_ids"] == ["987654"]
    assert manifest["git"]["dirty"] is True
    assert manifest["seed"] == 42


def test_secret_environment_values_are_redacted():
    assert task_tracker.redact("SOME_API_TOKEN", "secret") == "<redacted>"
    assert task_tracker.redact("OMP_NUM_THREADS", "4") == "4"
