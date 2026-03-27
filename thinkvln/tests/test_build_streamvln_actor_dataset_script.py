import subprocess
import sys
from pathlib import Path


def test_build_streamvln_actor_dataset_script_runs_help_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "build_streamvln_actor_dataset.py"

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
