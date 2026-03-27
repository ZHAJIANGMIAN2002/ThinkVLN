import subprocess
import sys
from pathlib import Path


def test_smoke_streamvln_actor_train_step_script_runs_help_from_repo_root():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "smoke_streamvln_actor_train_step.py"

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
