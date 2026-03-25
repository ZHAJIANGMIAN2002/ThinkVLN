from pathlib import Path


def test_run_watcher_manual_annotation_uses_module_invocation():
    script_path = Path("scripts/run_watcher_manual_annotation.sh")
    script_text = script_path.read_text(encoding="utf-8")
    assert 'python -m thinkvln.datagen.generation.watcher_manual_done_annotation' in script_text
