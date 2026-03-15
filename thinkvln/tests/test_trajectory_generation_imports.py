from pathlib import Path


def test_trajectory_generation_uses_packaged_habitat_extensions_import():
    source = Path("thinkvln/datagen/generation/trajectory_generation.py").read_text()

    assert "from thinkvln.habitat_extensions import measures" in source
