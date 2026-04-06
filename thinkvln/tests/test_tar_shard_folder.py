import json
import tarfile
from pathlib import Path

import pytest

import thinkvln.datagen.generation.extract_tar_shard_folder as extract_tar_shard_folder
import thinkvln.datagen.generation.tar_shard_folder as tar_shard_folder


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_shard_folder_preserves_relative_paths_and_manifest(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_file(input_root / "scene_a" / "000000_rgb.jpg", b"a" * 10)
    _write_file(input_root / "scene_a" / "000001_rgb.jpg", b"b" * 10)

    result = tar_shard_folder.shard_folder(
        input_root=input_root,
        output_root=output_root,
        max_shard_bytes=64,
    )

    assert result["num_input_files"] == 2
    assert result["num_shards"] == 1

    shard_path = output_root / "shards" / "shard-00000.tar"
    assert shard_path.exists()

    with tarfile.open(shard_path, "r") as handle:
        assert sorted(handle.getnames()) == [
            "scene_a/000000_rgb.jpg",
            "scene_a/000001_rgb.jpg",
        ]

    manifest_rows = _read_jsonl(output_root / "manifest.jsonl")
    assert manifest_rows == [
        {
            "relpath": "scene_a/000000_rgb.jpg",
            "shard_name": "shard-00000.tar",
            "member_name": "scene_a/000000_rgb.jpg",
            "size": 10,
        },
        {
            "relpath": "scene_a/000001_rgb.jpg",
            "shard_name": "shard-00000.tar",
            "member_name": "scene_a/000001_rgb.jpg",
            "size": 10,
        },
    ]


def test_shard_folder_splits_by_byte_budget(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_file(input_root / "a.jpg", b"a" * 50)
    _write_file(input_root / "b.jpg", b"b" * 40)
    _write_file(input_root / "c.jpg", b"c" * 20)

    result = tar_shard_folder.shard_folder(
        input_root=input_root,
        output_root=output_root,
        max_shard_bytes=80,
    )

    assert result["num_shards"] == 2
    manifest_rows = _read_jsonl(output_root / "manifest.jsonl")
    assert [row["shard_name"] for row in manifest_rows] == [
        "shard-00000.tar",
        "shard-00001.tar",
        "shard-00001.tar",
    ]


def test_shard_folder_places_oversized_file_in_own_shard(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_file(input_root / "big.jpg", b"x" * 120)
    _write_file(input_root / "small.jpg", b"y" * 20)

    result = tar_shard_folder.shard_folder(
        input_root=input_root,
        output_root=output_root,
        max_shard_bytes=80,
    )

    assert result["num_shards"] == 2
    manifest_rows = _read_jsonl(output_root / "manifest.jsonl")
    assert manifest_rows[0]["shard_name"] == "shard-00000.tar"
    assert manifest_rows[1]["shard_name"] == "shard-00001.tar"

    with tarfile.open(output_root / "shards" / "shard-00000.tar", "r") as handle:
        assert handle.getnames() == ["big.jpg"]


def test_shard_folder_removes_stale_generated_shards_on_rerun(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    shards_dir = output_root / "shards"
    _write_file(input_root / "only.jpg", b"a" * 10)
    shards_dir.mkdir(parents=True, exist_ok=True)
    (shards_dir / "shard-00000.tar").write_bytes(b"stale-0")
    (shards_dir / "shard-00001.tar").write_bytes(b"stale-1")

    result = tar_shard_folder.shard_folder(
        input_root=input_root,
        output_root=output_root,
        max_shard_bytes=64,
    )

    assert result["num_shards"] == 1
    assert sorted(path.name for path in shards_dir.glob("*.tar")) == ["shard-00000.tar"]


def test_extract_shard_folder_restores_original_tree_from_manifest(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    shard_root = tmp_path / "sharded"
    restore_root = tmp_path / "restored"
    _write_file(input_root / "images" / "scene" / "000000_rgb.jpg", b"a" * 10)
    _write_file(input_root / "manifest" / "watcher_rollout_manifest.jsonl", b'{"sample_id":"x"}\n')

    tar_shard_folder.shard_folder(
        input_root=input_root,
        output_root=shard_root,
        max_shard_bytes=64,
    )

    result = extract_tar_shard_folder.extract_shard_folder(
        input_root=shard_root,
        output_root=restore_root,
    )

    assert result["num_files"] == 2
    assert result["num_shards"] == 1
    assert (restore_root / "images" / "scene" / "000000_rgb.jpg").read_bytes() == b"a" * 10
    assert (restore_root / "manifest" / "watcher_rollout_manifest.jsonl").read_text(encoding="utf-8") == '{"sample_id":"x"}\n'


def test_extract_shard_folder_fails_when_manifest_references_missing_shard(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    shard_root = tmp_path / "sharded"
    restore_root = tmp_path / "restored"
    _write_file(input_root / "a.txt", b"hello")

    tar_shard_folder.shard_folder(
        input_root=input_root,
        output_root=shard_root,
        max_shard_bytes=64,
    )
    (shard_root / "shards" / "shard-00000.tar").unlink()

    with pytest.raises(FileNotFoundError, match="missing shard"):
        extract_tar_shard_folder.extract_shard_folder(
            input_root=shard_root,
            output_root=restore_root,
        )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("128", 128),
        ("2kb", 2 * 1024),
        ("3MB", 3 * 1024 * 1024),
        ("1g", 1024 * 1024 * 1024),
    ],
)
def test_parse_size_bytes_supports_suffixes(raw_value: str, expected: int) -> None:
    assert tar_shard_folder.parse_size_bytes(raw_value) == expected
