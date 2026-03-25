from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from thinkvln.datagen.generation.watcher_utils import append_jsonl


_SIZE_SUFFIXES = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024 * 1024,
    "mb": 1024 * 1024,
    "g": 1024 * 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
}


def parse_size_bytes(raw_value: str | int) -> int:
    text = str(raw_value).strip().lower()
    if not text:
        raise ValueError("max shard size is empty")
    if text.isdigit():
        value = int(text)
        if value < 1:
            raise ValueError("max shard size must be >= 1")
        return value
    suffix = "".join(ch for ch in text if ch.isalpha())
    number = text[: len(text) - len(suffix)]
    if suffix not in _SIZE_SUFFIXES or not number:
        raise ValueError(f"invalid max shard size: {raw_value}")
    value = int(number) * _SIZE_SUFFIXES[suffix]
    if value < 1:
        raise ValueError("max shard size must be >= 1")
    return value


def iter_input_files(input_root: Path) -> Iterable[Tuple[str, Path, int]]:
    for path in sorted(item for item in input_root.rglob("*") if item.is_file()):
        relpath = path.relative_to(input_root).as_posix()
        yield relpath, path, path.stat().st_size


def _build_shard_groups(
    files: Sequence[Tuple[str, Path, int]],
    max_shard_bytes: int,
) -> List[List[Tuple[str, Path, int]]]:
    shards: List[List[Tuple[str, Path, int]]] = []
    current: List[Tuple[str, Path, int]] = []
    current_size = 0
    for item in files:
        _, _, size = item
        if current and current_size + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(item)
        current_size += size
        if size >= max_shard_bytes:
            shards.append(current)
            current = []
            current_size = 0
    if current:
        shards.append(current)
    return shards


def _write_shard(shard_path: Path, files: Sequence[Tuple[str, Path, int]]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as handle:
        for relpath, file_path, _ in files:
            handle.add(file_path, arcname=relpath, recursive=False)


def _clear_generated_shards(shards_dir: Path) -> None:
    if not shards_dir.exists():
        return
    for path in shards_dir.glob("shard-*.tar"):
        path.unlink()


def shard_folder(
    input_root: Path,
    output_root: Path,
    max_shard_bytes: int,
) -> Dict[str, int | str]:
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    max_shard_bytes = int(max_shard_bytes)
    if max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be >= 1")

    shards_dir = output_root / "shards"
    manifest_path = output_root / "manifest.jsonl"
    files = list(iter_input_files(input_root))
    shard_groups = _build_shard_groups(files, max_shard_bytes=max_shard_bytes)

    shards_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_shards(shards_dir)
    if manifest_path.exists():
        manifest_path.unlink()

    for shard_index, shard_files in enumerate(shard_groups):
        shard_name = f"shard-{shard_index:05d}.tar"
        shard_path = shards_dir / shard_name
        _write_shard(shard_path, shard_files)
        for relpath, _, size in shard_files:
            append_jsonl(
                manifest_path,
                {
                    "relpath": relpath,
                    "shard_name": shard_name,
                    "member_name": relpath,
                    "size": int(size),
                },
            )

    return {
        "num_input_files": len(files),
        "num_shards": len(shard_groups),
        "manifest_file": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack a folder into byte-budgeted tar shards.")
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_shard_size", type=str, default="512MB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = shard_folder(
        input_root=args.input_root,
        output_root=args.output_root,
        max_shard_bytes=parse_size_bytes(args.max_shard_size),
    )
    print(
        f"tar shards written: files={result['num_input_files']} shards={result['num_shards']} "
        f"manifest={result['manifest_file']}"
    )


if __name__ == "__main__":
    main()
