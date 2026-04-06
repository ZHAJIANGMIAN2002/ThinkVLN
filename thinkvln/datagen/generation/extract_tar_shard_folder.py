from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, List


def _read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_extract(handle: tarfile.TarFile, output_root: Path) -> None:
    output_root = output_root.resolve()
    for member in handle.getmembers():
        target = (output_root / member.name).resolve()
        if target != output_root and output_root not in target.parents:
            raise ValueError(f"unsafe tar member path: {member.name}")
    handle.extractall(output_root)


def extract_shard_folder(input_root: Path, output_root: Path) -> Dict[str, int | str]:
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = input_root / "manifest.jsonl"
    shards_dir = input_root / "shards"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"shards dir not found: {shards_dir}")

    rows = _read_jsonl(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)

    shard_names: List[str] = []
    for row in rows:
        shard_name = str(row.get("shard_name") or "").strip()
        if shard_name and shard_name not in shard_names:
            shard_names.append(shard_name)

    for shard_name in shard_names:
        shard_path = shards_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"missing shard: {shard_path}")
        with tarfile.open(shard_path, "r") as handle:
            _safe_extract(handle, output_root)

    for row in rows:
        relpath = str(row.get("relpath") or "").strip()
        if relpath and not (output_root / relpath).exists():
            raise FileNotFoundError(f"missing extracted file: {output_root / relpath}")

    return {
        "num_files": len(rows),
        "num_shards": len(shard_names),
        "output_root": str(output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract tar shards back into the original folder tree.")
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = extract_shard_folder(args.input_root, args.output_root)
    print(
        f"tar shards extracted: files={result['num_files']} shards={result['num_shards']} "
        f"output={result['output_root']}"
    )


if __name__ == "__main__":
    main()
