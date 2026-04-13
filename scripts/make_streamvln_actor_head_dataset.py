#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the first N full StreamVLN actor recurrent sequences to a new JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input materialized actor JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=10,
        help="Number of full (episode_key, subtask_position) sequences to keep.",
    )
    parser.add_argument(
        "--drop_fields",
        nargs="*",
        default=[],
        help="Optional row fields to remove from the output dataset.",
    )
    return parser.parse_args()


def _sequence_key(row: Dict) -> Tuple[str, str]:
    return (
        str(row.get("episode_key", "")),
        str(row.get("subtask_position", row.get("subtask", ""))),
    )


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_head_sequences(rows: Iterable[Dict], max_sequences: int) -> List[Dict]:
    kept: List[Dict] = []
    seen_keys: List[Tuple[str, str]] = []
    active_key = None
    limit = max(1, int(max_sequences))
    for row in rows:
        key = _sequence_key(row)
        if key != active_key:
            if key not in seen_keys:
                if len(seen_keys) >= limit:
                    break
                seen_keys.append(key)
            active_key = key
        kept.append(row)
    return kept


def drop_fields(rows: Iterable[Dict], fields: List[str]) -> List[Dict]:
    field_set = {str(field) for field in fields if str(field)}
    if not field_set:
        return [dict(row) for row in rows]
    cleaned: List[Dict] = []
    for row in rows:
        cleaned.append({key: value for key, value in row.items() if key not in field_set})
    return cleaned


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    kept = collect_head_sequences(iter_jsonl(args.input), max_sequences=args.max_sequences)
    cleaned = drop_fields(kept, args.drop_fields)
    row_count = write_jsonl(args.output, cleaned)
    seq_count = len({_sequence_key(row) for row in cleaned})
    print(f"wrote {row_count} rows across {seq_count} sequences to {args.output}")


if __name__ == "__main__":
    main()
