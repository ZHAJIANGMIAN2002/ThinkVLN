#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _to_subtask_idx(value: Any) -> int:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return 1
    return idx if idx > 0 else 1


def _make_stat(success: int, total: int) -> Dict[str, Any]:
    return {"success": success, "total": total, "rate": (success / total) if total > 0 else 0.0}


def collect_stats(run_dir: str, pattern: str = "subtask_closed_loop_rank*.jsonl") -> Dict[str, Any]:
    root = Path(run_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {run_dir}")

    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' under: {run_dir}")

    total_success = 0
    total_count = 0
    by_file: Dict[str, Dict[str, Any]] = {}
    by_subtask_raw = defaultdict(lambda: [0, 0])  # idx -> [success, total]

    for file_path in files:
        file_success = 0
        file_count = 0
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {file_path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    continue

                success = _to_bool(record.get("success", False))
                subtask_idx = _to_subtask_idx(record.get("subtask_idx", 1))

                file_count += 1
                file_success += int(success)
                by_subtask_raw[subtask_idx][0] += int(success)
                by_subtask_raw[subtask_idx][1] += 1

        total_count += file_count
        total_success += file_success
        by_file[file_path.name] = _make_stat(file_success, file_count)

    by_subtask_idx = {
        idx: _make_stat(success=success, total=total)
        for idx, (success, total) in sorted(by_subtask_raw.items(), key=lambda item: item[0])
    }

    return {
        "directory": str(root),
        "pattern": pattern,
        "num_files": len(files),
        "overall": _make_stat(total_success, total_count),
        "by_file": by_file,
        "by_subtask_idx": by_subtask_idx,
    }


def _format_rate(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def _print_stats(stats: Dict[str, Any]) -> None:
    overall = stats["overall"]
    print(f"Directory: {stats['directory']}")
    print(f"Pattern:   {stats['pattern']}")
    print(f"Files:     {stats['num_files']}")
    print(
        f"Overall:   {overall['success']}/{overall['total']} "
        f"({_format_rate(overall['rate'])})"
    )

    print("\nBy file:")
    for name, item in stats["by_file"].items():
        print(f"  {name}: {item['success']}/{item['total']} ({_format_rate(item['rate'])})")

    print("\nBy subtask_idx:")
    for idx, item in stats["by_subtask_idx"].items():
        print(f"  {idx}: {item['success']}/{item['total']} ({_format_rate(item['rate'])})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize success rates from subtask_closed_loop_rank*.jsonl files."
    )
    parser.add_argument("run_dir", help="Directory containing subtask_closed_loop_rank*.jsonl")
    parser.add_argument(
        "--pattern",
        default="subtask_closed_loop_rank*.jsonl",
        help="Glob pattern for result files (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = collect_stats(args.run_dir, pattern=args.pattern)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        _print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
