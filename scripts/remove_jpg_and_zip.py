#!/usr/bin/env python3
import argparse
import os
import zipfile


def remove_jpgs(root_dir: str) -> int:
    removed = 0
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".jpg"):
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
    return removed


def zip_directory(source_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for name in files:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, source_dir)
                zf.write(path, rel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove JPGs and zip directory.")
    parser.add_argument("--dir", required=True, help="Target directory to clean")
    parser.add_argument("--zip_path", required=True, help="Output zip path")
    args = parser.parse_args()

    removed = remove_jpgs(args.dir)
    print(f"Removed {removed} jpg files from {args.dir}")
    zip_directory(args.dir, args.zip_path)
    print(f"Zip written to {args.zip_path}")


if __name__ == "__main__":
    main()
