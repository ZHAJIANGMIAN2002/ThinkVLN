from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from thinkvln.datagen.manual_switch_web.app import ManualSwitchConfig, create_app, prepare_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a manual watcher switch-annotation web app.")
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--manifest_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_full_path", type=Path, default=None)
    parser.add_argument("--image_stride", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--title", type=str, default="Watcher Manual Switch Annotation")
    parser.add_argument("--page_size", type=int, default=20)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ManualSwitchConfig:
    return ManualSwitchConfig(
        bundle_root=args.bundle_root.resolve(),
        manifest_file=args.manifest_file.resolve(),
        output_file=args.output_file.resolve(),
        summary_full_path=args.summary_full_path.resolve() if args.summary_full_path is not None else None,
        image_stride=int(args.image_stride),
        max_samples=args.max_samples,
        shuffle=bool(args.shuffle),
        seed=args.seed,
        title=str(args.title),
        page_size=int(args.page_size),
    )


def main() -> None:
    args = parse_args()
    app = create_app(build_config(args))
    print(f"[manual-switch] serving http://{args.host}:{args.port}")
    print(f"[manual-switch] bundle_root={args.bundle_root}")
    print(f"[manual-switch] manifest_file={args.manifest_file}")
    print(f"[manual-switch] output_file={args.output_file}")
    uvicorn.run(app, host=args.host, port=int(args.port))


__all__ = ["build_config", "create_app", "parse_args", "prepare_sample"]


if __name__ == "__main__":
    main()
