from __future__ import annotations

import argparse
import json

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bat-tracker",
        description="Track moving bats in IR monochrome cave videos.",
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--config", required=False, default=None, help="Path to config YAML")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    meta = run_pipeline(input_video=args.input, output_dir=args.output, config_path=args.config)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
