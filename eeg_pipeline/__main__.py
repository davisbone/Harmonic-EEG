"""Run with ``uv run --extra preprocessing python -m eeg_pipeline --help``."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "audit", "build"])
    parser.add_argument("--data-dir", type=Path, default=Path("Data/on003570"))
    parser.add_argument("--subjects", nargs="+", help="For example: sub-002 sub-009")
    parser.add_argument("--output-dir", type=Path, help="Build destination (default: results)")
    args = parser.parse_args()
    if args.command == "build" and args.subjects and args.output_dir is None:
        parser.error("A participant subset needs --output-dir to keep it separate from the full table")

    if args.command == "download":
        from .source import download
        download(args.data_dir, args.subjects)
    elif args.command == "audit":
        from .source import audit
        participants, events = audit(args.data_dir, args.subjects)
        print(participants.to_string(index=False))
        print(f"{len(events):,} candidate stimulus events")
    else:
        from .build import build
        build(args.data_dir, args.output_dir or Path("results"), args.subjects)


if __name__ == "__main__":
    main()
