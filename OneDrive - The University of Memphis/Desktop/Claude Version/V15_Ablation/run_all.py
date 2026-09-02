"""Run the V15 one-step feature-ablation folders sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=list(range(13, 0, -1)),
        help="Feature counts to run; default is 13 12 ... 1.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Do not aggregate the completed pure-test metrics at the end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list the folders without starting training.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    counts = []
    for count in arguments.counts:
        if not 1 <= count <= 13:
            raise ValueError("Every feature count must be between 1 and 13.")
        if count not in counts:
            counts.append(count)

    for position, count in enumerate(counts, start=1):
        folder = HERE / f"F{count:02d}"
        main_file = folder / "main.py"
        if not main_file.exists():
            raise FileNotFoundError(f"Missing model entry point: {main_file}")
        print("\n" + "=" * 88, flush=True)
        print(
            f"ABLATION RUN {position}/{len(counts)}: {count} FEATURES",
            flush=True,
        )
        print(f"Folder: {folder}", flush=True)
        print("=" * 88, flush=True)
        if arguments.dry_run:
            continue
        subprocess.run(
            [sys.executable, str(main_file)],
            cwd=folder,
            check=True,
        )

    if arguments.dry_run:
        print("\nDry run complete; no model was trained or pushed.")
        return

    if not arguments.skip_comparison:
        subprocess.run(
            [sys.executable, str(HERE / "compare.py")],
            cwd=HERE,
            check=True,
        )

    print("\nRequested feature-ablation runs completed successfully.")


if __name__ == "__main__":
    main()
