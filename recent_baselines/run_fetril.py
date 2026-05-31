from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import (
    FETRIL_REPO,
    fetril_pipeline_commands,
    format_commands,
    missing_paths,
)


REQUIRED_SCRIPTS = [
    "codes/scratch.py",
    "codes/compute_distances.py",
    "codes/prepare_train.py",
    "codes/train_classifiers.py",
    "codes/compute_predictions.py",
    "codes/eval.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or execute the official FeTrIL pipeline."
    )
    parser.add_argument("--repo", required=True, help="Path to a local FeTrIL checkout.")
    parser.add_argument("--config", required=True, help="Config path relative to the FeTrIL repo.")
    parser.add_argument("--python", default="python", help="Python executable in the FeTrIL env.")
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Run codes/clean_train.py between classifier training and prediction.",
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    required = list(REQUIRED_SCRIPTS)
    if args.include_clean:
        required.append("codes/clean_train.py")

    missing = missing_paths(repo, required + [args.config])
    if missing:
        print(f"FeTrIL repo is not ready: {repo}")
        print("Missing paths:")
        for item in missing:
            print(f"  - {item}")
        print(f"Official repo: {FETRIL_REPO}")
        raise SystemExit(2)

    commands = fetril_pipeline_commands(args.config, args.python, args.include_clean)
    print("FeTrIL commands:")
    print(format_commands(commands))

    if not args.execute:
        print("Dry run only. Add --execute after checking the config and environment.")
        return

    for command in commands:
        subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
