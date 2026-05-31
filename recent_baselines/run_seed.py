from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from common import SEED_REPO, seed_script_for_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or execute an official SEED experiment script."
    )
    parser.add_argument("--repo", required=True, help="Path to a local SEED checkout.")
    parser.add_argument(
        "--protocol",
        default="equal_10_tasks",
        choices=["equal_10_tasks", "equal_20_tasks", "equal_50_tasks", "large_first", "custom"],
        help="Known SEED protocol to run.",
    )
    parser.add_argument(
        "--script",
        help="Shell script relative to the SEED repo. Required for --protocol custom.",
    )
    parser.add_argument("--execute", action="store_true", help="Run the script instead of printing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    script = args.script or seed_script_for_protocol(args.protocol)
    if not script:
        print("--script is required when --protocol custom is used.")
        raise SystemExit(2)

    script_path = repo / script
    if not script_path.exists():
        print(f"SEED repo is not ready: {repo}")
        print(f"Missing script: {script}")
        print(f"Official repo: {SEED_REPO}")
        raise SystemExit(2)

    command = ["bash", script]
    print("SEED command:")
    print(" ".join(command))

    if not args.execute:
        print("Dry run only. Add --execute after checking the FACIL/SEED environment.")
        return

    env = os.environ.copy()
    subprocess.run(command, cwd=repo, check=True, env=env)


if __name__ == "__main__":
    main()
