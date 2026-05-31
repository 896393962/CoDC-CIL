from __future__ import annotations

import argparse
from pathlib import Path

from common import FIELDNAMES, official_records, write_records_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export recent FeTrIL/SEED reference results for the CoDC rebuttal."
    )
    parser.add_argument(
        "--output",
        default="official_recent_baseline_results.csv",
        help="CSV path to write.",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Also print the records as a Markdown table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = official_records()
    output = write_records_csv(records, Path(args.output))
    print(f"Wrote {len(records)} records to {output}")
    print(
        "Note: protocol=cifar100_b20_inc20_n5 is marked third-party-reproduced, "
        "not official-original."
    )
    if args.print_markdown:
        print(markdown_table(records))


def markdown_table(records: list[dict[str, str]]) -> str:
    headers = ["protocol", "method", "top1", "std", "source_type", "notes"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(record.get(field, "") for field in headers) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
