from __future__ import annotations

import csv
import shlex
from pathlib import Path
from typing import Iterable, Sequence


FIELDNAMES = [
    "dataset",
    "protocol",
    "method",
    "top1",
    "std",
    "metric",
    "source_type",
    "source",
    "notes",
]

FETRIL_REPO = "https://github.com/GregoirePetit/FeTrIL"
SEED_REPO = "https://github.com/grypesc/SEED"
SEED_PAPER = "https://openreview.net/pdf?id=sSyytcewxe"
DIFFCLASS_PAPER = "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11819.pdf"


def official_records() -> list[dict[str, str]]:
    """Return recent baseline numbers that can be cited before rerunning.

    The CoDC CIFAR-100 protocol is closest to the equally split N=5 protocol
    reported by DiffClass. Those values are kept separate from original
    FeTrIL/SEED numbers because they are third-party reproduced results.
    """
    return [
        _record(
            "cifar100",
            "cifar100_equal_10_tasks",
            "FeTrIL",
            "46.3",
            "0.3",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 1, equal split, ResNet32, T=10.",
        ),
        _record(
            "cifar100",
            "cifar100_equal_10_tasks",
            "SEED",
            "61.7",
            "0.4",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 1, equal split, ResNet32, T=10.",
        ),
        _record(
            "cifar100",
            "cifar100_equal_20_tasks",
            "FeTrIL",
            "38.7",
            "0.3",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 1, equal split, ResNet32, T=20.",
        ),
        _record(
            "cifar100",
            "cifar100_equal_20_tasks",
            "SEED",
            "56.2",
            "0.3",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 1, equal split, ResNet32, T=20.",
        ),
        _record(
            "cifar100",
            "cifar100_equal_50_tasks",
            "FeTrIL",
            "27.0",
            "1.2",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 1, equal split, ResNet32, T=50.",
        ),
        _record(
            "cifar100",
            "cifar100_equal_50_tasks",
            "SEED",
            "42.6",
            "1.4",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 1, equal split, ResNet32, T=50.",
        ),
        _record(
            "cifar100",
            "cifar100_b50_inc10",
            "FeTrIL",
            "66.3",
            "",
            "task-agnostic average incremental accuracy",
            "official-original",
            FETRIL_REPO,
            "FeTrIL official README CIFAR-100 T=5; same value is reused in SEED paper Table 2.",
        ),
        _record(
            "cifar100",
            "cifar100_b50_inc10",
            "SEED",
            "70.9",
            "0.3",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 2, large first task, ResNet18, T=6, |C1|=50.",
        ),
        _record(
            "cifar100",
            "cifar100_b50_inc5",
            "FeTrIL",
            "65.2",
            "",
            "task-agnostic average incremental accuracy",
            "official-original",
            FETRIL_REPO,
            "FeTrIL official README CIFAR-100 T=10; same value is reused in SEED paper Table 2.",
        ),
        _record(
            "cifar100",
            "cifar100_b50_inc5",
            "SEED",
            "69.3",
            "0.5",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 2, large first task, ResNet18, T=11, |C1|=50.",
        ),
        _record(
            "cifar100",
            "cifar100_b40_inc3",
            "FeTrIL",
            "61.5",
            "",
            "task-agnostic average incremental accuracy",
            "official-original",
            FETRIL_REPO,
            "FeTrIL official README CIFAR-100 T=20; same value is reused in SEED paper Table 2.",
        ),
        _record(
            "cifar100",
            "cifar100_b40_inc3",
            "SEED",
            "62.9",
            "0.9",
            "task-agnostic average incremental accuracy",
            "official-original",
            SEED_PAPER,
            "SEED paper Table 2, large first task, ResNet18, T=21, |C1|=40.",
        ),
        _record(
            "cifar100",
            "cifar100_b20_inc20_n5",
            "FeTrIL",
            "58.68",
            "",
            "average incremental accuracy",
            "third-party-reproduced",
            DIFFCLASS_PAPER,
            "DiffClass ECCV 2024 Table 2, equally split 100 classes into N=5 tasks.",
        ),
        _record(
            "cifar100",
            "cifar100_b20_inc20_n5",
            "SEED",
            "63.05",
            "",
            "average incremental accuracy",
            "third-party-reproduced",
            DIFFCLASS_PAPER,
            "DiffClass ECCV 2024 Table 2, equally split 100 classes into N=5 tasks.",
        ),
    ]


def write_records_csv(records: Iterable[dict[str, str]], output_path: Path | str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in FIELDNAMES})
    return output


def fetril_pipeline_commands(
    config_path: str,
    python_executable: str = "python",
    include_clean: bool = False,
) -> list[list[str]]:
    scripts = [
        "scratch.py",
        "compute_distances.py",
        "prepare_train.py",
        "train_classifiers.py",
    ]
    if include_clean:
        scripts.append("clean_train.py")
    scripts.extend(["compute_predictions.py", "eval.py"])
    return [[python_executable, str(Path("codes") / script), config_path] for script in scripts]


def seed_script_for_protocol(protocol: str) -> str | None:
    return {
        "equal_10_tasks": "cifar10x10.sh",
        "equal_20_tasks": "cifar20x5.sh",
        "equal_50_tasks": "cifar50x2.sh",
        "large_first": "table2.sh",
        "custom": None,
    }.get(protocol)


def repo_has_paths(repo: Path | str, relative_paths: Sequence[str]) -> bool:
    root = Path(repo)
    return all((root / item).exists() for item in relative_paths)


def missing_paths(repo: Path | str, relative_paths: Sequence[str]) -> list[str]:
    root = Path(repo)
    return [item for item in relative_paths if not (root / item).exists()]


def format_commands(commands: Sequence[Sequence[str]]) -> str:
    return "\n".join(" ".join(shlex.quote(part) for part in command) for command in commands)


def _record(
    dataset: str,
    protocol: str,
    method: str,
    top1: str,
    std: str,
    metric: str,
    source_type: str,
    source: str,
    notes: str,
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "protocol": protocol,
        "method": method,
        "top1": top1,
        "std": std,
        "metric": metric,
        "source_type": source_type,
        "source": source,
        "notes": notes,
    }
