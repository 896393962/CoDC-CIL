import csv
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    fetril_pipeline_commands,
    official_records,
    seed_script_for_protocol,
    write_records_csv,
)


class RecentBaselineCommonTests(unittest.TestCase):
    def test_official_records_cover_recent_protocols(self):
        records = official_records()
        lookup = {
            (row["protocol"], row["method"], row["source_type"]): row
            for row in records
        }

        self.assertAlmostEqual(
            float(lookup[("cifar100_b50_inc10", "FeTrIL", "official-original")]["top1"]),
            66.3,
        )
        self.assertAlmostEqual(
            float(lookup[("cifar100_b50_inc10", "SEED", "official-original")]["top1"]),
            70.9,
        )
        self.assertAlmostEqual(
            float(
                lookup[
                    ("cifar100_b20_inc20_n5", "SEED", "third-party-reproduced")
                ]["top1"]
            ),
            63.05,
        )

    def test_write_records_csv(self):
        output_path = ROOT / "test_official_reference.csv"
        try:
            write_records_csv(official_records(), output_path)

            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        finally:
            output_path.unlink(missing_ok=True)

        self.assertGreaterEqual(len(rows), 5)
        self.assertIn("method", rows[0])
        self.assertIn("protocol", rows[0])
        self.assertTrue({"FeTrIL", "SEED"}.issubset({row["method"] for row in rows}))

    def test_fetril_commands_are_in_official_pipeline_order(self):
        commands = fetril_pipeline_commands("configs/cifar100_b50_inc10.cf")
        script_names = [Path(command[1]).name for command in commands]

        self.assertEqual(
            script_names,
            [
                "scratch.py",
                "compute_distances.py",
                "prepare_train.py",
                "train_classifiers.py",
                "compute_predictions.py",
                "eval.py",
            ],
        )
        self.assertTrue(all(command[-1] == "configs/cifar100_b50_inc10.cf" for command in commands))

    def test_seed_protocol_mapping(self):
        self.assertEqual(seed_script_for_protocol("equal_10_tasks"), "cifar10x10.sh")
        self.assertEqual(seed_script_for_protocol("equal_20_tasks"), "cifar20x5.sh")
        self.assertEqual(seed_script_for_protocol("equal_50_tasks"), "cifar50x2.sh")
        self.assertIsNone(seed_script_for_protocol("custom"))


if __name__ == "__main__":
    unittest.main()
