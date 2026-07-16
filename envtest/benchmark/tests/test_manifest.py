import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "envtest" / "benchmark" / "manifests"


class ManifestTest(unittest.TestCase):
    def read(self, name):
        with open(MANIFEST / name, newline="") as stream:
            return list(csv.DictReader(stream))

    def test_expected_case_counts(self):
        self.assertEqual(len(self.read("ablation_validation_cases.csv")), 150)
        self.assertEqual(len(self.read("comparison_test_cases.csv")), 350)

    def test_case_ids_and_pairing_are_unique(self):
        rows = self.read("comparison_test_cases.csv")
        self.assertEqual(len({row["case_id"] for row in rows}), len(rows))
        for scenario in {row["scenario_id"] for row in rows}:
            subset = [row for row in rows if row["scenario_id"] == scenario]
            self.assertEqual(len(subset), 50)

    def test_medium_density_is_100_trees(self):
        rows = self.read("comparison_test_cases.csv")
        self.assertTrue(all(row["tree_count"] == "100" for row in rows if row["forest_density"] == "medium"))


if __name__ == "__main__":
    unittest.main()
