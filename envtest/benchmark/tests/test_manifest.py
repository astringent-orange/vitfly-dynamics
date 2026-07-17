import csv
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "envtest" / "benchmark" / "manifests"


class ManifestTest(unittest.TestCase):
    SCENARIOS = {
        "baseline",
        "dynamic_speed_1mps",
        "dynamic_speed_3mps",
        "forest_density_low",
        "forest_density_high",
        "flight_speed_3",
        "flight_speed_7",
    }

    def read(self, name):
        with open(MANIFEST / name, newline="") as stream:
            return list(csv.DictReader(stream))

    def test_expected_case_counts(self):
        self.assertEqual(len(self.read("ablation_validation_cases.csv")), 140)
        self.assertEqual(len(self.read("comparison_test_cases.csv")), 140)

    def test_each_split_uses_two_phase_seeds(self):
        self.assertEqual(
            {row["phase_seed"] for row in self.read("ablation_validation_cases.csv")},
            {"8000", "8001"},
        )
        self.assertEqual(
            {row["phase_seed"] for row in self.read("comparison_test_cases.csv")},
            {"9000", "9001"},
        )

    def test_case_ids_and_pairing_are_unique(self):
        for name in ("ablation_validation_cases.csv", "comparison_test_cases.csv"):
            rows = self.read(name)
            self.assertEqual(len({row["case_id"] for row in rows}), len(rows))
            counts = Counter(row["scenario_id"] for row in rows)
            self.assertEqual(set(counts), self.SCENARIOS)
            self.assertEqual(set(counts.values()), {20})
            self.assertEqual(counts["baseline"], 20)

    def test_each_scenario_changes_only_one_baseline_factor(self):
        rows = self.read("ablation_validation_cases.csv")
        signatures = {}
        for row in rows:
            signature = (
                row["desired_speed"],
                row["forest_density"],
                row["tree_count"],
                row["dynamic_speed_mps"],
            )
            signatures.setdefault(row["scenario_id"], set()).add(signature)
        self.assertTrue(all(len(values) == 1 for values in signatures.values()))
        actual = {scenario: next(iter(values)) for scenario, values in signatures.items()}
        self.assertEqual(actual["baseline"], ("5.0", "medium", "100", "2.0"))
        self.assertEqual(actual["dynamic_speed_1mps"], ("5.0", "medium", "100", "1.0"))
        self.assertEqual(actual["dynamic_speed_3mps"], ("5.0", "medium", "100", "3.0"))
        self.assertEqual(actual["forest_density_low"], ("5.0", "low", "50", "2.0"))
        self.assertEqual(actual["forest_density_high"], ("5.0", "high", "150", "2.0"))
        self.assertEqual(actual["flight_speed_3"], ("3.0", "medium", "100", "2.0"))
        self.assertEqual(actual["flight_speed_7"], ("7.0", "medium", "100", "2.0"))

    def test_medium_density_is_100_trees(self):
        rows = self.read("comparison_test_cases.csv")
        self.assertTrue(all(row["tree_count"] == "100" for row in rows if row["forest_density"] == "medium"))

    def test_all_cases_use_named_positive_dynamic_speeds(self):
        rows = self.read("ablation_validation_cases.csv") + self.read("comparison_test_cases.csv")
        self.assertEqual({row["dynamic_speed_mps"] for row in rows}, {"1.0", "2.0", "3.0"})
        self.assertTrue(all(row["dynamic_profile"].startswith("dynamic_speed_") for row in rows))


if __name__ == "__main__":
    unittest.main()
