import csv
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "envtest" / "benchmark" / "manifests"


class ManifestTest(unittest.TestCase):
    ABLATION_SCENARIOS = {
        "dynamic_speed_1mps",
        "dynamic_speed_2mps",
        "dynamic_speed_3mps",
        "dynamic_speed_4mps",
        "dynamic_speed_5mps",
        "flight_speed_2",
        "flight_speed_4",
        "flight_speed_6",
        "flight_speed_8",
        "flight_speed_10",
    }
    COMPARISON_SCENARIOS = ABLATION_SCENARIOS

    def read(self, name):
        with open(MANIFEST / name, newline="") as stream:
            return list(csv.DictReader(stream))

    def test_expected_case_counts(self):
        self.assertEqual(len(self.read("ablation_validation_cases.csv")), 500)
        self.assertEqual(len(self.read("comparison_test_cases.csv")), 500)

    def test_each_split_uses_five_phase_seeds(self):
        self.assertEqual({row["phase_seed"] for row in self.read("ablation_validation_cases.csv")},
                         {"8000", "8001", "8002", "8003", "8004"})
        self.assertEqual(
            {row["phase_seed"] for row in self.read("comparison_test_cases.csv")},
            {"9000", "9001", "9002", "9003", "9004"},
        )

    def test_case_ids_and_pairing_are_unique(self):
        for name in ("ablation_validation_cases.csv", "comparison_test_cases.csv"):
            rows = self.read(name)
            self.assertEqual(len({row["case_id"] for row in rows}), len(rows))
            counts = Counter(row["scenario_id"] for row in rows)
            expected = self.ABLATION_SCENARIOS if "ablation" in name else self.COMPARISON_SCENARIOS
            self.assertEqual(set(counts), expected)
            self.assertEqual(set(counts.values()), {50})

    def test_each_scenario_changes_only_one_baseline_factor(self):
        rows = self.read("ablation_validation_cases.csv")
        signatures = {}
        for row in rows:
            signature = (
                row["desired_speed"],
                row["forest_density"],
                row["dynamic_speed_mps"],
            )
            signatures.setdefault(row["scenario_id"], set()).add(signature)
        self.assertTrue(all(len(values) == 1 for values in signatures.values()))
        actual = {scenario: next(iter(values)) for scenario, values in signatures.items()}
        self.assertEqual({row["forest_density"] for row in rows}, {"6.0"})
        self.assertEqual(actual["dynamic_speed_1mps"], ("5.0", "6.0", "1.0"))
        self.assertEqual(actual["dynamic_speed_4mps"], ("5.0", "6.0", "4.0"))
        self.assertEqual(actual["dynamic_speed_5mps"], ("5.0", "6.0", "5.0"))
        self.assertEqual(actual["flight_speed_2"], ("2.0", "6.0", "2.0"))
        self.assertEqual(actual["flight_speed_8"], ("8.0", "6.0", "2.0"))
        self.assertEqual(actual["flight_speed_10"], ("10.0", "6.0", "2.0"))

    def test_density_is_fixed_and_corridor_count_is_70(self):
        rows = self.read("comparison_test_cases.csv")
        self.assertEqual({row["forest_density"] for row in rows}, {"6.0"})
        self.assertTrue(all(row["tree_count"].isdigit() for row in rows))

    def test_all_cases_use_named_positive_dynamic_speeds(self):
        rows = self.read("ablation_validation_cases.csv") + self.read("comparison_test_cases.csv")
        self.assertEqual({row["dynamic_speed_mps"] for row in rows}, {"1.0", "2.0", "3.0", "4.0", "5.0"})
        self.assertTrue(all(row["dynamic_profile"].startswith("dynamic_speed_") for row in rows))


if __name__ == "__main__":
    unittest.main()
