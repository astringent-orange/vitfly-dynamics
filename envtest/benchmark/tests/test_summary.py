import csv
import tempfile
import unittest
from pathlib import Path

import yaml

from envtest.benchmark.summarize_results import paired_comparisons, summarize, select_best


class SummaryTest(unittest.TestCase):
    def test_summary_counts(self):
        rows = [
            {"policy_id": "single", "scenario_id": "dynamic_collection", "success": "1", "collision": "0", "flight_time": "10"},
            {"policy_id": "single", "scenario_id": "dynamic_collection", "success": "0", "collision": "1", "flight_time": ""},
        ]
        result = summarize(rows)[0]
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["collision_count"], 1)

    def test_selection_writes_policy(self):
        rows = []
        for policy, success in (("single", 1), ("adjacent", 2)):
            for scenario in ("dynamic_collection", "dynamic_high", "dynamic_off"):
                for _ in range(2):
                    rows.append({"policy_id": policy, "scenario_id": scenario, "success": str(int(success > 0)), "collision": "0", "flight_time": "10"})
                success -= 1
        summaries = summarize(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with open(path / "policies.yaml", "w") as stream:
                yaml.safe_dump([{"id": "single", "adapter": "vitfly_neural", "checkpoint": "a"}, {"id": "adjacent", "adapter": "vitfly_neural", "checkpoint": "b"}], stream)
            winner = select_best(summaries, path)
            self.assertEqual(winner, "adjacent")
            self.assertTrue((path / "selected_policy.yaml").is_file())

    def test_paired_difference_uses_case_id(self):
        rows = [
            {"case_id": "c1", "policy_id": "a", "scenario_id": "baseline", "success": "1", "collision": "0", "flight_time": "10"},
            {"case_id": "c1", "policy_id": "b", "scenario_id": "baseline", "success": "0", "collision": "1", "flight_time": ""},
            {"case_id": "c2", "policy_id": "a", "scenario_id": "baseline", "success": "0", "collision": "1", "flight_time": ""},
            {"case_id": "c2", "policy_id": "b", "scenario_id": "baseline", "success": "1", "collision": "0", "flight_time": "11"},
        ]
        result = paired_comparisons(rows)[0]
        self.assertEqual(result["paired_total"], 2)
        self.assertEqual(result["success_difference_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
