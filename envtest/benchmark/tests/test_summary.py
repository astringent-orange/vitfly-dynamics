import csv
import unittest

from envtest.benchmark.summarize_results import paired_comparisons, summarize


class SummaryTest(unittest.TestCase):
    def test_summary_counts(self):
        rows = [
            {"policy_id": "single", "scenario_id": "dynamic_speed_2mps", "success": "1", "collision": "0", "flight_time": "10"},
            {"policy_id": "single", "scenario_id": "dynamic_speed_2mps", "success": "0", "collision": "1", "flight_time": ""},
        ]
        result = summarize(rows)[0]
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["collision_count"], 1)

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
