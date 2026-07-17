import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from envtest.benchmark.summarize_results import (
    DEFAULT_TABLE_OUTPUT,
    FACTOR_SPECS,
    build_parser as build_summary_parser,
    build_factor_figure,
    latest_policy_result_paths,
    paired_comparisons,
    plot_factor_sweeps,
    read_result_files,
    summarize,
)


class SummaryTest(unittest.TestCase):
    def test_summary_defaults_to_latest_policy_results_and_table_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for index, policy in enumerate(("single", "adjacent", "skip_one")):
                old = root / f"{policy}_old" / "results.csv"
                new = root / f"{policy}_new" / "results.csv"
                old.parent.mkdir()
                new.parent.mkdir()
                old.write_text("old")
                new.write_text("new")
                os.utime(old, (100 + index, 100 + index))
                os.utime(new, (200 + index, 200 + index))
                expected.append(new)
            self.assertEqual(latest_policy_result_paths(root), expected)

        args = build_summary_parser().parse_args([])
        self.assertIsNone(args.results)
        self.assertEqual(args.output, DEFAULT_TABLE_OUTPUT)

    def synthetic_summaries(self):
        rows = []
        scenarios = {
            scenario
            for spec in FACTOR_SPECS
            for scenario in spec["scenarios"]
        }
        for policy_index, policy in enumerate(("single", "adjacent", "skip_one")):
            for scenario_index, scenario in enumerate(sorted(scenarios)):
                success_rate = 0.6 + policy_index * 0.05 - scenario_index * 0.01
                rows.append({
                    "policy_id": policy,
                    "scenario_id": scenario,
                    "total": 20,
                    "success_count": round(success_rate * 20),
                    "success_rate": success_rate,
                    "collision_count": round((1.0 - success_rate) * 20),
                    "collision_rate": 1.0 - success_rate,
                    "successful_time_mean": 12.0 + scenario_index,
                    "successful_time_median": 11.5 + scenario_index,
                    "success_ci_low": max(0.0, success_rate - 0.08),
                    "success_ci_high": min(1.0, success_rate + 0.08),
                })
        return rows

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

    @patch("envtest.benchmark.summarize_results.read_csv")
    def test_multiple_per_policy_result_files_are_merged(self, mocked_read):
        mocked_read.side_effect = [
            [{"policy_id": "single", "case_id": "c1"}],
            [{"policy_id": "adjacent", "case_id": "c1"}],
            [{"policy_id": "skip_one", "case_id": "c1"}],
        ]
        rows = read_result_files(["single.csv", "adjacent.csv", "skip_one.csv"])
        self.assertEqual([row["policy_id"] for row in rows], ["single", "adjacent", "skip_one"])

    @patch("envtest.benchmark.summarize_results.read_csv")
    def test_multiple_result_files_reject_duplicate_rollouts(self, mocked_read):
        duplicate = {"policy_id": "single", "case_id": "c1"}
        mocked_read.side_effect = [[duplicate], [duplicate]]
        with self.assertRaisesRegex(ValueError, "duplicate result"):
            read_result_files(["first.csv", "second.csv"])

    def test_factor_plots_have_expected_axes_ticks_and_policy_lines(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib is not installed")

        summaries = self.synthetic_summaries()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plot_factor_sweeps(summaries, output)
            self.assertEqual(
                {path.name for path in output.glob("*.png")},
                {spec["filename"] for spec in FACTOR_SPECS},
            )

        for spec in FACTOR_SPECS:
            figure, axes = build_factor_figure(summaries, spec)
            self.assertEqual(len(axes), 3)
            self.assertEqual(list(axes[2].get_xticks()), list(spec["x_values"]))
            self.assertEqual(len(axes[0].get_legend_handles_labels()[1]), 3)
            self.assertEqual(len(axes[1].lines), 3)
            self.assertEqual(len(axes[2].lines), 3)
            plt.close(figure)

    def test_missing_successful_time_is_an_empty_plot_point(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib is not installed")

        summaries = self.synthetic_summaries()
        target = next(
            row for row in summaries
            if row["policy_id"] == "single" and row["scenario_id"] == "dynamic_speed_1mps"
        )
        target["successful_time_median"] = ""
        figure, axes = build_factor_figure(summaries, FACTOR_SPECS[0])
        self.assertTrue(math.isnan(axes[2].lines[0].get_ydata()[0]))
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
