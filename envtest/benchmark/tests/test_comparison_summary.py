import os
import tempfile
import unittest
from pathlib import Path

from envtest.benchmark.summarize_comparison import (
    COMPARISON_FACTOR_SPECS,
    COMPARISON_POLICY_ORDER,
    DEFAULT_TABLE_OUTPUT,
    build_parser,
    latest_comparison_result_paths,
    validate_comparison_coverage,
)
from envtest.benchmark.summarize_results import (
    build_factor_figure,
    plot_factor_sweeps,
)


class ComparisonSummaryTest(unittest.TestCase):
    def synthetic_summaries(self):
        scenarios = {
            scenario
            for spec in COMPARISON_FACTOR_SPECS
            for scenario in spec["scenarios"]
        }
        rows = []
        for policy_index, policy in enumerate(COMPARISON_POLICY_ORDER):
            for scenario_index, scenario in enumerate(sorted(scenarios)):
                success = 0.55 + 0.05 * policy_index - 0.01 * scenario_index
                rows.append({
                    "policy_id": policy,
                    "scenario_id": scenario,
                    "success_rate": success,
                    "collision_rate": 1.0 - success,
                    "successful_time_mean": 12.0 + scenario_index,
                    "success_ci_low": max(0.0, success - 0.08),
                    "success_ci_high": min(1.0, success + 0.08),
                })
        return rows

    def test_latest_four_model_results_and_default_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for index, policy in enumerate(COMPARISON_POLICY_ORDER):
                old = root / f"{policy}_old" / "results.csv"
                new = root / f"{policy}_new" / "results.csv"
                old.parent.mkdir()
                new.parent.mkdir()
                old.write_text("old")
                new.write_text("new")
                os.utime(old, (100 + index, 100 + index))
                os.utime(new, (200 + index, 200 + index))
                expected.append(new)
            self.assertEqual(latest_comparison_result_paths(root), expected)

        args = build_parser().parse_args([])
        self.assertIsNone(args.results)
        self.assertEqual(args.output, DEFAULT_TABLE_OUTPUT)

    def test_two_comparison_figures_contain_five_points_for_four_models(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib is not installed")

        summaries = self.synthetic_summaries()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plot_factor_sweeps(
                summaries,
                output,
                policy_order=COMPARISON_POLICY_ORDER,
                factor_specs=COMPARISON_FACTOR_SPECS,
            )
            self.assertEqual(
                {path.name for path in output.glob("*.png")},
                {
                    *[spec["filename"] for spec in COMPARISON_FACTOR_SPECS],
                },
            )
            for spec in COMPARISON_FACTOR_SPECS:
                self.assertTrue((output / spec["filename"]).stat().st_size > 0)
                figure, axes = build_factor_figure(
                    summaries,
                    spec,
                    policy_order=COMPARISON_POLICY_ORDER,
                )
                self.assertEqual(len(axes), 3)
                self.assertEqual(len(axes[0].get_legend_handles_labels()[1]), 4)
                self.assertEqual(len(axes[1].lines), 4)
                self.assertEqual(len(axes[2].lines), 4)
                self.assertTrue(all(len(line.get_xdata()) == 5 for axis in axes for line in axis.lines))
                self.assertEqual(
                    [line.get_marker() for line in axes[0].lines],
                    ["s", "D", "P", "X"],
                )
                for axis in axes[:2]:
                    tick_spacing = axis.get_yticks()[1] - axis.get_yticks()[0]
                    self.assertAlmostEqual(tick_spacing, 20.0)
                plt.close(figure)
        plt.close("all")

    def test_comparison_summary_requires_complete_paired_cases(self):
        expected = {"c1", "c2"}
        rows = [
            {"policy_id": policy, "case_id": case_id}
            for policy in COMPARISON_POLICY_ORDER
            for case_id in expected
        ]
        validate_comparison_coverage(rows, expected)
        rows.pop()
        with self.assertRaisesRegex(ValueError, "incomplete comparison results"):
            validate_comparison_coverage(rows, expected)


if __name__ == "__main__":
    unittest.main()
