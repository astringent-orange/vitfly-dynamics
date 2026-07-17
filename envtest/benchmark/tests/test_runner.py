import unittest
from datetime import datetime

from envtest.benchmark.run_benchmark import (
    DEFAULT_CASES,
    DEFAULT_CONFIG,
    build_parser,
    resolve_output_path,
)


class RunnerTest(unittest.TestCase):
    def test_default_inputs_are_forest_ablation_files(self):
        args = build_parser().parse_args(["--policy", "single"])
        self.assertEqual(args.config, DEFAULT_CONFIG)
        self.assertEqual(args.cases, DEFAULT_CASES)

    def test_ablation_uses_policy_and_timestamp_as_default_output(self):
        output, used_default = resolve_output_path(
            "envtest/benchmark/manifests/ablation_validation_cases.csv",
            [{"id": "single"}],
            now=datetime(2026, 7, 17, 15, 12, 30),
        )
        self.assertEqual(str(output), "result/ablation/single_20260717_151230")
        self.assertTrue(used_default)

    def test_ablation_rejects_multiple_models(self):
        with self.assertRaisesRegex(ValueError, "exactly one --policy"):
            resolve_output_path(
                "ablation_validation_cases.csv",
                [{"id": "single"}, {"id": "adjacent"}],
            )

    def test_explicit_output_is_preserved(self):
        output, used_default = resolve_output_path(
            "ablation_validation_cases.csv",
            [{"id": "skip_one"}],
            requested_output="result/ablation/resume_skip_one",
        )
        self.assertEqual(str(output), "result/ablation/resume_skip_one")
        self.assertFalse(used_default)


if __name__ == "__main__":
    unittest.main()
