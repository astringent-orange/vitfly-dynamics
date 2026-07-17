import unittest
from datetime import datetime

from envtest.benchmark.run_benchmark import RESULT_FIELDS, normalize_policy_paths
from envtest.benchmark.policy_adapters import validate_policy
from envtest.benchmark.run_comparison import (
    DEFAULT_CASES,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    build_parser,
    default_output_path,
    resolve_model,
)
from envtest.benchmark.run_benchmark import load


class ComparisonRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = normalize_policy_paths(load(DEFAULT_CONFIG))

    def test_defaults_use_main_comparison_manifest(self):
        args = build_parser().parse_args(["--model", "vitfly"])
        self.assertEqual(args.config, DEFAULT_CONFIG)
        self.assertEqual(args.cases, DEFAULT_CASES)
        self.assertIn("source_policy_id", RESULT_FIELDS)
        self.assertIn("adapter", RESULT_FIELDS)
        self.assertIn("frame_offset", RESULT_FIELDS)

    def test_best_model_requires_and_preserves_manual_selection(self):
        self.assertEqual(resolve_model(self.config, "best", "adjacent"), ("adjacent", "best_ours"))
        with self.assertRaisesRegex(ValueError, "--best-policy"):
            resolve_model(self.config, "best")

    def test_vitfly_maps_to_official_legacy_policy(self):
        self.assertEqual(resolve_model(self.config, "vitfly"), ("original_vitfly", "vitfly"))

    def test_unintegrated_planners_fail_before_rollout(self):
        for model in ("fastplanner", "egoplanner"):
            with self.assertRaisesRegex(ValueError, "is not integrated"):
                resolve_model(self.config, model)

    def test_planner_command_contract_is_validated(self):
        valid = validate_policy({
            "id": "planner",
            "adapter": "fastplanner_ros",
            "enabled": False,
            "command_topic": "/kingfisher/dodgeros_pilot/feedthrough_command",
            "command_type": "dodgeros_msgs/Command",
            "command_mode": 2,
            "command_frame": "world",
        })
        self.assertFalse(valid["enabled"])
        with self.assertRaisesRegex(ValueError, "world-frame"):
            validate_policy({
                "id": "planner",
                "adapter": "fastplanner_ros",
                "enabled": False,
                "command_mode": 1,
            })

    def test_default_output_uses_comparation_directory(self):
        output = default_output_path("best_ours", datetime(2026, 7, 17, 16, 30, 45))
        self.assertEqual(output, DEFAULT_OUTPUT_ROOT / "best_ours_20260717_163045")


if __name__ == "__main__":
    unittest.main()
