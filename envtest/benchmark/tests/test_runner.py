import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from envtest.benchmark.run_benchmark import (
    DEFAULT_ABLATION_OUTPUT,
    DEFAULT_CASES,
    DEFAULT_CONFIG,
    build_parser,
    empty_result,
    result_needs_rerun,
    resolve_output_path,
    run_one,
    run_attempt,
    upsert_result,
)


class RunnerTest(unittest.TestCase):
    def test_default_inputs_are_forest_ablation_files(self):
        args = build_parser().parse_args(["--policy", "single"])
        self.assertEqual(args.config, DEFAULT_CONFIG)
        self.assertEqual(args.cases, DEFAULT_CASES)
        self.assertEqual(args.simulator_retries, 1)
        self.assertEqual(args.case_id, [])

    def test_exact_case_filter_is_repeatable(self):
        args = build_parser().parse_args([
            "--policy", "single",
            "--case-id", "c1",
            "--case-id", "c2",
        ])
        self.assertEqual(args.case_id, ["c1", "c2"])

    def test_ablation_uses_policy_and_timestamp_as_default_output(self):
        output, used_default = resolve_output_path(
            "envtest/benchmark/manifests/ablation_validation_cases.csv",
            [{"id": "single"}],
            now=datetime(2026, 7, 17, 15, 12, 30),
        )
        self.assertEqual(output, DEFAULT_ABLATION_OUTPUT / "single_20260717_151230")
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
            requested_output="results/ablation/resume_skip_one",
        )
        self.assertEqual(str(output), "results/ablation/resume_skip_one")
        self.assertFalse(used_default)

    @patch("envtest.benchmark.run_benchmark.run_attempt")
    def test_simulator_failure_is_retried_once(self, mocked_attempt):
        mocked_attempt.side_effect = [
            empty_result("simulator_error", 3),
            {**empty_result("goal_reached", 0), "success": 1},
        ]
        result = run_one({}, {"id": "single"}, {"case_id": "c1"}, Path("out"))
        self.assertEqual(mocked_attempt.call_count, 2)
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["success"], 1)

    @patch("envtest.benchmark.run_benchmark.run_attempt")
    def test_controller_error_is_not_retried(self, mocked_attempt):
        mocked_attempt.return_value = empty_result("controller_error", 2)
        result = run_one({}, {"id": "single"}, {"case_id": "c1"}, Path("out"))
        self.assertEqual(mocked_attempt.call_count, 1)
        self.assertEqual(result["attempt_count"], 1)

    @patch("envtest.benchmark.run_benchmark.run_attempt")
    def test_repeated_simulator_failure_exhausts_retry_budget(self, mocked_attempt):
        mocked_attempt.return_value = empty_result("simulator_error", 3)
        result = run_one({}, {"id": "single"}, {"case_id": "c1"}, Path("out"))
        self.assertEqual(mocked_attempt.call_count, 2)
        self.assertEqual(result["attempt_count"], 2)
        self.assertTrue(result_needs_rerun(result))

    def test_resume_replaces_only_infrastructure_failure(self):
        infrastructure = {
            "policy_id": "single", "case_id": "c1",
            "termination_reason": "missing_result", "runner_returncode": "0",
        }
        controller = {
            "policy_id": "single", "case_id": "c2",
            "termination_reason": "controller_error", "runner_returncode": "2",
        }
        self.assertTrue(result_needs_rerun(infrastructure))
        self.assertFalse(result_needs_rerun(controller))
        replacement = {
            "policy_id": "single", "case_id": "c1",
            "termination_reason": "goal_reached", "runner_returncode": "0",
        }
        rows = upsert_result([infrastructure, controller], replacement)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            next(row for row in rows if row["case_id"] == "c1")["termination_reason"],
            "goal_reached",
        )

    @patch("envtest.benchmark.run_benchmark.terminate_process_group")
    @patch("envtest.benchmark.run_benchmark.subprocess.Popen")
    @patch("envtest.benchmark.run_benchmark.policy_environment", return_value={})
    def test_keyboard_interrupt_cleans_process_group(
        self, _mocked_environment, mocked_popen, mocked_terminate
    ):
        process = mocked_popen.return_value
        process.stdout = io.StringIO("")
        process.wait.side_effect = KeyboardInterrupt
        case = {
            "case_id": "c1", "scene_id": "scene", "map_id": "0",
            "phase_seed": "8000", "desired_speed": "5",
            "evaluation_profile": "strict",
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                run_attempt({}, {"id": "single"}, case, Path(directory), 1)
        mocked_terminate.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
