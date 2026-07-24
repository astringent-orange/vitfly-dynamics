#!/usr/bin/python3

import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import yaml

from curate_dataset import HARD_VALIDATION_OPTIONS, apply_curation, curate_dataset


class CurateDatasetTest(unittest.TestCase):
    def test_training_gate_matches_strict_expert_quality_standard(self):
        self.assertEqual(HARD_VALIDATION_OPTIONS["max_post_goal_rows"], 0)
        self.assertEqual(HARD_VALIDATION_OPTIONS["max_negative_xcmd_rows"], 0)
        self.assertEqual(HARD_VALIDATION_OPTIONS["max_collision_rows"], 0)
        self.assertEqual(HARD_VALIDATION_OPTIONS["max_low_speed_ratio"], 0.15)
        self.assertEqual(HARD_VALIDATION_OPTIONS["max_negative_path_speed_ratio"], 0.02)
        self.assertEqual(HARD_VALIDATION_OPTIONS["max_path_backtrack_distance"], 0.3)
        self.assertEqual(HARD_VALIDATION_OPTIONS["max_path_cross_track_error"], 0.8)

    def _make_folder(self, root, name, environment):
        folder = os.path.join(root, name)
        os.mkdir(folder)
        with open(os.path.join(folder, "data.csv"), "w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["env_level", "env_folder", "env_seed", "dynamic_phase_seed", "nearest_obstacle_margin", "path_cross_track_error"],
            )
            writer.writeheader()
            writer.writerow({
                "env_level": "dynamic_astar_medium",
                "env_folder": environment,
                "env_seed": "10",
                "dynamic_phase_seed": "1000",
                "nearest_obstacle_margin": "0.5",
                "path_cross_track_error": "0.2",
            })
        return folder

    @patch("curate_dataset._path_metrics", return_value=(0.0, 0.0))
    @patch("curate_dataset.validate_trajectory")
    def test_apply_keeps_only_hard_safe_rollouts(self, validate, _metrics):
        validate.side_effect = [([], 1, 0, {"environment_0"}), ([], 1, 0, {"environment_0"}),
                                ([], 1, 0, {"environment_1"}), ([], 1, 0, {"environment_1"})]
        with tempfile.TemporaryDirectory() as root:
            self._make_folder(root, "trajectory_1", "environment_0")
            self._make_folder(root, "trajectory_2", "environment_1")
            evaluation_path = os.path.join(root, "evaluation.yaml")
            with open(evaluation_path, "w") as stream:
                yaml.safe_dump({
                    "rollout_1": {"Success": True, "number_crashes": 0},
                    "rollout_2": {"Success": False, "number_crashes": 1},
                }, stream)
            records = curate_dataset(root, evaluation_path)
            summary = apply_curation(root, records)

        self.assertEqual([record["status"] for record in records], ["accepted", "rejected"])
        self.assertIn("number_crashes=1", records[1]["hard_reasons"])
        self.assertEqual(summary["collection_runs"], 1)
        self.assertEqual(summary["accepted_trajectories"], 1)
        self.assertEqual(summary["rejected_trajectories"], 1)


if __name__ == "__main__":
    unittest.main()
