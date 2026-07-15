#!/usr/bin/python3

import csv
import os
import tempfile
import unittest

import cv2
import numpy as np

from dataset_log_utils import cleanup_orphan_depth_images
from validate_dataset import REQUIRED_COLUMNS, validate_trajectory


class ValidateCandidateDatasetTest(unittest.TestCase):
    def make_row(self, timestamp="1.000"):
        row = {column: 0 for column in REQUIRED_COLUMNS}
        row.update(
            {
                "timestamp": timestamp,
                "desired_vel": 5.0,
                "pos_x": 2.0,
                "pos_y": 0.0,
                "pos_z": 3.0,
                "vel_x": 5.0,
                "vel_y": 0.0,
                "vel_z": 0.0,
                "velcmd_x": 5.0,
                "astar_success": 1,
                "nearest_obstacle_margin": 1.0,
                "candidate_selected_speed": 5.0,
                "candidate_safe_count": 11,
                "candidate_min_clearance": 999.0,
                "candidate_emergency_stop": 0,
                "candidate_prediction_horizon": 3.0,
                "candidate_raw_selected_speed": 5.0,
                "candidate_yield_active": 0,
                "candidate_applied_speed": 5.0,
                "path_cross_track_error": 0.0,
                "path_turn_angle_deg": 0.0,
                "path_speed_ceiling": 5.0,
                "candidate_control_delay": 0.25,
                "candidate_brake_decel": 1.5,
                "candidate_reverse_drift_buffer": 0.4,
                "candidate_initial_path_speed": 5.0,
                "candidate_predicted_stop_distance": 9.9833333333,
            }
        )
        return row

    def validate_rows(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = os.path.join(folder, "data.csv")
            with open(csv_path, "w", newline="") as output:
                fieldnames = list(dict.fromkeys(REQUIRED_COLUMNS + ["pos_y", "pos_z", "vel_y", "vel_z"]))
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            for row in rows:
                image = np.ones((2, 2), dtype=np.uint16)
                cv2.imwrite(os.path.join(folder, f"{row['timestamp']}.png"), image)
            return validate_trajectory(
                folder,
                path_points=[[0.0, 0.0, 3.0], [20.0, 0.0, 3.0]],
                **kwargs,
            )[0]

    def test_valid_candidate_row(self):
        self.assertEqual(self.validate_rows([self.make_row()]), [])

    def test_rejects_out_of_range_candidate(self):
        row = self.make_row()
        row["candidate_selected_speed"] = -0.5
        errors = self.validate_rows([row])
        self.assertTrue(any("out-of-range" in error for error in errors))

    def test_rejects_unmarked_emergency_stop(self):
        row = self.make_row()
        row["candidate_selected_speed"] = 0.0
        row["candidate_safe_count"] = 1
        row["v_slowdown_x"] = 1.0
        row["v_slowdown_dynamic_x"] = 1.0
        row["avoidance_active"] = 1
        errors = self.validate_rows([row], max_low_speed_ratio=1.0)
        self.assertTrue(any("missing candidate emergency-stop" in error for error in errors))

    def test_rejects_yield_holding_zero_after_positive_safe_candidate(self):
        rows = [self.make_row(f"{idx}.000") for idx in range(3)]
        for row in rows:
            row["candidate_selected_speed"] = 0.0
            row["candidate_safe_count"] = 2
            row["candidate_emergency_stop"] = 0
            row["candidate_yield_active"] = 1
            row["candidate_applied_speed"] = 0.0
            row["v_slowdown_x"] = 1.0
            row["v_slowdown_dynamic_x"] = 1.0
            row["avoidance_active"] = 1
        errors = self.validate_rows(rows, max_low_speed_ratio=1.0)
        self.assertTrue(any("yield policy held zero" in error for error in errors))

    def test_rejects_excessive_actual_backtrack(self):
        rows = [self.make_row(f"{idx}.000") for idx in range(4)]
        for idx, row in enumerate(rows):
            row["pos_x"] = 10.0 - 0.2 * idx
            row["vel_x"] = -0.5
        errors = self.validate_rows(
            rows,
            max_negative_path_speed_ratio=1.0,
            max_path_backtrack_distance=0.3,
        )
        self.assertTrue(any("max continuous path backtrack" in error for error in errors))

    def test_uses_path_tangent_instead_of_world_x_axis(self):
        rows = [self.make_row(f"{idx}.000") for idx in range(3)]
        for idx, row in enumerate(rows):
            row["pos_x"] = 10.0 - idx
            row["vel_x"] = -1.0
        with tempfile.TemporaryDirectory() as folder:
            csv_path = os.path.join(folder, "data.csv")
            fieldnames = list(dict.fromkeys(REQUIRED_COLUMNS + ["pos_y", "pos_z", "vel_y", "vel_z"]))
            with open(csv_path, "w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            for row in rows:
                cv2.imwrite(os.path.join(folder, f"{row['timestamp']}.png"), np.ones((2, 2), dtype=np.uint16))
            errors = validate_trajectory(
                folder,
                path_points=[[20.0, 0.0, 3.0], [0.0, 0.0, 3.0]],
            )[0]
        self.assertFalse(any("negative path-speed" in error for error in errors))

    def test_rejects_applied_speed_acceleration_spike(self):
        rows = [self.make_row("1.000"), self.make_row("1.100")]
        rows[0]["candidate_applied_speed"] = 0.0
        rows[1]["candidate_applied_speed"] = 1.0
        errors = self.validate_rows(rows)
        self.assertTrue(any("applied path-speed accel" in error for error in errors))

    def test_cleanup_removes_orphan_depth_image(self):
        with tempfile.TemporaryDirectory() as folder:
            kept_path = os.path.join(folder, "1.0.png")
            orphan_path = os.path.join(folder, "2.0.png")
            cv2.imwrite(kept_path, np.ones((2, 2), dtype=np.uint8))
            cv2.imwrite(orphan_path, np.ones((2, 2), dtype=np.uint8))
            removed = cleanup_orphan_depth_images(folder, [1.0])
            self.assertEqual(removed, [orphan_path])
            self.assertTrue(os.path.exists(kept_path))
            self.assertFalse(os.path.exists(orphan_path))


if __name__ == "__main__":
    unittest.main()
