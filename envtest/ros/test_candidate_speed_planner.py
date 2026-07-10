#!/usr/bin/python3

import unittest

import numpy as np

from candidate_speed_planner import CandidateSpeedPlanner
from dynamic_obstacle_predictor import DynamicTrajectory


class CandidateSpeedPlannerTest(unittest.TestCase):
    def setUp(self):
        self.planner = CandidateSpeedPlanner()
        self.path = np.array([[0.0, 0.0, 3.0], [20.0, 0.0, 3.0]])
        self.position = np.array([0.0, 0.0, 3.0])

    def obstacle(self, position, radius=0.2):
        positions = np.repeat(np.asarray(position, dtype=float)[None, :], len(self.planner.time_offsets), axis=0)
        return {"radius": radius, "positions": positions}

    def test_no_obstacle_selects_desired_speed(self):
        result = self.planner.select_speed(self.path, self.position, np.zeros(3), 5.0, [])
        self.assertAlmostEqual(result.selected_speed, 5.0)
        self.assertEqual(result.safe_count, 11)
        self.assertFalse(result.emergency_stop)

    def test_selects_fastest_safe_lower_speed(self):
        obstacle = self.obstacle([6.0, 0.0, 3.0])
        result = self.planner.select_speed(self.path, self.position, np.zeros(3), 5.0, [obstacle])
        self.assertGreater(result.selected_speed, 0.0)
        self.assertLess(result.selected_speed, 5.0)
        self.assertGreaterEqual(result.min_clearance, 0.0)

    def test_obstacle_that_moves_away_allows_full_speed(self):
        positions = np.repeat(np.array([[6.0, 4.0, 3.0]]), len(self.planner.time_offsets), axis=0)
        result = self.planner.select_speed(
            self.path,
            self.position,
            np.zeros(3),
            5.0,
            [{"radius": 0.2, "positions": positions}],
        )
        self.assertAlmostEqual(result.selected_speed, 5.0)

    def test_unsafe_zero_candidate_requests_emergency_stop(self):
        obstacle = self.obstacle(self.position)
        result = self.planner.select_speed(self.path, self.position, np.zeros(3), 5.0, [obstacle])
        self.assertEqual(result.selected_speed, 0.0)
        self.assertEqual(result.safe_count, 0)
        self.assertTrue(result.emergency_stop)

    def test_negative_actual_speed_is_kept_in_prediction(self):
        positions, initial_speed = self.planner.predict_drone_trajectory(
            self.path,
            np.array([5.0, 0.0, 3.0]),
            np.array([-1.0, 0.0, 0.0]),
            1.0,
        )
        self.assertAlmostEqual(initial_speed, -1.0)
        self.assertLess(positions[1, 0], positions[0, 0])

    def test_candidate_targets_are_never_negative(self):
        candidates = self.planner.candidate_speeds(5.0)
        self.assertTrue(np.all(candidates >= 0.0))
        self.assertAlmostEqual(candidates[-1], 5.0)

    def test_loop_trajectory_position_and_velocity_are_finite(self):
        trajectory = DynamicTrajectory(
            "loop",
            [0.0, 1.0, 2.0],
            [[5.0, -1.0, 3.0], [5.0, 0.0, 3.0], [5.0, 1.0, 3.0]],
            scale=0.5,
            loop=True,
        )
        for phase in [1.99, 2.0, 2.01, 4.01]:
            self.assertTrue(np.isfinite(trajectory.position_at(phase)).all())
            self.assertTrue(np.isfinite(trajectory.velocity_at(phase)).all())


if __name__ == "__main__":
    unittest.main()
