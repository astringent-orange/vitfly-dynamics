#!/usr/bin/python3

import unittest

import numpy as np

from candidate_speed_planner import CandidateSpeedPlanner, CandidateSpeedResult, CandidateYieldPolicy, PathSpeedController, PolylinePath
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

    def test_arc_length_reference_keeps_intermediate_corner(self):
        path = PolylinePath([[0.0, 0.0, 3.0], [5.0, 0.0, 3.0], [5.0, 5.0, 3.0]])
        reference = path.reference_from([4.5, 0.0, 3.0], 1.2)
        np.testing.assert_allclose(reference["reference"], [5.0, 0.7, 3.0], atol=1e-6)
        self.assertAlmostEqual(reference["cross_track_error"], 0.0)

    def test_turn_angle_preview_detects_upcoming_corner(self):
        path = PolylinePath([[0.0, 0.0, 3.0], [5.0, 0.0, 3.0], [5.0, 5.0, 3.0]])
        self.assertAlmostEqual(path.turn_angle_ahead([3.0, 0.0, 3.0], 2.5), 90.0)
        self.assertAlmostEqual(path.turn_angle_ahead([1.0, 0.0, 3.0], 2.5), 0.0)

    def test_speed_controller_reuses_command_for_same_timestamp(self):
        controller = PathSpeedController(max_accel=3.0)
        first, first_speed = controller.apply(5.0, [1.0, 0.0, 0.0], 1.0, [2.0, 0.0, 0.0])
        repeated, repeated_speed = controller.apply(0.0, [1.0, 0.0, 0.0], 1.0, [2.0, 0.0, 0.0])
        np.testing.assert_allclose(repeated, first)
        self.assertAlmostEqual(repeated_speed, first_speed)

    def test_speed_controller_matches_prediction_acceleration(self):
        controller = PathSpeedController(max_accel=3.0)
        controller.apply(5.0, [1.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0])
        command, applied_speed = controller.apply(5.0, [1.0, 0.0, 0.0], 1.1, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(applied_speed, 0.3)
        self.assertAlmostEqual(command[0], 0.3)

    def test_speed_controller_caps_initial_actual_speed(self):
        controller = PathSpeedController(max_accel=3.0)
        command, applied_speed = controller.apply(
            5.0,
            [1.0, 0.0, 0.0],
            1.0,
            [7.0, 0.0, 0.0],
            speed_limit=5.0,
        )
        self.assertAlmostEqual(applied_speed, 5.0)
        self.assertAlmostEqual(command[0], 5.0)

    def candidate_result(self, selected, safe_speeds, desired=5.0):
        evaluations = tuple(
            (speed, 0.5 if speed in safe_speeds else -0.5, speed in safe_speeds)
            for speed in np.linspace(0.0, desired, 11)
        )
        return CandidateSpeedResult(
            selected_speed=selected,
            safe_count=len(safe_speeds),
            min_clearance=0.5,
            emergency_stop=not any(speed > 0.0 for speed in safe_speeds),
            initial_path_speed=selected,
            evaluations=evaluations,
        )

    def test_yield_policy_holds_safe_lower_speed_until_release(self):
        policy = CandidateYieldPolicy(release_frames=3)
        slowed = self.candidate_result(2.5, [0.0, 1.0, 1.5, 2.0, 2.5])
        speed, active = policy.apply(slowed, 5.0)
        self.assertEqual(speed, 2.5)
        self.assertTrue(active)

        full_safe = self.candidate_result(5.0, list(np.linspace(0.0, 5.0, 11)))
        self.assertEqual(policy.apply(full_safe, 5.0), (2.5, True))
        self.assertEqual(policy.apply(full_safe, 5.0), (2.5, True))
        self.assertEqual(policy.apply(full_safe, 5.0), (5.0, False))

    def test_yield_policy_does_not_switch_to_faster_safe_candidate(self):
        policy = CandidateYieldPolicy()
        policy.apply(self.candidate_result(2.5, [0.0, 1.0, 2.0, 2.5]), 5.0)
        result = self.candidate_result(4.0, [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(policy.apply(result, 5.0), (2.0, True))

    def test_yield_policy_stops_when_no_candidate_is_safe(self):
        policy = CandidateYieldPolicy()
        policy.apply(self.candidate_result(2.0, [0.0, 1.0, 2.0]), 5.0)
        result = self.candidate_result(0.0, [])
        self.assertEqual(policy.apply(result, 5.0), (0.0, True))

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
