import unittest

from envtest.benchmark.benchmark_evaluator import EvaluationProfile, RolloutEvaluator, evaluate_rows


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.profile = EvaluationProfile()

    def test_success_requires_goal_and_zero_collision(self):
        evaluator = RolloutEvaluator(self.profile)
        evaluator.on_state(0.0, (0.0, 0.0, 3.0))
        evaluator.on_state(1.0, (1.0, 0.0, 3.0))
        evaluator.on_state(4.0, (60.0, 0.0, 3.0))
        self.assertEqual(evaluator.result()["success"], 1)
        self.assertEqual(evaluator.result()["flight_time"], 3.0)

    def test_collision_is_edge_triggered_and_invalidates_success(self):
        evaluator = RolloutEvaluator(self.profile)
        evaluator.on_state(0.0, (1.0, 0.0, 3.0))
        evaluator.on_obstacle_margin(-0.1)
        evaluator.on_obstacle_margin(-0.2)
        evaluator.on_obstacle_margin(0.1)
        evaluator.on_obstacle_margin(-0.1)
        evaluator.on_state(4.0, (60.0, 0.0, 3.0))
        result = evaluator.result()
        self.assertEqual(result["collision_count"], 2)
        self.assertEqual(result["success"], 0)
        self.assertEqual(result["termination_reason"], "goal_reached_with_collision")

    def test_rows_timeout(self):
        profile = EvaluationProfile(timeout_seconds=2.0)
        rows = [
            {"timestamp": "0", "pos_x": "0", "pos_y": "0", "pos_z": "3", "nearest_obstacle_margin": "1"},
            {"timestamp": "1", "pos_x": "1", "pos_y": "0", "pos_z": "3", "nearest_obstacle_margin": "1"},
            {"timestamp": "4", "pos_x": "2", "pos_y": "0", "pos_z": "3", "nearest_obstacle_margin": "1"},
        ]
        self.assertEqual(evaluate_rows(rows, profile)["termination_reason"], "timeout")


if __name__ == "__main__":
    unittest.main()
