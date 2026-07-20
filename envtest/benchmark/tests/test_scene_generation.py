import csv
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SCENES = ROOT / "flightmare" / "flightpy" / "configs" / "vision" / "forest_benchmark_v1"


class SceneGenerationTest(unittest.TestCase):
    def count(self, scene):
        with open(SCENES / scene / "static_obstacles.csv", newline="") as stream:
            return sum(1 for row in csv.reader(stream) if row)

    def test_fixed_corridor_density(self):
        scene = SCENES / "map_000_density_6_dynamic_speed_1mps"
        with open(scene / "scene_metadata.yaml") as stream:
            metadata = yaml.safe_load(stream)
        self.assertEqual(metadata["roi_tree_count"], 70)
        self.assertAlmostEqual(metadata["forest_density_trees_per_100m2"], 6.0)

    def test_speed_profiles_only_scale_trajectory_time(self):
        def duration(scene):
            with open(SCENES / scene / "csvtrajs" / "traj_dyn_000.csv", newline="") as stream:
                rows = list(csv.DictReader(stream))
            return float(rows[-1]["t"])
        speed_1 = duration("map_000_density_6_dynamic_speed_1mps")
        speed_2 = duration("map_000_density_6_dynamic_speed_2mps")
        speed_3 = duration("map_000_density_6_dynamic_speed_3mps")
        speed_4 = duration("map_000_density_6_dynamic_speed_4mps")
        self.assertAlmostEqual(speed_1 / speed_2, 2.0, places=5)
        self.assertAlmostEqual(speed_2 / speed_3, 1.5, places=5)
        self.assertAlmostEqual(speed_2 / speed_4, 2.0, places=5)

    def test_every_speed_profile_has_dynamic_obstacles(self):
        for speed in (1, 2, 3, 4):
            scene = SCENES / f"map_000_density_6_dynamic_speed_{speed}mps"
            with open(scene / "dynamic_obstacles.yaml") as stream:
                config = yaml.safe_load(stream)
            self.assertEqual(config["N"], 18)

    def test_dynamic_obstacles_cover_three_height_bands(self):
        scene = SCENES / "map_000_density_6_dynamic_speed_2mps"
        with open(scene / "scene_metadata.yaml") as stream:
            metadata = yaml.safe_load(stream)
        self.assertEqual(metadata["dynamic_obstacle_count"], 18)
        self.assertEqual(metadata["dynamic_height_bands"], {"low": 6, "medium": 6, "high": 6})
        self.assertGreaterEqual(metadata["dynamic_z_amplitude_range"][0], 0.8)
        self.assertLessEqual(metadata["dynamic_z_amplitude_range"][1], 1.8)

    def test_dynamic_geometry_is_shared_across_profiles(self):
        def trajectory(scene):
            with open(SCENES / scene / "csvtrajs" / "traj_dyn_009.csv", newline="") as stream:
                return [(row["x"], row["y"], row["z"]) for row in csv.DictReader(stream)]
        geometry_1 = trajectory("map_000_density_6_dynamic_speed_1mps")
        geometry_4 = trajectory("map_000_density_6_dynamic_speed_4mps")
        self.assertEqual(geometry_1[0], geometry_4[0])
        self.assertEqual(geometry_1[-1], geometry_4[-1])


if __name__ == "__main__":
    unittest.main()
