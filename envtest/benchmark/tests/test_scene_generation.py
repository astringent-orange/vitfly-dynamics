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

    def test_density_counts(self):
        self.assertEqual(self.count("map_000_density_low_dynamic_speed_1mps"), 50)
        self.assertEqual(self.count("map_000_density_medium_dynamic_speed_1mps"), 100)
        self.assertEqual(self.count("map_000_density_high_dynamic_speed_1mps"), 150)

    def test_speed_profiles_only_scale_trajectory_time(self):
        def duration(scene):
            with open(SCENES / scene / "csvtrajs" / "traj_dyn_000.csv", newline="") as stream:
                rows = list(csv.DictReader(stream))
            return float(rows[-1]["t"])
        speed_1 = duration("map_000_density_medium_dynamic_speed_1mps")
        speed_2 = duration("map_000_density_medium_dynamic_speed_2mps")
        speed_3 = duration("map_000_density_medium_dynamic_speed_3mps")
        self.assertAlmostEqual(speed_1 / speed_2, 2.0, places=5)
        self.assertAlmostEqual(speed_2 / speed_3, 1.5, places=5)

    def test_every_speed_profile_has_dynamic_obstacles(self):
        for speed in (1, 2, 3):
            scene = SCENES / f"map_000_density_medium_dynamic_speed_{speed}mps"
            with open(scene / "dynamic_obstacles.yaml") as stream:
                config = yaml.safe_load(stream)
            self.assertEqual(config["N"], 8)


if __name__ == "__main__":
    unittest.main()
