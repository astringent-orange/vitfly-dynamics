import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCENES = ROOT / "flightmare" / "flightpy" / "configs" / "vision" / "forest_benchmark_v1"


class SceneGenerationTest(unittest.TestCase):
    def count(self, scene):
        with open(SCENES / scene / "static_obstacles.csv", newline="") as stream:
            return sum(1 for row in csv.reader(stream) if row)

    def test_density_counts(self):
        self.assertEqual(self.count("map_000_density_low_dynamic_off"), 50)
        self.assertEqual(self.count("map_000_density_medium_dynamic_off"), 100)
        self.assertEqual(self.count("map_000_density_high_dynamic_off"), 150)

    def test_high_profile_is_one_point_five_times_faster(self):
        def duration(scene):
            with open(SCENES / scene / "csvtrajs" / "traj_dyn_000.csv", newline="") as stream:
                rows = list(csv.DictReader(stream))
            return float(rows[-1]["t"])
        collection = duration("map_000_density_medium_dynamic_collection")
        high = duration("map_000_density_medium_dynamic_high")
        self.assertAlmostEqual(collection / high, 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
