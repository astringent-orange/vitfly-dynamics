#!/usr/bin/python3

import csv
import os
import tempfile
import unittest

from calibrate_dynamics import fit


class DynamicsCalibrationTest(unittest.TestCase):
    def test_fit_uses_forward_to_zero_events(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "data.csv")
            rows = []
            timestamp = 0.0
            position = 0.0
            for _ in range(20):
                for command, velocity in ((4.0, 0.0), (4.0, 2.0), (0.0, 1.5), (0.0, 0.0)):
                    rows.append({"timestamp": timestamp, "pos_x": position, "vel_x": velocity, "velcmd_x": command})
                    timestamp += 0.1
                    position += velocity * 0.1
            with open(path, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            values = fit([path])

        self.assertEqual(values["calibration_stop_events"], 20)
        self.assertGreater(values["max_accel"], 0.0)
        self.assertGreater(values["max_brake_decel"], 0.0)


if __name__ == "__main__":
    unittest.main()
