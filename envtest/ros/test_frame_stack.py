import unittest

import numpy as np

from frame_stack import build_frame_stack


class FrameStackTest(unittest.TestCase):
    def test_named_offsets_and_warmup(self):
        current = np.full((6, 9), 3.0, dtype=np.float32)
        previous = np.full((6, 9), 2.0, dtype=np.float32)
        second_previous = np.full((6, 9), 1.0, dtype=np.float32)

        self.assertEqual(build_frame_stack([], current, 0).shape, (1, 60, 90))
        self.assertIsNone(build_frame_stack([], current, 1))
        self.assertIsNone(build_frame_stack([previous], current, 2))

        one_back = build_frame_stack([previous], current, 1)
        two_back = build_frame_stack([previous, second_previous], current, 2)
        self.assertEqual(one_back.shape, (2, 60, 90))
        self.assertEqual(two_back.shape, (2, 60, 90))
        self.assertAlmostEqual(float(one_back[0].mean()), 2.0)
        self.assertAlmostEqual(float(one_back[1].mean()), 3.0)
        self.assertAlmostEqual(float(two_back[0].mean()), 2.0)
        self.assertAlmostEqual(float(two_back[1].mean()), 3.0)


if __name__ == '__main__':
    unittest.main()
