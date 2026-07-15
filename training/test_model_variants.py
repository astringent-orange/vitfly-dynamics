import unittest
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from model import CurrentFrameViTLSTM, PreviousFrameViTLSTM, SecondPreviousFrameViTLSTM


class ModelVariantTest(unittest.TestCase):
    def test_three_named_models_have_matching_channels(self):
        for model_class, channels in (
            (CurrentFrameViTLSTM, 1),
            (PreviousFrameViTLSTM, 2),
            (SecondPreviousFrameViTLSTM, 2),
        ):
            model = model_class().float()
            images = torch.rand(2, channels, 60, 90, requires_grad=True)
            desired_velocity = torch.ones(2, 1)
            quaternion = torch.zeros(2, 4)
            output, hidden = model([images, desired_velocity, quaternion])
            self.assertEqual(tuple(output.shape), (2, 3))
            self.assertIsNotNone(hidden)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
