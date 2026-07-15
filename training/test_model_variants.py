import unittest
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from model import CurrentFrameViTLSTM, PreviousFrameViTLSTM, SecondPreviousFrameViTLSTM
from model import frame_mode_spec


class ModelVariantTest(unittest.TestCase):
    def test_offset_is_the_single_mode_mapping(self):
        self.assertEqual(frame_mode_spec(0), ('CurrentFrameViTLSTM', 1, 'current_frame_vitlstm'))
        self.assertEqual(frame_mode_spec(1), ('PreviousFrameViTLSTM', 2, 'previous_frame_vitlstm'))
        self.assertEqual(frame_mode_spec(2), ('SecondPreviousFrameViTLSTM', 2, 'second_previous_frame_vitlstm'))
        with self.assertRaises(ValueError):
            frame_mode_spec(3)

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
