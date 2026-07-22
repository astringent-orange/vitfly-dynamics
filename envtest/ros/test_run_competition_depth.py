import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_controller_module():
    rospy = types.ModuleType("rospy")
    rospy.exceptions = types.SimpleNamespace(ROSException=Exception)

    dodgeros_msgs = types.ModuleType("dodgeros_msgs")
    dodgeros_msgs_msg = types.ModuleType("dodgeros_msgs.msg")
    dodgeros_msgs_msg.Command = type("Command", (), {})
    dodgeros_msgs_msg.QuadState = type("QuadState", (), {})

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.TwistStamped = type("TwistStamped", (), {})
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = type("Image", (), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Empty = type("Empty", (), {})
    std_msgs_msg.String = type("String", (), {})
    envsim_msgs = types.ModuleType("envsim_msgs")
    envsim_msgs_msg = types.ModuleType("envsim_msgs.msg")
    envsim_msgs_msg.ObstacleArray = type("ObstacleArray", (), {})

    dataset_log_utils = types.ModuleType("dataset_log_utils")
    dataset_log_utils.cleanup_orphan_depth_images = lambda *_args, **_kwargs: None
    user_code = types.ModuleType("user_code")
    user_code.AStarDynamicExpert = object
    user_code.compute_command_vision_based = lambda *_args, **_kwargs: None
    user_code.compute_command_state_based = lambda *_args, **_kwargs: None
    user_code.default_planner_info = lambda: {}
    utils = types.ModuleType("utils")
    utils.AgileCommandMode = types.SimpleNamespace(SRT=0, CTBR=1, LINVEL=2)
    utils.AgileQuadState = object
    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object
    pandas = types.ModuleType("pandas")

    modules = {
        "rospy": rospy,
        "dodgeros_msgs": dodgeros_msgs,
        "dodgeros_msgs.msg": dodgeros_msgs_msg,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "envsim_msgs": envsim_msgs,
        "envsim_msgs.msg": envsim_msgs_msg,
        "dataset_log_utils": dataset_log_utils,
        "user_code": user_code,
        "utils": utils,
        "cv_bridge": cv_bridge,
        "pandas": pandas,
        "torch": None,
    }
    spec = importlib.util.spec_from_file_location(
        "vitfly_test_run_competition",
        ROOT / "envtest" / "ros" / "run_competition.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class DepthFrameTest(unittest.TestCase):
    def test_sparse_zero_pixels_are_usable(self):
        module = load_controller_module()
        image = np.full((240, 320), 0.03, dtype=np.float32)
        image[0, 0] = 0.0
        prepared, status = module.prepare_depth_frame(image)
        self.assertEqual(status, "partial_zero")
        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.shape, (240, 320))
        self.assertEqual(float(prepared[0, 0]), 0.0)

    def test_all_zero_pixels_are_rejected(self):
        module = load_controller_module()
        prepared, status = module.prepare_depth_frame(np.zeros((60, 90), dtype=np.float32))
        self.assertIsNone(prepared)
        self.assertEqual(status, "all_zero")

    def test_nan_negative_infinity_or_non_image_frames_are_rejected(self):
        module = load_controller_module()
        invalid = np.ones((60, 90), dtype=np.float32)
        invalid[0, 0] = np.nan
        prepared, status = module.prepare_depth_frame(invalid)
        self.assertIsNone(prepared)
        self.assertEqual(status, "invalid")
        invalid[0, 0] = -np.inf
        prepared, status = module.prepare_depth_frame(invalid)
        self.assertIsNone(prepared)
        self.assertEqual(status, "invalid")
        prepared, status = module.prepare_depth_frame(np.ones((2, 30, 45), dtype=np.float32))
        self.assertIsNone(prepared)
        self.assertEqual(status, "invalid")

    def test_positive_infinity_is_clipped_to_far_depth(self):
        module = load_controller_module()
        image = np.full((240, 320), 0.03, dtype=np.float32)
        image[0, 0] = np.inf
        prepared, status = module.prepare_depth_frame(image)
        self.assertEqual(status, "valid")
        self.assertEqual(float(prepared[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
