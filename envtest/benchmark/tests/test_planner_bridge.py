import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def load_bridge():
    rospy = types.ModuleType("rospy")
    rospy.get_param = lambda _name, default=None: default
    rospy.Duration = lambda value: value
    rospy.Time = types.SimpleNamespace(now=lambda: 0)
    rospy.logwarn_throttle = lambda *_args, **_kwargs: None
    rospy.loginfo = lambda *_args, **_kwargs: None
    rospy.signal_shutdown = lambda *_args, **_kwargs: None

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

    rospy.Publisher = Dummy
    rospy.Subscriber = Dummy
    rospy.Timer = Dummy

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = type("CvBridge", (), {})

    geometry = types.ModuleType("geometry_msgs.msg")
    geometry.PoseStamped = type("PoseStamped", (), {})
    geometry.TwistStamped = type("TwistStamped", (), {})
    nav = types.ModuleType("nav_msgs.msg")
    nav.Odometry = type("Odometry", (), {})
    nav.Path = type("Path", (), {})
    sensor = types.ModuleType("sensor_msgs.msg")
    sensor.Image = type("Image", (), {})
    std = types.ModuleType("std_msgs.msg")
    std.Bool = type("Bool", (), {})
    std.Empty = type("Empty", (), {})
    std.String = type("String", (), {})
    quad = types.ModuleType("quadrotor_msgs.msg")
    quad.PositionCommand = type("PositionCommand", (), {})

    modules = {
        "rospy": rospy,
        "cv_bridge": cv_bridge,
        "geometry_msgs.msg": geometry,
        "nav_msgs.msg": nav,
        "sensor_msgs.msg": sensor,
        "std_msgs.msg": std,
        "quadrotor_msgs.msg": quad,
    }
    # Parent packages are needed by Python's import machinery.
    for package, child in (
        ("geometry_msgs", geometry), ("nav_msgs", nav),
        ("sensor_msgs", sensor), ("std_msgs", std), ("quadrotor_msgs", quad),
    ):
        parent = types.ModuleType(package)
        parent.msg = child
        modules[package] = parent
    spec = importlib.util.spec_from_file_location(
        "vitfly_test_planner_bridge", ROOT / "planner_bridge" / "scripts" / "planner_bridge.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class PlannerBridgeConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_bridge()

    def test_depth_conversion_uses_metric_range_and_uint16_mm(self):
        raw = np.array([[0.01, 0.05, 0.08, 0.10, np.nan]], dtype=np.float32)
        result = self.module.raw_depth_to_millimetres(raw)
        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(result.tolist(), [[1000, 5000, 8000, 0, 0]])

    def test_camera_extrinsic_matches_flightmare_translation(self):
        rotation, translation = self.module.camera_extrinsic()
        np.testing.assert_allclose(translation, [0.3, 0.0, 0.0])
        np.testing.assert_allclose(rotation.dot(rotation.T), np.eye(3), atol=1e-7)

    def test_position_command_vector_fallback_is_safe(self):
        self.assertEqual(
            self.module._vector3(types.SimpleNamespace(x=1.0, y=2.0, z=3.0)).tolist(),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(self.module._vector3(None).tolist(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
