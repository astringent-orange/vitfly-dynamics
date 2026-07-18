import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_evaluation_node():
    rospy = types.ModuleType("rospy")
    rospy.signal_shutdown = lambda _reason: None
    rospy.init_node = lambda *_args, **_kwargs: None

    dodgeros_msgs = types.ModuleType("dodgeros_msgs")
    dodgeros_msgs_msg = types.ModuleType("dodgeros_msgs.msg")
    dodgeros_msgs_msg.QuadState = type("QuadState", (), {})
    envsim_msgs = types.ModuleType("envsim_msgs")
    envsim_msgs_msg = types.ModuleType("envsim_msgs.msg")
    envsim_msgs_msg.ObstacleArray = type("ObstacleArray", (), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Empty = type("Empty", (), {})
    uniplot = types.ModuleType("uniplot")
    uniplot.plot = lambda *_args, **_kwargs: None

    modules = {
        "rospy": rospy,
        "dodgeros_msgs": dodgeros_msgs,
        "dodgeros_msgs.msg": dodgeros_msgs_msg,
        "envsim_msgs": envsim_msgs,
        "envsim_msgs.msg": envsim_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "uniplot": uniplot,
    }
    spec = importlib.util.spec_from_file_location(
        "vitfly_test_evaluation_node",
        ROOT / "envtest" / "ros" / "evaluation_node.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, rospy


class EvaluationLifecycleTest(unittest.TestCase):
    def test_evaluator_persists_result_before_stopping_controller(self):
        module, rospy = load_evaluation_node()
        events = []
        rospy.signal_shutdown = lambda _reason: events.append("shutdown")

        evaluator = types.SimpleNamespace(
            printSummary=lambda: events.append("summary"),
            finish_pub=types.SimpleNamespace(publish=lambda: events.append("finish")),
            printPlots=lambda: events.append("plots"),
            config={"plots": False},
        )
        module.Evaluator.publishFinish(evaluator)
        self.assertEqual(events, ["summary", "finish", "shutdown"])

    def test_controller_callback_state_and_publishers_precede_subscribers(self):
        source = (ROOT / "envtest" / "ros" / "run_competition.py").read_text()
        depth_subscriber = source.index("self.img_sub = rospy.Subscriber")
        for marker in (
            "self.ctr = 0",
            "self.keyboard_input = ''",
            "self.rgb_img = None",
            "self.debug_img1_pub = rospy.Publisher",
            "self.debug_img2_pub = rospy.Publisher",
        ):
            self.assertLess(source.index(marker), depth_subscriber)

    def test_launcher_waits_for_result_before_controller_error(self):
        source = (ROOT / "launch_evaluation.bash").read_text()
        controller_branch = source.index("if ! ps -p $COMP_PID")
        wait = source.index("wait_for_evaluator_result 5", controller_branch)
        failure = source.index("write_rollout_failure_summary controller_error", controller_branch)
        self.assertLess(wait, failure)


if __name__ == "__main__":
    unittest.main()
