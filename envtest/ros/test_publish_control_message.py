import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_module():
    rospy = types.ModuleType("rospy")
    rospy.is_shutdown = lambda: False
    rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")

    class Empty:
        pass

    class Bool:
        def __init__(self, data=False):
            self.data = data

    std_msgs_msg.Empty = Empty
    std_msgs_msg.Bool = Bool
    spec = importlib.util.spec_from_file_location(
        "vitfly_test_publish_control_message",
        ROOT / "envtest" / "ros" / "publish_control_message.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"rospy": rospy, "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg},
    ):
        spec.loader.exec_module(module)
    return module


class FakePublisher:
    def __init__(self, connections):
        self.connections = iter(connections)
        self.last_connections = 0
        self.messages = []

    def get_num_connections(self):
        try:
            self.last_connections = next(self.connections)
        except StopIteration:
            pass
        return self.last_connections

    def publish(self, message):
        self.messages.append(message)

    def unregister(self):
        pass


class PublishControlMessageTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_waits_for_required_subscribers_and_publishes_once(self):
        publisher = FakePublisher([0, 1, 2])
        now = [0.0]

        connections = self.module.wait_for_subscribers(
            publisher,
            2,
            1.0,
            clock=lambda: now[0],
            sleep=lambda delay: now.__setitem__(0, now[0] + delay),
            is_shutdown=lambda: False,
        )
        self.assertEqual(connections, 2)

        message = self.module.build_message("empty")
        publisher.publish(message)
        self.assertEqual(publisher.messages, [message])

    def test_main_publishes_exactly_once_after_connection(self):
        publisher = FakePublisher([1])
        self.module.rospy.init_node = lambda *_args, **_kwargs: None
        self.module.rospy.Publisher = lambda *_args, **_kwargs: publisher
        with patch.object(self.module.time, "sleep", return_value=None):
            returncode = self.module.main(
                [
                    "--topic", "/control",
                    "--type", "bool",
                    "--value", "true",
                    "--min-subscribers", "1",
                    "--connect-timeout", "1",
                    "--delivery-wait", "0",
                ]
            )
        self.assertEqual(returncode, 0)
        self.assertEqual(len(publisher.messages), 1)
        self.assertTrue(publisher.messages[0].data)

    def test_timeout_does_not_publish(self):
        publisher = FakePublisher([0])
        now = [0.0]
        connections = self.module.wait_for_subscribers(
            publisher,
            1,
            0.1,
            clock=lambda: now[0],
            sleep=lambda delay: now.__setitem__(0, now[0] + delay),
            is_shutdown=lambda: False,
        )
        self.assertEqual(connections, 0)
        self.assertEqual(publisher.messages, [])

    def test_main_returns_failure_without_publishing_on_timeout(self):
        publisher = FakePublisher([0])
        self.module.rospy.init_node = lambda *_args, **_kwargs: None
        self.module.rospy.Publisher = lambda *_args, **_kwargs: publisher
        with patch.object(self.module, "wait_for_subscribers", return_value=0):
            returncode = self.module.main(
                [
                    "--topic", "/control",
                    "--type", "empty",
                    "--min-subscribers", "1",
                    "--connect-timeout", "1",
                ]
            )
        self.assertEqual(returncode, 1)
        self.assertEqual(publisher.messages, [])

    def test_builds_empty_and_bool_messages(self):
        self.assertEqual(type(self.module.build_message("empty")).__name__, "Empty")
        self.assertTrue(self.module.build_message("bool", True).data)
        self.assertFalse(self.module.build_message("bool", False).data)

    def test_waiting_uses_injected_wall_clock(self):
        publisher = FakePublisher([0])
        wall_time = [10.0]
        sleeps = []

        self.module.wait_for_subscribers(
            publisher,
            1,
            0.11,
            clock=lambda: wall_time[0],
            sleep=lambda delay: (sleeps.append(delay), wall_time.__setitem__(0, wall_time[0] + delay)),
            is_shutdown=lambda: False,
        )
        self.assertEqual(sleeps, [0.05, 0.05, unittest.mock.ANY])
        self.assertAlmostEqual(sum(sleeps), 0.11)

    def test_ack_predicates_cover_pilot_states(self):
        module = self.module
        telemetry = types.SimpleNamespace(
            bridge_armed=types.SimpleNamespace(data=False),
            num_references_in_queue=0,
        )
        self.assertTrue(module.ack_predicate("pilot_off")(telemetry))
        telemetry.bridge_armed.data = True
        self.assertTrue(module.ack_predicate("pilot_enabled")(telemetry))

        state = types.SimpleNamespace(
            pose=types.SimpleNamespace(
                position=types.SimpleNamespace(x=0.02, y=-0.01)
            ),
            velocity=types.SimpleNamespace(
                linear=types.SimpleNamespace(x=0.01, y=0.0, z=0.02)
            ),
        )
        self.assertTrue(module.ack_predicate("pilot_reset")(state))

    def test_ack_requires_fresh_stable_messages(self):
        state = {"count": 0, "latest": object()}
        now = [0.0]
        result = self.module.wait_for_ack(
            state, 0, lambda _message: True, 0.11, stable_samples=2,
            clock=lambda: now[0],
            sleep=lambda delay: (
                state.__setitem__("count", state["count"] + 1),
                now.__setitem__(0, now[0] + delay),
            ),
            is_shutdown=lambda: False,
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
