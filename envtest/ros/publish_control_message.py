#!/usr/bin/env python3
"""Publish one ROS control message after subscribers have connected."""

import argparse
import sys
import time

import rospy
from std_msgs.msg import Bool, Empty


ACK_TOPICS = {
    "pilot_off": ("/kingfisher/dodgeros_pilot/telemetry", "telemetry"),
    "pilot_enabled": ("/kingfisher/dodgeros_pilot/telemetry", "telemetry"),
    "pilot_reset": ("/kingfisher/dodgeros_pilot/state", "state"),
}


def parse_bool(value):
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("boolean value must be 'true' or 'false'")


def build_message(message_type, value=None):
    if message_type == "empty":
        return Empty()
    if value is None:
        raise ValueError("--value is required when --type=bool")
    return Bool(data=value)


def _nested_bool(message, name):
    value = getattr(message, name, False)
    return bool(getattr(value, "data", value))


def _near_zero(value, tolerance):
    try:
        return abs(float(value)) <= tolerance
    except (TypeError, ValueError):
        return False


def ack_predicate(ack):
    """Return a predicate for a fresh pilot state/telemetry message."""
    if ack == "pilot_off":
        return lambda message: (
            not _nested_bool(message, "bridge_armed")
            and int(getattr(message, "num_references_in_queue", 0)) == 0
        )
    if ack == "pilot_enabled":
        return lambda message: _nested_bool(message, "bridge_armed")
    if ack == "pilot_reset":
        def reset_ready(message):
            pose = getattr(message, "pose", None)
            velocity = getattr(message, "velocity", None)
            position = getattr(pose, "position", None)
            linear = getattr(velocity, "linear", None)
            if position is None or linear is None:
                return False
            return all(
                _near_zero(getattr(position, axis, None), 0.15)
                for axis in ("x", "y")
            ) and all(
                _near_zero(getattr(linear, axis, None), 0.15)
                for axis in ("x", "y", "z")
            )
        return reset_ready
    raise ValueError(f"unknown acknowledgement: {ack}")


def wait_for_ack(
    subscriber,
    baseline_count,
    predicate,
    timeout,
    stable_samples=2,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    is_shutdown=rospy.is_shutdown,
):
    """Wait for a predicate on messages received after publication."""
    deadline = clock() + timeout
    consecutive = 0
    while not is_shutdown():
        count, latest = subscriber["count"], subscriber["latest"]
        if count > baseline_count and latest is not None and predicate(latest):
            consecutive += 1
            if consecutive >= stable_samples:
                return True
        else:
            consecutive = 0
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleep(min(0.05, remaining))
    return False


def wait_for_subscribers(
    publisher,
    minimum,
    timeout,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    is_shutdown=rospy.is_shutdown,
):
    deadline = clock() + timeout
    while not is_shutdown():
        connections = publisher.get_num_connections()
        if connections >= minimum:
            return connections
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleep(min(0.05, remaining))
    return publisher.get_num_connections()


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--type", choices=("empty", "bool"), required=True, dest="message_type")
    parser.add_argument("--value", type=parse_bool)
    parser.add_argument("--min-subscribers", type=int, default=1)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--delivery-wait", type=float, default=0.2)
    parser.add_argument("--ack", choices=tuple(ACK_TOPICS))
    parser.add_argument("--ack-timeout", type=float, default=5.0)
    parser.add_argument("--stable-samples", type=int, default=2)
    return parser


def validate_args(parser, args):
    if args.message_type == "bool" and args.value is None:
        parser.error("--value is required when --type=bool")
    if args.message_type == "empty" and args.value is not None:
        parser.error("--value is only valid when --type=bool")
    if args.min_subscribers < 1:
        parser.error("--min-subscribers must be at least 1")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be positive")
    if args.delivery_wait < 0:
        parser.error("--delivery-wait must be nonnegative")
    if args.ack_timeout <= 0:
        parser.error("--ack-timeout must be positive")
    if args.stable_samples < 1:
        parser.error("--stable-samples must be at least 1")


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    message = build_message(args.message_type, args.value)

    try:
        rospy.init_node("publish_control_message", anonymous=True)
        ack_subscription = None
        ack_message_type = None
        ack_subscriber = None
        if args.ack:
            from dodgeros_msgs.msg import QuadState, Telemetry
            ack_topic, ack_kind = ACK_TOPICS[args.ack]
            ack_message_type = Telemetry if ack_kind == "telemetry" else QuadState
            ack_subscription = {"count": 0, "latest": None}

            def ack_callback(message):
                ack_subscription["count"] += 1
                ack_subscription["latest"] = message

            ack_subscriber = rospy.Subscriber(
                ack_topic, ack_message_type, ack_callback, queue_size=1
            )

        publisher = rospy.Publisher(
            args.topic,
            type(message),
            queue_size=1,
            latch=False,
            tcp_nodelay=True,
        )
        connections = wait_for_subscribers(
            publisher,
            args.min_subscribers,
            args.connect_timeout,
        )
        if connections < args.min_subscribers:
            print(
                "[ROS CONTROL] Timed out waiting for subscribers: "
                f"topic={args.topic} connected={connections} "
                f"required={args.min_subscribers}",
                file=sys.stderr,
            )
            return 1

        baseline_count = ack_subscription["count"] if ack_subscription else 0
        publisher.publish(message)
        if ack_subscription and not wait_for_ack(
            ack_subscription,
            baseline_count,
            ack_predicate(args.ack),
            args.ack_timeout,
            args.stable_samples,
        ):
            print(
                f"[ROS CONTROL] Timed out waiting for acknowledgement: {args.ack}",
                file=sys.stderr,
            )
            publisher.unregister()
            return 1
        time.sleep(args.delivery_wait)
        publisher.unregister()
        print(
            "[ROS CONTROL] Published once: "
            f"topic={args.topic} subscribers={connections}"
            + (f" ack={args.ack}" if args.ack else "")
        )
        return 0
    except rospy.ROSInterruptException:
        return 130
    except Exception as error:
        print(f"[ROS CONTROL] Publish failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
