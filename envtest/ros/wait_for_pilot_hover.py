#!/usr/bin/env python3
"""Wait until the low-level pilot has completed takeoff and is hovering."""

import argparse
import math
import sys
import time

import rospy
from dodgeros_msgs.msg import QuadState


class HoverWaiter:
    def __init__(self, min_height, max_speed, stable_samples):
        self.min_height = min_height
        self.max_speed = max_speed
        self.stable_samples = stable_samples
        self.consecutive = 0
        self.ready = False
        self.last_height = float("nan")
        self.last_speed = float("nan")

    def callback(self, state):
        velocity = state.velocity.linear
        self.last_height = state.pose.position.z
        self.last_speed = math.sqrt(
            velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
        )
        stable = self.last_height >= self.min_height and self.last_speed <= self.max_speed
        self.consecutive = self.consecutive + 1 if stable else 0
        self.ready = self.consecutive >= self.stable_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/kingfisher/dodgeros_pilot/state")
    parser.add_argument("--min-height", type=float, default=3.2)
    parser.add_argument("--max-speed", type=float, default=0.35)
    parser.add_argument("--stable-samples", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    rospy.init_node("wait_for_pilot_hover", anonymous=True, disable_signals=True)
    waiter = HoverWaiter(args.min_height, args.max_speed, args.stable_samples)
    rospy.Subscriber(args.topic, QuadState, waiter.callback, queue_size=1, tcp_nodelay=True)

    deadline = time.monotonic() + args.timeout
    rate = rospy.Rate(50)
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if waiter.ready:
            print(
                "[WAIT_FOR_PILOT_HOVER] Ready: "
                f"z={waiter.last_height:.3f}m speed={waiter.last_speed:.3f}m/s "
                f"stable_samples={waiter.consecutive}"
            )
            return 0
        rate.sleep()

    print(
        "[WAIT_FOR_PILOT_HOVER] Timed out: "
        f"z={waiter.last_height:.3f}m speed={waiter.last_speed:.3f}m/s "
        f"stable_samples={waiter.consecutive}/{args.stable_samples}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
