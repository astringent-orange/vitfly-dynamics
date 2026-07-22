#!/usr/bin/env python3
"""Bridge Flightmare observations to FastPlanner/EGO-Planner.

The repository's pilot consumes world-frame ``TwistStamped`` commands.  The
upstream planners instead publish ``quadrotor_msgs/PositionCommand``.  This
node is deliberately the only place where that external message is imported
or converted, so the ViTFly controller and evaluator remain unchanged.
"""

import argparse
import copy
import math
import os
import threading
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, String

try:
    from quadrotor_msgs.msg import PositionCommand
except ImportError:  # pragma: no cover - exercised only in an incomplete overlay
    PositionCommand = None


def quaternion_matrix(x, y, z, w):
    """Return a 3x3 rotation matrix without depending on tf_conversions."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_extrinsic():
    """Flightmare's local camera transform (body -> optical camera)."""
    rz = math.radians(-90.0)
    rx = math.radians(-90.0)
    rz_m = np.array([[math.cos(rz), -math.sin(rz), 0.0],
                     [math.sin(rz), math.cos(rz), 0.0], [0.0, 0.0, 1.0]])
    rx_m = np.array([[1.0, 0.0, 0.0],
                     [0.0, math.cos(rx), -math.sin(rx)],
                     [0.0, math.sin(rx), math.cos(rx)]])
    return rz_m.dot(rx_m), np.array([0.3, 0.0, 0.0])


def raw_depth_to_millimetres(raw, far_clip=100.0, minimum=0.2, maximum=8.0):
    """Convert Flightmare normalized depth to uint16 millimetres.

    Invalid, zero, NaN and out-of-range values become zero, which is the
    convention used by both upstream local-map implementations.
    """
    values = np.asarray(raw, dtype=np.float32) * float(far_clip)
    valid = np.isfinite(values) & (values >= minimum) & (values <= maximum)
    result = np.zeros(values.shape, dtype=np.uint16)
    result[valid] = np.rint(values[valid] * 1000.0).astype(np.uint16)
    return result


def _vector3(value, default=(0.0, 0.0, 0.0)):
    if value is None:
        return np.asarray(default, dtype=float)
    if hasattr(value, "x"):
        return np.array([value.x, value.y, value.z], dtype=float)
    try:
        values = list(value)
        return np.asarray(values[:3], dtype=float)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=float)


class PlannerBridge:
    def __init__(self, planner, desired_speed, goal_x=60.0, goal_y=0.0, goal_z=3.0):
        if PositionCommand is None:
            raise RuntimeError("quadrotor_msgs/PositionCommand is unavailable; source the planner overlay")
        self.planner = planner
        self.desired_speed = max(float(desired_speed), 0.1)
        self.goal = (float(goal_x), float(goal_y), float(goal_z))
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.current_odom = None
        self.latest_command = None
        self.started = False
        self.finished = False
        self.started_wall = None
        self.last_command_wall = None
        self.shutdown_reason = ""
        self.exit_code = 3
        self.depth_count = 0
        self.odom_count = 0
        self.command_count = 0
        self.diagnostics_path = os.environ.get("VITFLY_CONTROLLER_DIAGNOSTICS_PATH", "")

        prefix = "/fastplanner" if planner == "fastplanner" else "/egoplanner"
        self.depth_topic = rospy.get_param("~depth_output_topic", prefix + "/depth")
        self.pose_topic = rospy.get_param("~pose_output_topic", prefix + "/camera_pose")
        self.command_topic = rospy.get_param("~command_topic", "/position_cmd")
        self.depth_pub = rospy.Publisher(self.depth_topic, Image, queue_size=1)
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.ready_topic = rospy.get_param("~ready_topic", prefix + "/ready")
        self.ready_pub = rospy.Publisher(self.ready_topic, Bool, queue_size=1, latch=True)
        self.diag_pub = rospy.Publisher(prefix + "/diagnostics", String, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher("/waypoint_generator/waypoints", Path, queue_size=1, latch=True)
        self.cmd_pub = rospy.Publisher(
            "/kingfisher/dodgeros_pilot/velocity_command", TwistStamped, queue_size=1
        )

        self.depth_sub = rospy.Subscriber(
            "/kingfisher/dodgeros_pilot/unity/depth", Image, self.depth_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.odom_sub = rospy.Subscriber(
            "/kingfisher/dodgeros_pilot/groundtruth/odometry", Odometry, self.odom_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.start_sub = rospy.Subscriber(
            "/kingfisher/start_navigation", Empty, self.start_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.finish_sub = rospy.Subscriber(
            "/kingfisher/finish_navigation", Empty, self.finish_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.command_sub = rospy.Subscriber(
            self.command_topic, PositionCommand, self.command_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.timer = rospy.Timer(rospy.Duration(0.02), self.publish_command)
        self.watchdog = rospy.Timer(rospy.Duration(0.2), self.watchdog_callback)

    def odom_callback(self, msg):
        with self.lock:
            self.current_odom = msg
            self.odom_count += 1
            self._publish_camera_pose(msg)
            self._publish_ready_if_possible()

    def _publish_camera_pose(self, msg):
        rotation, translation = camera_extrinsic()
        body_position = np.array([
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
        ])
        body_rotation = quaternion_matrix(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
        )
        camera_position = body_position + body_rotation.dot(translation)
        camera_rotation = body_rotation.dot(rotation)
        pose = PoseStamped()
        pose.header = copy.deepcopy(msg.header)
        pose.header.frame_id = "world"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = camera_position
        # A compact matrix -> quaternion conversion avoids tf dependency at runtime.
        trace = np.trace(camera_rotation)
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            q = (camera_rotation[2, 1] - camera_rotation[1, 2]) / s, \
                (camera_rotation[0, 2] - camera_rotation[2, 0]) / s, \
                (camera_rotation[1, 0] - camera_rotation[0, 1]) / s, 0.25 * s
        else:
            q = (0.0, 0.0, 0.0, 1.0)
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
        self.pose_pub.publish(pose)

    def depth_callback(self, msg):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            converted = raw_depth_to_millimetres(raw)
            output = Image()
            output.header = msg.header
            output.height = int(converted.shape[0])
            output.width = int(converted.shape[1])
            output.encoding = "16UC1"
            output.is_bigendian = 0
            output.step = output.width * 2
            output.data = converted.tobytes()
            self.depth_pub.publish(output)
            with self.lock:
                self.depth_count += 1
                self._publish_ready_if_possible()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[planner_bridge] depth conversion failed: %s", exc)

    def _publish_ready_if_possible(self):
        if not (self.depth_count and self.odom_count):
            return
        try:
            planner_connected = self.depth_pub.get_num_connections() > 0
            command_connected = self.command_sub.get_num_connections() > 0
        except AttributeError:
            # Keep the bridge testable with lightweight publisher stubs.
            planner_connected = command_connected = True
        if planner_connected and command_connected:
            self.ready_pub.publish(Bool(data=True))

    def start_callback(self, _msg):
        with self.lock:
            if self.started or self.finished:
                return
            self.started = True
            self.started_wall = time.monotonic()
            self.last_command_wall = None
            path = Path()
            path.header.stamp = rospy.Time.now()
            path.header.frame_id = "world"
            point = PoseStamped()
            point.header = path.header
            point.pose.position.x, point.pose.position.y, point.pose.position.z = self.goal
            point.pose.orientation.w = 1.0
            path.poses.append(point)
            self.goal_pub.publish(path)
            rospy.loginfo("[planner_bridge] start_navigation received; goal=(%.2f, %.2f, %.2f)", *self.goal)

    def command_callback(self, msg):
        with self.lock:
            self.latest_command = msg
            self.last_command_wall = time.monotonic()

    def publish_command(self, _event):
        with self.lock:
            if not self.started or self.finished:
                return
            output = TwistStamped()
            output.header.stamp = rospy.Time.now()
            output.header.frame_id = "world"
            if self.latest_command is not None and self.current_odom is not None:
                current = np.array([
                    self.current_odom.pose.pose.position.x,
                    self.current_odom.pose.pose.position.y,
                    self.current_odom.pose.pose.position.z,
                ])
                desired = _vector3(getattr(self.latest_command, "position", None), current)
                feedforward = _vector3(
                    getattr(self.latest_command, "velocity", getattr(self.latest_command, "vel", None))
                )
                velocity = feedforward + 1.5 * (desired - current)
                norm = float(np.linalg.norm(velocity))
                if norm > self.desired_speed:
                    velocity *= self.desired_speed / norm
                output.twist.linear.x, output.twist.linear.y, output.twist.linear.z = velocity
                self.command_count += 1
            self.cmd_pub.publish(output)

    def watchdog_callback(self, _event):
        with self.lock:
            if not self.started or self.finished or self.last_command_wall is not None:
                return
            timeout = float(os.environ.get("VITFLY_PLANNER_NO_COMMAND_TIMEOUT_SECONDS", "5"))
            if time.monotonic() - self.started_wall > timeout:
                if self.depth_count == 0 or self.odom_count == 0:
                    self.shutdown_reason = "planner input topics produced no usable data"
                    self.exit_code = 3
                else:
                    self.shutdown_reason = "planner produced no PositionCommand"
                    self.exit_code = 2
                self.finished = True
                self._finish(self.exit_code)

    def finish_callback(self, _msg):
        with self.lock:
            if self.finished:
                return
            self.finished = True
            self.shutdown_reason = "finish_navigation"
            self.exit_code = 0
            self._finish(0)

    def _finish(self, code):
        zero = TwistStamped()
        zero.header.stamp = rospy.Time.now()
        zero.header.frame_id = "world"
        self.cmd_pub.publish(zero)
        self.diag_pub.publish(String(data=self.shutdown_reason))
        self.write_diagnostics(code)
        rospy.signal_shutdown(self.shutdown_reason)

    def write_diagnostics(self, code):
        if not self.diagnostics_path:
            return
        import json
        path = self.diagnostics_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w") as stream:
            json.dump({
                "planner": self.planner,
                "depth_frames_received": self.depth_count,
                "odom_messages_received": self.odom_count,
                "command_publish_count": self.command_count,
                "controller_failure_detail": self.shutdown_reason,
                "exit_code": code,
            }, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=("fastplanner", "egoplanner"), required=True)
    parser.add_argument("--desired-speed", type=float, default=float(os.environ.get("VITFLY_DES_VEL", 5.0)))
    parser.add_argument("--command-topic", default="")
    parser.add_argument("--goal-x", type=float, default=float(os.environ.get("VITFLY_EVAL_GOAL_X", 60.0)))
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-z", type=float, default=3.0)
    args = parser.parse_args(argv)
    rospy.init_node("vitfly_%s_bridge" % args.planner, anonymous=False)
    if args.command_topic:
        rospy.set_param("~command_topic", args.command_topic)
    bridge = PlannerBridge(args.planner, args.desired_speed, args.goal_x, args.goal_y, args.goal_z)
    rospy.loginfo("[planner_bridge] ready for %s", args.planner)
    rospy.spin()
    return bridge.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
