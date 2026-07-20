import os
import threading
import yaml
import rospy
import numpy as np

from dodgeros_msgs.msg import QuadState
from envsim_msgs.msg import ObstacleArray
from std_msgs.msg import Empty

from uniplot import plot


def parse_bool(value, name):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError("%s must be a boolean value" % name)


def terminal_plots_enabled(config):
    configured = config.get("plots", True)
    if "VITFLY_EVAL_PLOTS" in os.environ:
        configured = os.environ["VITFLY_EVAL_PLOTS"]
    if os.environ.get("VITFLY_BENCHMARK_MODE", "0") == "1":
        return False
    return parse_bool(configured, "VITFLY_EVAL_PLOTS")


class Evaluator:
    def __init__(self, config):
        rospy.init_node("evaluator", anonymous=False)
        self.config = config
        self.plots_enabled = terminal_plots_enabled(config)
        self.benchmark_mode = os.environ.get("VITFLY_BENCHMARK_MODE", "0") == "1"
        collision_setting = config.get("terminate_on_collision", False)
        if "VITFLY_EVAL_TERMINATE_ON_COLLISION" in os.environ:
            collision_setting = os.environ["VITFLY_EVAL_TERMINATE_ON_COLLISION"]
        self.terminate_on_collision = self.benchmark_mode and parse_bool(
            collision_setting, "VITFLY_EVAL_TERMINATE_ON_COLLISION"
        )
        self.finish_lock = threading.Lock()
        self.finished = False

        self.xmax = int(float(os.environ.get("VITFLY_EVAL_GOAL_X", self.config["target"])))

        self.is_active = False
        self.pos = []
        self.dist = []
        self.time_array = (self.xmax + 1) * [np.nan]

        self.hit_obstacle = False
        self.crash = 0
        self.dynamic_collision = False
        self.dynamic_encounter_margin = float(
            os.environ.get("VITFLY_DYNAMIC_ENCOUNTER_MARGIN", "2.0")
        )
        self.altitudes = []
        self.dynamic_clearances = []
        self.dynamic_encounter_count = 0
        self.dynamic_interaction_time = 0.0
        self.dynamic_in_encounter = False
        self.dynamic_last_timestamp = None
        self.timeout = float(os.environ.get("VITFLY_EVAL_TIMEOUT_SECONDS", self.config["timeout"]))
        self.collision_margin = float(os.environ.get("VITFLY_EVAL_COLLISION_MARGIN", "0.0"))
        bounds_text = os.environ.get("VITFLY_EVAL_BOUNDING_BOX", "")
        bounds = [float(value) for value in bounds_text.split(",")] if bounds_text else self.config["bounding_box"]
        if len(bounds) != 6:
            raise ValueError("VITFLY_EVAL_BOUNDING_BOX must contain six comma-separated values")
        self.bounding_box = np.reshape(np.array(bounds, dtype=float), (3, 2)).T
        self.start_x = float(os.environ.get("VITFLY_EVAL_START_X", "0.5"))

        self.start_time_mark = False
        self.goal_reached = False
        self.termination_reason = None

        # The finish publisher and callback state must be ready before live
        # simulator topics can invoke any subscriber callback.
        self._initPublishers(config["topics"])
        self._initSubscribers(config["topics"])

    def _initSubscribers(self, config):
        self.state_sub = rospy.Subscriber(
            "/%s/%s" % (config["quad_name"], config["state"]),
            QuadState,
            self.callbackState,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.obstacle_sub = rospy.Subscriber(
            "/%s/%s" % (config["quad_name"], config["obstacles"]),
            ObstacleArray,
            self.callbackObstacles,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.dynamic_obstacle_sub = rospy.Subscriber(
            "/%s/dodgeros_pilot/groundtruth/dynamic_obstacles" % config["quad_name"],
            ObstacleArray,
            self.callbackDynamicObstacles,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.start_sub = rospy.Subscriber(
            "/%s/%s" % (config["quad_name"], config["start"]),
            Empty,
            self.callbackStart,
            queue_size=1,
            tcp_nodelay=True,
        )

    def _initPublishers(self, config):
        self.finish_pub = rospy.Publisher(
            "/%s/%s" % (config["quad_name"], config["finish"]),
            Empty,
            queue_size=1,
            tcp_nodelay=True,
        )

    def publishFinish(self):
        # Persist the evaluator result before asking the controller to stop.
        # Otherwise the launcher can observe the controller exit first and
        # overwrite a successful rollout with controller_error.
        with self.finish_lock:
            if self.finished:
                return
            self.finished = True
            self.is_active = False
            self.printSummary()
            self.finish_pub.publish()
            if self.plots_enabled:
                self.printPlots()
            rospy.signal_shutdown("Completed Evaluation")

    def callbackState(self, msg):

        self.pos_x = msg.pose.position.x

        # mark start time based on position rather than start signal
        if self.pos_x > self.start_x and not self.start_time_mark:
            self.time_array[0] = rospy.get_rostime().to_sec()
            self.start_time_mark = True

        if not self.is_active:
            return

        pos = np.array(
            [
                msg.header.stamp.to_sec(),
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ]
        )
        self.pos.append(pos)
        self.altitudes.append(float(msg.pose.position.z))

        bin_x = int(max(min(np.floor(self.pos_x), self.xmax), 0))
        if np.isnan(self.time_array[bin_x]):
            self.time_array[bin_x] = rospy.get_rostime().to_sec()
        if self.pos_x >= self.xmax:
            self.goal_reached = True
            self.is_active = False
            self.publishFinish()
            return

        if rospy.get_time() - self.time_array[0] > self.timeout:
            self.abortRun()

        outside = ((pos[1:] > self.bounding_box[1, :]) | (pos[1:] < self.bounding_box[0, :])
        ).any(axis=-1)
        if (outside == True).any():
            self.termination_reason = "out_of_bounds"
            self.abortRun()

    # Note, the start signal may need to be sent multiple times. Sometimes once doesn't work.
    # So, self.time_array[0] is set in callbackState based on position instead.
    def callbackStart(self, msg):
        if not self.is_active:
            self.is_active = True
        # self.time_array[0] = rospy.get_rostime().to_sec()

    def callbackObstacles(self, msg):
        if not self.is_active or self.finished:
            return

        if not msg.obstacles:
            return
        margin = min(
            np.linalg.norm(np.array([obs.position.x, obs.position.y, obs.position.z])) - obs.scale
            for obs in msg.obstacles
        )
        self.dist.append([msg.header.stamp.to_sec(), margin])
        if margin < self.collision_margin:
            if not self.hit_obstacle:
                self.crash += 1
                print("Crashed")
            self.hit_obstacle = True
            if self.terminate_on_collision and self.crash == 1:
                self.termination_reason = "collision"
                self.abortRun()
                return
        else:
            self.hit_obstacle = False

    def callbackDynamicObstacles(self, msg):
        """Record dynamic-obstacle encounters separately from total collisions."""
        if not self.is_active or self.finished:
            return
        timestamp = getattr(msg, "t", None)
        if timestamp in (None, 0):
            timestamp = msg.header.stamp.to_sec()
        timestamp = float(timestamp)
        if self.dynamic_last_timestamp is not None and self.dynamic_in_encounter:
            self.dynamic_interaction_time += max(0.0, timestamp - self.dynamic_last_timestamp)
        self.dynamic_last_timestamp = timestamp

        if not msg.obstacles:
            self.dynamic_in_encounter = False
            return
        margin = min(
            np.linalg.norm(np.array([obs.position.x, obs.position.y, obs.position.z])) - obs.scale
            for obs in msg.obstacles
        )
        self.dynamic_clearances.append(float(margin))
        if margin < self.collision_margin:
            self.dynamic_collision = True
        in_encounter = margin <= self.dynamic_encounter_margin
        if in_encounter and not self.dynamic_in_encounter:
            self.dynamic_encounter_count += 1
        self.dynamic_in_encounter = in_encounter

    def diagnostic_summary(self):
        return {
            "altitude_min": min(self.altitudes) if self.altitudes else "",
            "altitude_mean": float(np.mean(self.altitudes)) if self.altitudes else "",
            "altitude_max": max(self.altitudes) if self.altitudes else "",
            "min_dynamic_clearance": min(self.dynamic_clearances) if self.dynamic_clearances else "",
            "dynamic_encounter_count": int(self.dynamic_encounter_count),
            "dynamic_interaction_time": float(self.dynamic_interaction_time),
            "dynamic_collision": bool(self.dynamic_collision),
        }

    def abortRun(self):
        with self.finish_lock:
            if self.finished:
                return
            self.finished = True
            print("You did not reach the goal!")
            self.is_active = False
            summary = {}
            summary["Success"] = False
            summary["goal_reached"] = bool(self.goal_reached)
            summary["collision"] = bool(self.crash > 0)
            summary["collision_count"] = int(self.crash)
            summary["termination_reason"] = self.termination_reason or ("collision" if self.crash else "timeout")
            summary.update(self.diagnostic_summary() if hasattr(self, "diagnostic_summary") else {})
            if self.time_array[0] == self.time_array[0]:
                summary["termination_elapsed_time"] = rospy.get_time() - self.time_array[0]
            self.writeSummary(summary)
            # Persist the result before stopping the controller, just as in
            # the successful path. This keeps collision termination a normal
            # rollout result instead of a controller_error.
            self.finish_pub.publish()
            rospy.signal_shutdown("Completed Evaluation")

    def writeSummary(self, summary):
        payload = summary
        if os.getenv("ROLLOUT_NAME") is not None:
            payload = {os.getenv("ROLLOUT_NAME"): summary}
        temporary_path = ".summary.yaml.tmp"
        with open(temporary_path, "w") as stream:
            yaml.safe_dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, "summary.yaml")

    def printSummary(self):
        
        ttf = self.time_array[-1] - self.time_array[0]
        summary = {}
        summary["Success"] = True if self.crash == 0 else False
        summary["goal_reached"] = True
        summary["collision"] = bool(self.crash > 0)
        summary["termination_reason"] = "goal_reached" if self.crash == 0 else "goal_reached_with_collision"
        print("You reached the goal in %5.3f seconds" % ttf)
        summary["time_to_finish"] = ttf
        print("Your intermediate times are:")
        print_distance = 10
        summary["segment_times"] = {}
        for i in range(print_distance, self.xmax + 1, print_distance):
            print("    %2i: %5.3fs " % (i, self.time_array[i] - self.time_array[0]))
            summary["segment_times"]["%i" % i] = self.time_array[i] - self.time_array[0]
        print("You hit %i obstacles" % self.crash)
        summary["number_crashes"] = self.crash
        summary["collision_count"] = self.crash
        summary["termination_elapsed_time"] = ttf
        summary.update(self.diagnostic_summary() if hasattr(self, "diagnostic_summary") else {})
        self.writeSummary(summary)

    def printPlots(self):
        print("Here is a plot of your trajectory in the xy plane")
        pos = np.array(self.pos)
        plot(xs=pos[:, 1], ys=pos[:, 2], color=True)

        print("Here is a plot of your average velocity per 1m x-segment")
        x = np.arange(1, self.xmax + 1)
        dt = np.array(self.time_array)
        y = 1 / (dt[1:] - dt[0:-1])
        plot(xs=x, ys=y, color=True)

        print("Here is a plot of the distance to the closest obstacles")
        dist = np.array(self.dist)
        plot(xs=dist[:, 0] - self.time_array[0], ys=dist[:, 1], color=True)


if __name__ == "__main__":
    with open("./evaluation_config.yaml") as f:
        config = yaml.safe_load(f)
    Evaluator(config)
    rospy.spin()
