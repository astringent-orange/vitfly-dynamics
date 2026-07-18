import os
import yaml
import rospy
import numpy as np

from dodgeros_msgs.msg import QuadState
from envsim_msgs.msg import ObstacleArray
from std_msgs.msg import Empty

from uniplot import plot


class Evaluator:
    def __init__(self, config):
        rospy.init_node("evaluator", anonymous=False)
        self.config = config

        self.xmax = int(float(os.environ.get("VITFLY_EVAL_GOAL_X", self.config["target"])))

        self.is_active = False
        self.pos = []
        self.dist = []
        self.time_array = (self.xmax + 1) * [np.nan]

        self.hit_obstacle = False
        self.crash = 0
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
        self.printSummary()
        self.finish_pub.publish()
        if self.config["plots"]:
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
        if not self.is_active:
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
        else:
            self.hit_obstacle = False

    def abortRun(self):
        print("You did not reach the goal!")
        self.is_active = False
        summary = {}
        summary["Success"] = False
        summary["goal_reached"] = bool(self.goal_reached)
        summary["collision"] = bool(self.crash > 0)
        summary["collision_count"] = int(self.crash)
        summary["termination_reason"] = self.termination_reason or ("collision" if self.crash else "timeout")
        if self.time_array[0] == self.time_array[0]:
            summary["termination_elapsed_time"] = rospy.get_time() - self.time_array[0]
        self.writeSummary(summary)
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
