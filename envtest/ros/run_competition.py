#!/usr/bin/python3
import argparse

import rospy
from dodgeros_msgs.msg import Command
from dodgeros_msgs.msg import QuadState
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from std_msgs.msg import String

from envsim_msgs.msg import ObstacleArray

# from rl_example import load_rl_policy
from dataset_log_utils import cleanup_orphan_depth_images
from user_code import AStarDynamicExpert, compute_command_vision_based, compute_command_state_based, default_planner_info
from utils import AgileCommandMode, AgileQuadState

import atexit
import json
import time
import threading
from collections import deque
import numpy as np
import pandas as pd
import os, sys
from os.path import join as opj
from copy import deepcopy
import cv2
from cv_bridge import CvBridge
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
    from model import *

PLANNER_FIELDS = [
    "lookahead_x",
    "lookahead_y",
    "lookahead_z",
    "v_path_x",
    "v_path_y",
    "v_path_z",
    "nearest_dyn_dist",
    "nearest_dyn_rel_speed",
    "ttc_min",
    "avoidance_active",
    "astar_replan_count",
    "astar_success",
    "astar_plan_time",
    "nearest_obstacle_margin",
    "nearest_static_dist",
    "dynamic_obstacle_count",
    "v_slowdown_x",
    "v_slowdown_dynamic_x",
    "configured_dynamic_obstacle_count",
    "candidate_selected_speed",
    "candidate_safe_count",
    "candidate_min_clearance",
    "candidate_emergency_stop",
    "candidate_prediction_horizon",
    "candidate_raw_selected_speed",
    "candidate_yield_active",
    "candidate_applied_speed",
    "path_cross_track_error",
    "path_turn_angle_deg",
    "path_speed_ceiling",
    "candidate_control_delay",
    "candidate_brake_decel",
    "candidate_reverse_drift_buffer",
    "candidate_initial_path_speed",
    "candidate_predicted_stop_distance",
]

REPOSITORY_ROOT = os.path.abspath(opj(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_PATHS = {
    0: opj(REPOSITORY_ROOT, "models", "current_frame", "current_frame_vitlstm_000099.pth"),
    1: opj(REPOSITORY_ROOT, "models", "previous_frame", "previous_frame_vitlstm_000099.pth"),
    2: opj(REPOSITORY_ROOT, "models", "second_previous_frame", "second_previous_frame_vitlstm_000099.pth"),
}


def resolve_model_path(offset, model_path=None):
    """Return an explicit checkpoint or the repository default for an offset."""
    offset = int(offset)
    if offset not in DEFAULT_MODEL_PATHS:
        raise ValueError(f"unsupported frame offset: {offset}")
    if model_path:
        expanded = os.path.expanduser(model_path)
        if not os.path.isabs(expanded):
            expanded = opj(REPOSITORY_ROOT, expanded)
        return os.path.abspath(expanded)
    return DEFAULT_MODEL_PATHS[offset]


def prepare_depth_frame(image, threshold=0.09):
    """Normalize one depth frame while preserving valid sparse zero pixels.

    Unity can use zero for individual far/background pixels.  The original
    controller still inferred on such a frame; only an entirely empty or
    malformed frame is unusable.
    """
    image = np.asarray(image)
    if (
        image.ndim != 2
        or image.size == 0
        or np.isnan(image).any()
        or np.isneginf(image).any()
    ):
        return None, "invalid"
    normalized = np.clip(image / threshold, 0, 1)
    if not np.any(normalized > 0.0):
        return None, "all_zero"
    if np.any(normalized <= 0.0):
        return normalized, "partial_zero"
    return normalized, "valid"


class AgilePilotNode:
    def __init__(self, vision_based=False, offset=0, model_path=None, desVel=None, keyboard=False):
        print("[RUN_COMPETITION] Initializing agile_pilot_node...")
        rospy.init_node("agile_pilot_node", anonymous=False)

        self.vision_based = vision_based
        self.rl_policy = None
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        self.keyboard = keyboard
        self.dynamic_obstacles = None
        self.is_shutting_down = False
        self.finished = False
        self.exit_code = 0
        self.controller_failure_detail = ""
        self.depth_frames_received = 0
        self.depth_frames_usable = 0
        self.depth_invalid_frames = 0
        self.depth_all_zero_frames = 0
        self.depth_partial_zero_frames = 0
        self.command_publish_count = 0
        self.first_command_wall_seconds = None
        self.navigation_started_monotonic = None
        self.watchdog_stop = threading.Event()
        self.watchdog_thread = None
        self.controller_diagnostics_path = os.environ.get("VITFLY_CONTROLLER_DIAGNOSTICS_PATH", "")

        quad_name = "kingfisher"

        self.init = 0
        self.col = None
        self.t1 = 0 #Time flag
        self.timestamp = 0 #Time stamp initial
        self.last_valid_img = None #Image that will be logged
        self.frame_history = deque(maxlen=2)
        self.saved_timestamps = set()
        self.last_saved_state_t = None
        self.data_log_lock = threading.RLock()
        data_log_format = {'timestamp':[],
                           'desired_vel':[],
                           'env_level':[],
                           'env_folder':[],
                           'env_seed':[],
                           'dynamic_phase_seed':[],
                           'dynamic_phase_mode':[],
                           'expert_strategy':[],
                           'quat_1':[],
                           'quat_2':[],
                           'quat_3':[],
                           'quat_4':[],
                           'pos_x':[],
                           'pos_y':[],
                           'pos_z':[],
                           'vel_x':[],
                           'vel_y':[],
                           'vel_z':[],
                           'velcmd_x':[],
                           'velcmd_y':[],
                           'velcmd_z':[],
                           'ct_cmd':[],
                           'br_cmd_x':[],
                           'br_cmd_y':[],
                           'br_cmd_z':[],
                           'is_collide': [],
        } 
        for field in PLANNER_FIELDS:
            data_log_format[field] = []
        self.data_log = pd.DataFrame(data_log_format) # store in the data frame
        self.count = 0 # counter for the csv
        
        # @NOTE: Dont log too fast, I have not tested that
        self.time_interval = .03 #Time interval for logging
        try:
            self.flush_every_samples = max(
                1, int(os.environ.get("VITFLY_FLUSH_EVERY_SAMPLES", "50"))
            )
        except ValueError:
            self.flush_every_samples = 50

        self.data_collection_xrange = [2, 60]

        # Benchmark rollouts are evaluated by the dedicated evaluator and must
        # never pollute the expert ``train_set``. State collection keeps the
        # historical behavior when this flag is absent.
        self.benchmark_mode = os.environ.get("VITFLY_BENCHMARK_MODE", "0") == "1"
        timing_default = "false" if self.benchmark_mode else "true"
        self.inference_timing_logs = os.environ.get(
            "VITFLY_INFERENCE_TIMING_LOGS", timing_default
        ).strip().lower() in ("1", "true", "yes", "on")

        # Create a trajectory directory only after the first valid sample.
        self.folder = None
        self.env_level = os.environ.get("VITFLY_ENV_LEVEL", "dynamic_astar_medium")
        self.env_folder = os.environ.get("VITFLY_ENV_FOLDER", "environment_0")
        self.env_seed = os.environ.get("VITFLY_ENV_SEED", "")
        self.dynamic_phase_seed = os.environ.get("VITFLY_DYNAMIC_PHASE_SEED", self.env_seed)
        self.dynamic_phase_mode = "seeded_navigation_reset"
        requested_expert = os.environ.get("VITFLY_STATE_EXPERT", "astar_dynamic").strip().lower()
        if requested_expert in ("vitfly", "original", "vitfly_original"):
            self.expert_strategy = "vitfly_original"
        elif requested_expert in ("astar", "astar_dynamic", "experiment"):
            self.expert_strategy = "astar_dynamic"
        else:
            raise ValueError(
                "VITFLY_STATE_EXPERT must be astar_dynamic or vitfly_original"
            )
        atexit.register(self.flush_data_log)
        rospy.on_shutdown(self.shutdown_callback)

        self.desiredVel = desVel #self.readVel("velocity.txt") #np.random.uniform(low=2.0, high=3.0)
        print()
        print(f"[RUN_COMPETITION] Desired velocity = {self.desiredVel}")
        print(f"[RUN_COMPETITION] Environment = {self.env_level}/{self.env_folder} seed={self.env_seed} phase_seed={self.dynamic_phase_seed}")
        print()

        self.frame_offset = int(offset)
        self.model_type, self.num_input_frames, self.checkpoint_prefix = frame_mode_spec(self.frame_offset)
        if self.vision_based:
            model_path = resolve_model_path(self.frame_offset, model_path)

        self.state_expert = None
        if not self.vision_based and not self.keyboard:
            if self.expert_strategy == "astar_dynamic":
                self.state_expert = AStarDynamicExpert()
                print("[RUN_COMPETITION] A* dynamic state expert initialized")
            elif self.expert_strategy == "vitfly_original":
                print("[RUN_COMPETITION] Original Vitfly grid-waypoint expert initialized")
            else:
                raise ValueError(
                    "VITFLY_STATE_EXPERT must be astar_dynamic or vitfly_original"
                )
        elif self.vision_based:
            self.expert_strategy = "vision_model"
        elif self.keyboard:
            self.expert_strategy = "keyboard"

        # load trained model here (copied over from user_code.py)
        if self.vision_based:
            if torch is None:
                raise RuntimeError("Torch is required for vision-based model inference.")
            print(f"[RUN_COMPETITION] Model loading from {model_path} ...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = globals()[self.model_type]().to(self.device).float()

            # Give full path if possible since the bash script runs from outside the folder
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            metadata_path = opj(os.path.dirname(os.path.abspath(model_path)), 'run_metadata.json')
            allow_legacy = os.environ.get("VITFLY_ALLOW_LEGACY_CHECKPOINT", "0") == "1"
            if not os.path.isfile(metadata_path) and not allow_legacy:
                raise ValueError(f'[RUN_COMPETITION] Missing run_metadata.json next to {model_path}')
            if os.path.isfile(metadata_path):
                with open(metadata_path) as stream:
                    metadata = json.load(stream)
                if metadata.get('model_type') != self.model_type or int(metadata.get('frame_offset', -1)) != self.frame_offset:
                    raise ValueError('[RUN_COMPETITION] checkpoint metadata does not match offset')
            elif allow_legacy and self.frame_offset != 0:
                raise ValueError('[RUN_COMPETITION] legacy checkpoints only support offset=0')
            self.model.eval()

            # Initialize hidden state
            self.model_hidden_state = None

            print(f"[RUN_COMPETITION] Model loaded")
            time.sleep(2)

        self.start_time = 0
        self.logged_time_flag = 0
        self.depth_im_threshold = 0.09

        self.curr_cmd = None
        self.ctr = 0
        self.keyboard_input = ''
        self.got_keypress = 0.0
        self.rgb_img = None
        self.save_rgb_debug = False
        self.debug_rgb_folder = None

        # Publishers and all callback-visible state must exist before any
        # subscriber is created: rospy may invoke a callback immediately from
        # a live Flightmare topic during Subscriber construction.
        self.cmd_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/feedthrough_command",
            Command,
            queue_size=1,
        )
        self.linvel_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/velocity_command",
            TwistStamped,
            queue_size=1,
        )
        self.debug_img1_pub = None
        self.debug_img2_pub = None
        if not self.benchmark_mode:
            self.debug_img1_pub = rospy.Publisher(
                "/debug_img1",
                Image,
                queue_size=1,
            )
            self.debug_img2_pub = rospy.Publisher(
                "/debug_img2",
                Image,
                queue_size=1,
            )

        # Logic subscribers
        self.start_sub = rospy.Subscriber(
            "/" + quad_name + "/start_navigation",
            Empty,
            self.start_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.finish_sub = rospy.Subscriber(
            "/" + quad_name + "/finish_navigation",
            Empty,
            self.finish_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Observation subscribers
        self.odom_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/state",
            QuadState,
            self.state_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.img_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/depth",
            Image,
            self.img_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.obstacle_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles",
            ObstacleArray,
            self.obstacle_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.dynamic_obstacle_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/groundtruth/dynamic_obstacles",
            ObstacleArray,
            self.dynamic_obstacle_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.cmd_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/command",
            Command,
            self.cmd_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.keyboard_sub = rospy.Subscriber(
            "/keyboard_input",
            String,
            self.keyboard_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.rgb_img_sub = None
        if not self.benchmark_mode:
            self.rgb_img_sub = rospy.Subscriber(
                "/" + quad_name + "/dodgeros_pilot/unity/image",
                Image,
                self.rgb_callback,
                queue_size=1,
                tcp_nodelay=True,
            )


        print("[RUN_COMPETITION] Initialization completed!")
    def rgb_callback(self, img):
        self.rgb_img = self.cv_bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

    def cmd_callback(self, msg):
        self.curr_cmd = msg

    def keyboard_callback(self, msg):
        self.got_keypress = rospy.Time().now().to_sec()
        self.keyboard_input = msg.data

    def readVel(self,file):
        with open(file,"r") as f:
            x = f.readlines()
            for i in range(len(x)):
                if i == 0:
                    return float(x[i].split("\n")[0])

    def current_low_level_cmd(self):
        if self.curr_cmd is None:
            return 0.0, 0.0, 0.0, 0.0
        return (
            self.curr_cmd.collective_thrust,
            self.curr_cmd.bodyrates.x,
            self.curr_cmd.bodyrates.y,
            self.curr_cmd.bodyrates.z,
        )

    def planner_log_values(self, planner_info):
        info = default_planner_info()
        if planner_info is not None:
            info.update(planner_info)
        return [info[field] for field in PLANNER_FIELDS]

    def ensure_data_folder(self):
        if self.benchmark_mode:
            return
        if self.folder is not None:
            return
        base_folder = "train_set"
        os.makedirs(base_folder, exist_ok=True)
        while True:
            candidate = opj(base_folder, str(int(time.time() * 100)))
            try:
                os.mkdir(candidate)
                self.folder = candidate
                self.debug_rgb_folder = opj(candidate, "debug_rgb")
                print(f"[RUN_COMPETITION] Recording trajectory in {candidate}")
                return
            except FileExistsError:
                time.sleep(0.01)

    def sanitize_data_log(self):
        with self.data_log_lock:
            if not hasattr(self, "data_log") or self.folder is None:
                return
            cleaned = self.data_log.drop_duplicates(subset=["timestamp"], keep="first")
            if "pos_x" in cleaned.columns:
                pos_x = pd.to_numeric(cleaned["pos_x"], errors="coerce")
                cleaned = cleaned[pos_x < self.data_collection_xrange[1]]

            cleanup_orphan_depth_images(self.folder, cleaned["timestamp"].tolist())

            self.data_log = cleaned.reset_index(drop=True)
            self.saved_timestamps = set()
            for timestamp in self.data_log["timestamp"].tolist():
                try:
                    self.saved_timestamps.add(float(timestamp))
                except (TypeError, ValueError):
                    pass

    def flush_data_log(self):
        if getattr(self, "benchmark_mode", False):
            return
        with self.data_log_lock:
            try:
                if self.folder is not None and hasattr(self, "data_log"):
                    self.sanitize_data_log()
                    self.data_log.to_csv(self.folder + "/data.csv", index=False)
            except Exception as exc:
                print(f"[RUN_COMPETITION] Failed to flush data.csv: {exc}")

    def shutdown_callback(self):
        self.is_shutting_down = True
        self.publish_commands = False
        self.watchdog_stop.set()
        self.write_controller_diagnostics()
        self.flush_data_log()

    def controller_diagnostics(self):
        return {
            "depth_frames_received": int(self.depth_frames_received),
            "depth_frames_usable": int(self.depth_frames_usable),
            "depth_invalid_frames": int(self.depth_invalid_frames),
            "depth_all_zero_frames": int(self.depth_all_zero_frames),
            "depth_partial_zero_frames": int(self.depth_partial_zero_frames),
            "command_publish_count": int(self.command_publish_count),
            "first_command_wall_seconds": self.first_command_wall_seconds,
            "controller_failure_detail": self.controller_failure_detail,
            "exit_code": int(self.exit_code),
        }

    def write_controller_diagnostics(self):
        if not self.controller_diagnostics_path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(self.controller_diagnostics_path))
            os.makedirs(parent, exist_ok=True)
            tmp_path = self.controller_diagnostics_path + ".tmp"
            with open(tmp_path, "w") as stream:
                json.dump(self.controller_diagnostics(), stream, sort_keys=True)
                stream.write("\n")
            os.replace(tmp_path, self.controller_diagnostics_path)
        except Exception as exc:
            print(f"[RUN_COMPETITION] Failed to write controller diagnostics: {exc}")

    def _watchdog_loop(self):
        timeout = float(os.environ.get("VITFLY_NO_COMMAND_TIMEOUT_SECONDS", "5"))
        while not self.watchdog_stop.wait(0.2):
            if (
                not self.benchmark_mode
                or self.finished
                or not self.publish_commands
                or self.navigation_started_monotonic is None
                or self.command_publish_count > 0
            ):
                continue
            if time.monotonic() - self.navigation_started_monotonic < timeout:
                continue
            if self.depth_frames_usable == 0:
                self.controller_failure_detail = "no_usable_depth"
                self.exit_code = 3
                reason = "No usable depth frame after navigation"
            else:
                self.controller_failure_detail = "no_command"
                self.exit_code = 2
                reason = "No velocity command after navigation"
            print(
                f"[RUN_COMPETITION] {reason}; received={self.depth_frames_received} "
                f"usable={self.depth_frames_usable} invalid={self.depth_invalid_frames}"
            )
            self.finish_run(reason)
            return

    def start_watchdog(self):
        if self.watchdog_thread is None and self.benchmark_mode:
            self.watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="benchmark_command_watchdog", daemon=True
            )
            self.watchdog_thread.start()

    def reached_goal(self, state=None):
        state = state if state is not None else self.state
        return state is not None and state.pos[0] >= self.data_collection_xrange[1]

    def publish_zero_velocity(self):
        if not hasattr(self, "linvel_pub"):
            return
        try:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time.now()
            vel_msg.twist.linear.x = 0.0
            vel_msg.twist.linear.y = 0.0
            vel_msg.twist.linear.z = 0.0
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = 0.0
            self.linvel_pub.publish(vel_msg)
        except rospy.exceptions.ROSException:
            pass

    def finish_run(self, reason):
        if self.finished:
            return
        self.finished = True
        self.watchdog_stop.set()
        print(f"[RUN_COMPETITION] Finishing run: {reason}")
        self.publish_zero_velocity()
        self.publish_commands = False
        self.write_controller_diagnostics()
        self.flush_data_log()
        rospy.signal_shutdown(reason)

    def try_log_sample(self, command, state_snapshot, planner_info=None, nearest_margin=None):
        if self.benchmark_mode:
            return False
        with self.data_log_lock:
            return self._try_log_sample_locked(command, state_snapshot, planner_info, nearest_margin)

    def _try_log_sample_locked(self, command, state_snapshot, planner_info=None, nearest_margin=None):
        if self.finished or state_snapshot is None or self.last_valid_img is None:
            return False
        if state_snapshot.pos[0] >= self.data_collection_xrange[1]:
            self.finish_run("Reached goal")
            return False
        if state_snapshot.pos[0] <= self.data_collection_xrange[0]:
            return False

        timestamp = round(state_snapshot.t, 3)
        if timestamp in self.saved_timestamps:
            return False

        self.ensure_data_folder()
        image_path = f"{self.folder}/{str(timestamp)}.png"
        depth_image = self.last_valid_img.copy()
        if not cv2.imwrite(image_path, (depth_image * 255).astype(np.uint8)):
            print(f"[RUN_COMPETITION] Failed to write depth image {image_path}")
            return False
        if self.save_rgb_debug and self.rgb_img is not None:
            os.makedirs(self.debug_rgb_folder, exist_ok=True)
            cv2.imwrite(f"{self.debug_rgb_folder}/{str(timestamp)}_rgb.png", (self.rgb_img * 255).astype(np.uint8))

        if nearest_margin is None:
            is_collide = int(self.col) if self.col is not None else 0
        else:
            is_collide = int(nearest_margin < 0.0)
        ct_cmd, br_x, br_y, br_z = self.current_low_level_cmd()
        self.data_log.loc[len(self.data_log)] = [
            timestamp,
            self.desiredVel,
            self.env_level,
            self.env_folder,
            self.env_seed,
            self.dynamic_phase_seed,
            self.dynamic_phase_mode,
            self.expert_strategy,
            state_snapshot.att[0],
            state_snapshot.att[1],
            state_snapshot.att[2],
            state_snapshot.att[3],
            state_snapshot.pos[0],
            state_snapshot.pos[1],
            state_snapshot.pos[2],
            state_snapshot.vel[0],
            state_snapshot.vel[1],
            state_snapshot.vel[2],
            command.velocity[0],
            command.velocity[1],
            command.velocity[2],
            ct_cmd,
            br_x,
            br_y,
            br_z,
            is_collide,
        ] + self.planner_log_values(planner_info)

        self.saved_timestamps.add(timestamp)
        self.last_saved_state_t = state_snapshot.t
        self.t1 = state_snapshot.t
        self.count += 1
        return True

    def img_callback(self, img_data):
        if rospy.is_shutdown() or self.is_shutting_down or self.finished:
            return
        self.ctr += 1
        self.depth_frames_received += 1
        raw_img = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
        img, depth_status = prepare_depth_frame(raw_img, self.depth_im_threshold)
        if img is None and depth_status == "invalid":
            self.depth_invalid_frames += 1
            return
        if depth_status == "all_zero":
            self.depth_invalid_frames += 1
            self.depth_all_zero_frames += 1
            return
        self.depth_frames_usable += 1
        if depth_status == "partial_zero":
            self.depth_partial_zero_frames += 1
        if self.last_valid_img is not None:
            self.frame_history.append(deepcopy(self.last_valid_img))
        self.last_valid_img = deepcopy(img)
        
        
        
        
        

        if not self.vision_based:
            return
        
        if self.state is None:
            return
        state_snapshot = deepcopy(self.state)
        if self.reached_goal(state_snapshot):
            self.finish_run("Reached goal")
            return
        
        # print('[RUN_COMPETITION] calling compute_command_vision_based')
        start_compute_time = time.time() if self.inference_timing_logs else None

        try:
            command, (debug_img1, debug_img2), self.model_hidden_state = compute_command_vision_based(
                state_snapshot, img, self.frame_history, self.desiredVel, self.model,
                self.model_hidden_state, self.frame_offset,
            )
        except Exception as exc:
            self.controller_failure_detail = f"inference_error:{type(exc).__name__}"
            self.exit_code = 2
            print(f"[RUN_COMPETITION] Inference failed: {exc}")
            self.finish_run("Controller inference error")
            return
        if command is None:
            return

        # publish debug images
        if not self.benchmark_mode:
            self.debug_img1_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img1, encoding="passthrough"))
            self.debug_img2_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img2, encoding="passthrough"))

        if self.inference_timing_logs and self.ctr % 30 == 0:
            print(f'[RUN_COMPETITION] compute_command_vision_based took {time.time() - start_compute_time} seconds')

        self.publish_command(command)
        # print(f'[RUN_COMPETITION] output: {command.velocity}')

        if state_snapshot.pos[0] < 0.1:
            self.start_time = command.t

        if state_snapshot.pos[0] >= 60 and self.logged_time_flag == 0 and not self.benchmark_mode:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        #if we exceed the time interval then save the data
        if state_snapshot.t - self.t1 > self.time_interval or self.t1 == 0:
            self.try_log_sample(command, state_snapshot, None, None)

        # Periodic full CSV rewrites are deliberately infrequent; finish and
        # shutdown paths still flush synchronously.
        if self.count % self.flush_every_samples == 0 and self.count != 0:
            self.flush_data_log()

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)
        if self.reached_goal():
            self.finish_run("Reached goal")

    def obstacle_callback(self, obs_data):
        if rospy.is_shutdown() or self.is_shutting_down or self.finished:
            return
        if self.state is None:
            return
        state_snapshot = deepcopy(self.state)
        if self.reached_goal(state_snapshot):
            self.finish_run("Reached goal")
            return
        nearest_margin = self.nearest_obstacle_margin(obs_data)
        self.col = int(nearest_margin < 0.0)
        if self.vision_based:
            return

        # try:
        #     self.desiredVel = self.readVel("velocity.txt") #Changed some thing
        # except:
        #     pass
        # usable keypress?
        if rospy.Time().now().to_sec() - self.got_keypress > 0.1:
            self.keyboard_input = ''

        command, planner_info = compute_command_state_based(
            state=state_snapshot,
            obstacles=obs_data,
            desiredVel=self.desiredVel,
            rl_policy=self.rl_policy,
            keyboard=self.keyboard,
            keyboard_input=self.keyboard_input,
            expert=self.state_expert,
            return_info=True,
            dynamic_obstacles=self.dynamic_obstacles,
        )
        if not self.publish_command(command):
            return

        if state_snapshot.pos[0] < 0.1:
            self.start_time = command.t
        if state_snapshot.pos[0] >= 60 and self.logged_time_flag == 0:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        # if we exceed the time interval then save the data
        if (state_snapshot.t - self.t1 > self.time_interval or self.t1 == 0) and (state_snapshot.pos[2] > 2.95 or self.init == 1):
            
            self.init = 1

            self.try_log_sample(command, state_snapshot, planner_info, nearest_margin)

        if self.count % self.flush_every_samples == 0 and self.count != 0:
            self.flush_data_log()

    def dynamic_obstacle_callback(self, obs_data):
        self.dynamic_obstacles = obs_data

    def finish_callback(self, _msg):
        self.finish_run("Received finish_navigation")

    def nearest_obstacle_margin(self, obstacles):
        if obstacles is None or not obstacles.obstacles:
            return float("inf")
        margins = []
        for obs in obstacles.obstacles:
            dist = np.linalg.norm(np.array([obs.position.x, obs.position.y, obs.position.z]))
            margins.append(dist - obs.scale)
        return float(min(margins)) if margins else float("inf")

    def if_collide(self, obs):
        """
        Borrowed and modified from evaluation_node
        """

        dist = np.linalg.norm(
            np.array([obs.position.x, obs.position.y, obs.position.z])
        )
        margin = dist - obs.scale
        # Ground hit condition
        if margin < 0 or self.state.pos[2] <= 0.01:
            hit_obstacle = True
        else:
            hit_obstacle = False

        return hit_obstacle

    def publish_command(self, command):
        if rospy.is_shutdown() or self.is_shutting_down or not self.publish_commands:
            return False
        if command.mode == AgileCommandMode.SRT:
            assert len(command.rotor_thrusts) == 4
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = True
            cmd_msg.thrusts = command.rotor_thrusts
            if self.publish_commands:
                try:
                    self.cmd_pub.publish(cmd_msg)
                except rospy.exceptions.ROSException:
                    return False
                self.record_command_published()
                return True
        elif command.mode == AgileCommandMode.CTBR:
            assert len(command.bodyrates) == 3
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = False
            cmd_msg.collective_thrust = command.collective_thrust
            cmd_msg.bodyrates.x = command.bodyrates[0]
            cmd_msg.bodyrates.y = command.bodyrates[1]
            cmd_msg.bodyrates.z = command.bodyrates[2]
            if self.publish_commands:
                try:
                    self.cmd_pub.publish(cmd_msg)
                except rospy.exceptions.ROSException:
                    return False
                self.record_command_published()
                return True
        elif command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                try:
                    self.linvel_pub.publish(vel_msg)
                except rospy.exceptions.ROSException:
                    return False
                self.record_command_published()
                return True
        else:
            assert False, "Unknown command mode specified"
        return False

    def start_callback(self, data):
        if rospy.is_shutdown() or self.is_shutting_down or self.finished:
            return
        self.frame_history.clear()
        self.last_valid_img = None
        if hasattr(self, 'model_hidden_state'):
            self.model_hidden_state = None
        print("[RUN_COMPETITION] Start publishing commands!")
        self.navigation_started_monotonic = time.monotonic()
        self.start_watchdog()
        self.publish_commands = True

    def record_command_published(self):
        self.command_publish_count += 1
        if self.first_command_wall_seconds is None and self.navigation_started_monotonic is not None:
            self.first_command_wall_seconds = time.monotonic() - self.navigation_started_monotonic
            self.write_controller_diagnostics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agile Pilot.")
    parser.add_argument("--vision_based", help="Fly vision-based", required=False, dest="vision_based", action="store_true")
    parser.add_argument('--offset', type=int, choices=(0, 1, 2), default=0, help='frame mode: 0=current, 1=previous, 2=second previous')
    parser.add_argument('--model_path', type=str, default=None, help='optional checkpoint path; defaults from --offset')
    parser.add_argument('--des_vel', type=float, default=None, help='desired velocity for quadrotor')
    parser.add_argument('--benchmark-mode', action='store_true', help='isolate benchmark rollout logging from train_set')
    parser.add_argument('--policy-config', type=str, default=None, help='optional benchmark policy YAML (metadata only)')
    parser.add_argument('--case-config', type=str, default=None, help='optional benchmark case YAML (metadata only)')
    parser.add_argument('--evaluation-profile', type=str, default=None, help='benchmark evaluation profile name')
    parser.add_argument("--keyboard", help="Fly state-based mode but take velocity commands from keyboard WASD", required=False, dest="keyboard", action="store_true")

    args = parser.parse_args()
    if args.benchmark_mode:
        os.environ["VITFLY_BENCHMARK_MODE"] = "1"
    for name, value in (("VITFLY_POLICY_CONFIG", args.policy_config), ("VITFLY_CASE_CONFIG", args.case_config), ("VITFLY_EVALUATION_PROFILE", args.evaluation_profile)):
        if value:
            os.environ[name] = value
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, offset=args.offset, model_path=args.model_path, desVel=args.des_vel, keyboard=args.keyboard)
    rospy.spin()
    agile_pilot_node.flush_data_log()
    sys.exit(agile_pilot_node.exit_code)
