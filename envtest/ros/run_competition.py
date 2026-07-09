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
from user_code import AStarDynamicExpert, compute_command_vision_based, compute_command_state_based, default_planner_info
from utils import AgileCommandMode, AgileQuadState

import atexit
import time
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
    "v_avoid_x",
    "v_avoid_y",
    "v_avoid_z",
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
    "v_slowdown_static_x",
    "local_fallback_control_enabled",
    "configured_dynamic_obstacle_count",
]


class AgilePilotNode:
    def __init__(self, vision_based=False, model_type=None, model_path=None, desVel=None, keyboard=False):
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

        quad_name = "kingfisher"

        self.init = 0
        self.col = None
        self.t1 = 0 #Time flag
        self.timestamp = 0 #Time stamp initial
        self.last_valid_img = None #Image that will be logged
        self.saved_timestamps = set()
        self.last_saved_state_t = None
        data_log_format = {'timestamp':[],
                           'desired_vel':[],
                           'env_level':[],
                           'env_folder':[],
                           'env_seed':[],
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

        self.data_collection_xrange = [2, 60]

        # make the folder for the epoch
        self.folder = f"train_set/{int(time.time()*100)}" 
        os.mkdir(self.folder)
        self.env_level = os.environ.get("VITFLY_ENV_LEVEL", "dynamic_astar_medium")
        self.env_folder = os.environ.get("VITFLY_ENV_FOLDER", "environment_0")
        self.env_seed = os.environ.get("VITFLY_ENV_SEED", "")
        atexit.register(self.flush_data_log)
        rospy.on_shutdown(self.shutdown_callback)

        self.desiredVel = desVel #self.readVel("velocity.txt") #np.random.uniform(low=2.0, high=3.0)
        print()
        print(f"[RUN_COMPETITION] Desired velocity = {self.desiredVel}")
        print(f"[RUN_COMPETITION] Environment = {self.env_level}/{self.env_folder} seed={self.env_seed}")
        print()

        self.state_expert = None
        if not self.vision_based and not self.keyboard:
            self.state_expert = AStarDynamicExpert()
            print("[RUN_COMPETITION] A* dynamic state expert initialized")

        # load trained model here (copied over from user_code.py)
        if self.vision_based and model_path is not None:
            if torch is None:
                raise RuntimeError("Torch is required for vision-based model inference.")
            print(f"[RUN_COMPETITION] Model loading from {model_path} ...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if model_type == 'LSTMNet':
                self.model = LSTMNet().to(self.device).float()
            elif model_type == 'UNetLSTM':
                self.model = UNetConvLSTMNet().to(self.device).float()
            elif model_type == 'ConvNet':
                self.model = ConvNet().to(self.device).float()                
            elif model_type == 'ViT':
                self.model = ViT().to(self.device).float()
            elif model_type == 'ViTLSTM':
                self.model = LSTMNetVIT().to(self.device).float()                
            else:
                print(f'[RUN_COMPETITION] Invalid model_type {model_type}. Exiting.')
                exit()

            # Give full path if possible since the bash script runs from outside the folder
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            # Initialize hidden state
            self.model_hidden_state = None

            print(f"[RUN_COMPETITION] Model loaded")
            time.sleep(2)

        self.start_time = 0
        self.logged_time_flag = 0
        self.depth_im_threshold = 0.09

        self.curr_cmd = None

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
        self.rgb_img_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/image",
            Image,
            self.rgb_callback,
            queue_size=1,
            tcp_nodelay=True,
        )


        # Command publishers
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
        print("[RUN_COMPETITION] Initialization completed!")

        self.ctr = 0

        self.keyboard_input = ''
        self.got_keypress = 0.0
        self.rgb_img = None
        self.save_rgb_debug = False
        self.debug_rgb_folder = opj(self.folder, "debug_rgb")

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

    def sanitize_data_log(self):
        if not hasattr(self, "data_log") or self.data_log.empty:
            return
        before_timestamps = {str(ts) for ts in self.data_log["timestamp"].tolist()}
        cleaned = self.data_log.drop_duplicates(subset=["timestamp"], keep="first")
        if "pos_x" in cleaned.columns:
            pos_x = pd.to_numeric(cleaned["pos_x"], errors="coerce")
            cleaned = cleaned[pos_x < self.data_collection_xrange[1]]

        kept_timestamps = {str(ts) for ts in cleaned["timestamp"].tolist()}
        for timestamp in before_timestamps - kept_timestamps:
            image_path = f"{self.folder}/{timestamp}.png"
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except OSError as exc:
                    print(f"[RUN_COMPETITION] Failed to remove dropped image {image_path}: {exc}")

        self.data_log = cleaned.reset_index(drop=True)
        self.saved_timestamps = set()
        for timestamp in self.data_log["timestamp"].tolist():
            try:
                self.saved_timestamps.add(float(timestamp))
            except (TypeError, ValueError):
                pass

    def flush_data_log(self):
        try:
            if hasattr(self, "folder") and hasattr(self, "data_log"):
                self.sanitize_data_log()
                self.data_log.to_csv(self.folder + "/data.csv", index=False)
        except Exception as exc:
            print(f"[RUN_COMPETITION] Failed to flush data.csv: {exc}")

    def shutdown_callback(self):
        self.is_shutting_down = True
        self.publish_commands = False
        self.flush_data_log()

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
        print(f"[RUN_COMPETITION] Finishing run: {reason}")
        self.publish_zero_velocity()
        self.publish_commands = False
        self.flush_data_log()
        rospy.signal_shutdown(reason)

    def try_log_sample(self, command, state_snapshot, planner_info=None, nearest_margin=None):
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

        image_path = f"{self.folder}/{str(timestamp)}.png"
        if not cv2.imwrite(image_path, (self.last_valid_img * 255).astype(np.uint8)):
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
        self.prevImg = deepcopy(self.last_valid_img)
        img = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
        img = np.clip(img/self.depth_im_threshold, 0, 1)
                
        if self.prevImg is None:
            self.prevImg = img

        self.last_valid_img = deepcopy(img) if img.min() > 0.0 else self.last_valid_img
        
        
        
        
        

        if not self.vision_based:
            return
        
        if self.state is None:
            return
        state_snapshot = deepcopy(self.state)
        if self.reached_goal(state_snapshot):
            self.finish_run("Reached goal")
            return
        
        # print('[RUN_COMPETITION] calling compute_command_vision_based')
        start_compute_time = time.time()

        command, (debug_img1, debug_img2), self.model_hidden_state = compute_command_vision_based(state_snapshot, img, self.prevImg,self.desiredVel, self.model, self.model_hidden_state)

        # publish debug images
        self.debug_img1_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img1, encoding="passthrough"))
        self.debug_img2_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img2, encoding="passthrough"))

        if self.ctr % 30 == 0:
            print(f'[RUN_COMPETITION] compute_command_vision_based took {time.time() - start_compute_time} seconds')

        self.publish_command(command)
        # print(f'[RUN_COMPETITION] output: {command.velocity}')

        if state_snapshot.pos[0] < 0.1:
            self.start_time = command.t

        if state_snapshot.pos[0] >= 60 and self.logged_time_flag == 0:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        #if we exceed the time interval then save the data
        if state_snapshot.t - self.t1 > self.time_interval or self.t1 == 0:
            self.try_log_sample(command, state_snapshot, None, None)

        # Save once every 10 instances - writing every instance can be expensive
        if self.count % 5 == 0 and self.count != 0:
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

        # Save once every 10 instances - writing every instance can be expensive
        if self.count % 2 == 0 and self.count != 0 or abs(state_snapshot.pos[0] - 20) < 1:
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
                return True
        else:
            assert False, "Unknown command mode specified"
        return False

    def start_callback(self, data):
        if rospy.is_shutdown() or self.is_shutting_down or self.finished:
            return
        print("[RUN_COMPETITION] Start publishing commands!")
        self.publish_commands = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agile Pilot.")
    parser.add_argument("--vision_based", help="Fly vision-based", required=False, dest="vision_based", action="store_true")
    parser.add_argument('--model_type', type=str, default='LSTMNet', help='string matching model name in lstmArch.py')
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to model checkpoint')
    parser.add_argument('--des_vel', type=float, default=None, help='desired velocity for quadrotor')
    parser.add_argument("--keyboard", help="Fly state-based mode but take velocity commands from keyboard WASD", required=False, dest="keyboard", action="store_true")

    args = parser.parse_args()
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, model_type=args.model_type, model_path=args.model_path, desVel=args.des_vel, keyboard=args.keyboard)
    rospy.spin()
    agile_pilot_node.flush_data_log()
