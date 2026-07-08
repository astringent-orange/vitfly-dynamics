#!/usr/bin/python3

from utils import AgileCommandMode, AgileCommand
import cv2
import numpy as np
try:
    import torch
except ImportError:
    torch = None

import glob, os, sys, time
from os.path import join as opj

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
if torch is not None:
    from model import *

from astar_planner import StaticAStarPlanner, default_astar_path_cache_path, default_static_map_path, read_path_csv

# 3D line determined by two points (x1, y1, z1) and (x2, y2, z2)
# sphere determined by a center point (x3, y3, z3) and radius r
# quantity b^2 - 4ac < 0 then there is no intersection, where:
# b = 2*( (x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3) )
# a = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
# c = x3^2 + y3^2 + z3^2 + x1^2 + y1^2 + z1^2 - 2*(x3*x1 + y3*y1 + z3*z1) - r^2
# line is a 2-tuple of 3-tuples, obstacle is a 2-tuple of the center 3-tuple and the radius float
def check_collision(line, obstacle):
    (x1, y1, z1), (x2, y2, z2) = line
    (x3, y3, z3), r = obstacle
    b = 2 * ((x2 - x1) * (x1 - x3) + (y2 - y1) * (y1 - y3) + (z2 - z1) * (z1 - z3))
    a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    c = (
        x3**2
        + y3**2
        + z3**2
        + x1**2
        + y1**2
        + z1**2
        - 2 * (x3 * x1 + y3 * y1 + z3 * z1)
        - r**2
    )
    return b**2 - 4 * a * c >= 0


def compute_command_vision_based(state, orig_img, prev_img, desiredVel, trained_model, hidden_state):
    if torch is None:
        raise RuntimeError("Torch is required for vision-based model inference.")

    # print("Computing command vision-based!")

    """
    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]
    """

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    # command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0
    command.mode = 2
    
    ###############
    ## Load data ##
    ###############

    q = np.array([state.att[0], state.att[1], state.att[2], state.att[3]])
    
    h, w = (60, 90)
    img = cv2.resize(orig_img, (w, h))
    img2 = orig_img.copy() # used for generating debugimg
    img = torch.from_numpy(np.array(img)).float().unsqueeze(0)

    device = next(trained_model.parameters()).device

    if 'LSTMNet' in trained_model.__class__.__name__:
        if trained_model.__class__.__name__ == 'LSTMNet':
            trained_model.lstm.num_layers = 2
            trained_model.lstm.hidden_size = 395
        elif trained_model.__class__.__name__ == 'LSTMNetVIT':
            trained_model.lstm.num_layers = 3
            trained_model.lstm.hidden_size = 128
        elif trained_model.__class__.__name__ == 'UNetConvLSTMNet':
            trained_model.lstm.num_layers = 2
            trained_model.lstm.hidden_size = 200
        else:
            raise Exception ("Incorrect Model specified!!")
        if state.pos[0] < 0.5 or hidden_state is None:
            hidden_state = (torch.zeros(trained_model.lstm.num_layers, trained_model.lstm.hidden_size).float().to(device), torch.zeros(trained_model.lstm.num_layers, trained_model.lstm.hidden_size).float().to(device))
        with torch.no_grad():
            x, hidden_state = trained_model([img.view(1, 1, h, w).to(device), torch.tensor(desiredVel).view(1, 1).float().to(device), torch.tensor(q).view(1,-1).float().to(device) ,hidden_state])

    else:

        with torch.no_grad():
            x, hidden_state = trained_model([img.view(1, 1, h, w).to(device), torch.tensor(desiredVel).view(1, 1).float().to(device), torch.tensor(q).view(1,-1).float().to(device)])


    x = x.squeeze().detach().cpu().numpy()
    x[0] = np.clip(x[0], -1, 1)
    x = x/np.linalg.norm(x)
    command.velocity = x*desiredVel

    # manual speedup
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if state.pos[0] < hardcoded_ctl_threshold:
        command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)
    

    # creating debug images,
    # debugimg1 of the stabilized, cropped image with a velocity vector, and 
    # debugimg2 of the original image with the four points used for stabilization

    h, w = img2.shape
    arrow_start = (int(w/2), int(h/2))    
    arrow_end = (int(w/2-command.velocity[1]*(w/3)), int(h/2-command.velocity[2]*(h/3)))
    debugimg1 = cv2.arrowedLine( img2, arrow_start, arrow_end, (0, 0, 255), 10, )

    debugimg2 = orig_img.copy()

    return command, (debugimg1, debugimg2), hidden_state

# helper function for vectorized expert policy (method_id = 1)
def find_closest_zero_index(arr):
    center = np.array(arr.shape) // 2  # find the center point of the array
    dist_to_center = np.abs(np.indices(arr.shape) - center.reshape(-1, 1, 1)).sum(0)  # calculate distance to center for each element
    zero_indices = np.argwhere(arr == 0)  # find indices of all zero elements
    if len(zero_indices) == 0:
        return None  # if no zero elements, return None
    dist_to_zeros = dist_to_center[tuple(zero_indices.T)]  # get distances to center for zero elements
    min_dist_indices = np.argwhere(dist_to_zeros == dist_to_zeros.min()).flatten()  # find indices of zero elements with minimum distance to center
    chosen_index = np.random.choice(min_dist_indices)  # randomly choose one of the zero elements with minimum distance to center
    return tuple(zero_indices[chosen_index])  # return index tuple

def limit_norm(vec, max_norm):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return vec
    if norm > max_norm:
        return vec / norm * max_norm
    return vec


def default_planner_info():
    return {
        "lookahead_x": 0.0,
        "lookahead_y": 0.0,
        "lookahead_z": 0.0,
        "v_path_x": 0.0,
        "v_path_y": 0.0,
        "v_path_z": 0.0,
        "v_avoid_x": 0.0,
        "v_avoid_y": 0.0,
        "v_avoid_z": 0.0,
        "nearest_dyn_dist": 0.0,
        "nearest_dyn_rel_speed": 0.0,
        "ttc_min": 0.0,
        "avoidance_active": 0,
        "astar_replan_count": 0,
        "astar_success": 0,
        "astar_plan_time": 0.0,
        "nearest_obstacle_margin": 0.0,
        "nearest_static_dist": 0.0,
        "dynamic_obstacle_count": 0,
        "v_slowdown_x": 0.0,
        "v_slowdown_dynamic_x": 0.0,
        "v_slowdown_static_x": 0.0,
    }


class AStarDynamicExpert:
    def __init__(
        self,
        static_csv=None,
        resolution=0.3,
        inflation_radius=0.5,
        goal=(60.0, 0.0, 3.0),
        lookahead_distance=4.5,
        dynamic_detection_radius=8.0,
        prediction_horizon=3.0,
        dynamic_safety_radius=2.2,
        repulsion_gain=2.4,
        max_avoid_speed=1.3,
        smoothing=0.65,
        static_speed_threshold=0.5,
        track_match_distance=2.0,
        track_max_misses=3,
        max_cmd_accel=5.0,
        cmd_smoothing=0.7,
        path_cache=None,
        local_obstacle_safety_radius=2.4,
        local_static_repulsion_gain=1.8,
        local_obstacle_slow_radius=0.8,
        min_ttc_for_slowdown=1.2,
        critical_ttc=0.4,
        min_forward_speed=0.4,
        goal_gate_distance=0.3,
        goal_slowdown_distance=1.5,
    ):
        self.static_csv = static_csv or default_static_map_path()
        if not os.path.exists(self.static_csv):
            self.static_csv = opj(
                os.path.dirname(os.path.abspath(__file__)),
                "../../flightmare/flightpy/configs/vision/spheres_medium/environment_0/static_obstacles.csv",
            )
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.path_cache = path_cache or default_astar_path_cache_path()
        self.cached_path = self._load_cached_path()
        self.planner = None
        self.goal = np.asarray(goal, dtype=float)
        self.lookahead_distance = lookahead_distance
        self.dynamic_detection_radius = dynamic_detection_radius
        self.prediction_horizon = prediction_horizon
        self.dynamic_safety_radius = dynamic_safety_radius
        self.repulsion_gain = repulsion_gain
        self.max_avoid_speed = max_avoid_speed
        self.smoothing = smoothing
        self.static_speed_threshold = static_speed_threshold
        self.track_match_distance = track_match_distance
        self.track_max_misses = track_max_misses
        self.max_cmd_accel = max_cmd_accel
        self.cmd_smoothing = cmd_smoothing
        self.local_obstacle_safety_radius = local_obstacle_safety_radius
        self.local_static_repulsion_gain = local_static_repulsion_gain
        self.local_obstacle_slow_radius = local_obstacle_slow_radius
        self.min_ttc_for_slowdown = min_ttc_for_slowdown
        self.critical_ttc = critical_ttc
        self.min_forward_speed = min_forward_speed
        self.goal_gate_distance = goal_gate_distance
        self.goal_slowdown_distance = goal_slowdown_distance
        self.path = []
        self.path_index = 0
        self.replan_count = 0
        self.last_plan_time = 0.0
        self.tracks = []
        self.next_track_id = 1
        self.prev_t = None
        self.prev_v_avoid = np.zeros(3)
        self.prev_cmd = None
        self.prev_cmd_t = None
        self.avoidance_latch = False

    def _load_cached_path(self):
        if not self.path_cache or not os.path.exists(self.path_cache):
            return []
        try:
            path = read_path_csv(self.path_cache)
        except Exception as exc:
            print(f"[AStarDynamicExpert] Failed to load cached A* path {self.path_cache}: {exc}")
            return []
        if len(path) < 2:
            return []
        print(f"[AStarDynamicExpert] Loaded cached A* path with {len(path)} points from {self.path_cache}")
        return path

    def _planner(self):
        if self.planner is None:
            self.planner = StaticAStarPlanner(
                self.static_csv,
                resolution=self.resolution,
                inflation_radius=self.inflation_radius,
            )
        return self.planner

    def reset_path(self):
        self.path = []
        self.path_index = 0
        self.tracks = []
        self.prev_t = None
        self.prev_v_avoid = np.zeros(3)
        self.prev_cmd = None
        self.prev_cmd_t = None
        self.avoidance_latch = False

    def _make_command(self, state, velocity):
        command = AgileCommand(AgileCommandMode.LINVEL)
        command.t = state.t
        command.yawrate = 0.0
        command.velocity = [float(v) for v in velocity]
        return command

    def _ensure_path(self, position):
        if len(self.path) == 0:
            start_time = time.time()
            if self.cached_path:
                self.path = [np.asarray(point, dtype=float) for point in self.cached_path]
                nearest_idx = int(np.argmin([np.linalg.norm(point - position) for point in self.path]))
                self.path_index = min(nearest_idx, len(self.path) - 1)
                self.last_plan_time = 0.0
            else:
                self.path = self._planner().plan(position, self.goal)
                self.path_index = 0
                self.last_plan_time = time.time() - start_time
            self.replan_count += 1
        return len(self.path) > 0

    def _relative_obstacle_measurements(self, obstacles):
        rel = []
        if obstacles is None:
            return rel
        for obst in obstacles.obstacles:
            pos = np.array([obst.position.x, obst.position.y, obst.position.z], dtype=float)
            if not np.all(np.isfinite(pos)):
                continue
            dist = np.linalg.norm(pos)
            if dist <= self.dynamic_detection_radius and pos[0] > -1.0:
                rel.append({"pos": pos, "scale": float(obst.scale)})
        return rel

    def _track_obstacles(self, measurements, t, drone_velocity, require_motion=True):
        if self.prev_t is None or t <= self.prev_t:
            self.tracks = []
            for meas in measurements:
                self.tracks.append(
                    {
                        "id": self.next_track_id,
                        "pos": meas["pos"].copy(),
                        "vel": np.zeros(3),
                        "scale": meas["scale"],
                        "misses": 0,
                        "age": 1,
                        "world_speed": 0.0,
                    }
                )
                self.next_track_id += 1
            self.prev_t = t
            return []

        dt = max(t - self.prev_t, 1e-3)
        unused_tracks = set(range(len(self.tracks)))
        updated_tracks = []

        for meas in measurements:
            matched_idx = None
            matched_dist = self.track_match_distance
            for idx in list(unused_tracks):
                dist = np.linalg.norm(meas["pos"] - self.tracks[idx]["pos"])
                if dist < matched_dist:
                    matched_idx = idx
                    matched_dist = dist

            if matched_idx is None:
                updated_tracks.append(
                    {
                        "id": self.next_track_id,
                        "pos": meas["pos"].copy(),
                        "vel": np.zeros(3),
                        "scale": meas["scale"],
                        "misses": 0,
                        "age": 1,
                        "world_speed": 0.0,
                    }
                )
                self.next_track_id += 1
                continue

            track = self.tracks[matched_idx]
            unused_tracks.remove(matched_idx)
            measured_vel = (meas["pos"] - track["pos"]) / dt
            rel_vel = 0.5 * track["vel"] + 0.5 * measured_vel if track["age"] > 1 else measured_vel
            world_vel = rel_vel + drone_velocity
            updated_tracks.append(
                {
                    "id": track["id"],
                    "pos": meas["pos"].copy(),
                    "vel": rel_vel,
                    "scale": meas["scale"],
                    "misses": 0,
                    "age": track["age"] + 1,
                    "world_speed": float(np.linalg.norm(world_vel)),
                }
            )

        for idx in unused_tracks:
            track = self.tracks[idx].copy()
            track["misses"] += 1
            if track["misses"] <= self.track_max_misses:
                updated_tracks.append(track)

        self.tracks = updated_tracks
        self.prev_t = t
        dynamic_tracks = [
            track
            for track in self.tracks
            if track["age"] >= 2
            and track["misses"] == 0
            and (not require_motion or track["world_speed"] >= self.static_speed_threshold)
        ]
        return dynamic_tracks

    def _dynamic_avoidance(self, rel_obstacles):
        v_avoid = np.zeros(3)
        slowdown_x = 0.0
        nearest_dist = float("inf")
        nearest_rel_speed = 0.0
        ttc_min = float("inf")
        raw_active = False

        for obs in rel_obstacles:
            p_rel = obs["pos"]
            v_rel = obs["vel"]
            dist = np.linalg.norm(p_rel)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_rel_speed = np.linalg.norm(v_rel)

            safety = self.dynamic_safety_radius + 0.5 * obs["scale"]
            v_rel_norm2 = float(np.dot(v_rel, v_rel))
            if v_rel_norm2 > 1e-6:
                ttc = float(np.clip(-np.dot(p_rel, v_rel) / v_rel_norm2, 0.0, self.prediction_horizon))
                closest = p_rel + v_rel * ttc
                closest_dist = np.linalg.norm(closest)
            else:
                ttc = float("inf")
                closest = p_rel
                closest_dist = dist

            dangerous = (ttc <= self.prediction_horizon and closest_dist < safety) or dist < safety
            if dangerous:
                raw_active = True
                ttc_min = min(ttc_min, ttc if np.isfinite(ttc) else self.prediction_horizon)
                if np.isfinite(ttc) and ttc < self.min_ttc_for_slowdown:
                    slowdown_x = max(slowdown_x, (self.min_ttc_for_slowdown - ttc) / self.min_ttc_for_slowdown)
                if closest_dist < safety:
                    slowdown_x = max(slowdown_x, (safety - closest_dist) / safety)
                away = -closest / max(np.linalg.norm(closest), 1e-6)
                away[0] = 0.0
                if np.linalg.norm(away) < 1e-6:
                    away = np.array([0.0, 1.0, 0.0])
                else:
                    away = away / np.linalg.norm(away)
                strength = self.repulsion_gain * max(0.0, safety - closest_dist) / safety
                if np.isfinite(ttc):
                    strength *= 1.0 + (self.prediction_horizon - ttc) / self.prediction_horizon
                v_avoid += away * strength

        if raw_active:
            self.avoidance_latch = True
        elif self.avoidance_latch:
            self.avoidance_latch = np.linalg.norm(self.prev_v_avoid) > 0.2

        if not self.avoidance_latch:
            v_avoid = np.zeros(3)

        v_avoid = limit_norm(v_avoid, self.max_avoid_speed)
        v_avoid = self.smoothing * self.prev_v_avoid + (1.0 - self.smoothing) * v_avoid
        if not self.avoidance_latch and np.linalg.norm(v_avoid) < 0.05:
            v_avoid = np.zeros(3)
        self.prev_v_avoid = v_avoid

        if not np.isfinite(nearest_dist):
            nearest_dist = 0.0
        if not np.isfinite(ttc_min):
            ttc_min = 0.0

        return v_avoid, {
            "nearest_dyn_dist": nearest_dist,
            "nearest_dyn_rel_speed": nearest_rel_speed,
            "ttc_min": ttc_min,
            "avoidance_active": int(self.avoidance_latch),
            "v_slowdown_dynamic_x": float(np.clip(slowdown_x, 0.0, 1.0)),
        }

    def _local_obstacle_fallback(self, all_measurements, dynamic_measurements):
        if not all_measurements:
            return np.zeros(3), 999.0, 0.0, 0.0

        dynamic_positions = [dyn["pos"] for dyn in dynamic_measurements]
        v_local = np.zeros(3)
        nearest_margin = float("inf")
        nearest_static_dist = float("inf")
        static_slowdown = 0.0

        for meas in all_measurements:
            pos = meas["pos"]
            dist = np.linalg.norm(pos)
            margin = dist - meas["scale"]
            nearest_margin = min(nearest_margin, margin)
            is_dynamic = any(np.linalg.norm(pos - dyn_pos) < 0.75 for dyn_pos in dynamic_positions)
            if not is_dynamic:
                nearest_static_dist = min(nearest_static_dist, dist)

            if margin < self.local_obstacle_slow_radius:
                static_slowdown = max(
                    static_slowdown,
                    (self.local_obstacle_slow_radius - margin) / max(self.local_obstacle_slow_radius, 1e-6),
                )

            safety = self.local_obstacle_safety_radius + meas["scale"]
            if dist < safety and pos[0] > -0.5:
                away = -pos / max(dist, 1e-6)
                away[0] = min(0.0, away[0])
                if np.linalg.norm(away[1:]) < 1e-6:
                    away[1] = 1.0
                strength = self.local_static_repulsion_gain * (safety - dist) / safety
                v_local += away * strength

        if not np.isfinite(nearest_margin):
            nearest_margin = 999.0
        if not np.isfinite(nearest_static_dist):
            nearest_static_dist = 0.0
        return limit_norm(v_local, self.max_avoid_speed), nearest_margin, nearest_static_dist, float(np.clip(static_slowdown, 0.0, 1.0))

    def _lookahead_on_path(self, position):
        if not self.path:
            return np.asarray(position, dtype=float)

        while self.path_index < len(self.path) - 1:
            if np.linalg.norm(np.asarray(self.path[self.path_index]) - position) > 1.0:
                break
            self.path_index += 1

        lookahead = np.asarray(self.path[-1], dtype=float)
        for idx in range(self.path_index, len(self.path)):
            point = np.asarray(self.path[idx], dtype=float)
            if point[0] < position[0] - 0.2:
                self.path_index = min(idx + 1, len(self.path) - 1)
                continue
            if np.linalg.norm(point - position) >= self.lookahead_distance:
                lookahead = point
                self.path_index = idx
                break
        return lookahead

    def _smooth_command(self, raw_cmd, t, desiredVel):
        raw_cmd = limit_norm(raw_cmd, desiredVel)
        if self.prev_cmd is None or self.prev_cmd_t is None or t <= self.prev_cmd_t:
            self.prev_cmd = raw_cmd.copy()
            self.prev_cmd_t = t
            return raw_cmd

        dt = max(t - self.prev_cmd_t, 1e-3)
        max_delta = self.max_cmd_accel * dt
        delta = limit_norm(raw_cmd - self.prev_cmd, max_delta)
        accel_limited = self.prev_cmd + delta
        smoothed = self.cmd_smoothing * self.prev_cmd + (1.0 - self.cmd_smoothing) * accel_limited
        smoothed = limit_norm(smoothed, desiredVel)
        self.prev_cmd = smoothed.copy()
        self.prev_cmd_t = t
        return smoothed

    def compute_command(self, state, obstacles, desiredVel, dynamic_obstacles=None):
        pos = np.asarray(state.pos, dtype=float)
        if pos[0] >= self.goal[0]:
            info = default_planner_info()
            info.update(
                {
                    "lookahead_x": self.goal[0],
                    "lookahead_y": self.goal[1],
                    "lookahead_z": self.goal[2],
                    "astar_success": 1,
                }
            )
            return self._make_command(state, np.zeros(3)), info
        if pos[0] < 0.5:
            self.reset_path()

        astar_success = self._ensure_path(pos)
        if astar_success:
            lookahead = self._lookahead_on_path(pos)
            path_vec = lookahead - pos
            v_path = limit_norm(path_vec, desiredVel)
            if np.linalg.norm(v_path) > 1e-6:
                v_path = v_path / np.linalg.norm(v_path) * desiredVel
        else:
            lookahead = np.array([pos[0] + 4.0, 0.0, 3.0])
            v_path = np.array([desiredVel, 0.0, 0.0])

        drone_velocity = np.asarray(state.vel, dtype=float)
        all_measurements = self._relative_obstacle_measurements(obstacles)
        dynamic_measurements = self._relative_obstacle_measurements(dynamic_obstacles) if dynamic_obstacles is not None else all_measurements
        if dynamic_obstacles is not None:
            rel_obstacles = self._track_obstacles(dynamic_measurements, state.t, drone_velocity, require_motion=False)
        else:
            rel_obstacles = self._track_obstacles(dynamic_measurements, state.t, drone_velocity, require_motion=True)
        v_avoid, avoid_info = self._dynamic_avoidance(rel_obstacles)
        v_local, nearest_margin, nearest_static_dist, static_slowdown = self._local_obstacle_fallback(all_measurements, dynamic_measurements)
        dynamic_slowdown = float(np.clip(avoid_info.get("v_slowdown_dynamic_x", 0.0), 0.0, 1.0))
        slowdown = max(dynamic_slowdown, static_slowdown)
        v_path[0] *= max(0.15, 1.0 - 0.85 * np.clip(slowdown, 0.0, 1.0))
        v_cmd = v_path + v_avoid + v_local

        if pos[2] < 2.0:
            v_cmd[2] = max(v_cmd[2], (2.0 - pos[2]) * 2.0)
        if pos[0] < 2.0:
            v_cmd[0] = max(1.0, (pos[0] / 2.0) * desiredVel)
        remaining_to_goal = self.goal[0] - pos[0]
        ttc_min = float(avoid_info.get("ttc_min", 0.0))
        critical_stop = nearest_margin < 0.0 or (ttc_min > 0.0 and ttc_min < self.critical_ttc)
        if remaining_to_goal > self.goal_gate_distance:
            min_forward = 0.0 if critical_stop else self.min_forward_speed
            v_cmd[0] = max(min_forward, v_cmd[0])
        else:
            v_cmd[0] = max(0.0, v_cmd[0])
        if 0.0 < remaining_to_goal < self.goal_slowdown_distance:
            max_goal_speed = desiredVel * max(0.2, remaining_to_goal / self.goal_slowdown_distance)
            v_cmd[0] = min(v_cmd[0], max_goal_speed)

        v_cmd = self._smooth_command(v_cmd, state.t, desiredVel)
        if remaining_to_goal > self.goal_gate_distance:
            min_forward = 0.0 if critical_stop else self.min_forward_speed
            v_cmd[0] = max(min_forward, v_cmd[0])
        else:
            v_cmd[0] = max(0.0, v_cmd[0])
        if 0.0 < remaining_to_goal < self.goal_slowdown_distance:
            max_goal_speed = desiredVel * max(0.2, remaining_to_goal / self.goal_slowdown_distance)
            v_cmd[0] = min(v_cmd[0], max_goal_speed)

        info = default_planner_info()
        info.update(
            {
                "lookahead_x": lookahead[0],
                "lookahead_y": lookahead[1],
                "lookahead_z": lookahead[2],
                "v_path_x": v_path[0],
                "v_path_y": v_path[1],
                "v_path_z": v_path[2],
                "v_avoid_x": v_avoid[0],
                "v_avoid_y": v_avoid[1],
                "v_avoid_z": v_avoid[2],
                "astar_replan_count": self.replan_count,
                "astar_success": int(astar_success),
                "astar_plan_time": self.last_plan_time,
                "nearest_obstacle_margin": nearest_margin,
                "nearest_static_dist": nearest_static_dist,
                "dynamic_obstacle_count": len(dynamic_measurements),
                "v_slowdown_x": slowdown,
                "v_slowdown_dynamic_x": dynamic_slowdown,
                "v_slowdown_static_x": static_slowdown,
            }
        )
        info.update(avoid_info)
        return self._make_command(state, v_cmd), info


def compute_command_state_based(state, obstacles, desiredVel, rl_policy=None, keyboard=False, keyboard_input='', expert=None, return_info=False, dynamic_obstacles=None):
    if expert is not None and not keyboard:
        command, planner_info = expert.compute_command(state, obstacles, desiredVel, dynamic_obstacles=dynamic_obstacles)
        return (command, planner_info) if return_info else command

    # print("Computing command based on obstacle information!")
    # print("Obstacles: ", obstacles)

    """
    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 10.0
    command.bodyrates = [0.0, 0.0, 0.0]
    """

    # LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.yawrate = 0.0

    obst_dist_threshold = 8
    obst_inflate_factor = 0.6 #0.4#0.6
    method_id = 1 # 0 = old spiral method, 1 = new re-factored, 2 = constant
    if keyboard:
        import select
        method_id = 3

    # calculate an obstacle-free waypoint
    x_displacement = 8 #5
    grid_center_offset = 8
    grid_displacement = 0.5
    y_vals = np.arange(-grid_center_offset, grid_center_offset + grid_displacement, grid_displacement)
    num_wpts = y_vals.size

    start = time.time()

    # old expert
    if method_id == 0:

        wpts_2d = np.zeros((num_wpts, num_wpts, 2))
        for xi, x in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
            for yi, y in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
                wpts_2d[yi, xi] = [x, y]

        # the first layer of wpts_2d is actually the world y axis, the second is z axis
        # the third, the x axis, should all be +5m forward
        x_slice = x_displacement * np.ones((num_wpts, num_wpts))
        wpts_2d = np.concatenate((x_slice[:, :, None], wpts_2d), axis=2)

        # try spiraling outward again but just using bounds instead, and selecting blocks
        idx_midpt = num_wpts // 2
        curr_x = idx_midpt
        curr_y = idx_midpt
        x_bound = 1
        y_bound = -1
        wpt_idxs_2d = []
        count = 0
        while curr_x < num_wpts:
            if count % 4 == 0:
                x_bound = count / 4 + 1
            if (count - 1) % 4 == 0:
                y_bound = -((count - 1) / 4 + 1)

            if not count % 2:  # x-dir vector
                xvals = np.arange(
                    curr_x, idx_midpt + x_bound, -1 if x_bound < 0 else 1, dtype=int
                )
                wpt_idxs_2d += [
                    pair for pair in zip(np.repeat(int(curr_y), xvals.size), xvals)
                ]
                curr_x = idx_midpt + x_bound
                x_bound *= -1
            else:  # y-dir vector
                yvals = np.arange(
                    curr_y, idx_midpt + y_bound, -1 if y_bound < 0 else 1, dtype=int
                )
                wpt_idxs_2d += [
                    pair for pair in zip(yvals, np.repeat(int(curr_x), yvals.size))
                ]
                curr_y = idx_midpt + y_bound
                y_bound *= -1

            count += 1

        # iterate through waypoints, spiraling outwards from center
        for wpt_idx in wpt_idxs_2d:
            found_valid_pt = True
            # check if the current wpt is valid for all obstacles ahead of our current position
            for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                if check_collision(((0, 0, 0), (wpts_2d[wpt_idx])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor)):
                    found_valid_pt = False
                    break
            if found_valid_pt:
                break

        # CHECK AGAIN WITH OBSTACLE SCALE REDUCED TO .17
        if not found_valid_pt:
            print("[EXPERT] Didn't find a feasible path, Searching again with less inflation!")
            for wpt_idx in wpt_idxs_2d:
                found_valid_pt = True
                # check if the current wpt is valid for all obstacles ahead of our current position
                for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                    if check_collision(((0, 0, 0), (wpts_2d[wpt_idx])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+0.17)):
                        found_valid_pt = False
                        break
                if found_valid_pt:
                    break
        
        # simplest controller: waypoint --PID--> linear velocity command
        yvel = 1.25 * (wpts_2d[wpt_idx][1])
        # x_scale_down_factor = (grid_center_offset - np.abs(yvel))/grid_center_offset
        xvel = max(desiredVel, 1 * (wpts_2d[wpt_idx][0]))
        zvel = 1.25 * wpts_2d[wpt_idx][2]

    # new expert
    elif method_id == 1:

        wpts_2d = np.zeros((num_wpts, num_wpts, 3))
        collisions = np.zeros((num_wpts, num_wpts))
        for xi, x in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
            for yi, y in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
                wpts_2d[yi, xi] = [x_displacement, x, y]
                for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                    # print(f'wpt: {wpts_2d[yi, xi]} \t obst: {obst.position.x, obst.position.y, obst.position.z, obst.scale+obst_inflate_factor}')
                    if check_collision(((0, 0, 0), (wpts_2d[yi, xi])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor)):
                        collisions[yi, xi] = 1
                        break


        if collisions.sum() == collisions.size:
            print(f'[EXPERT] No collision-free path found')
            xvel = 0.5
            yvel = 0
            zvel = 0.25            
        else:
            wpt_idx = find_closest_zero_index(collisions)
            wpt = wpts_2d[wpt_idx[0], wpt_idx[1]]

            # make the desired velocity vector of magnitude desiredVel
            wpt = (wpt / np.linalg.norm(wpt)) * desiredVel
            xvel = wpt[0]
            yvel = wpt[1]
            zvel = wpt[2]

    # just fly forward
    elif method_id == 2:

        xvel, yvel, zvel = (4.0, 0., 0.)

    elif method_id == 3:

        xvel, yvel, zvel = (2., 0., 0.)

        # print(f'[EXPERT] Keyboard input: {keyboard_input}')

        # Check if there is any keypress
        if keyboard_input == 'w':
            zvel = 1.0
        elif keyboard_input == 's':
            zvel = -1.0
        elif keyboard_input == 'a':
            yvel = 1.0
        elif keyboard_input == 'd':
            yvel = -1.0

        # norm the command vector up to desiredVel
        scaler = desiredVel/np.linalg.norm([xvel, yvel, zvel])
        xvel, yvel, zvel = (xvel*scaler, yvel*scaler, zvel*scaler)


    if time.time() - int(time.time()) < 0.1: # print this as infrequently as possible
        print(f'[EXPERT] Expert method {method_id} took {time.time() - start:.3f} seconds')

    command.velocity = [xvel, yvel, zvel]

    # recover altitude if too low
    if state.pos[2] < 2:
        command.velocity[2] = (2 - state.pos[2]) * 2

    # manual speedup
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if state.pos[0] < hardcoded_ctl_threshold:
        command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)
        

    ################################################
    # !!! End !!!
    ###############################################

    if return_info:
        return command, default_planner_info()
    return command
