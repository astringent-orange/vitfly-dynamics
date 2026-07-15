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
import yaml

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
if torch is not None:
    from model import *

from astar_planner import DEFAULT_STATIC_INFLATION, StaticAStarPlanner, default_astar_path_cache_path, default_static_map_path, read_path_csv
from candidate_speed_planner import CandidateSpeedPlanner, CandidateYieldPolicy, PathSpeedController, PolylinePath
from dynamic_obstacle_predictor import DynamicObstacleTrajectoryPredictor
from frame_stack import build_frame_stack, MODEL_FRAME_OFFSETS


DEFAULT_EXPERT_DYNAMICS = {
    "control_delay": 0.25,
    "max_accel": 3.0,
    "max_brake_decel": 1.5,
    "reverse_drift_distance": 0.4,
    "cross_track_decay_time": 0.8,
}


def load_expert_dynamics(config_path=None):
    config_path = config_path or opj(os.path.dirname(os.path.abspath(__file__)), "expert_dynamics.yaml")
    values = DEFAULT_EXPERT_DYNAMICS.copy()
    try:
        with open(config_path, "r") as stream:
            loaded = yaml.safe_load(stream) or {}
        for key in values:
            if key in loaded:
                values[key] = float(loaded[key])
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"[AStarDynamicExpert] Using fallback dynamics values: {exc}")
    if values["max_accel"] <= 0.0 or values["max_brake_decel"] <= 0.0:
        raise ValueError("expert dynamics acceleration limits must be positive")
    if values["control_delay"] < 0.0 or values["reverse_drift_distance"] < 0.0:
        raise ValueError("expert dynamics delay and reverse-drift buffer must be nonnegative")
    if values["cross_track_decay_time"] <= 0.0:
        raise ValueError("expert dynamics cross-track decay time must be positive")
    return values

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


def compute_command_vision_based(state, orig_img, frame_history, desiredVel, trained_model, hidden_state, frame_offset=0):
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
    img_stack = build_frame_stack(frame_history, orig_img, frame_offset)
    if img_stack is None:
        return None, (orig_img.copy(), orig_img.copy()), hidden_state
    img = img_stack[-1]
    img2 = orig_img.copy() # used for generating debugimg
    img = torch.from_numpy(img_stack).float()

    device = next(trained_model.parameters()).device

    if trained_model.__class__.__name__ not in MODEL_FRAME_OFFSETS:
        raise ValueError(f'Unsupported inference model {trained_model.__class__.__name__}')
    if MODEL_FRAME_OFFSETS[trained_model.__class__.__name__] != frame_offset:
        raise ValueError(
            f'Model {trained_model.__class__.__name__} requires frame_offset '
            f'{MODEL_FRAME_OFFSETS[trained_model.__class__.__name__]}, got {frame_offset}'
        )
    if state.pos[0] < 0.5:
        hidden_state = None
    model_input = [
        img.view(1, img.shape[0], h, w).to(device),
        torch.tensor(desiredVel).view(1, 1).float().to(device),
        torch.tensor(q).view(1, -1).float().to(device),
    ]
    if hidden_state is not None:
        model_input.append(hidden_state)
    with torch.no_grad():
        x, hidden_state = trained_model(model_input)


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
        "configured_dynamic_obstacle_count": 0,
        "candidate_selected_speed": 0.0,
        "candidate_safe_count": 0,
        "candidate_min_clearance": 0.0,
        "candidate_emergency_stop": 0,
        "candidate_prediction_horizon": 0.0,
        "candidate_raw_selected_speed": 0.0,
        "candidate_yield_active": 0,
        "candidate_applied_speed": 0.0,
        "path_cross_track_error": 0.0,
        "path_turn_angle_deg": 0.0,
        "path_speed_ceiling": 0.0,
        "candidate_control_delay": 0.0,
        "candidate_brake_decel": 0.0,
        "candidate_reverse_drift_buffer": 0.0,
        "candidate_initial_path_speed": 0.0,
        "candidate_predicted_stop_distance": 0.0,
    }


class AStarDynamicExpert:
    def __init__(
        self,
        static_csv=None,
        resolution=0.3,
        inflation_radius=DEFAULT_STATIC_INFLATION,
        goal=(60.0, 0.0, 3.0),
        lookahead_distance=1.2,
        static_speed_threshold=0.5,
        track_match_distance=2.0,
        track_max_misses=3,
        path_cache=None,
        candidate_prediction_horizon=2.2,
        candidate_prediction_dt=0.1,
        candidate_prediction_accel=None,
        dynamics_config_path=None,
        candidate_safety_margin=1.0,
        candidate_step=0.1,
        turn_preview_distance=2.5,
        medium_turn_angle_deg=15.0,
        sharp_turn_angle_deg=30.0,
        medium_turn_speed=4.0,
        sharp_turn_speed=3.0,
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
        self.static_speed_threshold = static_speed_threshold
        self.track_match_distance = track_match_distance
        self.track_max_misses = track_max_misses
        self.goal_gate_distance = goal_gate_distance
        self.goal_slowdown_distance = goal_slowdown_distance
        self.turn_preview_distance = float(turn_preview_distance)
        self.medium_turn_angle_deg = float(medium_turn_angle_deg)
        self.sharp_turn_angle_deg = float(sharp_turn_angle_deg)
        self.medium_turn_speed = float(medium_turn_speed)
        self.sharp_turn_speed = float(sharp_turn_speed)
        self.expert_dynamics = load_expert_dynamics(dynamics_config_path)
        if candidate_prediction_accel is not None:
            self.expert_dynamics["max_accel"] = float(candidate_prediction_accel)
        self.candidate_planner = CandidateSpeedPlanner(
            horizon=candidate_prediction_horizon,
            prediction_dt=candidate_prediction_dt,
            max_accel=self.expert_dynamics["max_accel"],
            max_brake_decel=self.expert_dynamics["max_brake_decel"],
            control_delay=self.expert_dynamics["control_delay"],
            reverse_drift_distance=self.expert_dynamics["reverse_drift_distance"],
            cross_track_decay_time=self.expert_dynamics["cross_track_decay_time"],
            safety_margin=candidate_safety_margin,
            candidate_step=candidate_step,
        )
        self.speed_controller = PathSpeedController(self.expert_dynamics["max_accel"])
        self.yield_policy = CandidateYieldPolicy(release_frames=5)
        self.path = []
        self.path_polyline = None
        self.path_index = 0
        self.replan_count = 0
        self.last_plan_time = 0.0
        self.tracks = []
        self.next_track_id = 1
        self.prev_t = None
        self.last_compute_t = None
        self.last_command_velocity = None
        self.last_planner_info = None
        self.dynamic_predictor = self._make_dynamic_predictor()
        print("[AStarDynamicExpert] Candidate-speed dynamic planner enabled")

    def _make_dynamic_predictor(self):
        try:
            env_dir = os.path.dirname(os.path.abspath(self.static_csv))
            predictor = DynamicObstacleTrajectoryPredictor(env_dir)
            return predictor if predictor.loaded else None
        except Exception as exc:
            print(f"[AStarDynamicExpert] Dynamic trajectory predictor disabled: {exc}")
            return None

    def _path_speed_ceiling(self, path, pos, desired_speed):
        if path is None:
            return float(desired_speed), 0.0
        turn_angle = path.turn_angle_ahead(pos, self.turn_preview_distance)
        if turn_angle > self.sharp_turn_angle_deg:
            ceiling = self.sharp_turn_speed
        elif turn_angle > self.medium_turn_angle_deg:
            ceiling = self.medium_turn_speed
        else:
            ceiling = float(desired_speed)
        return min(float(desired_speed), ceiling), turn_angle

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
        self.path_polyline = None
        self.path_index = 0
        self.tracks = []
        self.prev_t = None
        self.last_compute_t = None
        self.last_command_velocity = None
        self.last_planner_info = None
        self.speed_controller.reset()
        self.yield_policy.reset()
        if self.dynamic_predictor is not None:
            self.dynamic_predictor.reset_calibration()

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
            if self.path:
                self.path_polyline = PolylinePath(self.path)
        return len(self.path) > 0

    def _relative_obstacle_measurements(self, obstacles, max_distance="dynamic", forward_only=True):
        rel = []
        if obstacles is None:
            return rel
        max_distance = None if max_distance == "dynamic" else max_distance
        for obst in obstacles.obstacles:
            pos = np.array([obst.position.x, obst.position.y, obst.position.z], dtype=float)
            if not np.all(np.isfinite(pos)):
                continue
            dist = np.linalg.norm(pos)
            if max_distance is not None and dist > max_distance:
                continue
            if forward_only and pos[0] <= -1.0:
                continue
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

    def _scene_dynamic_obstacles(self, state, dynamic_obstacles, drone_velocity):
        if self.dynamic_predictor is None:
            return []
        return self.dynamic_predictor.relative_measurements(
            state,
            dynamic_obstacles,
            drone_velocity,
            max_distance=None,
            forward_only=False,
        )

    def _configured_dynamic_obstacle_count(self):
        if self.dynamic_predictor is None:
            return 0
        return self.dynamic_predictor.configured_count()

    def _dynamic_diagnostics(self, rel_obstacles):
        nearest_dist = float("inf")
        nearest_rel_speed = 0.0
        ttc_min = float("inf")

        for obs in rel_obstacles:
            p_rel = obs["pos"]
            v_rel = obs["vel"]
            dist = np.linalg.norm(p_rel)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_rel_speed = np.linalg.norm(v_rel)

            v_rel_norm2 = float(np.dot(v_rel, v_rel))
            if v_rel_norm2 > 1e-6:
                ttc = float(-np.dot(p_rel, v_rel) / v_rel_norm2)
                if 0.0 < ttc <= self.candidate_planner.horizon:
                    ttc_min = min(ttc_min, ttc)

        if not np.isfinite(nearest_dist):
            nearest_dist = 0.0
        if not np.isfinite(ttc_min):
            ttc_min = 0.0

        return {
            "nearest_dyn_dist": nearest_dist,
            "nearest_dyn_rel_speed": nearest_rel_speed,
            "ttc_min": ttc_min,
        }

    def _obstacle_diagnostics(self, all_measurements, dynamic_measurements):
        if not all_measurements:
            return 999.0, 0.0

        dynamic_positions = [dyn["pos"] for dyn in dynamic_measurements]
        nearest_margin = float("inf")
        nearest_static_dist = float("inf")

        for meas in all_measurements:
            pos = meas["pos"]
            dist = np.linalg.norm(pos)
            margin = dist - meas["scale"]
            nearest_margin = min(nearest_margin, margin)
            is_dynamic = any(np.linalg.norm(pos - dyn_pos) < 0.75 for dyn_pos in dynamic_positions)
            if not is_dynamic:
                nearest_static_dist = min(nearest_static_dist, dist)

        if not np.isfinite(nearest_margin):
            nearest_margin = 999.0
        if not np.isfinite(nearest_static_dist):
            nearest_static_dist = 0.0
        return nearest_margin, nearest_static_dist

    def _linear_obstacle_predictions(self, state, rel_obstacles):
        drone_pos = np.asarray(state.pos, dtype=float)
        drone_velocity = np.asarray(state.vel, dtype=float)
        offsets = self.candidate_planner.time_offsets
        predictions = []
        for idx, obstacle in enumerate(rel_obstacles):
            world_pos = drone_pos + obstacle["pos"]
            world_velocity = drone_velocity + obstacle["vel"]
            positions = world_pos[None, :] + offsets[:, None] * world_velocity[None, :]
            predictions.append(
                {
                    "index": idx,
                    "name": f"tracked_{idx}",
                    "radius": obstacle["scale"],
                    "positions": positions,
                }
            )
        return predictions

    def _dynamic_predictions(self, state, dynamic_obstacles, rel_obstacles):
        if self.dynamic_predictor is not None:
            predictions = self.dynamic_predictor.world_predictions(
                state,
                dynamic_obstacles,
                self.candidate_planner.time_offsets,
            )
            if predictions:
                return predictions
        return self._linear_obstacle_predictions(state, rel_obstacles)

    def _safe_candidate_at_or_below(self, candidate_result, upper_bound, fallback):
        safe_speeds = [
            float(speed)
            for speed in candidate_result.safe_speeds
            if speed <= upper_bound + 1e-6
        ]
        if safe_speeds:
            return max(safe_speeds)
        return float(fallback)

    def compute_command(self, state, obstacles, desiredVel, dynamic_obstacles=None):
        pos = np.asarray(state.pos, dtype=float)
        state_t = float(state.t)
        if self.last_compute_t is not None and abs(state_t - self.last_compute_t) <= 1e-9:
            return (
                self._make_command(state, self.last_command_velocity),
                self.last_planner_info.copy(),
            )
        if self.last_compute_t is not None and state_t < self.last_compute_t:
            self.reset_path()
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
        astar_success = self._ensure_path(pos)
        if astar_success:
            path_reference = self.path_polyline.reference_from(pos, self.lookahead_distance)
            lookahead = path_reference["reference"]
            path_vec = lookahead - pos
            cross_track_error = path_reference["cross_track_error"]
        else:
            lookahead = np.array([pos[0] + 4.0, 0.0, 3.0])
            path_vec = lookahead - pos
            cross_track_error = 0.0

        drone_velocity = np.asarray(state.vel, dtype=float)
        all_measurements = self._relative_obstacle_measurements(obstacles, max_distance=None, forward_only=False)
        dynamic_measurements = (
            self._relative_obstacle_measurements(dynamic_obstacles, max_distance=None, forward_only=False)
            if dynamic_obstacles is not None
            else self._relative_obstacle_measurements(obstacles, max_distance=None, forward_only=False)
        )
        scene_rel_obstacles = self._scene_dynamic_obstacles(state, dynamic_obstacles, drone_velocity)
        if scene_rel_obstacles:
            rel_obstacles = scene_rel_obstacles
        elif dynamic_obstacles is not None:
            rel_obstacles = self._track_obstacles(dynamic_measurements, state.t, drone_velocity, require_motion=False)
        else:
            rel_obstacles = self._track_obstacles(dynamic_measurements, state.t, drone_velocity, require_motion=True)
        dynamic_info = self._dynamic_diagnostics(rel_obstacles)
        nearest_margin, nearest_static_dist = self._obstacle_diagnostics(all_measurements, dynamic_measurements)
        obstacle_predictions = self._dynamic_predictions(state, dynamic_obstacles, rel_obstacles)
        candidate_path = self.path_polyline if self.path_polyline is not None else PolylinePath([pos, self.goal])
        path_speed_ceiling, path_turn_angle_deg = self._path_speed_ceiling(
            candidate_path,
            pos,
            desiredVel,
        )
        candidate_result = self.candidate_planner.select_speed(
            candidate_path,
            pos,
            drone_velocity,
            path_speed_ceiling,
            obstacle_predictions,
        )

        remaining_to_goal = self.goal[0] - pos[0]
        selected_speed, yield_active = self.yield_policy.apply(candidate_result, path_speed_ceiling)
        candidate_clearance = candidate_result.clearance_for(selected_speed)
        if cross_track_error > 0.4:
            recovery_ratio = float(np.clip((0.8 - cross_track_error) / 0.4, 0.0, 1.0))
            recovery_cap = max(1.0, desiredVel * recovery_ratio)
            if selected_speed > recovery_cap:
                selected_speed = self._safe_candidate_at_or_below(
                    candidate_result,
                    recovery_cap,
                    selected_speed,
                )
        if 0.0 < remaining_to_goal < self.goal_slowdown_distance:
            max_goal_speed = desiredVel * max(0.2, remaining_to_goal / self.goal_slowdown_distance)
            if selected_speed > max_goal_speed:
                selected_speed = self._safe_candidate_at_or_below(
                    candidate_result,
                    max_goal_speed,
                    selected_speed,
                )

        if np.linalg.norm(path_vec) > 1e-6:
            path_direction = path_vec / np.linalg.norm(path_vec)
        elif astar_success:
            path_direction = path_reference["reference_tangent"]
        else:
            path_direction = np.array([1.0, 0.0, 0.0])
        v_path = path_direction * selected_speed
        v_cmd, applied_speed = self.speed_controller.apply(
            selected_speed,
            path_direction,
            state.t,
            drone_velocity,
            speed_limit=desiredVel,
        )

        dynamic_slowdown = 0.0 if desiredVel <= 1e-6 else 1.0 - selected_speed / desiredVel
        dynamic_slowdown = float(np.clip(dynamic_slowdown, 0.0, 1.0))

        info = default_planner_info()
        info.update(
            {
                "lookahead_x": lookahead[0],
                "lookahead_y": lookahead[1],
                "lookahead_z": lookahead[2],
                "v_path_x": v_path[0],
                "v_path_y": v_path[1],
                "v_path_z": v_path[2],
                "astar_replan_count": self.replan_count,
                "astar_success": int(astar_success),
                "astar_plan_time": self.last_plan_time,
                "nearest_obstacle_margin": nearest_margin,
                "nearest_static_dist": nearest_static_dist,
                "dynamic_obstacle_count": len(dynamic_measurements),
                "v_slowdown_x": dynamic_slowdown,
                "v_slowdown_dynamic_x": dynamic_slowdown,
                "configured_dynamic_obstacle_count": self._configured_dynamic_obstacle_count(),
                "avoidance_active": int(selected_speed < desiredVel - 1e-6),
                "candidate_selected_speed": selected_speed,
                "candidate_safe_count": candidate_result.safe_count,
                "candidate_min_clearance": candidate_clearance,
                "candidate_emergency_stop": int(candidate_result.emergency_stop),
                "candidate_prediction_horizon": self.candidate_planner.horizon,
                "candidate_raw_selected_speed": candidate_result.selected_speed,
                "candidate_yield_active": int(yield_active),
                "candidate_applied_speed": applied_speed,
                "path_cross_track_error": cross_track_error,
                "path_turn_angle_deg": path_turn_angle_deg,
                "path_speed_ceiling": path_speed_ceiling,
                "candidate_control_delay": self.candidate_planner.control_delay,
                "candidate_brake_decel": self.candidate_planner.max_brake_decel,
                "candidate_reverse_drift_buffer": self.candidate_planner.reverse_drift_distance,
                "candidate_initial_path_speed": candidate_result.initial_path_speed,
                "candidate_predicted_stop_distance": candidate_result.predicted_stop_distance,
            }
        )
        info.update(dynamic_info)
        self.last_compute_t = state_t
        self.last_command_velocity = np.asarray(v_cmd, dtype=float).copy()
        self.last_planner_info = info.copy()
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
