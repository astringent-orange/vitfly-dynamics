#!/usr/bin/python3

import csv
import math
import os
import re

import numpy as np
import yaml


def _object_sort_key(name):
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else name


def _state_pos(state):
    return np.asarray(state.pos, dtype=float)


class DynamicTrajectory:
    def __init__(self, name, times, positions, scale, loop=True, max_reasonable_speed=8.0):
        self.name = name
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.scale = float(scale)
        self.loop = bool(loop)
        self.max_reasonable_speed = float(max_reasonable_speed)
        self.period = float(self.times[-1]) if len(self.times) else 0.0
        self.dt = self._median_dt()
        self.segment_velocities = self._segment_velocities()
        self.fallback_velocity = self._fallback_velocity()

    def _median_dt(self):
        if len(self.times) < 2:
            return 0.02
        diffs = np.diff(self.times)
        diffs = diffs[diffs > 1e-6]
        if len(diffs) == 0:
            return 0.02
        return float(np.median(diffs))

    def _segment_velocities(self):
        if len(self.times) < 2:
            return np.zeros((0, 3))
        dt = np.diff(self.times)
        dp = np.diff(self.positions, axis=0)
        velocities = np.zeros_like(dp)
        valid = dt > 1e-6
        velocities[valid] = dp[valid] / dt[valid, None]
        return velocities

    def _fallback_velocity(self):
        if len(self.segment_velocities) == 0:
            return np.zeros(3)
        speeds = np.linalg.norm(self.segment_velocities, axis=1)
        valid = np.isfinite(speeds) & (speeds <= self.max_reasonable_speed)
        if np.any(valid):
            return np.median(self.segment_velocities[valid], axis=0)
        return np.zeros(3)

    def _normalize_phase(self, phase):
        if self.period <= 0.0:
            return 0.0
        if self.loop:
            return float(phase % self.period)
        return float(np.clip(phase, self.times[0], self.period))

    def position_at(self, phase):
        if len(self.positions) == 0:
            return np.zeros(3)
        if len(self.positions) == 1 or self.period <= 0.0:
            return self.positions[0].copy()
        phase = self._normalize_phase(phase)
        return np.array(
            [
                np.interp(phase, self.times, self.positions[:, axis])
                for axis in range(3)
            ],
            dtype=float,
        )

    def velocity_at(self, phase):
        if len(self.segment_velocities) == 0 or self.period <= 0.0:
            return np.zeros(3)
        phase = self._normalize_phase(phase)
        idx = int(np.searchsorted(self.times, phase, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.segment_velocities) - 1))
        velocity = self.segment_velocities[idx]
        if not np.all(np.isfinite(velocity)) or np.linalg.norm(velocity) > self.max_reasonable_speed:
            return self.fallback_velocity.copy()
        return velocity.copy()

    def closest_phase(self, world_pos):
        if len(self.positions) == 0:
            return 0.0, float("inf")
        world_pos = np.asarray(world_pos, dtype=float)
        dists = np.linalg.norm(self.positions - world_pos, axis=1)
        idx = int(np.argmin(dists))
        return float(self.times[idx]), float(dists[idx])


class DynamicObstacleTrajectoryPredictor:
    def __init__(self, env_dir, recalibration_error=0.5):
        self.env_dir = env_dir
        self.recalibration_error = float(recalibration_error)
        self.trajectories = []
        self.calibration = {}
        self.loaded = False
        self._load()

    def _load(self):
        if not self.env_dir:
            return
        yaml_path = os.path.join(self.env_dir, "dynamic_obstacles.yaml")
        if not os.path.exists(yaml_path):
            return

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f) or {}

        object_names = sorted(
            [key for key in config.keys() if key.startswith("Object")],
            key=_object_sort_key,
        )
        for object_name in object_names:
            obj = config.get(object_name) or {}
            traj_name = obj.get("csvtraj")
            if not traj_name:
                continue
            traj_path = os.path.join(self.env_dir, "csvtrajs", f"{traj_name}.csv")
            if not os.path.exists(traj_path):
                continue
            times, positions = self._read_traj_csv(traj_path)
            if len(times) == 0:
                continue
            scale_values = obj.get("scale") or [0.0]
            scale = float(scale_values[0]) * 0.5
            self.trajectories.append(
                DynamicTrajectory(
                    traj_name,
                    times,
                    positions,
                    scale,
                    loop=bool(obj.get("loop", True)),
                )
            )
        self.loaded = len(self.trajectories) > 0
        if self.loaded:
            print(
                f"[DynamicObstacleTrajectoryPredictor] Loaded {len(self.trajectories)} trajectories from {self.env_dir}"
            )

    def _read_traj_csv(self, traj_path):
        times = []
        positions = []
        with open(traj_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    times.append(float(row["t"]))
                    positions.append([float(row["x"]), float(row["y"]), float(row["z"])])
                except (KeyError, TypeError, ValueError):
                    continue
        return times, positions

    def configured_count(self):
        return len(self.trajectories)

    def _topic_world_positions(self, state, dynamic_obstacles_msg):
        if dynamic_obstacles_msg is None:
            return []
        drone_pos = _state_pos(state)
        world_positions = []
        for obstacle in dynamic_obstacles_msg.obstacles:
            rel = np.array(
                [obstacle.position.x, obstacle.position.y, obstacle.position.z],
                dtype=float,
            )
            if np.all(np.isfinite(rel)):
                world_positions.append(drone_pos + rel)
        return world_positions

    def _ordered_matches(self, world_positions):
        if len(world_positions) == len(self.trajectories):
            matches = []
            for idx, world_pos in enumerate(world_positions):
                phase, error = self.trajectories[idx].closest_phase(world_pos)
                matches.append((idx, phase, error))
            if matches and max(match[2] for match in matches) <= max(1.0, self.recalibration_error * 2.0):
                return matches

        unmatched = set(range(len(self.trajectories)))
        matches = []
        for world_pos in world_positions:
            best = None
            for idx in unmatched:
                phase, error = self.trajectories[idx].closest_phase(world_pos)
                if best is None or error < best[2]:
                    best = (idx, phase, error)
            if best is not None:
                unmatched.remove(best[0])
                matches.append(best)
        return matches

    def calibrate(self, state, dynamic_obstacles_msg):
        if not self.loaded:
            return False
        world_positions = self._topic_world_positions(state, dynamic_obstacles_msg)
        if not world_positions:
            return False

        state_t = float(state.t)
        matches = self._ordered_matches(world_positions)
        for idx, phase, error in matches:
            self.calibration[idx] = {
                "phase": float(phase),
                "state_t": state_t,
                "error": float(error),
            }
        return len(self.calibration) > 0

    def _predicted_world_position(self, idx, state_t):
        calib = self.calibration.get(idx)
        if calib is None:
            return None
        traj = self.trajectories[idx]
        phase = calib["phase"] + (state_t - calib["state_t"])
        return traj.position_at(phase)

    def maybe_recalibrate(self, state, dynamic_obstacles_msg):
        if not self.loaded:
            return False
        if not self.calibration:
            return self.calibrate(state, dynamic_obstacles_msg)

        world_positions = self._topic_world_positions(state, dynamic_obstacles_msg)
        if not world_positions:
            return len(self.calibration) > 0

        state_t = float(state.t)
        errors = []
        if len(world_positions) == len(self.trajectories):
            for idx, world_pos in enumerate(world_positions):
                predicted = self._predicted_world_position(idx, state_t)
                if predicted is not None:
                    errors.append(float(np.linalg.norm(predicted - world_pos)))
        else:
            predicted_positions = [
                (idx, self._predicted_world_position(idx, state_t))
                for idx in range(len(self.trajectories))
            ]
            predicted_positions = [
                (idx, pos) for idx, pos in predicted_positions if pos is not None
            ]
            for world_pos in world_positions:
                if predicted_positions:
                    errors.append(
                        min(float(np.linalg.norm(pos - world_pos)) for _, pos in predicted_positions)
                    )

        if errors and max(errors) > self.recalibration_error:
            return self.calibrate(state, dynamic_obstacles_msg)
        return len(self.calibration) > 0

    def relative_measurements(
        self,
        state,
        dynamic_obstacles_msg,
        drone_velocity,
        max_distance=None,
        forward_only=True,
    ):
        if not self.maybe_recalibrate(state, dynamic_obstacles_msg):
            return []

        drone_pos = _state_pos(state)
        drone_velocity = np.asarray(drone_velocity, dtype=float)
        state_t = float(state.t)
        measurements = []

        for idx, traj in enumerate(self.trajectories):
            calib = self.calibration.get(idx)
            if calib is None:
                continue
            phase = calib["phase"] + (state_t - calib["state_t"])
            world_pos = traj.position_at(phase)
            world_vel = traj.velocity_at(phase)
            rel_pos = world_pos - drone_pos
            rel_vel = world_vel - drone_velocity
            if not np.all(np.isfinite(rel_pos)) or not np.all(np.isfinite(rel_vel)):
                continue
            dist = np.linalg.norm(rel_pos)
            if max_distance is not None and dist > max_distance:
                continue
            if forward_only and rel_pos[0] <= -1.0:
                continue
            measurements.append(
                {
                    "pos": rel_pos,
                    "vel": rel_vel,
                    "scale": traj.scale,
                    "source": "scene_csv",
                    "phase": float(traj._normalize_phase(phase)),
                    "calibration_error": float(calib.get("error", math.inf)),
                }
            )
        return measurements
