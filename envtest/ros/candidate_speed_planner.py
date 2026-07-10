#!/usr/bin/python3

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateSpeedResult:
    selected_speed: float
    safe_count: int
    min_clearance: float
    emergency_stop: bool
    initial_path_speed: float
    evaluations: tuple

    @property
    def safe_speeds(self):
        return tuple(speed for speed, _, safe in self.evaluations if safe)

    def clearance_for(self, speed):
        if not self.evaluations:
            return 999.0
        return float(min(self.evaluations, key=lambda item: abs(item[0] - speed))[1])


class CandidateYieldPolicy:
    def __init__(self, release_frames=5):
        self.release_frames = int(release_frames)
        if self.release_frames <= 0:
            raise ValueError("release_frames must be positive")
        self.reset()

    def reset(self):
        self.active = False
        self.target_speed = 0.0
        self.full_speed_safe_frames = 0

    def apply(self, result, desired_speed):
        desired_speed = max(0.0, float(desired_speed))
        raw_speed = max(0.0, float(result.selected_speed))
        safe_speeds = sorted(float(speed) for speed in result.safe_speeds)
        full_speed_safe = any(abs(speed - desired_speed) <= 1e-6 for speed in safe_speeds)

        if not self.active:
            self.target_speed = raw_speed
            if raw_speed < desired_speed - 1e-6:
                self.active = True
                self.full_speed_safe_frames = 0
            return self.target_speed, self.active

        if full_speed_safe:
            self.full_speed_safe_frames += 1
        else:
            self.full_speed_safe_frames = 0

        if self.full_speed_safe_frames >= self.release_frames:
            self.active = False
            self.target_speed = desired_speed
            self.full_speed_safe_frames = 0
            return self.target_speed, self.active

        nonincreasing_safe = [speed for speed in safe_speeds if speed <= self.target_speed + 1e-6]
        if nonincreasing_safe:
            self.target_speed = max(nonincreasing_safe)
        elif safe_speeds:
            self.target_speed = min(safe_speeds)
        else:
            self.target_speed = 0.0
        return self.target_speed, self.active


class PathSpeedController:
    def __init__(self, max_accel=3.0):
        self.max_accel = float(max_accel)
        if self.max_accel <= 0.0:
            raise ValueError("max_accel must be positive")
        self.reset()

    def reset(self):
        self.applied_speed = None
        self.previous_t = None
        self.previous_command = np.zeros(3)

    def apply(self, target_speed, direction, t, actual_velocity):
        target_speed = max(0.0, float(target_speed))
        direction = np.asarray(direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        t = float(t)

        if self.applied_speed is None or self.previous_t is None:
            actual_velocity = np.asarray(actual_velocity, dtype=float)
            self.applied_speed = max(0.0, float(np.dot(actual_velocity, direction)))
            self.previous_t = t
        elif t <= self.previous_t:
            return self.previous_command.copy(), float(self.applied_speed)
        else:
            dt = t - self.previous_t
            delta = float(
                np.clip(
                    target_speed - self.applied_speed,
                    -self.max_accel * dt,
                    self.max_accel * dt,
                )
            )
            self.applied_speed = max(0.0, self.applied_speed + delta)
            self.previous_t = t

        command = direction * self.applied_speed
        command[0] = max(0.0, command[0])
        self.previous_command = command.copy()
        return command, float(self.applied_speed)


class PolylinePath:
    def __init__(self, points):
        raw_points = np.asarray(points, dtype=float)
        if raw_points.ndim != 2 or raw_points.shape[1] != 3 or len(raw_points) == 0:
            raise ValueError("path must contain at least one 3D point")

        kept = [raw_points[0]]
        for point in raw_points[1:]:
            if np.linalg.norm(point - kept[-1]) > 1e-6:
                kept.append(point)
        self.points = np.asarray(kept, dtype=float)
        if len(self.points) == 1:
            self.segment_vectors = np.zeros((0, 3))
            self.segment_lengths = np.zeros(0)
            self.cumulative_lengths = np.array([0.0])
        else:
            self.segment_vectors = np.diff(self.points, axis=0)
            self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
            self.cumulative_lengths = np.concatenate(
                ([0.0], np.cumsum(self.segment_lengths))
            )

    @property
    def length(self):
        return float(self.cumulative_lengths[-1])

    def project(self, position):
        position = np.asarray(position, dtype=float)
        if len(self.points) == 1:
            return 0.0, self.points[0].copy(), np.array([1.0, 0.0, 0.0])

        best = None
        for idx, (start, vector, length) in enumerate(
            zip(self.points[:-1], self.segment_vectors, self.segment_lengths)
        ):
            ratio = float(np.clip(np.dot(position - start, vector) / (length * length), 0.0, 1.0))
            projected = start + ratio * vector
            distance = float(np.linalg.norm(position - projected))
            if best is None or distance < best[0]:
                progress = self.cumulative_lengths[idx] + ratio * length
                best = (distance, float(progress), projected, vector / length)
        return best[1], best[2].copy(), best[3].copy()

    def position_at(self, progress):
        if len(self.points) == 1:
            return self.points[0].copy()
        progress = float(np.clip(progress, 0.0, self.length))
        idx = int(np.searchsorted(self.cumulative_lengths, progress, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.segment_lengths) - 1))
        local = progress - self.cumulative_lengths[idx]
        ratio = local / self.segment_lengths[idx]
        return self.points[idx] + ratio * self.segment_vectors[idx]

    def tangent_at(self, progress):
        if len(self.points) == 1:
            return np.array([1.0, 0.0, 0.0])
        progress = float(np.clip(progress, 0.0, self.length))
        idx = int(np.searchsorted(self.cumulative_lengths, progress, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.segment_lengths) - 1))
        return self.segment_vectors[idx] / self.segment_lengths[idx]

    def reference_from(self, position, lookahead_distance):
        progress, projected, tangent = self.project(position)
        reference_progress = min(self.length, progress + max(0.0, float(lookahead_distance)))
        reference = self.position_at(reference_progress)
        reference_tangent = self.tangent_at(reference_progress)
        cross_track_error = float(np.linalg.norm(np.asarray(position, dtype=float) - projected))
        return {
            "progress": progress,
            "projected": projected,
            "tangent": tangent,
            "reference_progress": reference_progress,
            "reference": reference,
            "reference_tangent": reference_tangent,
            "cross_track_error": cross_track_error,
        }


class CandidateSpeedPlanner:
    def __init__(
        self,
        horizon=3.0,
        prediction_dt=0.1,
        prediction_accel=3.0,
        safety_margin=1.0,
        candidate_step=0.1,
        cross_track_convergence=1.0,
    ):
        self.horizon = float(horizon)
        self.prediction_dt = float(prediction_dt)
        self.prediction_accel = float(prediction_accel)
        self.safety_margin = float(safety_margin)
        self.candidate_step = float(candidate_step)
        self.cross_track_convergence = float(cross_track_convergence)
        if self.horizon <= 0.0 or self.prediction_dt <= 0.0:
            raise ValueError("prediction horizon and dt must be positive")
        if self.prediction_accel <= 0.0:
            raise ValueError("prediction acceleration must be positive")
        if not 0.0 < self.candidate_step <= 1.0:
            raise ValueError("candidate_step must be in (0, 1]")

    @property
    def time_offsets(self):
        count = int(np.ceil(self.horizon / self.prediction_dt))
        return np.linspace(0.0, self.horizon, count + 1)

    def candidate_speeds(self, desired_speed):
        desired_speed = max(0.0, float(desired_speed))
        if desired_speed <= 1e-6:
            return np.array([0.0])
        fractions = np.arange(0.0, 1.0 + 0.5 * self.candidate_step, self.candidate_step)
        fractions = np.unique(np.clip(np.append(fractions, 1.0), 0.0, 1.0))
        return fractions * desired_speed

    def predict_drone_trajectory(self, path, position, velocity, target_speed):
        polyline = path if isinstance(path, PolylinePath) else PolylinePath(path)
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        progress, projected, tangent = polyline.project(position)
        path_offset = position - projected
        initial_speed = float(np.dot(velocity, tangent))
        target_speed = max(0.0, float(target_speed))

        positions = []
        speed = initial_speed
        previous_time = 0.0
        for tau in self.time_offsets:
            step_dt = float(tau - previous_time)
            if step_dt > 0.0:
                speed_delta = np.clip(
                    target_speed - speed,
                    -self.prediction_accel * step_dt,
                    self.prediction_accel * step_dt,
                )
                next_speed = speed + speed_delta
                progress += 0.5 * (speed + next_speed) * step_dt
                speed = next_speed
            if self.cross_track_convergence > 1e-6:
                offset_ratio = max(0.0, 1.0 - float(tau) / self.cross_track_convergence)
            else:
                offset_ratio = 0.0
            positions.append(polyline.position_at(progress) + path_offset * offset_ratio)
            previous_time = float(tau)
        return np.asarray(positions), initial_speed

    def _candidate_clearance(self, drone_positions, obstacle_predictions):
        min_clearance = float("inf")
        for obstacle in obstacle_predictions:
            positions = np.asarray(obstacle.get("positions", []), dtype=float)
            if positions.shape != drone_positions.shape:
                continue
            radius = max(0.0, float(obstacle.get("radius", 0.0)))
            distances = np.linalg.norm(drone_positions - positions, axis=1)
            clearance = float(np.min(distances - radius - self.safety_margin))
            min_clearance = min(min_clearance, clearance)
        return min_clearance

    def select_speed(self, path, position, velocity, desired_speed, obstacle_predictions):
        polyline = path if isinstance(path, PolylinePath) else PolylinePath(path)
        candidates = self.candidate_speeds(desired_speed)
        evaluations = []
        initial_path_speed = 0.0

        for candidate in candidates:
            drone_positions, initial_path_speed = self.predict_drone_trajectory(
                polyline,
                position,
                velocity,
                candidate,
            )
            clearance = self._candidate_clearance(drone_positions, obstacle_predictions)
            evaluations.append((float(candidate), clearance, clearance >= 0.0))

        safe = [evaluation for evaluation in evaluations if evaluation[2]]
        safe_positive = [evaluation for evaluation in safe if evaluation[0] > 1e-6]
        emergency_stop = len(safe_positive) == 0 and float(desired_speed) > 1e-6
        if safe_positive:
            selected = max(safe_positive, key=lambda item: item[0])
        else:
            zero = min(evaluations, key=lambda item: item[0])
            selected = zero

        min_clearance = selected[1]
        if not np.isfinite(min_clearance):
            min_clearance = 999.0
        return CandidateSpeedResult(
            selected_speed=max(0.0, selected[0]),
            safe_count=len(safe),
            min_clearance=float(min_clearance),
            emergency_stop=emergency_stop,
            initial_path_speed=float(initial_path_speed),
            evaluations=tuple(evaluations),
        )
