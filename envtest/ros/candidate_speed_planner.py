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
    predicted_stop_distance: float = 0.0

    @property
    def safe_speeds(self):
        return tuple(speed for speed, _, safe in self.evaluations if safe)

    def clearance_for(self, speed):
        if not self.evaluations:
            return 999.0
        return float(min(self.evaluations, key=lambda item: abs(item[0] - speed))[1])


class CandidateYieldPolicy:
    def __init__(self, release_frames=5, resume_frames=2):
        self.release_frames = int(release_frames)
        if self.release_frames <= 0:
            raise ValueError("release_frames must be positive")
        self.resume_frames = int(resume_frames)
        if self.resume_frames <= 0:
            raise ValueError("resume_frames must be positive")
        self.reset()

    def reset(self):
        self.active = False
        self.target_speed = 0.0
        self.full_speed_safe_frames = 0
        self.positive_safe_frames = 0

    def apply(self, result, desired_speed):
        desired_speed = max(0.0, float(desired_speed))
        raw_speed = max(0.0, float(result.selected_speed))
        safe_speeds = sorted(float(speed) for speed in result.safe_speeds)
        positive_safe = [speed for speed in safe_speeds if speed > 1e-6]
        full_speed_safe = any(abs(speed - desired_speed) <= 1e-6 for speed in safe_speeds)

        if not self.active:
            self.target_speed = raw_speed
            if raw_speed < desired_speed - 1e-6:
                self.active = True
                self.full_speed_safe_frames = 0
                self.positive_safe_frames = 0
            return self.target_speed, self.active

        if not positive_safe:
            self.target_speed = 0.0
            self.full_speed_safe_frames = 0
            self.positive_safe_frames = 0
            return self.target_speed, self.active

        if full_speed_safe:
            self.full_speed_safe_frames += 1
        else:
            self.full_speed_safe_frames = 0

        if self.full_speed_safe_frames >= self.release_frames:
            self.active = False
            self.target_speed = desired_speed
            self.full_speed_safe_frames = 0
            self.positive_safe_frames = 0
            return self.target_speed, self.active

        if self.target_speed <= 1e-6:
            self.positive_safe_frames += 1
            if self.positive_safe_frames < self.resume_frames:
                return 0.0, self.active
            self.target_speed = min(positive_safe)
            return self.target_speed, self.active

        self.positive_safe_frames = 0
        nonincreasing_safe = [speed for speed in safe_speeds if speed <= self.target_speed + 1e-6]
        if nonincreasing_safe:
            self.target_speed = max(nonincreasing_safe)
        faster_safe = [speed for speed in positive_safe if speed > self.target_speed + 1e-6]
        if faster_safe:
            self.target_speed = min(faster_safe)
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

    def apply(self, target_speed, direction, t, actual_velocity, speed_limit=None):
        target_speed = max(0.0, float(target_speed))
        speed_limit = float("inf") if speed_limit is None else max(0.0, float(speed_limit))
        target_speed = min(target_speed, speed_limit)
        direction = np.asarray(direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        t = float(t)

        if self.applied_speed is None or self.previous_t is None:
            actual_velocity = np.asarray(actual_velocity, dtype=float)
            self.applied_speed = float(
                np.clip(np.dot(actual_velocity, direction), 0.0, speed_limit)
            )
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
            self.applied_speed = float(np.clip(self.applied_speed + delta, 0.0, speed_limit))
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

    def turn_angle_ahead(self, position, preview_distance):
        """Return the path-direction change visible within the preview distance."""
        progress, _, tangent = self.project(position)
        preview_progress = min(self.length, progress + max(0.0, float(preview_distance)))
        preview_tangent = self.tangent_at(preview_progress)
        cosine = float(np.clip(np.dot(tangent, preview_tangent), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))


class CandidateSpeedPlanner:
    def __init__(
        self,
        horizon=3.0,
        prediction_dt=0.1,
        max_accel=3.0,
        max_brake_decel=1.5,
        control_delay=0.25,
        reverse_drift_distance=0.4,
        cross_track_decay_time=0.8,
        safety_margin=1.0,
        candidate_step=0.1,
    ):
        self.horizon = float(horizon)
        self.prediction_dt = float(prediction_dt)
        self.max_accel = float(max_accel)
        self.max_brake_decel = float(max_brake_decel)
        self.control_delay = float(control_delay)
        self.reverse_drift_distance = float(reverse_drift_distance)
        self.cross_track_decay_time = float(cross_track_decay_time)
        self.safety_margin = float(safety_margin)
        self.candidate_step = float(candidate_step)
        if self.horizon <= 0.0 or self.prediction_dt <= 0.0:
            raise ValueError("prediction horizon and dt must be positive")
        if self.max_accel <= 0.0 or self.max_brake_decel <= 0.0:
            raise ValueError("candidate acceleration limits must be positive")
        if self.control_delay < 0.0 or self.reverse_drift_distance < 0.0:
            raise ValueError("candidate delay and reverse-drift buffer must be nonnegative")
        if self.cross_track_decay_time <= 0.0:
            raise ValueError("cross-track decay time must be positive")
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

    @staticmethod
    def _position_at_prediction_progress(polyline, progress):
        if progress < 0.0:
            return polyline.points[0] + progress * polyline.tangent_at(0.0)
        if progress > polyline.length:
            return polyline.points[-1] + (progress - polyline.length) * polyline.tangent_at(polyline.length)
        return polyline.position_at(progress)

    def _advance_speed(self, speed, target_speed, duration):
        if duration <= 0.0:
            return speed, 0.0
        accel_limit = self.max_accel if target_speed >= speed else self.max_brake_decel
        delta = float(np.clip(target_speed - speed, -accel_limit * duration, accel_limit * duration))
        next_speed = speed + delta
        return next_speed, 0.5 * (speed + next_speed) * duration

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
                delay_dt = min(step_dt, max(0.0, self.control_delay - previous_time))
                progress += speed * delay_dt
                response_dt = step_dt - delay_dt
                speed, response_progress = self._advance_speed(speed, target_speed, response_dt)
                progress += response_progress
            offset_ratio = float(0.2 ** (float(tau) / self.cross_track_decay_time))
            predicted = self._position_at_prediction_progress(polyline, progress)
            if target_speed <= 1e-6 and self.reverse_drift_distance > 0.0:
                drift_ratio = min(1.0, float(tau) / max(self.control_delay, self.prediction_dt))
                predicted = predicted - polyline.tangent_at(progress) * self.reverse_drift_distance * drift_ratio
            positions.append(predicted + path_offset * offset_ratio)
            previous_time = float(tau)
        return np.asarray(positions), initial_speed

    def predicted_stop_distance(self, initial_path_speed):
        forward_speed = max(0.0, float(initial_path_speed))
        return float(
            forward_speed * self.control_delay
            + forward_speed * forward_speed / (2.0 * self.max_brake_decel)
            + self.reverse_drift_distance
        )

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
            predicted_stop_distance=self.predicted_stop_distance(initial_path_speed),
        )
