#!/usr/bin/env python3
"""Deterministic rollout evaluation reference used by offline tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProfile:
    name: str = "strict"
    start_x: float = 0.5
    goal_x: float = 60.0
    timeout_seconds: float = 60.0
    collision_margin: float = 0.0
    terminate_on_collision: bool = False
    bounding_box: tuple = (-5.0, 65.0, -10.0, 10.0, 0.0, 10.0)

    @classmethod
    def from_mapping(cls, name, mapping):
        bounds = tuple(float(value) for value in mapping.get("bounding_box", cls.bounding_box))
        if len(bounds) != 6:
            raise ValueError("bounding_box must contain six values")
        terminate_on_collision = mapping.get("terminate_on_collision", cls.terminate_on_collision)
        if isinstance(terminate_on_collision, str):
            normalized = terminate_on_collision.strip().lower()
            if normalized in ("1", "true", "yes", "on"):
                terminate_on_collision = True
            elif normalized in ("0", "false", "no", "off"):
                terminate_on_collision = False
            else:
                raise ValueError("terminate_on_collision must be a boolean")
        return cls(
            name=name,
            start_x=float(mapping.get("start_x", cls.start_x)),
            goal_x=float(mapping.get("goal_x", cls.goal_x)),
            timeout_seconds=float(mapping.get("timeout_seconds", cls.timeout_seconds)),
            collision_margin=float(mapping.get("collision_margin", cls.collision_margin)),
            terminate_on_collision=bool(terminate_on_collision),
            bounding_box=bounds,
        )


class RolloutEvaluator:
    """State machine for one rollout.

    Timestamps are simulator timestamps.  Collision count is edge-triggered,
    matching the original evaluator's one-count-per-contact behavior.
    """

    def __init__(self, profile):
        self.profile = profile
        self.start_time = None
        self.goal_time = None
        self.last_time = None
        self.goal_reached = False
        self.finished = False
        self.collision_count = 0
        self._in_collision = False
        self.termination_reason = None
        self.last_position = None

    def _out_of_bounds(self, position):
        x, y, z = position
        xmin, xmax, ymin, ymax, zmin, zmax = self.profile.bounding_box
        return not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax)

    def on_state(self, timestamp, position):
        if self.finished:
            return
        timestamp = float(timestamp)
        position = tuple(float(value) for value in position[:3])
        self.last_time = timestamp
        self.last_position = position
        if self.start_time is None and position[0] > self.profile.start_x:
            self.start_time = timestamp
        if position[0] >= self.profile.goal_x:
            self.goal_reached = True
            self.goal_time = timestamp
            self.finished = True
            self.termination_reason = "goal_reached_with_collision" if self.collision_count else "goal_reached"
            return
        if self._out_of_bounds(position):
            self.finished = True
            self.termination_reason = "out_of_bounds"
            return
        if self.start_time is not None and timestamp - self.start_time > self.profile.timeout_seconds:
            self.finished = True
            self.termination_reason = "timeout"

    def on_obstacle_margin(self, margin, timestamp=None):
        if self.finished:
            return
        if timestamp is not None:
            self.last_time = float(timestamp)
        colliding = float(margin) < self.profile.collision_margin
        if colliding and not self._in_collision:
            self.collision_count += 1
        self._in_collision = colliding
        if colliding and self.profile.terminate_on_collision:
            self.finished = True
            self.termination_reason = "collision"

    def on_collision(self, colliding, timestamp=None):
        self.on_obstacle_margin(-1.0 if colliding else 1.0, timestamp)

    def fail(self, reason):
        if not self.finished:
            self.finished = True
            self.termination_reason = str(reason)

    def result(self):
        termination_elapsed_time = None
        if self.start_time is not None:
            end_time = self.goal_time if self.goal_reached else self.last_time
            if end_time is not None:
                termination_elapsed_time = max(0.0, float(end_time) - self.start_time)
        flight_time = termination_elapsed_time if self.goal_reached else None
        success = bool(self.goal_reached and self.collision_count == 0)
        collision = bool(self.collision_count > 0)
        if self.termination_reason is None:
            reason = "missing_result"
        else:
            reason = self.termination_reason
        return {
            "goal_reached": int(self.goal_reached),
            "success": int(success),
            "collision": int(collision),
            "collision_count": int(self.collision_count),
            "flight_time": flight_time,
            "termination_elapsed_time": termination_elapsed_time,
            "termination_reason": reason,
        }


def evaluate_rows(rows, profile):
    """Evaluate a CSV-like iterable containing telemetry rows."""
    evaluator = RolloutEvaluator(profile)
    for row in rows:
        timestamp = float(row["timestamp"])
        evaluator.on_state(timestamp, (float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])))
        if "nearest_obstacle_margin" in row and row["nearest_obstacle_margin"] not in ("", None):
            evaluator.on_obstacle_margin(float(row["nearest_obstacle_margin"]), timestamp)
        elif "is_collide" in row:
            evaluator.on_collision(float(row["is_collide"]) > 0, timestamp)
        if evaluator.finished:
            break
    return evaluator.result()
