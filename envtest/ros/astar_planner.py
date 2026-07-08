#!/usr/bin/python3
import csv
import heapq
import math
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class StaticObstacle:
    center: np.ndarray
    radius: float


class StaticAStarPlanner:
    def __init__(
        self,
        static_csv,
        resolution=0.3,
        inflation_radius=0.5,
        bounds=((0.0, 62.0), (-9.5, 9.5), (1.0, 8.0)),
    ):
        self.static_csv = static_csv
        self.resolution = float(resolution)
        self.inflation_radius = float(inflation_radius)
        self.bounds = np.array(bounds, dtype=float)
        self.origin = self.bounds[:, 0]
        self.shape = np.floor((self.bounds[:, 1] - self.bounds[:, 0]) / self.resolution).astype(int) + 1
        self.obstacles = self._read_static_obstacles(static_csv)
        self.occupancy = self._build_occupancy()

    def _read_static_obstacles(self, static_csv):
        obstacles = []
        with open(static_csv, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 11:
                    continue
                center = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=float)
                if not np.all(np.isfinite(center)) or np.any(np.abs(center) > 500.0):
                    continue
                scale = max(float(row[8]), float(row[9]), float(row[10]))
                obstacles.append(StaticObstacle(center=center, radius=0.5 * scale + self.inflation_radius))
        return obstacles

    def _build_occupancy(self):
        xs = self.origin[0] + np.arange(self.shape[0]) * self.resolution
        ys = self.origin[1] + np.arange(self.shape[1]) * self.resolution
        zs = self.origin[2] + np.arange(self.shape[2]) * self.resolution
        occupancy = np.zeros(tuple(self.shape), dtype=bool)

        yy, zz = np.meshgrid(ys, zs, indexing="ij")
        for obstacle in self.obstacles:
            min_idx = self.world_to_grid(obstacle.center - obstacle.radius)
            max_idx = self.world_to_grid(obstacle.center + obstacle.radius)
            min_idx = np.maximum(min_idx, 0)
            max_idx = np.minimum(max_idx, self.shape - 1)
            for ix in range(min_idx[0], max_idx[0] + 1):
                dx2 = (xs[ix] - obstacle.center[0]) ** 2
                y_slice = yy[min_idx[1] : max_idx[1] + 1, min_idx[2] : max_idx[2] + 1]
                z_slice = zz[min_idx[1] : max_idx[1] + 1, min_idx[2] : max_idx[2] + 1]
                occupied = dx2 + (y_slice - obstacle.center[1]) ** 2 + (z_slice - obstacle.center[2]) ** 2 <= obstacle.radius**2
                occupancy[ix, min_idx[1] : max_idx[1] + 1, min_idx[2] : max_idx[2] + 1] |= occupied
        return occupancy

    def world_to_grid(self, point):
        point = np.asarray(point, dtype=float)
        return np.rint((point - self.origin) / self.resolution).astype(int)

    def grid_to_world(self, idx):
        return self.origin + np.asarray(idx, dtype=float) * self.resolution

    def in_bounds(self, idx):
        idx = np.asarray(idx, dtype=int)
        return np.all(idx >= 0) and np.all(idx < self.shape)

    def is_free_idx(self, idx):
        idx = tuple(np.asarray(idx, dtype=int))
        return self.in_bounds(idx) and not self.occupancy[idx]

    def nearest_free(self, idx, max_radius=12):
        idx = np.asarray(idx, dtype=int)
        if self.is_free_idx(idx):
            return tuple(idx)
        for radius in range(1, max_radius + 1):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != radius:
                            continue
                        cand = idx + np.array([dx, dy, dz])
                        if self.is_free_idx(cand):
                            candidates.append(tuple(cand))
            if candidates:
                return min(candidates, key=lambda c: np.linalg.norm(np.asarray(c) - idx))
        return None

    def _neighbors(self, idx):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nxt = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                    if self.is_free_idx(nxt):
                        yield nxt, math.sqrt(dx * dx + dy * dy + dz * dz) * self.resolution

    def _heuristic(self, a, b):
        return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) * self.resolution))

    def plan(self, start, goal):
        start_idx = self.nearest_free(self.world_to_grid(start))
        goal_idx = self.nearest_free(self.world_to_grid(goal))
        if start_idx is None or goal_idx is None:
            return []

        open_heap = [(self._heuristic(start_idx, goal_idx), 0.0, start_idx)]
        came_from = {}
        g_score = {start_idx: 0.0}
        visited = set()

        while open_heap:
            _, cost, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            if current == goal_idx:
                return self.simplify_path(self._reconstruct(came_from, current))
            visited.add(current)
            for neighbor, step_cost in self._neighbors(current):
                tentative = cost + step_cost
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    priority = tentative + self._heuristic(neighbor, goal_idx)
                    heapq.heappush(open_heap, (priority, tentative, neighbor))
        return []

    def _reconstruct(self, came_from, current):
        path = [self.grid_to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.grid_to_world(current))
        path.reverse()
        return path

    def segment_is_free(self, start, end, step=None):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        dist = np.linalg.norm(end - start)
        if dist <= 1e-6:
            return True
        step = step or self.resolution
        for alpha in np.linspace(0.0, 1.0, max(2, int(math.ceil(dist / step)) + 1)):
            idx = self.world_to_grid(start + alpha * (end - start))
            if not self.is_free_idx(idx):
                return False
        return True

    def simplify_path(self, path):
        if len(path) <= 2:
            return [np.asarray(p, dtype=float) for p in path]
        simplified = [np.asarray(path[0], dtype=float)]
        anchor = 0
        probe = 2
        while probe < len(path):
            if not self.segment_is_free(path[anchor], path[probe]):
                simplified.append(np.asarray(path[probe - 1], dtype=float))
                anchor = probe - 1
            probe += 1
        simplified.append(np.asarray(path[-1], dtype=float))
        return simplified

    def first_lookahead(self, path, position, lookahead_distance=3.0):
        if not path:
            return np.asarray(position, dtype=float)
        position = np.asarray(position, dtype=float)
        for point in path:
            if np.linalg.norm(np.asarray(point) - position) >= lookahead_distance and point[0] >= position[0] - 0.5:
                return np.asarray(point, dtype=float)
        return np.asarray(path[-1], dtype=float)


def default_static_map_path():
    env_level = os.environ.get("VITFLY_ENV_LEVEL", "dynamic_astar_medium")
    env_folder = os.environ.get("VITFLY_ENV_FOLDER", "environment_0")
    flightmare_path = os.environ.get("FLIGHTMARE_PATH")
    if flightmare_path:
        return os.path.join(
            flightmare_path,
            "flightpy",
            "configs",
            "vision",
            env_level,
            env_folder,
            "static_obstacles.csv",
        )
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "flightmare",
            "flightpy",
            "configs",
            "vision",
            env_level,
            env_folder,
            "static_obstacles.csv",
        )
    )
