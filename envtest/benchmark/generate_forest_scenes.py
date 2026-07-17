#!/usr/bin/env python3
"""Generate immutable forest benchmark scenes from the original ``trees`` maps.

The generator keeps the medium map byte-for-byte identical to the source map,
derives low density by deterministic sub-sampling, and adds deterministic tree
rows for high density. Dynamic obstacle geometry is shared by all speed
profiles; only the trajectory time scale changes.
"""

import argparse
import csv
import hashlib
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
ROS_DIR = ROOT / "envtest" / "ros"
sys.path.insert(0, str(ROS_DIR))
from astar_planner import StaticAStarPlanner, write_path_csv  # noqa: E402


def load_config(path):
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


def dynamic_profiles(cfg):
    profiles = cfg.get("scenario", {}).get("dynamic", {}).get("profiles", {})
    if not profiles:
        raise ValueError("scenario.dynamic.profiles must define dynamic obstacle speeds")
    checked = {}
    for name, values in profiles.items():
        speed = float(values["nominal_speed_mps"])
        multiplier = float(values["speed_multiplier"])
        if speed <= 0.0 or multiplier <= 0.0:
            raise ValueError(f"dynamic profile {name} must have positive speed")
        checked[name] = {"nominal_speed_mps": speed, "speed_multiplier": multiplier}
    return checked


def source_rows(path):
    with open(path, newline="") as stream:
        return [row for row in csv.reader(stream) if len(row) >= 11]


def write_rows(path, rows):
    with open(path, "w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def parse_xyz(row):
    return np.asarray([float(row[1]), float(row[2]), float(row[3])], dtype=float)


def stable_rng(seed, map_id, density):
    value = f"{seed}:{map_id}:{density}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(value).digest()[:8], "big")
    return np.random.default_rng(derived)


def clear_of_endpoints(point, roi, clearance):
    x, y = point[:2]
    return (
        roi[0] <= x <= roi[1]
        and roi[2] <= y <= roi[3]
        and np.linalg.norm(np.asarray([x, y]) - np.asarray([0.0, 0.0])) >= clearance
        and np.linalg.norm(np.asarray([x, y]) - np.asarray([60.0, 0.0])) >= clearance
    )


def make_static_rows(source, density, target, cfg, map_id, attempt=0):
    if density == "medium":
        # This is the central experimental condition: preserve the source map.
        return list(source)

    rng = stable_rng(cfg.get("scene_seed", 0) + attempt * 104729, map_id, density)
    if density == "low":
        indices = np.sort(rng.choice(len(source), size=target, replace=False))
        return [source[int(index)] for index in indices]

    if density != "high":
        raise ValueError(f"unsupported density {density}")

    rows = list(source)
    roi = cfg["scenario"]["forest_roi"]
    spacing = float(cfg["scenario"].get("min_tree_spacing", 1.0))
    clearance = float(cfg["scenario"].get("endpoint_clearance", 3.0))
    existing = [parse_xyz(row) for row in rows]
    attempts = 0
    while len(rows) < target and attempts < 50000:
        attempts += 1
        point = np.asarray(
            [rng.uniform(roi[0], roi[1]), rng.uniform(roi[2], roi[3]), 0.0],
            dtype=float,
        )
        if not clear_of_endpoints(point, roi, clearance):
            continue
        if any(np.linalg.norm(point[:2] - other[:2]) < spacing for other in existing):
            continue
        angle = rng.uniform(-math.pi, math.pi)
        row = [
            "rpg_tree",
            f"{point[0]:.9f}", f"{point[1]:.9f}", "0.0",
            f"{math.cos(angle / 2):.9f}", "0.0",
            f"{math.sin(angle / 2):.9f}", "0.0",
            "0.5", "0.5", "0.5",
        ]
        rows.append(row)
        existing.append(point)
    if len(rows) != target:
        raise RuntimeError(f"could not create {target} trees for map {map_id} after {attempts} attempts")
    return rows


def write_dynamic_assets(scene_dir, map_id, cfg, profile):
    dynamic_cfg = cfg["scenario"]["dynamic"]
    profile_cfg = dynamic_profiles(cfg)[profile]
    multiplier = profile_cfg["speed_multiplier"]
    count = int(dynamic_cfg.get("count", 8))
    periods = dynamic_cfg.get("medium_period_seconds", [6.0, 10.0])
    # Keep positions, directions and scales identical across dynamic profiles;
    # only the trajectory time scale changes.
    rng = stable_rng(cfg.get("scene_seed", 0) + 7919, map_id, "dynamic_geometry")
    trajectories = scene_dir / "csvtrajs"
    trajectories.mkdir(parents=True, exist_ok=True)
    objects = []
    for index, x in enumerate(np.linspace(10.0, 55.0, count)):
        y0, y1 = ((-8.0, 8.0) if index % 2 == 0 else (8.0, -8.0))
        y0 += rng.uniform(-0.7, 0.7)
        y1 += rng.uniform(-0.7, 0.7)
        z_mid = rng.uniform(2.4, 5.6)
        z_amp = rng.uniform(0.3, 0.8)
        period = rng.uniform(float(periods[0]), float(periods[1])) / multiplier
        scale = rng.uniform(0.6, 1.2)
        name = f"traj_dyn_{index:03d}"
        with open(trajectories / f"{name}.csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["t", "x", "y", "z", "qw", "qx", "qy", "qz"])
            for step in range(int(math.ceil(period / 0.02)) + 1):
                t = min(step * 0.02, period)
                phase = t / period
                y = y0 + (y1 - y0) * phase
                z = z_mid + z_amp * math.sin(2.0 * math.pi * phase)
                writer.writerow([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", "1", "0", "0", "0"])
        objects.append((name, x, y0, z_mid, scale))

    with open(scene_dir / "dynamic_obstacles.yaml", "w") as stream:
        stream.write(f"N: {count}\n")
        for index, (name, x, y, z, scale) in enumerate(objects, 1):
            stream.write(f"Object{index}:\n  csvtraj: {name}\n  loop: true\n")
            stream.write("  position:\n")
            for value in (x, y, z):
                stream.write(f"  - {value:.6f}\n")
            stream.write("  prefab: rpg_box01\n  rotation:\n")
            for value in (1.0, 0.0, 0.0, 0.0):
                stream.write(f"  - {value:.6f}\n")
            stream.write("  scale:\n")
            for _ in range(3):
                stream.write(f"  - {scale:.6f}\n")


def check_path(static_path, cfg):
    planner = StaticAStarPlanner(
        str(static_path),
        resolution=float(cfg["scenario"].get("astar_resolution", 0.3)),
        inflation_radius=float(cfg["scenario"].get("static_inflation", 0.9)),
    )
    path = planner.plan([0.0, 0.0, 3.0], [60.0, 0.0, 3.0])
    if not path:
        raise RuntimeError(f"A* found no path for {static_path}")
    if not all(planner.segment_margin(a, b) >= -1e-9 for a, b in zip(path, path[1:])):
        raise RuntimeError(f"A* path clips an inflated obstacle for {static_path}")
    write_path_csv(static_path.parent / "astar_path.csv", path)


def generate(config_path, map_ids=None, overwrite=False):
    cfg = load_config(config_path)
    source_root = ROOT / "flightmare" / "flightpy" / "configs" / "vision" / cfg.get("source_level", "trees")
    output_root = ROOT / "flightmare" / "flightpy" / "configs" / "vision" / cfg.get("output_level", "forest_benchmark_v1")
    profiles = dynamic_profiles(cfg)
    map_ids = list(map_ids if map_ids is not None else sorted(set(cfg["scenario"]["map_ids"]["validation"] + cfg["scenario"]["map_ids"]["test"])))
    for map_id in map_ids:
        source_dir = source_root / f"environment_{map_id}"
        source_file = source_dir / "static_obstacles.csv"
        source = source_rows(source_file)
        for density, target in cfg["scenario"]["density_counts"].items():
            for profile, profile_cfg in profiles.items():
                scene_id = f"map_{map_id:03d}_density_{density}_{profile}"
                scene_dir = output_root / scene_id
                if scene_dir.exists():
                    if not overwrite:
                        continue
                    shutil.rmtree(scene_dir)
                generated = False
                last_error = None
                for attempt in range(20 if density == "high" else 1):
                    staging = output_root / f".{scene_id}.staging"
                    if staging.exists():
                        shutil.rmtree(staging)
                    staging.mkdir(parents=True)
                    rows = make_static_rows(source, density, int(target), cfg, map_id, attempt=attempt)
                    if density == "medium":
                        # Preserve the original map byte-for-byte for the reference
                        # density; this also preserves its exact line endings.
                        shutil.copy2(source_file, staging / "static_obstacles.csv")
                    else:
                        write_rows(staging / "static_obstacles.csv", rows)
                    try:
                        write_dynamic_assets(staging, map_id, cfg, profile)
                        check_path(staging / "static_obstacles.csv", cfg)
                    except RuntimeError as exc:
                        last_error = exc
                        shutil.rmtree(staging)
                        continue
                    (staging / "scene_metadata.yaml").write_text(yaml.safe_dump({
                        "scene_id": scene_id,
                        "map_id": map_id,
                        "density": density,
                        "tree_count": len(rows),
                        "dynamic_profile": profile,
                        "dynamic_speed_mps": profile_cfg["nominal_speed_mps"],
                        "dynamic_speed_multiplier": profile_cfg["speed_multiplier"],
                        "source_map": f"environment_{map_id}",
                        "generator_config": str(config_path),
                    }, sort_keys=True))
                    staging.rename(scene_dir)
                    generated = True
                    print(f"[FOREST_SCENE] generated {scene_id}")
                    break
                if not generated:
                    raise RuntimeError(f"could not generate {scene_id}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--map-ids", default="", help="comma-separated map ids; default uses config splits")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    map_ids = None if not args.map_ids else [int(value) for value in args.map_ids.split(",")]
    generate(args.config, map_ids=map_ids, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
