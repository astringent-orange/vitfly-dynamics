#!/usr/bin/env python3
"""Generate fixed-density forest benchmark scenes from the original ``trees`` maps.

Each derived scene normalizes the evaluation corridor to the configured tree
count. Dynamic obstacle geometry is shared by all speed profiles; only the
trajectory time scale changes.
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
    if density == "density_6":
        roi = cfg["scenario"]["forest_roi"]
        spacing = float(cfg["scenario"].get("min_tree_spacing", 1.0))
        clearance = float(cfg["scenario"].get("endpoint_clearance", 3.0))
        rng = stable_rng(cfg.get("scene_seed", 0) + attempt * 104729, map_id, density)
        rows = list(source)
        existing = [parse_xyz(row) for row in rows]
        inside = [point for point in existing if roi[0] <= point[0] <= roi[1]
                  and roi[2] <= point[1] <= roi[3]]
        attempts = 0
        while len(inside) < target and attempts < 100000:
            attempts += 1
            point = np.asarray([rng.uniform(roi[0], roi[1]), rng.uniform(roi[2], roi[3]), 0.0])
            if not clear_of_endpoints(point, roi, clearance):
                continue
            if any(np.linalg.norm(point[:2] - other[:2]) < spacing for other in existing):
                continue
            angle = rng.uniform(-math.pi, math.pi)
            rows.append([
                "rpg_tree", f"{point[0]:.9f}", f"{point[1]:.9f}", "0.0",
                f"{math.cos(angle / 2):.9f}", "0.0", f"{math.sin(angle / 2):.9f}", "0.0",
                "0.5", "0.5", "0.5",
            ])
            existing.append(point)
            inside.append(point)
        if len(inside) != target:
            raise RuntimeError(f"could not create corridor density {target} for map {map_id}")
        return rows

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
    reference_speed = float(dynamic_cfg.get("reference_speed_mps", 2.0))
    target_speed = profile_cfg["nominal_speed_mps"]
    stations = [float(value) for value in dynamic_cfg.get(
        "station_x", np.linspace(10.0, 55.0, int(dynamic_cfg.get("stations", 6)))
    )]
    bands = dynamic_cfg.get("height_bands", {})
    band_order = ("low", "medium", "high")
    if set(bands) != set(band_order):
        raise ValueError("dynamic.height_bands must define low, medium and high")
    count = len(stations) * len(band_order)
    configured_count = int(dynamic_cfg.get("count", count))
    if configured_count != count:
        raise ValueError(f"dynamic.count must equal stations*height_bands ({count})")
    scale_range = tuple(float(value) for value in dynamic_cfg.get("scale_range", [0.6, 1.2]))
    jitter_range = tuple(float(value) for value in dynamic_cfg.get("period_jitter_range", [0.85, 1.15]))
    z_bounds = tuple(float(value) for value in dynamic_cfg.get("z_bounds", [0.0, 10.0]))
    boundary_margin = float(dynamic_cfg.get("boundary_margin", 0.2))
    if len(scale_range) != 2 or len(jitter_range) != 2 or scale_range[0] <= 0:
        raise ValueError("dynamic scale/period ranges are invalid")
    if target_speed <= 0 or reference_speed <= 0:
        raise ValueError("dynamic speeds must be positive")

    # Geometry is generated once per map and shared across all speed profiles;
    # only timestamps are re-scaled to the requested nominal 3-D speed.
    rng = stable_rng(cfg.get("scene_seed", 0) + 7919, map_id, "dynamic_geometry")
    trajectories = scene_dir / "csvtrajs"
    trajectories.mkdir(parents=True, exist_ok=True)
    objects = []
    for station_index, x in enumerate(stations):
        for band_index, band_name in enumerate(band_order):
            index = station_index * len(band_order) + band_index
            band_cfg = bands[band_name]
            center_range = tuple(float(value) for value in band_cfg["center_range"])
            amplitude_range = tuple(float(value) for value in band_cfg["amplitude_range"])
            if len(center_range) != 2 or len(amplitude_range) != 2:
                raise ValueError(f"invalid height range for dynamic band {band_name}")
            z_mid = rng.uniform(*center_range)
            z_amp = rng.uniform(*amplitude_range)
            scale = rng.uniform(*scale_range)
            z_phase = rng.uniform(-math.pi, math.pi)
            z_min = z_mid - z_amp - scale / 2.0
            z_max = z_mid + z_amp + scale / 2.0
            if z_min < z_bounds[0] + boundary_margin or z_max > z_bounds[1] - boundary_margin:
                raise RuntimeError(
                    f"dynamic obstacle {index} leaves z bounds: {z_min:.3f}..{z_max:.3f}"
                )

            y0, y1 = ((-8.0, 8.0) if station_index % 2 == 0 else (8.0, -8.0))
            y0 += rng.uniform(-0.7, 0.7)
            y1 += rng.uniform(-0.7, 0.7)

            phases = np.linspace(0.0, 1.0, 1001)
            ys = y0 + (y1 - y0) * phases
            zs = z_mid + z_amp * np.sin(2.0 * math.pi * phases + z_phase)
            path_length = float(np.linalg.norm(np.diff(np.column_stack((ys, zs)), axis=0), axis=1).sum())
            jitter = rng.uniform(*jitter_range)
            base_period = path_length / reference_speed * jitter
            period = base_period * reference_speed / target_speed
            name = f"traj_dyn_{index:03d}"
            with open(trajectories / f"{name}.csv", "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["t", "x", "y", "z", "qw", "qx", "qy", "qz"])
                for step in range(int(math.ceil(period / 0.02)) + 1):
                    t = min(step * 0.02, period)
                    phase = t / period
                    y = y0 + (y1 - y0) * phase
                    z = z_mid + z_amp * math.sin(2.0 * math.pi * phase + z_phase)
                    writer.writerow([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", "1", "0", "0", "0"])
            objects.append({
                "name": name,
                "x": x,
                "y": y0,
                "z": z_mid + z_amp * math.sin(z_phase),
                "scale": scale,
                "height_band": band_name,
                "z_center": z_mid,
                "z_amplitude": z_amp,
                "z_phase": z_phase,
                "path_length": path_length,
                "period": period,
            })

    with open(scene_dir / "dynamic_obstacles.yaml", "w") as stream:
        stream.write(f"N: {count}\n")
        for index, obj in enumerate(objects, 1):
            stream.write(f"Object{index}:\n  csvtraj: {obj['name']}\n  loop: true\n")
            stream.write("  position:\n")
            for value in (obj["x"], obj["y"], obj["z"]):
                stream.write(f"  - {value:.6f}\n")
            stream.write("  prefab: rpg_box01\n  rotation:\n")
            for value in (1.0, 0.0, 0.0, 0.0):
                stream.write(f"  - {value:.6f}\n")
            stream.write("  scale:\n")
            for _ in range(3):
                stream.write(f"  - {obj['scale']:.6f}\n")
    return objects


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
        density = "density_6"
        target = int(cfg["scenario"]["forest_density"]["roi_tree_count"])
        for profile, profile_cfg in profiles.items():
            scene_id = f"map_{map_id:03d}_{density}_{profile}"
            scene_dir = output_root / scene_id
            if scene_dir.exists():
                if not overwrite:
                    continue
                shutil.rmtree(scene_dir)
            generated = False
            last_error = None
            for attempt in range(20):
                staging = output_root / f".{scene_id}.staging"
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir(parents=True)
                rows = make_static_rows(source, density, int(target), cfg, map_id, attempt=attempt)
                write_rows(staging / "static_obstacles.csv", rows)
                try:
                    dynamic_objects = write_dynamic_assets(staging, map_id, cfg, profile)
                    check_path(staging / "static_obstacles.csv", cfg)
                except RuntimeError as exc:
                    last_error = exc
                    shutil.rmtree(staging)
                    continue
                roi_count = int(sum(
                        cfg["scenario"]["forest_roi"][0] <= parse_xyz(row)[0] <= cfg["scenario"]["forest_roi"][1]
                        and cfg["scenario"]["forest_roi"][2] <= parse_xyz(row)[1] <= cfg["scenario"]["forest_roi"][3]
                        for row in rows
                ))
                (staging / "scene_metadata.yaml").write_text(yaml.safe_dump({
                        "scene_id": scene_id,
                        "map_id": map_id,
                        "density": density,
                        "tree_count": len(rows),
                        "roi_tree_count": roi_count,
                        "forest_density_trees_per_100m2": float(cfg["scenario"]["forest_density"]["trees_per_100m2"]),
                        "dynamic_profile": profile,
                        "dynamic_speed_mps": profile_cfg["nominal_speed_mps"],
                        "dynamic_speed_multiplier": profile_cfg["speed_multiplier"],
                        "dynamic_obstacle_count": len(dynamic_objects),
                        "dynamic_height_bands": {
                            band: sum(obj["height_band"] == band for obj in dynamic_objects)
                            for band in ("low", "medium", "high")
                        },
                        "dynamic_z_center_range": [
                            min(obj["z_center"] for obj in dynamic_objects),
                            max(obj["z_center"] for obj in dynamic_objects),
                        ],
                        "dynamic_z_amplitude_range": [
                            min(obj["z_amplitude"] for obj in dynamic_objects),
                            max(obj["z_amplitude"] for obj in dynamic_objects),
                        ],
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
