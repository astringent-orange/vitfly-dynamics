#!/usr/bin/python3
import argparse
import csv
import math
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from astar_planner import DEFAULT_STATIC_INFLATION, StaticAStarPlanner, read_path_csv, write_path_csv


DIFFICULTY_CONFIG = {
    "easy": {"period": (9.0, 12.0), "scale": (0.5, 0.9), "z_amp": (0.2, 0.5)},
    "medium": {"period": (6.0, 10.0), "scale": (0.6, 1.2), "z_amp": (0.3, 0.8)},
    "hard": {"period": (4.5, 7.0), "scale": (0.8, 1.4), "z_amp": (0.4, 1.0)},
}


def parse_env_ids(value):
    env_ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start_id = int(start)
            end_id = int(end)
            step = 1 if end_id >= start_id else -1
            env_ids.extend(range(start_id, end_id + step, step))
        else:
            env_ids.append(int(part))
    if not env_ids:
        raise argparse.ArgumentTypeError("env id list is empty")
    return env_ids


def repo_root():
    return Path(__file__).resolve().parents[2]


def write_traj_csv(path, x, y0, y1, z_mid, z_amp, period, dt):
    steps = int(math.ceil(period / dt)) + 1
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "qw", "qx", "qy", "qz"])
        for i in range(steps):
            t = min(i * dt, period)
            phase = t / period
            y = y0 + (y1 - y0) * phase
            z = z_mid + z_amp * math.sin(2.0 * math.pi * phase)
            writer.writerow([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", "1.0", "0.0", "0.0", "0.0"])


def write_dynamic_yaml(path, objects):
    with path.open("w") as f:
        f.write(f"N: {len(objects)}\n")
        for idx, obj in enumerate(objects, start=1):
            f.write(f"Object{idx}:\n")
            f.write(f"  csvtraj: {obj['csvtraj']}\n")
            f.write("  loop: true\n")
            f.write("  position:\n")
            for value in obj["position"]:
                f.write(f"  - {value:.6f}\n")
            f.write("  prefab: rpg_box01\n")
            f.write("  rotation:\n")
            for value in [1.0, 0.0, 0.0, 0.0]:
                f.write(f"  - {value:.6f}\n")
            f.write("  scale:\n")
            for _ in range(3):
                f.write(f"  - {obj['scale']:.6f}\n")


def plan_astar_path(args, static_csv, output_dir):
    planner = StaticAStarPlanner(
        str(static_csv),
        resolution=args.astar_resolution,
        inflation_radius=args.static_inflation,
    )
    planned_path = planner.plan(args.astar_start, args.astar_goal)
    if not planned_path:
        raise RuntimeError(f"A* failed for {output_dir}")
    if not all(
        planner.segment_margin(planned_path[idx], planned_path[idx + 1]) >= -1e-9
        for idx in range(len(planned_path) - 1)
    ):
        raise RuntimeError(f"A* path enters an inflated obstacle for {output_dir}")
    return planned_path


def environment_paths(args, env_id):
    root = repo_root()
    source_dir = root / "flightmare" / "flightpy" / "configs" / "vision" / args.source_level / f"environment_{env_id}"
    output_dir = root / "flightmare" / "flightpy" / "configs" / "vision" / f"dynamic_astar_{args.difficulty}" / f"environment_{env_id}"
    return source_dir, output_dir


def preflight_paths(args):
    planned_paths = {}
    for env_id in args.env_ids:
        source_dir, output_dir = environment_paths(args, env_id)
        static_csv = source_dir / "static_obstacles.csv"
        if not static_csv.is_file():
            raise FileNotFoundError(f"Source static map not found for environment_{env_id}: {static_csv}")
        if output_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing {output_dir}; use --overwrite only when intentional"
            )
        print(f"[GEN_DYNAMIC_ASTAR] Preflighting environment_{env_id}")
        planned_paths[env_id] = plan_astar_path(args, static_csv, output_dir)
    return planned_paths


def generate_one(args, env_id, planned_path):
    rng = np.random.default_rng(args.seed + env_id * 9973)
    source_dir, output_dir = environment_paths(args, env_id)
    staging_dir = output_dir.parent / f".{output_dir.name}.staging"
    csv_dir = staging_dir / "csvtrajs"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)
    csv_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "static_obstacles.csv", staging_dir / "static_obstacles.csv")

    cfg = DIFFICULTY_CONFIG[args.difficulty]
    x_positions = np.linspace(10.0, 55.0, args.num_dynamic)
    x_positions += rng.uniform(-1.5, 1.5, size=args.num_dynamic)

    objects = []
    for idx, x in enumerate(x_positions):
        y0, y1 = (-8.0, 8.0) if idx % 2 == 0 else (8.0, -8.0)
        y0 += rng.uniform(-0.7, 0.7)
        y1 += rng.uniform(-0.7, 0.7)
        z_mid = rng.uniform(2.4, 5.6)
        z_amp = rng.uniform(*cfg["z_amp"])
        period = rng.uniform(*cfg["period"])
        scale = rng.uniform(*cfg["scale"])
        traj_name = f"traj_dyn_{idx:03d}"
        write_traj_csv(csv_dir / f"{traj_name}.csv", x, y0, y1, z_mid, z_amp, period, args.dt)
        objects.append(
            {
                "csvtraj": traj_name,
                "position": [x, y0, z_mid],
                "scale": scale,
            }
        )

    write_dynamic_yaml(staging_dir / "dynamic_obstacles.yaml", objects)
    write_path_csv(staging_dir / "astar_path.csv", planned_path)
    verify_environment(args, env_id, staging_dir, planned_path)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to replace existing {output_dir}")
        backup_dir = output_dir.parent / f".{output_dir.name}.backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        output_dir.rename(backup_dir)
        try:
            staging_dir.rename(output_dir)
        except Exception:
            backup_dir.rename(output_dir)
            raise
        shutil.rmtree(backup_dir)
    else:
        staging_dir.rename(output_dir)
    print(f"[GEN_DYNAMIC_ASTAR] Wrote {args.num_dynamic} dynamic obstacles to {output_dir}")


def verify_environment(args, env_id, output_dir=None, planned_path=None):
    _, default_output_dir = environment_paths(args, env_id)
    output_dir = output_dir or default_output_dir
    static_csv = output_dir / "static_obstacles.csv"
    yaml_path = output_dir / "dynamic_obstacles.yaml"
    path_csv = output_dir / "astar_path.csv"
    required_files = [static_csv, yaml_path, path_csv]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"environment_{env_id} missing required files: {missing}")
    with yaml_path.open() as stream:
        config = yaml.safe_load(stream) or {}
    if int(config.get("N", -1)) != args.num_dynamic:
        raise RuntimeError(f"environment_{env_id} dynamic YAML count does not match {args.num_dynamic}")
    trajectory_names = [config.get(f"Object{index}", {}).get("csvtraj") for index in range(1, args.num_dynamic + 1)]
    if any(not name for name in trajectory_names):
        raise RuntimeError(f"environment_{env_id} dynamic YAML has missing trajectory names")
    for name in trajectory_names:
        trajectory_path = output_dir / "csvtrajs" / f"{name}.csv"
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"environment_{env_id} missing trajectory {trajectory_path}")
        with trajectory_path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        if len(rows) < 3 or rows[0] != ["t", "x", "y", "z", "qw", "qx", "qy", "qz"]:
            raise RuntimeError(f"environment_{env_id} has invalid trajectory CSV {trajectory_path}")
        try:
            values = np.asarray([[float(value) for value in row[:4]] for row in rows[1:]], dtype=float)
        except ValueError as exc:
            raise RuntimeError(f"environment_{env_id} has non-numeric trajectory {trajectory_path}") from exc
        if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) < 0.0):
            raise RuntimeError(f"environment_{env_id} has invalid trajectory samples {trajectory_path}")

    path = planned_path or read_path_csv(path_csv)
    if len(path) < 2:
        raise RuntimeError(f"environment_{env_id} A* path has fewer than two points")
    planner = StaticAStarPlanner(
        str(static_csv), resolution=args.astar_resolution, inflation_radius=args.static_inflation
    )
    if not all(planner.segment_margin(path[index], path[index + 1]) >= -1e-9 for index in range(len(path) - 1)):
        raise RuntimeError(f"environment_{env_id} A* path enters an inflated obstacle")
    return len(path)


def rebuild_paths_only(args):
    root = repo_root()
    output_root = root / "flightmare" / "flightpy" / "configs" / "vision" / f"dynamic_astar_{args.difficulty}"
    planned_paths = []
    for env_id in args.env_ids:
        output_dir = output_root / f"environment_{env_id}"
        static_csv = output_dir / "static_obstacles.csv"
        if not static_csv.exists():
            raise FileNotFoundError(f"Static map not found: {static_csv}")
        planned_paths.append((output_dir, plan_astar_path(args, static_csv, output_dir)))

    for output_dir, planned_path in planned_paths:
        write_path_csv(output_dir / "astar_path.csv", planned_path)
        print(
            f"[GEN_DYNAMIC_ASTAR] Rebuilt A* path for {output_dir} "
            f"with inflation={args.static_inflation:.2f}m points={len(planned_path)}"
        )


def generate(args):
    if args.path_only:
        rebuild_paths_only(args)
        return
    planned_paths = preflight_paths(args)
    for env_id in args.env_ids:
        generate_one(args, env_id, planned_paths[env_id])


def verify(args):
    for env_id in args.env_ids:
        point_count = verify_environment(args, env_id)
        print(f"[GEN_DYNAMIC_ASTAR] Verified environment_{env_id} points={point_count}")


def main():
    parser = argparse.ArgumentParser(description="Generate dynamic A* data-collection environment.")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-dynamic", type=int, default=8)
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTY_CONFIG.keys()), default="medium")
    parser.add_argument("--source-level", default="spheres_medium")
    parser.add_argument("--env-ids", type=parse_env_ids, default=[0], help="Environment ids, e.g. 0, 0-9, or 0,2,4.")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--astar-resolution", type=float, default=0.3)
    parser.add_argument("--static-inflation", type=float, default=DEFAULT_STATIC_INFLATION)
    parser.add_argument("--astar-start", type=float, nargs=3, default=[0.0, 0.0, 3.0])
    parser.add_argument("--astar-goal", type=float, nargs=3, default=[60.0, 0.0, 3.0])
    parser.add_argument("--path-only", action="store_true", help="Only rebuild astar_path.csv in existing dynamic environments.")
    parser.add_argument("--verify-only", action="store_true", help="Validate existing dynamic environments without modifying them.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing environments after staging validation.")
    args = parser.parse_args()
    if args.path_only and args.verify_only:
        parser.error("--path-only and --verify-only cannot be combined")
    if args.verify_only:
        verify(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
