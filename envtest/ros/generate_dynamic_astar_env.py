#!/usr/bin/python3
import argparse
import csv
import math
import os
import shutil
from pathlib import Path

import numpy as np


DIFFICULTY_CONFIG = {
    "easy": {"period": (9.0, 12.0), "scale": (0.5, 0.9), "z_amp": (0.2, 0.5)},
    "medium": {"period": (6.0, 10.0), "scale": (0.6, 1.2), "z_amp": (0.3, 0.8)},
    "hard": {"period": (4.5, 7.0), "scale": (0.8, 1.4), "z_amp": (0.4, 1.0)},
}


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


def generate(args):
    rng = np.random.default_rng(args.seed)
    root = repo_root()
    source_dir = root / "flightmare" / "flightpy" / "configs" / "vision" / "spheres_medium" / "environment_0"
    output_dir = root / "flightmare" / "flightpy" / "configs" / "vision" / f"dynamic_astar_{args.difficulty}" / "environment_0"
    csv_dir = output_dir / "csvtrajs"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source environment not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "static_obstacles.csv", output_dir / "static_obstacles.csv")

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

    write_dynamic_yaml(output_dir / "dynamic_obstacles.yaml", objects)
    print(f"[GEN_DYNAMIC_ASTAR] Wrote {args.num_dynamic} dynamic obstacles to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate dynamic A* data-collection environment.")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-dynamic", type=int, default=8)
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTY_CONFIG.keys()), default="medium")
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
