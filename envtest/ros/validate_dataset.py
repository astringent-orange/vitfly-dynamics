#!/usr/bin/python3
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np


REQUIRED_COLUMNS = [
    "timestamp",
    "env_level",
    "env_folder",
    "env_seed",
    "velcmd_x",
    "velcmd_y",
    "velcmd_z",
    "lookahead_x",
    "lookahead_y",
    "lookahead_z",
    "v_path_x",
    "v_path_y",
    "v_path_z",
    "v_avoid_x",
    "v_avoid_y",
    "v_avoid_z",
    "nearest_dyn_dist",
    "nearest_dyn_rel_speed",
    "ttc_min",
    "avoidance_active",
    "astar_replan_count",
    "astar_success",
]


def validate_trajectory(path):
    csv_path = os.path.join(path, "data.csv")
    if not os.path.exists(csv_path):
        return [f"{path}: missing data.csv"], 0, 0

    errors = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else []
    depth_pngs = sorted(p for p in glob.glob(os.path.join(path, "*.png")) if not p.endswith("_rgb.png"))
    rgb_pngs = sorted(glob.glob(os.path.join(path, "*_rgb.png")))

    missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        errors.append(f"{path}: missing columns {missing}")
    if rgb_pngs:
        errors.append(f"{path}: found RGB debug PNGs in trajectory root")
    if len(depth_pngs) != len(rows):
        errors.append(f"{path}: png count {len(depth_pngs)} != csv rows {len(rows)}")
    if len(depth_pngs) > 0:
        sample = cv2.imread(depth_pngs[0], cv2.IMREAD_UNCHANGED)
        if sample is None or sample.size == 0 or not np.isfinite(sample).all() or sample.max() <= 0:
            errors.append(f"{path}: invalid depth PNG sample")
    if {"velcmd_x", "velcmd_y", "velcmd_z"}.issubset(fieldnames):
        try:
            vel_values = np.array(
                [[float(row["velcmd_x"]), float(row["velcmd_y"]), float(row["velcmd_z"])] for row in rows],
                dtype=float,
            )
        except ValueError:
            errors.append(f"{path}: non-finite velocity command")
        else:
            if not np.isfinite(vel_values).all():
                errors.append(f"{path}: non-finite velocity command")
    astar_success = 0
    if "astar_success" in fieldnames:
        astar_success = sum(int(float(row["astar_success"])) for row in rows if row.get("astar_success", "") != "")
    if "astar_success" in fieldnames and astar_success == 0:
        errors.append(f"{path}: no successful A* samples")
    avoidance_rows = 0
    if "avoidance_active" in fieldnames:
        avoidance_rows = sum(int(float(row["avoidance_active"])) for row in rows if row.get("avoidance_active", "") != "")
    return errors, len(rows), avoidance_rows


def main():
    parser = argparse.ArgumentParser(description="Validate dynamic A* expert dataset folders.")
    parser.add_argument("dataset_dir", help="Directory containing trajectory subfolders.")
    args = parser.parse_args()

    folders = sorted(p for p in glob.glob(os.path.join(args.dataset_dir, "*")) if os.path.isdir(p))
    if not folders:
        print(f"[VALIDATE_DATASET] No trajectory folders under {args.dataset_dir}")
        return 1

    all_errors = []
    rows = 0
    avoidance_rows = 0
    for folder in folders:
        errors, count, avoidance_count = validate_trajectory(folder)
        all_errors.extend(errors)
        rows += count
        avoidance_rows += avoidance_count

    if all_errors:
        print("[VALIDATE_DATASET] Failed:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print(f"[VALIDATE_DATASET] OK: {len(folders)} trajectories, {rows} rows, {avoidance_rows} avoidance-active rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
