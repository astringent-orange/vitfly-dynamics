#!/usr/bin/python3
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
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
        return [f"{path}: missing data.csv"], 0

    errors = []
    data = pd.read_csv(csv_path)
    depth_pngs = sorted(p for p in glob.glob(os.path.join(path, "*.png")) if not p.endswith("_rgb.png"))
    rgb_pngs = sorted(glob.glob(os.path.join(path, "*_rgb.png")))

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        errors.append(f"{path}: missing columns {missing}")
    if rgb_pngs:
        errors.append(f"{path}: found RGB debug PNGs in trajectory root")
    if len(depth_pngs) != len(data):
        errors.append(f"{path}: png count {len(depth_pngs)} != csv rows {len(data)}")
    if len(depth_pngs) > 0:
        sample = cv2.imread(depth_pngs[0], cv2.IMREAD_UNCHANGED)
        if sample is None or sample.size == 0 or not np.isfinite(sample).all() or sample.max() <= 0:
            errors.append(f"{path}: invalid depth PNG sample")
    if {"velcmd_x", "velcmd_y", "velcmd_z"}.issubset(data.columns):
        if not np.isfinite(data[["velcmd_x", "velcmd_y", "velcmd_z"]].to_numpy()).all():
            errors.append(f"{path}: non-finite velocity command")
    if "astar_success" in data.columns and data["astar_success"].sum() == 0:
        errors.append(f"{path}: no successful A* samples")
    return errors, len(data)


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
        errors, count = validate_trajectory(folder)
        all_errors.extend(errors)
        rows += count
        csv_path = os.path.join(folder, "data.csv")
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            if "avoidance_active" in data.columns:
                avoidance_rows += int(data["avoidance_active"].sum())

    if all_errors:
        print("[VALIDATE_DATASET] Failed:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print(f"[VALIDATE_DATASET] OK: {len(folders)} trajectories, {rows} rows, {avoidance_rows} avoidance-active rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
