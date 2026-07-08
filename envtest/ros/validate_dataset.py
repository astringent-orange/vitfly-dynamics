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
    "astar_plan_time",
    "nearest_obstacle_margin",
    "nearest_static_dist",
    "dynamic_obstacle_count",
    "v_slowdown_x",
    "v_slowdown_dynamic_x",
    "v_slowdown_static_x",
]

ENV_COLUMNS = ["env_level", "env_folder", "env_seed"]


def validate_trajectory(
    path,
    require_env_fields=False,
    max_post_goal_rows=0,
    max_negative_xcmd_rows=0,
    max_low_speed_ratio=0.15,
):
    csv_path = os.path.join(path, "data.csv")
    if not os.path.exists(csv_path):
        return [f"{path}: missing data.csv"], 0, 0, set()

    errors = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else []
    depth_pngs = sorted(p for p in glob.glob(os.path.join(path, "*.png")) if not p.endswith("_rgb.png"))
    rgb_pngs = sorted(glob.glob(os.path.join(path, "*_rgb.png")))

    missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        errors.append(f"{path}: missing columns {missing}")
    missing_env = [col for col in ENV_COLUMNS if col not in fieldnames]
    if require_env_fields and missing_env:
        errors.append(f"{path}: missing env columns {missing_env}")
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
            negative_xcmd_rows = int(np.sum(vel_values[:, 0] < -1e-3))
            if negative_xcmd_rows > max_negative_xcmd_rows:
                errors.append(f"{path}: negative velcmd_x rows {negative_xcmd_rows} > {max_negative_xcmd_rows}")
            low_speed_rows = int(np.sum(np.linalg.norm(vel_values, axis=1) < 0.3))
            low_speed_ratio = low_speed_rows / max(len(rows), 1)
            if low_speed_ratio > max_low_speed_ratio:
                errors.append(f"{path}: low-speed ratio {low_speed_ratio:.3f} > {max_low_speed_ratio:.3f}")
    if "pos_x" in fieldnames:
        try:
            post_goal_rows = sum(1 for row in rows if float(row["pos_x"]) >= 60.0)
        except ValueError:
            errors.append(f"{path}: invalid pos_x")
        else:
            if post_goal_rows > max_post_goal_rows:
                errors.append(f"{path}: post-goal rows {post_goal_rows} > {max_post_goal_rows}")
    astar_success = 0
    if "astar_success" in fieldnames:
        astar_success = sum(int(float(row["astar_success"])) for row in rows if row.get("astar_success", "") != "")
    if "astar_success" in fieldnames and astar_success == 0:
        errors.append(f"{path}: no successful A* samples")
    avoidance_rows = 0
    if "avoidance_active" in fieldnames:
        avoidance_rows = sum(int(float(row["avoidance_active"])) for row in rows if row.get("avoidance_active", "") != "")
    env_folders = set()
    if "env_folder" in fieldnames:
        env_folders = {row["env_folder"] for row in rows if row.get("env_folder", "")}
    return errors, len(rows), avoidance_rows, env_folders


def main():
    parser = argparse.ArgumentParser(description="Validate dynamic A* expert dataset folders.")
    parser.add_argument("dataset_dir", help="Directory containing trajectory subfolders.")
    parser.add_argument("--require-env-fields", action="store_true", help="Fail trajectories that do not contain env_level/env_folder/env_seed columns.")
    parser.add_argument("--require-multiple-envs", action="store_true", help="Fail unless at least two env_folder values are present.")
    parser.add_argument("--max-post-goal-rows", type=int, default=0)
    parser.add_argument("--max-negative-xcmd-rows", type=int, default=0)
    parser.add_argument("--max-low-speed-ratio", type=float, default=0.15)
    args = parser.parse_args()

    folders = sorted(p for p in glob.glob(os.path.join(args.dataset_dir, "*")) if os.path.isdir(p))
    if not folders:
        print(f"[VALIDATE_DATASET] No trajectory folders under {args.dataset_dir}")
        return 1

    all_errors = []
    rows = 0
    avoidance_rows = 0
    env_folders = set()
    for folder in folders:
        errors, count, avoidance_count, folder_envs = validate_trajectory(
            folder,
            require_env_fields=args.require_env_fields,
            max_post_goal_rows=args.max_post_goal_rows,
            max_negative_xcmd_rows=args.max_negative_xcmd_rows,
            max_low_speed_ratio=args.max_low_speed_ratio,
        )
        all_errors.extend(errors)
        rows += count
        avoidance_rows += avoidance_count
        env_folders.update(folder_envs)

    if args.require_multiple_envs and len(env_folders) < 2:
        all_errors.append(f"{args.dataset_dir}: only found env folders {sorted(env_folders)}")

    if all_errors:
        print("[VALIDATE_DATASET] Failed:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    env_summary = f", env_folders={sorted(env_folders)}" if env_folders else ""
    print(f"[VALIDATE_DATASET] OK: {len(folders)} trajectories, {rows} rows, {avoidance_rows} avoidance-active rows{env_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
