#!/usr/bin/python3
import argparse
from collections import Counter
import csv
import glob
import os
import sys

import cv2
import numpy as np


REQUIRED_COLUMNS = [
    "timestamp",
    "desired_vel",
    "pos_x",
    "vel_x",
    "velcmd_x",
    "velcmd_y",
    "velcmd_z",
    "lookahead_x",
    "lookahead_y",
    "lookahead_z",
    "v_path_x",
    "v_path_y",
    "v_path_z",
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
    "candidate_selected_speed",
    "candidate_safe_count",
    "candidate_min_clearance",
    "candidate_emergency_stop",
    "candidate_prediction_horizon",
    "candidate_raw_selected_speed",
    "candidate_yield_active",
    "candidate_applied_speed",
    "path_cross_track_error",
    "path_turn_angle_deg",
    "path_speed_ceiling",
    "is_collide",
]

ENV_COLUMNS = ["env_level", "env_folder", "env_seed"]


def validate_trajectory(
    path,
    require_env_fields=False,
    max_post_goal_rows=0,
    max_negative_xcmd_rows=0,
    max_low_speed_ratio=0.15,
    max_collision_rows=0,
    min_nearest_margin=0.0,
    max_negative_actual_x_ratio=0.02,
    max_backtrack_distance=0.3,
    max_path_cross_track_error=0.8,
    max_applied_speed_accel=3.5,
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
    timestamps = [row.get("timestamp", "") for row in rows]
    duplicate_timestamps = {ts: count for ts, count in Counter(timestamps).items() if ts and count > 1}
    if duplicate_timestamps:
        duplicate_rows = sum(count - 1 for count in duplicate_timestamps.values())
        sample = sorted(duplicate_timestamps.items())[:5]
        errors.append(f"{path}: duplicate timestamp rows {duplicate_rows}, sample {sample}")
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
    if {"vel_x", "pos_x"}.issubset(fieldnames):
        try:
            actual_vel_x = np.asarray([float(row["vel_x"]) for row in rows], dtype=float)
            positions_x = np.asarray([float(row["pos_x"]) for row in rows], dtype=float)
        except ValueError:
            errors.append(f"{path}: invalid actual x velocity or position")
        else:
            negative_actual = actual_vel_x < -0.05
            negative_actual_ratio = float(np.sum(negative_actual)) / max(len(rows), 1)
            if negative_actual_ratio > max_negative_actual_x_ratio:
                errors.append(
                    f"{path}: negative actual vel_x ratio {negative_actual_ratio:.3f} "
                    f"> {max_negative_actual_x_ratio:.3f}"
                )
            max_backtrack = 0.0
            backtrack_start = None
            for idx, is_negative in enumerate(negative_actual):
                if is_negative and backtrack_start is None:
                    backtrack_start = positions_x[max(0, idx - 1)]
                if is_negative:
                    max_backtrack = max(max_backtrack, float(backtrack_start - positions_x[idx]))
                else:
                    backtrack_start = None
            if max_backtrack > max_backtrack_distance:
                errors.append(
                    f"{path}: max continuous backtrack {max_backtrack:.3f}m "
                    f"> {max_backtrack_distance:.3f}m"
                )
    candidate_fields = {
        "candidate_selected_speed",
        "candidate_safe_count",
        "candidate_min_clearance",
        "candidate_emergency_stop",
        "candidate_prediction_horizon",
        "candidate_raw_selected_speed",
        "candidate_yield_active",
        "candidate_applied_speed",
        "path_cross_track_error",
        "path_turn_angle_deg",
        "path_speed_ceiling",
        "desired_vel",
    }
    if candidate_fields.issubset(fieldnames):
        try:
            selected_speeds = np.asarray([float(row["candidate_selected_speed"]) for row in rows])
            desired_speeds = np.asarray([float(row["desired_vel"]) for row in rows])
            safe_count_values = np.asarray([float(row["candidate_safe_count"]) for row in rows])
            safe_counts = safe_count_values.astype(int)
            emergency_stops = np.asarray([int(float(row["candidate_emergency_stop"])) for row in rows])
            min_clearances = np.asarray([float(row["candidate_min_clearance"]) for row in rows])
            horizons = np.asarray([float(row["candidate_prediction_horizon"]) for row in rows])
            raw_selected_speeds = np.asarray([float(row["candidate_raw_selected_speed"]) for row in rows])
            yield_active = np.asarray([int(float(row["candidate_yield_active"])) for row in rows])
            applied_speeds = np.asarray([float(row["candidate_applied_speed"]) for row in rows])
            cross_track_errors = np.asarray([float(row["path_cross_track_error"]) for row in rows])
            turn_angles = np.asarray([float(row["path_turn_angle_deg"]) for row in rows])
            path_speed_ceilings = np.asarray([float(row["path_speed_ceiling"]) for row in rows])
            slowdown_dynamic = np.asarray([float(row["v_slowdown_dynamic_x"]) for row in rows])
            slowdown_total = np.asarray([float(row["v_slowdown_x"]) for row in rows])
            avoidance_active = np.asarray([int(float(row["avoidance_active"])) for row in rows])
        except ValueError:
            errors.append(f"{path}: invalid candidate-speed diagnostics")
        else:
            finite_values = np.concatenate(
                (
                    selected_speeds,
                    desired_speeds,
                    min_clearances,
                    horizons,
                    raw_selected_speeds,
                    applied_speeds,
                    cross_track_errors,
                    turn_angles,
                    path_speed_ceilings,
                )
            )
            if not np.isfinite(finite_values).all():
                errors.append(f"{path}: non-finite candidate-speed diagnostics")
            out_of_range = np.sum((selected_speeds < -1e-6) | (selected_speeds > desired_speeds + 1e-6))
            if out_of_range:
                errors.append(f"{path}: candidate speed out-of-range rows {int(out_of_range)}")
            raw_out_of_range = np.sum(
                (raw_selected_speeds < -1e-6) | (raw_selected_speeds > desired_speeds + 1e-6)
            )
            if raw_out_of_range:
                errors.append(f"{path}: raw candidate speed out-of-range rows {int(raw_out_of_range)}")
            applied_out_of_range = np.sum(
                (applied_speeds < -1e-6) | (applied_speeds > desired_speeds + 1e-6)
            )
            if applied_out_of_range:
                errors.append(f"{path}: applied candidate speed out-of-range rows {int(applied_out_of_range)}")
            if np.any((turn_angles < -1e-6) | (turn_angles > 180.0 + 1e-6)):
                errors.append(f"{path}: invalid path turn angle")
            if np.any((path_speed_ceilings < -1e-6) | (path_speed_ceilings > desired_speeds + 1e-6)):
                errors.append(f"{path}: invalid path speed ceiling")
            invalid_emergency = np.sum(
                (emergency_stops != 0)
                & ((selected_speeds > 1e-6) | (safe_counts > 1))
            )
            if invalid_emergency:
                errors.append(f"{path}: invalid candidate emergency-stop rows {int(invalid_emergency)}")
            missing_emergency = np.sum(
                (desired_speeds > 1e-6)
                & (selected_speeds <= 1e-6)
                & (emergency_stops == 0)
                & (yield_active == 0)
            )
            if missing_emergency:
                errors.append(f"{path}: missing candidate emergency-stop rows {int(missing_emergency)}")
            if np.any(np.abs(safe_count_values - safe_counts) > 1e-6) or np.any((safe_counts < 0) | (safe_counts > 11)):
                errors.append(f"{path}: invalid candidate safe count")
            if np.any(horizons <= 0.0):
                errors.append(f"{path}: non-positive candidate prediction horizon")
            if np.any((yield_active != 0) & (yield_active != 1)):
                errors.append(f"{path}: invalid candidate yield state")
            if np.any(cross_track_errors < -1e-6):
                errors.append(f"{path}: negative path cross-track error")
            elif len(cross_track_errors) and np.max(cross_track_errors) > max_path_cross_track_error:
                errors.append(
                    f"{path}: max path cross-track error {np.max(cross_track_errors):.3f}m "
                    f"> {max_path_cross_track_error:.3f}m"
                )
            if len(rows) > 1:
                timestamps_float = np.asarray([float(row["timestamp"]) for row in rows])
                dt = np.diff(timestamps_float)
                valid_dt = dt > 1e-4
                applied_accel = np.zeros_like(dt)
                applied_accel[valid_dt] = np.abs(np.diff(applied_speeds)[valid_dt] / dt[valid_dt])
                if np.any(applied_accel[valid_dt] > max_applied_speed_accel):
                    errors.append(
                        f"{path}: max applied path-speed accel {np.max(applied_accel[valid_dt]):.3f}m/s^2 "
                        f"> {max_applied_speed_accel:.3f}m/s^2"
                    )
            expected_slowdown = np.zeros_like(selected_speeds)
            moving = desired_speeds > 1e-6
            expected_slowdown[moving] = 1.0 - selected_speeds[moving] / desired_speeds[moving]
            if np.any(np.abs(slowdown_dynamic - expected_slowdown) > 1e-5) or np.any(
                np.abs(slowdown_total - expected_slowdown) > 1e-5
            ):
                errors.append(f"{path}: candidate slowdown diagnostics mismatch")
            expected_active = (selected_speeds < desired_speeds - 1e-6).astype(int)
            if np.any(avoidance_active != expected_active):
                errors.append(f"{path}: candidate avoidance_active mismatch")
    if "pos_x" in fieldnames:
        try:
            post_goal_rows = sum(1 for row in rows if float(row["pos_x"]) >= 60.0)
        except ValueError:
            errors.append(f"{path}: invalid pos_x")
        else:
            if post_goal_rows > max_post_goal_rows:
                errors.append(f"{path}: post-goal rows {post_goal_rows} > {max_post_goal_rows}")
    collision_rows = 0
    if "is_collide" in fieldnames:
        try:
            collision_rows = sum(1 for row in rows if int(float(row["is_collide"])) != 0)
        except ValueError:
            errors.append(f"{path}: invalid is_collide")
        else:
            if collision_rows > max_collision_rows:
                errors.append(f"{path}: collision rows {collision_rows} > {max_collision_rows}")
    if "nearest_obstacle_margin" in fieldnames:
        try:
            margins = [float(row["nearest_obstacle_margin"]) for row in rows if row.get("nearest_obstacle_margin", "") != ""]
        except ValueError:
            errors.append(f"{path}: invalid nearest_obstacle_margin")
        else:
            if margins and min(margins) < min_nearest_margin:
                errors.append(f"{path}: min nearest_obstacle_margin {min(margins):.3f} < {min_nearest_margin:.3f}")
    if {"is_collide", "nearest_obstacle_margin"}.issubset(fieldnames):
        try:
            mismatch_rows = sum(
                1
                for row in rows
                if int(float(row["is_collide"])) != 0 and float(row["nearest_obstacle_margin"]) >= 0.0
            )
        except ValueError:
            errors.append(f"{path}: invalid collision/margin values")
        else:
            if mismatch_rows:
                errors.append(f"{path}: collision/margin mismatch rows {mismatch_rows}")
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
    parser.add_argument("--max-collision-rows", type=int, default=0)
    parser.add_argument("--min-nearest-margin", type=float, default=0.0)
    parser.add_argument("--max-negative-actual-x-ratio", type=float, default=0.02)
    parser.add_argument("--max-backtrack-distance", type=float, default=0.3)
    parser.add_argument("--max-path-cross-track-error", type=float, default=0.8)
    parser.add_argument("--max-applied-speed-accel", type=float, default=3.5)
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
            max_collision_rows=args.max_collision_rows,
            min_nearest_margin=args.min_nearest_margin,
            max_negative_actual_x_ratio=args.max_negative_actual_x_ratio,
            max_backtrack_distance=args.max_backtrack_distance,
            max_path_cross_track_error=args.max_path_cross_track_error,
            max_applied_speed_accel=args.max_applied_speed_accel,
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
