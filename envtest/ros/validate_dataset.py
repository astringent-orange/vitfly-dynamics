#!/usr/bin/python3
import argparse
from collections import Counter
import csv
import glob
import os
import sys

import cv2
import numpy as np

from astar_planner import read_path_csv
from candidate_speed_planner import PolylinePath


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
    "candidate_control_delay",
    "candidate_brake_decel",
    "candidate_reverse_drift_buffer",
    "candidate_initial_path_speed",
    "candidate_predicted_stop_distance",
    "is_collide",
]

ENV_COLUMNS = ["env_level", "env_folder", "env_seed", "dynamic_phase_seed", "dynamic_phase_mode"]


def _max_consecutive_true(values):
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _trajectory_path(rows, supplied_points=None):
    if supplied_points is not None:
        return PolylinePath(supplied_points), None
    if not rows:
        return None, "empty trajectory"
    env_levels = {row.get("env_level", "") for row in rows if row.get("env_level", "")}
    env_folders = {row.get("env_folder", "") for row in rows if row.get("env_folder", "")}
    if len(env_levels) != 1 or len(env_folders) != 1:
        return None, "missing or inconsistent env_level/env_folder"
    env_level = next(iter(env_levels))
    env_folder = next(iter(env_folders))
    path_file = os.path.abspath(
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
            "astar_path.csv",
        )
    )
    if not os.path.isfile(path_file):
        return None, f"missing A* path cache {path_file}"
    try:
        points = read_path_csv(path_file)
        return PolylinePath(points), None
    except (OSError, ValueError, KeyError) as exc:
        return None, f"invalid A* path cache {path_file}: {exc}"


def _path_motion_metrics(rows, path):
    positions = np.asarray(
        [[float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])] for row in rows],
        dtype=float,
    )
    velocities = np.asarray(
        [[float(row["vel_x"]), float(row["vel_y"]), float(row["vel_z"])] for row in rows],
        dtype=float,
    )
    progress = np.empty(len(rows), dtype=float)
    path_speed = np.empty(len(rows), dtype=float)
    for index, (position, velocity) in enumerate(zip(positions, velocities)):
        progress[index], _, tangent = path.project(position)
        path_speed[index] = float(np.dot(velocity, tangent))

    reversing = path_speed < -0.05
    negative_ratio = float(np.sum(reversing)) / max(len(rows), 1)
    max_backtrack = 0.0
    backtrack_start = None
    for index, is_reversing in enumerate(reversing):
        if is_reversing and backtrack_start is None:
            backtrack_start = progress[max(0, index - 1)]
        if is_reversing:
            max_backtrack = max(max_backtrack, float(backtrack_start - progress[index]))
        else:
            backtrack_start = None
    return negative_ratio, max_backtrack, path_speed, progress


def validate_trajectory(
    path,
    require_env_fields=False,
    max_post_goal_rows=0,
    max_negative_xcmd_rows=0,
    max_low_speed_ratio=0.15,
    max_collision_rows=0,
    min_nearest_margin=0.0,
    max_negative_path_speed_ratio=0.02,
    max_path_backtrack_distance=0.3,
    max_path_cross_track_error=0.8,
    max_applied_speed_accel=3.5,
    max_yield_zero_positive_frames=2,
    path_points=None,
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
    if {"vel_x", "vel_y", "vel_z", "pos_x", "pos_y", "pos_z"}.issubset(fieldnames):
        try:
            path_polyline, path_error = _trajectory_path(rows, supplied_points=path_points)
            if path_polyline is None:
                if require_env_fields or path_points is not None:
                    errors.append(f"{path}: cannot validate path-projected motion: {path_error}")
            else:
                negative_path_ratio, max_backtrack, _, _ = _path_motion_metrics(rows, path_polyline)
        except ValueError:
            errors.append(f"{path}: invalid path-projected velocity or position")
        else:
            if path_polyline is not None:
                if negative_path_ratio > max_negative_path_speed_ratio:
                    errors.append(
                        f"{path}: negative path-speed ratio {negative_path_ratio:.3f} "
                        f"> {max_negative_path_speed_ratio:.3f}"
                    )
                if max_backtrack > max_path_backtrack_distance:
                    errors.append(
                        f"{path}: max continuous path backtrack {max_backtrack:.3f}m "
                        f"> {max_path_backtrack_distance:.3f}m"
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
        "candidate_control_delay",
        "candidate_brake_decel",
        "candidate_reverse_drift_buffer",
        "candidate_initial_path_speed",
        "candidate_predicted_stop_distance",
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
            control_delays = np.asarray([float(row["candidate_control_delay"]) for row in rows])
            brake_decels = np.asarray([float(row["candidate_brake_decel"]) for row in rows])
            reverse_drift_buffers = np.asarray([float(row["candidate_reverse_drift_buffer"]) for row in rows])
            initial_path_speeds = np.asarray([float(row["candidate_initial_path_speed"]) for row in rows])
            predicted_stop_distances = np.asarray([float(row["candidate_predicted_stop_distance"]) for row in rows])
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
                    control_delays,
                    brake_decels,
                    reverse_drift_buffers,
                    initial_path_speeds,
                    predicted_stop_distances,
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
            if np.any(control_delays < 0.0) or np.any(brake_decels <= 0.0) or np.any(reverse_drift_buffers < 0.0):
                errors.append(f"{path}: invalid candidate braking diagnostics")
            if np.any(predicted_stop_distances < reverse_drift_buffers - 1e-6):
                errors.append(f"{path}: invalid candidate predicted stop distance")
            if np.any((yield_active != 0) & (yield_active != 1)):
                errors.append(f"{path}: invalid candidate yield state")
            held_zero_with_positive_safe = (
                (selected_speeds <= 1e-6)
                & (emergency_stops == 0)
                & (safe_counts > 1)
                & (yield_active != 0)
            )
            max_held_zero_frames = _max_consecutive_true(held_zero_with_positive_safe)
            if max_held_zero_frames > max_yield_zero_positive_frames:
                errors.append(
                    f"{path}: yield policy held zero after positive safe candidate for "
                    f"{max_held_zero_frames} frames > {max_yield_zero_positive_frames}"
                )
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
    parser.add_argument(
        "--max-negative-path-speed-ratio",
        "--max-negative-actual-x-ratio",
        dest="max_negative_path_speed_ratio",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-path-backtrack-distance",
        "--max-backtrack-distance",
        dest="max_path_backtrack_distance",
        type=float,
        default=0.3,
    )
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
            max_negative_path_speed_ratio=args.max_negative_path_speed_ratio,
            max_path_backtrack_distance=args.max_path_backtrack_distance,
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
