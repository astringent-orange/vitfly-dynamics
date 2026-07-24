#!/usr/bin/python3
"""Validate the latest dynamic A* batch and retain only accepted trajectories."""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

from validate_dataset import (
    _actual_motion_metrics,
    _path_motion_metrics,
    _trajectory_path,
    validate_trajectory,
)


HARD_VALIDATION_OPTIONS = {
    "require_env_fields": True,
    # Training data must satisfy the same safety and motion-quality gates as
    # the standalone validator, rather than merely being collision-free.
    "max_post_goal_rows": 0,
    "max_negative_xcmd_rows": 0,
    "max_low_speed_ratio": 0.15,
    "max_collision_rows": 0,
    "min_nearest_margin": 0.0,
    "max_negative_path_speed_ratio": 0.02,
    "max_path_backtrack_distance": 0.3,
    "max_path_cross_track_error": 0.8,
    "max_applied_speed_accel": 3.5,
    "max_yield_zero_positive_frames": 2,
}

WARNING_VALIDATION_OPTIONS = {
    "require_env_fields": True,
    # Advisory thresholds are intentionally tighter than the training gate;
    # they remain in collection_summary for diagnosing borderline rollouts.
    "max_low_speed_ratio": 0.10,
    "min_nearest_margin": 0.3,
    "max_negative_path_speed_ratio": 0.01,
    "max_path_backtrack_distance": 0.15,
    "max_path_cross_track_error": 0.5,
}

SUMMARY_FILENAME = "collection_summary.json"


def _rollout_key(name):
    match = re.fullmatch(r"rollout_(\d+)", str(name))
    return int(match.group(1)) if match else sys.maxsize


def _latest_trajectory_folders(dataset_dir, count):
    folders = [path for path in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(path)]
    folders.sort(key=os.path.getmtime)
    if count > len(folders):
        raise ValueError(f"requested {count} trajectories but only found {len(folders)} under {dataset_dir}")
    return folders[-count:]


def _load_rows(folder):
    csv_path = os.path.join(folder, "data.csv")
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="") as stream:
        return list(csv.DictReader(stream))


def _float_metric(rows, column, default=np.nan, reducer=min):
    try:
        values = [float(row[column]) for row in rows if row.get(column, "") != ""]
        return float(reducer(values)) if values else float(default)
    except (KeyError, TypeError, ValueError):
        return float(default)


def _path_metrics(rows):
    try:
        strategies = {
            row.get("expert_strategy", "").strip().lower()
            for row in rows
            if row.get("expert_strategy", "").strip()
        }
        if strategies == {"vitfly_original"}:
            return _actual_motion_metrics(rows)
        path, error = _trajectory_path(rows)
        if path is None:
            return np.nan, np.nan
        negative_ratio, backtrack, _, _ = _path_motion_metrics(rows, path)
        return negative_ratio, backtrack
    except (KeyError, TypeError, ValueError):
        return np.nan, np.nan


def _strip_folder_prefix(errors, folder):
    prefix = f"{folder}: "
    return [error[len(prefix):] if error.startswith(prefix) else error for error in errors]


def curate_dataset(dataset_dir, evaluation_path, latest=None):
    with open(evaluation_path) as stream:
        evaluation = yaml.safe_load(stream) or {}
    rollout_names = sorted((name for name in evaluation if _rollout_key(name) != sys.maxsize), key=_rollout_key)
    if not rollout_names:
        raise ValueError(f"no rollout_N entries found in {evaluation_path}")
    if latest is None:
        latest = len(rollout_names)
    if latest != len(rollout_names):
        raise ValueError("--latest must equal the number of rollout entries in the evaluation file")

    folders = _latest_trajectory_folders(dataset_dir, latest)
    records = []
    for rollout, folder in zip(rollout_names, folders):
        rows = _load_rows(folder)
        evaluator = evaluation[rollout] or {}
        hard_errors, _, _, _ = validate_trajectory(folder, **HARD_VALIDATION_OPTIONS)
        warning_errors, _, _, _ = validate_trajectory(folder, **WARNING_VALIDATION_OPTIONS)
        hard_reasons = _strip_folder_prefix(hard_errors, folder)
        success = bool(evaluator.get("Success", False))
        crashes = int(evaluator.get("number_crashes", 0))
        if not success:
            hard_reasons.append("evaluator Success is false")
        if crashes != 0:
            hard_reasons.append(f"evaluator number_crashes={crashes}")

        warning_set = set(_strip_folder_prefix(warning_errors, folder))
        warning_set.difference_update(hard_reasons)
        negative_ratio, backtrack = _path_metrics(rows)
        record = {
            "status": "accepted" if not hard_reasons else "rejected",
            "rollout": rollout,
            "trajectory_dir": os.path.basename(folder),
            "env_level": rows[0].get("env_level", "") if rows else "",
            "env_folder": rows[0].get("env_folder", "") if rows else "",
            "env_seed": rows[0].get("env_seed", "") if rows else "",
            "dynamic_phase_seed": rows[0].get("dynamic_phase_seed", "") if rows else "",
            "expert_strategy": rows[0].get("expert_strategy", "astar_dynamic") if rows else "",
            "row_count": len(rows),
            "png_count": len([path for path in glob.glob(os.path.join(folder, "*.png")) if not path.endswith("_rgb.png")]),
            "evaluator_success": int(success),
            "evaluator_crashes": crashes,
            "min_obstacle_margin": _float_metric(rows, "nearest_obstacle_margin", reducer=min),
            "negative_path_speed_ratio": negative_ratio,
            "max_path_backtrack_m": backtrack,
            "max_cross_track_error_m": _float_metric(rows, "path_cross_track_error", reducer=max),
            "hard_reasons": "; ".join(hard_reasons),
            "quality_warnings": "; ".join(sorted(warning_set)),
        }
        records.append(record)
    return records


def _load_summary(path):
    if not os.path.isfile(path):
        return {
            "schema_version": 1,
            "collection_runs": 0,
            "trajectories_total": 0,
            "accepted_trajectories": 0,
            "rejected_trajectories": 0,
            "retained_trajectories": 0,
        }
    with open(path) as stream:
        return json.load(stream)


def apply_curation(dataset_dir, records):
    summary_path = os.path.join(dataset_dir, SUMMARY_FILENAME)
    summary = _load_summary(summary_path)
    accepted = sum(record["status"] == "accepted" for record in records)
    rejected_records = [record for record in records if record["status"] == "rejected"]
    for record in rejected_records:
        folder = os.path.join(dataset_dir, record["trajectory_dir"])
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"rejected trajectory is missing: {folder}")
    for record in rejected_records:
        shutil.rmtree(os.path.join(dataset_dir, record["trajectory_dir"]))
    summary.update({
        "schema_version": 1,
        "collection_runs": int(summary.get("collection_runs", 0)) + 1,
        "trajectories_total": int(summary.get("trajectories_total", 0)) + len(records),
        "accepted_trajectories": int(summary.get("accepted_trajectories", 0)) + accepted,
        "rejected_trajectories": int(summary.get("rejected_trajectories", 0)) + len(rejected_records),
        "retained_trajectories": int(summary.get("retained_trajectories", 0)) + accepted,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    with open(summary_path, "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Raw train_set directory containing trajectory folders.")
    parser.add_argument("--evaluation", default="evaluation.yaml", help="Evaluation YAML for the current batch.")
    parser.add_argument("--latest", type=int, help="Number of latest trajectories to associate with evaluation rollouts.")
    parser.add_argument("--apply", action="store_true", help="Delete rejected trajectories and update collection_summary.json.")
    args = parser.parse_args()

    try:
        records = curate_dataset(args.dataset_dir, args.evaluation, latest=args.latest)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    accepted = sum(record["status"] == "accepted" for record in records)
    rejected = len(records) - accepted
    for record in records:
        detail = record["hard_reasons"] or record["quality_warnings"] or "clean"
        print(f"[CURATE_DATASET] {record['rollout']} {record['status']}: {detail}")
    if args.apply:
        summary = apply_curation(args.dataset_dir, records)
        print(f"[CURATE_DATASET] Updated {os.path.join(args.dataset_dir, SUMMARY_FILENAME)}")
        print(f"[CURATE_DATASET] cumulative={summary}")
    else:
        print("[CURATE_DATASET] Dry run only; pass --apply to delete rejected trajectories.")
    print(f"[CURATE_DATASET] accepted={accepted} rejected={rejected} total={len(records)}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
