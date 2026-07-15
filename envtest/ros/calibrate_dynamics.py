#!/usr/bin/python3
"""Fit conservative candidate-planner dynamics from dedicated velocity-step logs."""

import argparse
import csv
import glob
import os

import numpy as np
import yaml


DEFAULTS = {
    "control_delay": 0.25,
    "max_accel": 3.0,
    "max_brake_decel": 1.5,
    "reverse_drift_distance": 0.4,
    "cross_track_decay_time": 0.8,
}


def _csv_paths(inputs):
    paths = []
    for value in inputs:
        if os.path.isdir(value):
            paths.extend(glob.glob(os.path.join(value, "**", "data.csv"), recursive=True))
        elif os.path.basename(value) == "data.csv" and os.path.isfile(value):
            paths.append(value)
    return sorted(set(paths))


def _numeric_rows(path):
    required = ("timestamp", "pos_x", "vel_x", "velcmd_x")
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(column not in rows[0] for column in required):
        return None
    try:
        return {
            key: np.asarray([float(row[key]) for row in rows], dtype=float)
            for key in required
        }
    except (KeyError, TypeError, ValueError):
        return None


def _quantile(values, quantile, default, lower, upper):
    if not values:
        return default
    return float(np.clip(np.quantile(np.asarray(values, dtype=float), quantile), lower, upper))


def fit(paths):
    delays, accelerations, decelerations, drifts = [], [], [], []
    used = 0
    for path in paths:
        data = _numeric_rows(path)
        if data is None:
            continue
        t, x, velocity, command = (data[key] for key in ("timestamp", "pos_x", "vel_x", "velcmd_x"))
        if len(t) < 4:
            continue
        dt = np.diff(t)
        valid = dt > 1e-4
        accel = np.zeros_like(dt)
        accel[valid] = np.diff(velocity)[valid] / dt[valid]
        accelerations.extend(accel[(command[1:] > 0.5) & (accel > 0.05)].tolist())

        stop_events = np.where((command[:-1] > 0.5) & (command[1:] <= 0.05))[0] + 1
        for event in stop_events:
            window = np.where((t >= t[event]) & (t <= t[event] + 5.0))[0]
            if len(window) < 2:
                continue
            changed = window[accel[np.maximum(window - 1, 0)] < -0.1]
            if len(changed):
                delays.append(float(t[changed[0]] - t[event]))
            decelerations.extend((-accel[np.maximum(window - 1, 0)][accel[np.maximum(window - 1, 0)] < -0.05]).tolist())
            settled = window[np.abs(velocity[window]) <= 0.1]
            end = settled[0] if len(settled) else window[-1]
            drifts.append(max(0.0, float(x[event] - np.min(x[event : end + 1]))))
            used += 1

    values = {
        "control_delay": _quantile(delays, 0.95, DEFAULTS["control_delay"], 0.05, 0.75),
        "max_accel": _quantile(accelerations, 0.10, DEFAULTS["max_accel"], 0.5, 5.0),
        "max_brake_decel": _quantile(decelerations, 0.10, DEFAULTS["max_brake_decel"], 0.5, 5.0),
        "reverse_drift_distance": _quantile(drifts, 0.95, DEFAULTS["reverse_drift_distance"], 0.0, 2.0),
        "cross_track_decay_time": DEFAULTS["cross_track_decay_time"],
        "calibration_stop_events": int(used),
        "calibration_source_files": len(paths),
    }
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Calibration trajectory folders, dataset roots, or data.csv files.")
    parser.add_argument("--output", default="expert_dynamics.yaml", help="Output YAML path.")
    args = parser.parse_args()
    paths = _csv_paths(args.inputs)
    if not paths:
        parser.error("no data.csv files found")
    values = fit(paths)
    if values["calibration_stop_events"] < 20:
        parser.error("need at least 20 detected forward-to-zero command events")
    with open(args.output, "w") as stream:
        yaml.safe_dump(values, stream, sort_keys=False)
    print(f"[CALIBRATE_DYNAMICS] Wrote {args.output} from {values['calibration_stop_events']} stop events")


if __name__ == "__main__":
    main()
