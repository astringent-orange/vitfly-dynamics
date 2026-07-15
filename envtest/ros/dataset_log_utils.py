#!/usr/bin/python3

import glob
import os


def cleanup_orphan_depth_images(folder, kept_timestamps):
    normalized_timestamps = set()
    for timestamp in kept_timestamps:
        try:
            normalized_timestamps.add(round(float(timestamp), 3))
        except (TypeError, ValueError):
            continue

    removed = []
    for image_path in glob.glob(os.path.join(folder, "*.png")):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        try:
            image_timestamp = round(float(stem), 3)
        except ValueError:
            continue
        if image_timestamp in normalized_timestamps:
            continue
        try:
            os.remove(image_path)
            removed.append(image_path)
        except OSError:
            continue
    return removed
