"""Trajectory-folder data loading for the STEP2 single- and two-frame models."""

import csv
import glob
import os
from os.path import join as opj
import random

import cv2
import numpy as np
import torch


TELEMETRY_REQUIRED_COLUMNS = (
    'timestamp', 'desired_vel',
    'quat_1', 'quat_2', 'quat_3', 'quat_4',
    'velcmd_x', 'velcmd_y', 'velcmd_z', 'is_collide',
)


def _parse_collision_flag(value):
    value = str(value).strip().lower()
    if value in ('0', 'false', 'f', 'no'):
        return False
    if value in ('1', 'true', 't', 'yes'):
        return True
    raise ValueError(f'invalid is_collide value {value!r}')


def _finite_float(value, column, folder):
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f'{folder}: non-finite {column}')
    return parsed


def _trajectory_folders(data_dir):
    """Return every trajectory directory in an accepted-only dataset root."""
    dataset_root = os.path.realpath(data_dir)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f'[TRAJECTORY_DATALOADER] dataset directory is missing: {data_dir}')
    folders = [
        os.path.realpath(path) for path in glob.glob(opj(dataset_root, '*'))
        if os.path.isdir(path)
    ]
    folders.sort()
    if not folders:
        raise ValueError(f'[TRAJECTORY_DATALOADER] no trajectory folders in {data_dir}')
    return folders


def _split_trajectory_folders(folders, val_split, short, seed, train_val_dirs):
    available = set(folders)
    if train_val_dirs is not None:
        train_dirs = [os.path.realpath(str(path)) for path in train_val_dirs[0]]
        val_dirs = [os.path.realpath(str(path)) for path in train_val_dirs[1]]
        requested = train_dirs + val_dirs
        if len(set(requested)) != len(requested):
            raise ValueError('[TRAJECTORY_DATALOADER] train/validation trajectory lists overlap')
        unknown = [path for path in requested if path not in available]
        if unknown:
            raise ValueError(
                '[TRAJECTORY_DATALOADER] saved split contains directories missing from the dataset: '
                + ', '.join(unknown)
            )
    else:
        folders = list(folders)
        random.Random(seed).shuffle(folders)
        if short > 0:
            if short > len(folders):
                raise ValueError(
                    f'[TRAJECTORY_DATALOADER] short={short} exceeds trajectory count={len(folders)}'
                )
            folders = folders[:short]
        num_val = int(val_split * len(folders))
        val_dirs = folders[:num_val]
        train_dirs = folders[num_val:]

    if not train_dirs or not val_dirs:
        raise ValueError(
            '[TRAJECTORY_DATALOADER] train and validation splits must both contain at least one trajectory'
        )
    return train_dirs, val_dirs


def _load_trajectory(folder, num_frames, frame_delta_s):
    if num_frames not in (1, 2):
        raise ValueError(f'[TRAJECTORY_DATALOADER] unsupported num_frames={num_frames}; expected 1 or 2')
    if frame_delta_s <= 0:
        raise ValueError('[TRAJECTORY_DATALOADER] frame_delta_s must be positive')

    csv_path = opj(folder, 'data.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'{folder}: missing data.csv')
    with open(csv_path, newline='') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = set(TELEMETRY_REQUIRED_COLUMNS) - fields
        if missing:
            raise ValueError(f'{folder}: data.csv missing columns {sorted(missing)}')
        metadata = []
        for row in reader:
            timestamp = _finite_float(row['timestamp'], 'timestamp', folder)
            desired_vel = _finite_float(row['desired_vel'], 'desired_vel', folder)
            if desired_vel <= 0:
                raise ValueError(f'{folder}: desired_vel must be positive')
            quaternion = np.asarray([
                _finite_float(row['quat_1'], 'quat_1', folder),
                _finite_float(row['quat_2'], 'quat_2', folder),
                _finite_float(row['quat_3'], 'quat_3', folder),
                _finite_float(row['quat_4'], 'quat_4', folder),
            ], dtype=np.float32)
            velocity_command = np.asarray([
                _finite_float(row['velcmd_x'], 'velcmd_x', folder),
                _finite_float(row['velcmd_y'], 'velcmd_y', folder),
                _finite_float(row['velcmd_z'], 'velcmd_z', folder),
            ], dtype=np.float32)
            metadata.append((timestamp, desired_vel, quaternion, velocity_command,
                             _parse_collision_flag(row['is_collide'])))

    image_files = [
        path for path in glob.glob(opj(folder, '*.png'))
        if not os.path.basename(path).endswith('_rgb.png')
    ]
    images = []
    for path in image_files:
        try:
            timestamp = _finite_float(os.path.splitext(os.path.basename(path))[0], 'PNG timestamp', folder)
        except ValueError as exc:
            raise ValueError(f'{folder}: invalid depth PNG name {path}') from exc
        images.append((timestamp, path))

    metadata.sort(key=lambda item: item[0])
    images.sort(key=lambda item: item[0])
    if not metadata or len(metadata) != len(images):
        raise ValueError(f'{folder}: PNG/CSV count mismatch ({len(images)} PNG, {len(metadata)} CSV rows)')

    frames = []
    for index, (meta, image) in enumerate(zip(metadata, images)):
        if index and (meta[0] - metadata[index - 1][0] <= 1e-6 or image[0] - images[index - 1][0] <= 1e-6):
            raise ValueError(f'{folder}: duplicate or non-increasing timestamp')
        if abs(meta[0] - image[0]) > 1e-3:
            raise ValueError(f'{folder}: PNG/CSV timestamp mismatch ({image[0]:.6f} vs {meta[0]:.6f})')
        depth_image = cv2.imread(image[1], cv2.IMREAD_GRAYSCALE)
        if depth_image is None:
            raise ValueError(f'{folder}: cannot read depth image {image[1]}')
        depth_image = cv2.resize(depth_image, (90, 60)).astype(np.float32) / 255.0
        frames.append((meta[0], depth_image, meta[1], meta[2], meta[3], meta[4]))

    samples, desired_vels, quaternions, labels = [], [], [], []
    for current_index, current in enumerate(frames):
        current_time, current_image, desired_vel, quaternion, velocity_command, collided = current
        if collided:
            continue
        if num_frames == 1:
            image_stack = current_image[np.newaxis, :, :]
        else:
            history_index = next((
                candidate_index for candidate_index in range(current_index - 1, -1, -1)
                if frames[candidate_index][0] <= current_time - frame_delta_s
                and not frames[candidate_index][5]
            ), None)
            if history_index is None:
                continue
            image_stack = np.stack((frames[history_index][1], current_image), axis=0)
        samples.append(image_stack)
        desired_vels.append(desired_vel)
        quaternions.append(quaternion)
        labels.append(velocity_command / desired_vel)

    if not samples:
        return None
    return (
        np.asarray(samples, dtype=np.float32),
        np.asarray(desired_vels, dtype=np.float32),
        np.asarray(quaternions, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def _load_split(folders, num_frames, frame_delta_s, split_name):
    images, desired_vels, quaternions, labels, lengths, loaded_dirs, skipped = [], [], [], [], [], [], []
    for folder in folders:
        try:
            trajectory = _load_trajectory(folder, num_frames, frame_delta_s)
        except (OSError, ValueError) as exc:
            skipped.append(str(exc))
            continue
        if trajectory is None:
            skipped.append(f'{folder}: no valid {num_frames}-frame samples')
            continue
        trajectory_images, trajectory_desvel, trajectory_quat, trajectory_labels = trajectory
        images.append(trajectory_images)
        desired_vels.append(trajectory_desvel)
        quaternions.append(trajectory_quat)
        labels.append(trajectory_labels)
        lengths.append(trajectory_images.shape[0])
        loaded_dirs.append(folder)
    if not images:
        raise ValueError(
            f'[TRAJECTORY_DATALOADER] no valid {split_name} trajectories after validation; '
            + '; '.join(skipped)
        )
    return (
        np.concatenate(images, axis=0),
        np.concatenate(desired_vels, axis=0),
        np.concatenate(quaternions, axis=0),
        np.concatenate(labels, axis=0),
        np.asarray(lengths, dtype=np.int32),
    ), skipped, loaded_dirs


def trajectory_dataloader(data_dir, num_frames, frame_delta_s=0.10, val_split=0.2,
                          short=0, seed=None, train_val_dirs=None):
    """Load a dataset root containing only accepted trajectory directories."""
    folders = _trajectory_folders(data_dir)
    train_dirs, val_dirs = _split_trajectory_folders(folders, val_split, short, seed, train_val_dirs)
    train_requested, val_requested = len(train_dirs), len(val_dirs)
    train_data, train_skipped, train_dirs = _load_split(train_dirs, num_frames, frame_delta_s, 'training')
    val_data, val_skipped, val_dirs = _load_split(val_dirs, num_frames, frame_delta_s, 'validation')
    stats = {
        'dataset_trajectories': len(folders),
        'train_requested_trajectories': train_requested,
        'val_requested_trajectories': val_requested,
        'train_loaded_trajectories': len(train_data[-1]),
        'val_loaded_trajectories': len(val_data[-1]),
        'train_samples': int(train_data[0].shape[0]),
        'val_samples': int(val_data[0].shape[0]),
        'train_skipped': train_skipped,
        'val_skipped': val_skipped,
    }
    return train_data, val_data, (train_dirs, val_dirs), stats


def preload(items, device='cpu'):
    return [torch.from_numpy(item).to(device).float() for item in items]
