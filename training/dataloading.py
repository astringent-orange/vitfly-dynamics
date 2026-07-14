"""
@authors: A Bhattacharya
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the dataloading routine that was used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import csv
import cv2
import glob, os, time
from os.path import join as opj
import numpy as np
import torch
import random
import getpass
uname = getpass.getuser()

def dataloader(data_dir, val_split=0., short=0, seed=None, train_val_dirs=None):
    cropHeight = 60
    cropWidth = 90

    if train_val_dirs is not None:
        traj_folders = train_val_dirs[0] + train_val_dirs[1]
        val_split = len(train_val_dirs[1]) / len(traj_folders)
    else:
        traj_folders = sorted(glob.glob(opj(data_dir, '*')))
        random.seed(seed)
        random.shuffle(traj_folders)

    if short > 0:
        assert short <= len(traj_folders), f"short={short} is greater than the number of folders={len(traj_folders)}"
        traj_folders = traj_folders[:short]
    desired_vels = []
    traj_ims_full = []
    traj_meta_full = []
    curr_quats = []    

    start_dataloading = time.time()

    skippedImages = 0
    skippedFolders = 0
    collisionImages = 0
    collisionFolders = 0

    for i, traj_folder in enumerate(traj_folders):
        if len(traj_folders)//10 > 0 and i % (len(traj_folders)//10) == 0:
            print(f'[DATALOADER] Loading folder {os.path.basename(traj_folder)}, folder # {i+1}/{len(traj_folders)}, time elapsed {time.time()-start_dataloading:.2f}s')
        im_files = sorted(glob.glob(opj(traj_folder, '*.png')))

        # check for empty folder
        if len(im_files) == 0:
            print(f'[DATALOADER] No images in {os.path.basename(traj_folder)}, skipping')
            continue

        csv_file = 'data.csv'
        # float64 is required to read ros timestamps without rounding
        # NOTE not sure if float64 will break training (torch dtypes)
        traj_meta = np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype=np.float64)[1:]
        traj_meta[:,-1] = np.int32(np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype="bool")[1:,-1])

        # check for collisions in trajectory
        # if traj_meta[:,-1].sum() > 0:
        #     print(f'[DATALOADER] Collision in {os.path.basename(traj_folder)}, skipping')
        #     collisionFolders += 1
        #     collisionImages += int(len(traj_meta[:,0]))
        #     continue

        # check for nan in metadata
        if np.isnan(traj_meta).any():
            print(f'[DATALOADER] NaN in {os.path.basename(traj_folder)}, skipping')
            traj_meta = traj_meta[:,:-1]
            

        # read png files and scale them by 255.0 to recover normalized (0, 1) range
        # for npy files, manually normalize them by a set value (0.09 for "old" dataset)
        traj_ims = np.asarray([cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) for im_file in im_files], dtype=np.float32) / 255.0

        # check for mismatch in number of images and telemetry entries
        if traj_ims.shape[0] != traj_meta.shape[0]:

            # usually the last image may not have a corresponding line of telemetry, so check specifically for that case
            last_im_timestamp = os.path.basename(im_files[-1])[:-4]
            if float(last_im_timestamp) > traj_meta[-1, 1]:
                traj_ims = traj_ims[:-1]
                print(f'[DATALOADER] Extra image found at end of data, cutting it from {os.path.basename(traj_folder)}')
            if traj_ims.shape[0] != traj_meta.shape[0]:
                print(f'[DATALOADER] Number of images and telemetry still do not match in {os.path.basename(traj_folder)}, skipping')
                skippedFolders += 1
                skippedImages += int(len(traj_meta[:,0]))
                continue
        temp = [cv2.resize(img, (cropWidth, cropHeight)) for img in traj_ims]

        traj_ims = np.array(temp)
        for ii in range(traj_meta.shape[0]):
            desired_vels.append(traj_meta[ii, 2])
            q = traj_meta[ii, 3:7]
            rmat = q 
            curr_quats.append(rmat)
        try:
            traj_ims_full.append(traj_ims)
            traj_meta_full.append(traj_meta)
        except:
            print(f'[DATALOADER] {traj_ims.shape}')
            print(f"[DATALOADER] Suspected empty image, folder {os.path.basename(traj_folder)}")

    print(skippedFolders, skippedImages)
    print(collisionFolders, collisionImages)

    print("[ANALYZER] Analyzing the data....")
    traj_lengths = np.array([traj_ims.shape[0] for traj_ims in traj_ims_full])
    traj_ims_full = np.concatenate(traj_ims_full).reshape(-1, cropHeight, cropWidth)
    traj_meta_full = np.concatenate(traj_meta_full).reshape(-1, traj_meta.shape[-1])
    desired_vels = np.array(desired_vels)
    curr_quats = np.array(curr_quats)


    #Col: mean, std
    #row: ct. brx/y/z
    stats_ctbr = np.zeros((4, 2))
    stats_ctbr[0, :] = np.mean(traj_meta_full[:, 16]), np.std(traj_meta_full[:, 16])
    stats_ctbr[1, :] = np.mean(traj_meta_full[:, 17]), np.std(traj_meta_full[:, 17])
    stats_ctbr[2, :] = np.mean(traj_meta_full[:, 18]), np.std(traj_meta_full[:, 18])
    stats_ctbr[3, :] = np.mean(traj_meta_full[:, 19]), np.std(traj_meta_full[:, 19])

    traj_meta_full[:, 16] = (traj_meta_full[:, 16] - stats_ctbr[0, 0]) / (2 * stats_ctbr[0, 1])
    traj_meta_full[:, 17] = (traj_meta_full[:, 17] - stats_ctbr[1, 0]) / (2 * stats_ctbr[1, 1])
    traj_meta_full[:, 18] = (traj_meta_full[:, 18] - stats_ctbr[2, 0]) / (2 * stats_ctbr[2, 1])
    traj_meta_full[:, 19] = (traj_meta_full[:, 19] - stats_ctbr[3, 0]) / (2 * stats_ctbr[3, 1])

    
    curr_ctbr = traj_meta_full[:, 16:20]

    #Col: mean, std
    #row: ct. brx/y/z
    stats_ctbr = np.zeros((4, 2))
    stats_ctbr[0, :] = np.mean(traj_meta[:, 16]), np.std(traj_meta[:, 16])
    stats_ctbr[1, :] = np.mean(traj_meta[:, 17]), np.std(traj_meta[:, 17])
    stats_ctbr[2, :] = np.mean(traj_meta[:, 18]), np.std(traj_meta[:, 18])
    stats_ctbr[3, :] = np.mean(traj_meta[:, 19]), np.std(traj_meta[:, 19])

    traj_meta[:, 16] = (traj_meta[:, 16] - stats_ctbr[0, 0]) / (2 * stats_ctbr[0, 1])
    traj_meta[:, 17] = (traj_meta[:, 17] - stats_ctbr[1, 0]) / (2 * stats_ctbr[1, 1])
    traj_meta[:, 18] = (traj_meta[:, 18] - stats_ctbr[2, 0]) / (2 * stats_ctbr[2, 1])
    traj_meta[:, 19] = (traj_meta[:, 19] - stats_ctbr[3, 0]) / (2 * stats_ctbr[3, 1])

    # make train-val split (relies on earlier shuffle of traj_folders to randomize selection)
    num_val_trajs = int(val_split * len(traj_lengths))
    val_idx = np.sum(traj_lengths[:num_val_trajs], dtype=np.int32)
    traj_meta_val = traj_meta_full[:val_idx]
    traj_meta_train = traj_meta_full[val_idx:]
    traj_ims_val = traj_ims_full[:val_idx]
    traj_ims_train = traj_ims_full[val_idx:]
    traj_lengths_val = traj_lengths[:num_val_trajs]
    traj_lengths_train = traj_lengths[num_val_trajs:]
    desired_vels_val = desired_vels[:val_idx]
    desired_vels_train = desired_vels[val_idx:]
    #curr_vels_val = curr_vels[:val_idx]
    #curr_vels_train = curr_vels[val_idx:]
    curr_quats_val = curr_quats[:val_idx]
    curr_quats_train = curr_quats[val_idx:]
    curr_ctbr_val = curr_ctbr[:val_idx]
    curr_ctbr_train = curr_ctbr[val_idx:]

    # Note, we return the is_png=1 flag since it indicates old vs new datasets, which indicates how to parse the metadata
    # We also return the traj_folder names for train and val sets, so that they can be saved and later used to specifically generate evaluate plots on each set
    return (traj_meta_train, traj_ims_train, traj_lengths_train, desired_vels_train, curr_quats_train, curr_ctbr_train), (traj_meta_val, traj_ims_val, traj_lengths_val, desired_vels_val, curr_quats_val, curr_ctbr_val), 1, (traj_folders[num_val_trajs:], traj_folders[:num_val_trajs])


MANIFEST_REQUIRED_COLUMNS = ('status', 'trajectory_dir')
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


def _accepted_trajectory_folders(data_dir, manifest_filename):
    manifest_path = opj(data_dir, manifest_filename)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f'[MANIFEST_DATALOADER] Missing required accepted manifest: {manifest_path}'
        )

    with open(manifest_path, newline='') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = set(MANIFEST_REQUIRED_COLUMNS) - fields
        if missing:
            raise ValueError(
                f'[MANIFEST_DATALOADER] {manifest_path} is missing columns: {sorted(missing)}'
            )
        records = list(reader)

    dataset_root = os.path.realpath(data_dir)
    folders = []
    seen = set()
    for record in records:
        if str(record.get('status', '')).strip().lower() != 'accepted':
            continue
        trajectory_dir = str(record.get('trajectory_dir', '')).strip()
        if not trajectory_dir:
            raise ValueError(
                f'[MANIFEST_DATALOADER] accepted record without trajectory_dir in {manifest_path}'
            )
        folder = os.path.realpath(opj(dataset_root, trajectory_dir))
        if os.path.commonpath((dataset_root, folder)) != dataset_root:
            raise ValueError(
                f'[MANIFEST_DATALOADER] trajectory_dir escapes dataset root: {trajectory_dir}'
            )
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f'[MANIFEST_DATALOADER] accepted trajectory is missing: {folder}'
            )
        if folder not in seen:
            folders.append(folder)
            seen.add(folder)

    if not folders:
        raise ValueError(
            f'[MANIFEST_DATALOADER] no status=accepted trajectories in {manifest_path}'
        )
    return folders


def _split_manifest_folders(folders, val_split, short, seed, train_val_dirs):
    accepted = set(folders)
    if train_val_dirs is not None:
        train_dirs = [os.path.realpath(str(path)) for path in train_val_dirs[0]]
        val_dirs = [os.path.realpath(str(path)) for path in train_val_dirs[1]]
        requested = train_dirs + val_dirs
        if len(set(requested)) != len(requested):
            raise ValueError('[MANIFEST_DATALOADER] train/validation trajectory lists overlap')
        unknown = [path for path in requested if path not in accepted]
        if unknown:
            raise ValueError(
                '[MANIFEST_DATALOADER] saved split contains trajectories not accepted by the current manifest: '
                + ', '.join(unknown)
            )
    else:
        folders = list(folders)
        random.Random(seed).shuffle(folders)
        if short > 0:
            if short > len(folders):
                raise ValueError(
                    f'[MANIFEST_DATALOADER] short={short} exceeds accepted trajectory count={len(folders)}'
                )
            folders = folders[:short]
        num_val = int(val_split * len(folders))
        val_dirs = folders[:num_val]
        train_dirs = folders[num_val:]

    if not train_dirs or not val_dirs:
        raise ValueError(
            '[MANIFEST_DATALOADER] train and validation splits must both contain at least one accepted trajectory'
        )
    return train_dirs, val_dirs


def _load_manifest_trajectory(folder, num_frames, frame_delta_s):
    if num_frames not in (1, 2):
        raise ValueError(f'[MANIFEST_DATALOADER] unsupported num_frames={num_frames}; expected 1 or 2')
    if frame_delta_s <= 0:
        raise ValueError('[MANIFEST_DATALOADER] frame_delta_s must be positive')

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
        raise ValueError(
            f'{folder}: PNG/CSV count mismatch ({len(images)} PNG, {len(metadata)} CSV rows)'
        )

    frames = []
    for index, (meta, image) in enumerate(zip(metadata, images)):
        if index and (meta[0] - metadata[index - 1][0] <= 1e-6 or
                      image[0] - images[index - 1][0] <= 1e-6):
            raise ValueError(f'{folder}: duplicate or non-increasing timestamp')
        if abs(meta[0] - image[0]) > 1e-3:
            raise ValueError(
                f'{folder}: PNG/CSV timestamp mismatch ({image[0]:.6f} vs {meta[0]:.6f})'
            )
        depth_image = cv2.imread(image[1], cv2.IMREAD_GRAYSCALE)
        if depth_image is None:
            raise ValueError(f'{folder}: cannot read depth image {image[1]}')
        depth_image = cv2.resize(depth_image, (90, 60)).astype(np.float32) / 255.0
        frames.append((meta[0], depth_image, meta[1], meta[2], meta[3], meta[4]))

    samples = []
    desired_vels = []
    quaternions = []
    labels = []
    for current_index, current in enumerate(frames):
        current_time, current_image, desired_vel, quaternion, velocity_command, collided = current
        if collided:
            continue
        if num_frames == 1:
            image_stack = current_image[np.newaxis, :, :]
        else:
            history_index = None
            for candidate_index in range(current_index - 1, -1, -1):
                candidate = frames[candidate_index]
                if candidate[0] > current_time - frame_delta_s:
                    continue
                if not candidate[5]:
                    history_index = candidate_index
                    break
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


def _load_manifest_split(folders, num_frames, frame_delta_s, split_name):
    images, desired_vels, quaternions, labels, lengths, loaded_dirs = [], [], [], [], [], []
    skipped = []
    for folder in folders:
        try:
            trajectory = _load_manifest_trajectory(folder, num_frames, frame_delta_s)
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
            f'[MANIFEST_DATALOADER] no valid {split_name} trajectories after validation; '
            + '; '.join(skipped)
        )
    return (
        np.concatenate(images, axis=0),
        np.concatenate(desired_vels, axis=0),
        np.concatenate(quaternions, axis=0),
        np.concatenate(labels, axis=0),
        np.asarray(lengths, dtype=np.int32),
    ), skipped, loaded_dirs


def manifest_dataloader(data_dir, num_frames, frame_delta_s=0.10, val_split=0.2,
                        short=0, seed=None, train_val_dirs=None,
                        manifest_filename='accepted_manifest.csv'):
    """Load accepted STEP1 trajectories as 1-frame or 2-frame samples.

    The dataset directory must contain ``accepted_manifest.csv`` and trajectory
    folders named by its ``trajectory_dir`` entries.  This loader deliberately
    never scans raw collection folders to decide what is trainable.
    """
    accepted_folders = _accepted_trajectory_folders(data_dir, manifest_filename)
    train_dirs, val_dirs = _split_manifest_folders(
        accepted_folders, val_split, short, seed, train_val_dirs
    )
    train_requested = len(train_dirs)
    val_requested = len(val_dirs)
    train_data, train_skipped, train_dirs = _load_manifest_split(
        train_dirs, num_frames, frame_delta_s, 'training'
    )
    val_data, val_skipped, val_dirs = _load_manifest_split(
        val_dirs, num_frames, frame_delta_s, 'validation'
    )
    stats = {
        'accepted_trajectories': len(accepted_folders),
        'train_requested_trajectories': train_requested,
        'val_requested_trajectories': val_requested,
        'train_loaded_trajectories': len(train_data[-1]),
        'val_loaded_trajectories': len(val_data[-1]),
        'train_samples': int(train_data[0].shape[0]),
        'val_samples': int(val_data[0].shape[0]),
        'train_skipped': train_skipped,
        'val_skipped': val_skipped,
        'manifest_filename': manifest_filename,
    }
    return train_data, val_data, (train_dirs, val_dirs), stats

def parse_meta_str(meta_str):

    meta = torch.zeros_like(meta_str)

    meta_str

    return meta


def preload(items, device='cpu'):

    return [torch.from_numpy(item).to(device).float() for item in items]
