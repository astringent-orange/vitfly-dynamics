import csv
import os
import tempfile
import unittest

import cv2
import numpy as np

try:
    from .dataloading import manifest_dataloader
except ImportError:
    from dataloading import manifest_dataloader


FIELDS = [
    'timestamp', 'desired_vel',
    'quat_1', 'quat_2', 'quat_3', 'quat_4',
    'velcmd_x', 'velcmd_y', 'velcmd_z', 'is_collide',
]


def _write_trajectory(root, name, pixel_offset=0, collision_index=None):
    folder = os.path.join(root, name)
    os.makedirs(folder)
    timestamps = [1000.000, 1000.040, 1000.080, 1000.120, 1000.160]
    with open(os.path.join(folder, 'data.csv'), 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            writer.writerow({
                'timestamp': f'{timestamp:.3f}', 'desired_vel': '4.0',
                'quat_1': '1.0', 'quat_2': '0.0', 'quat_3': '0.0', 'quat_4': '0.0',
                'velcmd_x': str(1.0 + index), 'velcmd_y': '0.0', 'velcmd_z': '0.0',
                'is_collide': int(index == collision_index),
            })
            image = np.full((12, 18), pixel_offset + index, dtype=np.uint8)
            cv2.imwrite(os.path.join(folder, f'{timestamp:.3f}.png'), image)
    return name


class ManifestDataloaderTest(unittest.TestCase):
    def test_two_frame_order_manifest_filter_and_trajectory_split(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            accepted = [_write_trajectory(dataset_dir, f'traj_{index}', pixel_offset=index * 10)
                        for index in range(4)]
            rejected = _write_trajectory(dataset_dir, 'rejected', pixel_offset=100)
            with open(os.path.join(dataset_dir, 'accepted_manifest.csv'), 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['status', 'trajectory_dir'])
                writer.writeheader()
                for name in accepted:
                    writer.writerow({'status': 'accepted', 'trajectory_dir': name})
                writer.writerow({'status': 'rejected', 'trajectory_dir': rejected})

            train, val, (train_dirs, val_dirs), stats = manifest_dataloader(
                dataset_dir, num_frames=2, frame_delta_s=0.10, val_split=0.25, seed=7
            )
            train_images, _, _, train_labels, train_lengths = train
            val_images, _, _, _, val_lengths = val
            self.assertEqual(train_images.shape[1:], (2, 60, 90))
            self.assertEqual(train_images.shape[0] + val_images.shape[0], 8)
            self.assertTrue(np.all(train_lengths == 2))
            self.assertTrue(np.all(val_lengths == 2))
            self.assertFalse(set(train_dirs) & set(val_dirs))
            self.assertEqual(stats['accepted_trajectories'], 4)
            self.assertTrue(np.allclose(train_labels[:, 0] * 4.0 % 1.0, 0.0))

            first_pair = train_images[0]
            self.assertLess(first_pair[0].mean(), first_pair[1].mean())

    def test_collision_current_frame_is_not_used(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            names = [_write_trajectory(dataset_dir, f'traj_{index}', collision_index=3)
                     for index in range(4)]
            with open(os.path.join(dataset_dir, 'accepted_manifest.csv'), 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['status', 'trajectory_dir'])
                writer.writeheader()
                for name in names:
                    writer.writerow({'status': 'accepted', 'trajectory_dir': name})

            train, val, _, _ = manifest_dataloader(
                dataset_dir, num_frames=2, frame_delta_s=0.10, val_split=0.25, seed=7
            )
            self.assertEqual(train[0].shape[0] + val[0].shape[0], 4)


if __name__ == '__main__':
    unittest.main()
