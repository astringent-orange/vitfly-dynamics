"""Frame-index selection shared by the Flightmare inference path."""

import cv2
import numpy as np


def build_frame_stack(history, current, frame_offset):
    """Build ``[history, current]`` channels from a prior-frame sequence.

    ``history`` contains only valid frames before ``current`` in chronological
    order.  ``None`` is returned until the requested historical frame exists.
    """
    if frame_offset not in (0, 1, 2):
        raise ValueError(f'frame_offset must be 0, 1, or 2, got {frame_offset}')
    if current is None:
        return None
    if frame_offset == 0:
        selected = [current]
    else:
        if len(history) < frame_offset:
            return None
        selected = [history[-frame_offset], current]
    resized = [cv2.resize(frame, (90, 60)).astype(np.float32) for frame in selected]
    return np.stack(resized, axis=0)
