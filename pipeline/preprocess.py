from __future__ import annotations

import numpy as np


def grid_average_downsample(points: np.ndarray, grid_step: float) -> np.ndarray:
    """MATLAB pcdownsample(..., 'gridAverage', gridStep) style downsampling."""
    points = np.asarray(points, dtype=float)
    if grid_step <= 0:
        return points.copy()

    origin = points.min(axis=0)
    voxel = np.floor((points - origin) / grid_step).astype(np.int64)
    _, inverse = np.unique(voxel, axis=0, return_inverse=True)

    counts = np.bincount(inverse).astype(float)
    out = np.zeros((counts.size, 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.bincount(inverse, weights=points[:, dim]) / counts
    return out


def center_and_ground(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center by mean position and shift z so the minimum is zero."""
    points = np.asarray(points, dtype=float)
    mean_pos = points.mean(axis=0)
    centered = points - mean_pos
    z_min = float(centered[:, 2].min())
    centered[:, 2] -= z_min
    return centered, mean_pos, z_min


def preprocess_points(
    raw_points: np.ndarray, grid_step: float
) -> tuple[np.ndarray, np.ndarray, float]:
    down = grid_average_downsample(raw_points, grid_step)
    return center_and_ground(down)

