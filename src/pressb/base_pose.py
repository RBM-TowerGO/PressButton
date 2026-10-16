"""Fixed robot root placement; yaw is counterclockwise about world +Z."""
import numpy as np


def root_transform(xyz, yaw_rad=0.):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(yaw_rad):
        raise ValueError("Finite XYZ and yaw are required")
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    transform[:3, 3] = xyz
    return transform


def world_target_in_root(world_from_root, xyz, rotation):
    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = xyz
    return np.linalg.inv(world_from_root) @ target
