"""Offline-only geometry and group-aware evaluation for the latent probe.

Panel axes are right, up, outward (toward the robot). For the authored
VIB820 panel these are -Y, +Z, -X. All returned errors are millimetres.
No simulator imports, policy changes, or privileged actor inputs live here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

PANEL_AXES = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


def pose_matrix(position, quaternion_wxyz):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    result[:3, 3] = position
    return result


def offsets_mm(point, face_center, axes=PANEL_AXES):
    point, center, axes = map(np.asarray, (point, face_center, axes))
    if point.shape != (3,) or center.shape != (3,) or axes.shape != (3, 3):
        raise ValueError("Expected points (3,) and axes (3,3)")
    if not all(np.isfinite(x).all() for x in (point, center, axes)):
        raise ValueError("Nonfinite geometry")
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
        raise ValueError("Panel axes must be orthonormal")
    return (point - center) @ axes * 1000.


def group_split(groups, seed=20261023):
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("At least five independent groups required")
    order = np.random.default_rng(seed).permutation(unique)
    a, b = int(.6 * len(order)), int(.8 * len(order))
    mapping = {g: split for split, part in enumerate((order[:a], order[a:b], order[b:])) for g in part}
    return np.array([mapping[g] for g in groups], dtype=np.int8)


def near_mask(labels):
    return np.all(np.abs(labels) <= 30., axis=1)


def balanced_accuracy(y, prediction, threshold):
    true = np.where(y < -threshold, -1, np.where(y > threshold, 1, 0))
    pred = np.where(prediction < -threshold, -1, np.where(prediction > threshold, 1, 0))
    return [float(np.mean([np.mean(pred[true[:, axis] == k, axis] == k)
                           for k in np.unique(true[:, axis])])) for axis in range(2)]


def metrics(y, prediction, groups):
    if not len(y):
        return dict(n=0, groups=0)
    error = np.abs(y - prediction)
    group_errors = np.array([error[groups == g].mean(0) for g in np.unique(groups)])
    return dict(n=len(y), groups=len(np.unique(groups)), mae_mm=error.mean(0).tolist(),
                group_macro_mae_mm=group_errors.mean(0).tolist(),
                balanced_accuracy={str(t): balanced_accuracy(y, prediction, t) for t in (1, 2, 4)},
                direction_counts_2mm=[[(y[:, a] < -2).sum().item(),
                                      (np.abs(y[:, a]) <= 2).sum().item(),
                                      (y[:, a] > 2).sum().item()] for a in range(2)])


def paired_improvement(y, baseline, augmented, groups, seed=20261023, draws=2000):
    """Equal group weight; positive differences mean latent helps."""
    unique = np.unique(groups)
    if not len(unique):
        return dict(groups=0)
    delta = np.abs(y - baseline) - np.abs(y - augmented)
    per_group = np.array([delta[groups == g].mean(0) for g in unique])
    rng = np.random.default_rng(seed)
    boots = per_group[rng.integers(len(unique), size=(draws, len(unique)))].mean(1)
    return dict(groups=len(unique), improvement_mm=per_group.mean(0).tolist(),
                ci95_mm=np.quantile(boots, [.025, .975], axis=0).T.tolist(),
                mean_axes_improvement_mm=float(per_group.mean()),
                mean_axes_ci95_mm=np.quantile(boots.mean(1), [.025, .975]).tolist())
