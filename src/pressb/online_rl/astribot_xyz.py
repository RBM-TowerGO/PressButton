"""Deterministic bounded Cartesian residual execution for the Astribot arm."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from pressb.astribot_kinematics import AstribotKinematics, UnreachablePose

XYZ_CONTRACT = dict(version=1, frame="astribot_torso_base", tcp="closed_hand_press_tcp",
                    action_dim=21, critic="base56_normalized_residual21",
                    max_correction_rad=.1, max_correction_change_rad=.03,
                    position_tolerance_m=.0005, rotation_tolerance_rad=.01,
                    backtracking=[1., .5, .25, .125, 0.])


def compose_xyz(base, action, tcp, previous, scale=(.01, .01, .01), *, kin=None):
    started = time.perf_counter()
    base = np.asarray(base, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32).reshape(7, 3)
    tcp = np.asarray(tcp, dtype=float)
    previous = np.asarray(previous, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if (base.shape != (7, 8) or tcp.shape != (4, 4) or previous.shape != (8,)
            or scale.shape != (3,) or not all(np.isfinite(v).all() for v in (base, action, tcp, previous, scale))
            or np.any(scale <= 0) or np.any(np.abs(action) > 1 + 1e-6)):
        raise ValueError("Invalid XYZ execution inputs")
    requested = action * scale
    diagnostics = dict(requested_xyz_m=requested.tolist(), factor=0., attempts=0,
                       position_error_m=0., rotation_error_rad=0., actual_xyz_m=np.zeros((7, 3)).tolist())
    result = base.copy()
    if np.any(action):
        kin = kin or AstribotKinematics()
        poses = [kin.tcp(q[:7], tcp) for q in base]
        for factor in XYZ_CONTRACT["backtracking"][:-1]:
            diagnostics["attempts"] += 1
            candidate = base.copy()
            try:
                for i, pose in enumerate(poses):
                    target = pose.copy()
                    target[:3, 3] += factor * requested[i]
                    candidate[i, :7] = kin.solve(target, base[i, :7], tcp)
            except UnreachablePose:
                continue
            correction = candidate[:, :7] - base[:, :7]
            if (np.any(np.abs(correction) > .1) or np.any(np.abs(np.diff(correction, axis=0)) > .03)
                    or np.any(np.abs(candidate[0, :7] - previous[:7]) > np.abs(base[0, :7] - previous[:7]) + .03)
                    or np.any(candidate[:, :7] < kin.lower) or np.any(candidate[:, :7] > kin.upper)):
                continue
            actual = [kin.tcp(q[:7], tcp) for q in candidate]
            delta = np.array([a[:3, 3] - b[:3, 3] for a, b in zip(actual, poses)])
            pe = float(np.max(np.linalg.norm(delta - factor * requested, axis=1)))
            re = max(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude() for a, b in zip(actual, poses))
            if pe > .0005 or re > .01:
                continue
            result = candidate
            diagnostics.update(factor=factor, position_error_m=pe, rotation_error_rad=re, actual_xyz_m=delta.tolist())
            break
    diagnostics.update(seconds=time.perf_counter() - started, zero_request=not bool(np.any(action)))
    return result, diagnostics
