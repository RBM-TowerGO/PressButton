"""Cartesian planning and strict S1B SDK kinematics adaptation (metres/radians)."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('invalid_vector')
    return result


def transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1)):
        raise ValueError('invalid_transform')
    return t


def pose_matrix(pose):
    p = vector(pose, 7)
    if abs(np.linalg.norm(p[3:]) - 1) > 1e-3:
        raise ValueError('invalid_quaternion')
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(p[3:]).as_matrix()
    t[:3, 3] = p[:3]
    return t


def pose_list(t):
    t = transform(t)
    return [*t[:3, 3], *Rotation.from_matrix(t[:3, :3]).as_quat()]


class Plane:
    def __init__(self, config):
        self.origin = vector(config['origin_m'], 3)
        self.basis = np.column_stack([vector(config[k], 3) for k in ('right', 'up', 'forward')])
        # right/up/forward is not necessarily a right-handed ordered basis.
        if not np.allclose(self.basis.T @ self.basis, np.eye(3), atol=1e-6):
            raise ValueError('nonorthonormal_plane')
        if not np.allclose(self.basis[:, 1], [0, 0, 1], atol=1e-6):
            raise ValueError('plane_not_vertical_in_level_chassis')
        self.rotation = transform(config['tcp_orientation'])[:3, :3]
        self.bounds = np.asarray(config['uv_bounds_m'], dtype=float)
        if (self.bounds.shape != (2, 2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError('invalid_plane_bounds')

    def target(self, uv, d=0.):
        uv = vector(uv, 2)
        if np.any(uv < self.bounds[:, 0]) or np.any(uv > self.bounds[:, 1]) or not np.isfinite(d):
            raise ValueError('outside_plane_workspace')
        t = np.eye(4)
        t[:3, :3] = self.rotation
        t[:3, 3] = self.origin + self.basis @ np.r_[uv, d]
        return t

    def coordinates(self, t):
        return self.basis.T @ (transform(t)[:3, 3] - self.origin)


class SDKKinematics:
    """Only accepts the verified (flag, named joints) SDK return contract."""
    arm = 'astribot_arm_right'
    torso = 'astribot_torso'

    def __init__(self, client, torso, left, tcp_in_wrist):
        self.client = client
        self.fixed_torso = vector(torso, 4)
        self.left = vector(left, 7)
        self.offset = transform(tcp_in_wrist)
        result = client.get_forward_kinematics([self.torso], [self.fixed_torso.tolist()])
        self.torso_pose = pose_list(pose_matrix(result[self.torso]))

    def fk(self, q):
        result = self.client.get_forward_kinematics(
            [self.torso, self.arm], [self.fixed_torso.tolist(), vector(q, 7).tolist()])
        return pose_matrix(result[self.arm]) @ self.offset

    def solve(self, target, seed):
        valid, rows = self.client.get_inverse_kinematics(
            [self.torso, self.arm], [self.torso_pose, pose_list(target @ np.linalg.inv(self.offset))])
        if not valid or not isinstance(rows, dict):
            raise ValueError('ik_failed')
        if np.max(np.abs(vector(rows[self.torso], 4) - self.fixed_torso)) > 1e-5:
            raise ValueError('ik_changed_torso')
        # The SDK has no seed parameter. The planner rejects discontinuous solutions.
        return vector(rows[self.arm], 7)

    def clearance(self, q):
        distance, *_ = self.client.get_self_closest_point(
            self.fixed_torso.tolist(), self.left.tolist(), vector(q, 7).tolist())
        if not np.isfinite(distance):
            raise ValueError('invalid_collision_distance')
        return float(distance)


@dataclass
class Point:
    q: np.ndarray
    target: np.ndarray


class Planner:
    def __init__(self, plane, kinematics, config):
        self.plane, self.kin, self.c = plane, kinematics, config

    def check_pose(self, actual, target):
        actual, target = transform(actual), transform(target)
        error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        angle = Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude()
        if error > self.c['position_tolerance_m'] or angle > self.c['rotation_tolerance_rad']:
            raise ValueError('tcp_tracking_error')
        return float(error), float(angle)

    def path(self, start, finish, seed, speed, cancelled=lambda: False):
        start, finish = vector(start, 3), vector(finish, 3)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('invalid_speed')
        distance = np.linalg.norm(finish - start)
        # Quintic derivative peaks at 1.875. No joint-space clipping is permitted.
        count = max(1, int(np.ceil(1.875 * distance / speed * self.c['frequency_hz'])))
        if count > self.c['max_path_points']:
            raise ValueError('path_too_long')
        q = vector(seed, 7)
        points = []
        for i in range(1, count + 1):
            if cancelled():
                raise ValueError('planning_cancelled')
            s = i / count
            x = start + (finish - start) * (10*s**3 - 15*s**4 + 6*s**5)
            target = self.plane.target(x[:2], x[2])
            next_q = vector(self.kin.solve(target, q), 7)
            delta = np.abs(next_q - q)
            if np.any(next_q < self.c['joint_lower']) or np.any(next_q > self.c['joint_upper']):
                raise ValueError('ik_joint_limit')
            if np.any(delta > self.c['max_joint_step_rad']):
                raise ValueError('ik_branch_jump')
            if np.any(delta * self.c['frequency_hz'] > np.asarray(self.c['joint_velocity_rad_s'])):
                raise ValueError('ik_velocity_limit')
            self.check_pose(self.kin.fk(next_q), target)
            clearance = self.kin.clearance(next_q)
            if not np.isfinite(clearance) or clearance < self.c['minimum_self_clearance_m']:
                raise ValueError('self_collision_clearance')
            points.append(Point(next_q, target))
            q = next_q
        return points
