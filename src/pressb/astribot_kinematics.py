"""URDF kinematics and coordinated gripper closure for the fixed-base S1 task.

No simulator imports: the same transforms can be checked against PhysX at runtime.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "vendor/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body_maniskill.urdf"
RIGHT_ARM = [f"astribot_arm_right_joint_{i}" for i in range(1, 8)]
GRIPPER_SIGNS = {"L1": -1, "L2": -1, "L11": 1, "R1": 1, "R2": 1, "R11": -1}


def gripper_targets(angle, side="right"):
    if not np.isfinite(angle) or not 0 <= angle <= 1.57:
        raise ValueError("Gripper angle must be in [0, 1.57] radians")
    return {f"astribot_gripper_{side}_joint_{name}": sign * angle
            for name, sign in GRIPPER_SIGNS.items()}


class UnreachablePose(ValueError):
    """A finite pose could not be reached within the IK tolerances."""


class AstribotKinematics:
    def __init__(self, urdf=URDF):
        self.joints = {}
        self.limits = {}
        for joint in ET.parse(urdf).findall("joint"):
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            axis = joint.find("axis")
            vector = np.fromstring(axis.get("xyz", "1 0 0"), sep=" ") if axis is not None else None
            name = joint.get("name")
            self.joints[joint.find("child").get("link")] = (
                joint.find("parent").get("link"), name, transform, vector)
            limit = joint.find("limit")
            if limit is not None:
                self.limits[name] = [float(limit.get(k)) for k in ("lower", "upper", "velocity")]
        self.lower, self.upper, self.velocity = np.array([self.limits[n] for n in RIGHT_ARM]).T

    def fk(self, link, positions=None, root="astribot_torso_base"):
        positions = positions or {}
        if link == root:
            return np.eye(4)
        parent, name, origin, axis = self.joints[link]
        turn = np.eye(4)
        if axis is not None:
            turn[:3, :3] = Rotation.from_rotvec(axis * positions.get(name, 0.)).as_matrix()
        return self.fk(parent, positions, root) @ origin @ turn

    def tcp(self, q, tcp_in_wrist):
        wrist = self.fk("astribot_arm_right_link_7", dict(zip(RIGHT_ARM, q)))
        return wrist @ tcp_in_wrist

    def solve(self, target, seed, tcp_in_wrist, *, diagnostics=None):
        """Bounded pose IK, retaining the nearby redundant-arm posture."""
        seed = np.clip(np.asarray(seed), self.lower + 1e-5, self.upper - 1e-5)

        def residual(q):
            actual = self.tcp(q, tcp_in_wrist)
            return np.r_[actual[:3, 3] - target[:3, 3],
                         .2 * Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec(),
                         1e-5 * (q - seed)]

        result = least_squares(residual, seed, bounds=(self.lower + 1e-5, self.upper - 1e-5),
                               max_nfev=180, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        actual = self.tcp(result.x, tcp_in_wrist)
        error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        rotation_error = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
        if diagnostics is not None:
            diagnostics.update(position_error_m=float(error), rotation_error_rad=float(rotation_error),
                               within_joint_limits=bool(np.all(result.x >= self.lower) and np.all(result.x <= self.upper)),
                               q_rad=result.x.tolist(), nfev=int(result.nfev),
                               target_in_robot_root=target.tolist())
        if error > .0005 or rotation_error > .01:
            raise UnreachablePose(f"Unreachable pose: position={error:.6f} m, rotation={rotation_error:.5f} rad")
        return result.x
