import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.astribot_kinematics import AstribotKinematics, RIGHT_ARM, URDF, gripper_targets
from pressb.astribot_press import smooth_fraction


@pytest.fixture
def kin():
    if not URDF.exists():
        pytest.skip("Pinned Astribot assets not downloaded")
    return AstribotKinematics()


def test_closed_hand_preserves_parallel_finger_orientation(kin):
    for angle in [0., .5, 1.005]:
        positions = gripper_targets(angle)
        left = kin.fk("astribot_gripper_right_Link_L11", positions, root="astribot_gripper_right_base")
        right = kin.fk("astribot_gripper_right_Link_R11", positions, root="astribot_gripper_right_base")
        np.testing.assert_allclose(left[:3, :3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(right[:3, :3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(left[0, 3], -right[0, 3], atol=1e-10)
    assert all(kin.limits[n][0] <= a <= kin.limits[n][1] for n, a in positions.items())


def test_ik_recovers_pose_with_seven_joints_and_tool_offset(kin):
    tool = np.eye(4)
    tool[:3, 3] = [.01, -.23, .002]
    q = np.array([.1, -.45, .9, 1.3, .8, .4, -.3])
    goal = kin.tcp(q, tool)
    actual_q = kin.solve(goal, q + .04, tool)
    actual = kin.tcp(actual_q, tool)
    assert len(actual_q) == len(RIGHT_ARM) == 7
    assert np.linalg.norm(actual[:3, 3] - goal[:3, 3]) < .0005
    assert Rotation.from_matrix(actual[:3, :3] @ goal[:3, :3].T).magnitude() < .01


def test_ik_rejects_unreachable_target(kin):
    goal = np.eye(4)
    goal[0, 3] = 10.
    with pytest.raises(ValueError, match="Unreachable"):
        kin.solve(goal, np.zeros(7), np.eye(4))


def test_gripper_rejects_invalid_commands():
    for value in [-.01, 2., float("nan")]:
        with pytest.raises(ValueError):
            gripper_targets(value)


def test_motion_ramp_endpoints_and_monotonicity():
    t = np.linspace(0, 1, 101)
    values = smooth_fraction(t)
    assert values[0] == 0 and values[-1] == 1
    assert np.all(np.diff(values) >= 0)
    assert values[1] < 1e-5 and 1-values[-2] < 1e-5
