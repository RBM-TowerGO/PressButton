import numpy as np
import pytest
from pressb.base_pose import root_transform, world_target_in_root


def test_quarter_turn_changes_position_and_orientation_in_robot_frame():
    base=root_transform([1,2,3], np.pi/2)
    local=world_target_in_root(base,[2,2,4],np.eye(3))
    np.testing.assert_allclose(local[:3,3],[0,-1,1],atol=1e-15)
    np.testing.assert_allclose(local[:3,:3],[[0,1,0],[-1,0,0],[0,0,1]],atol=1e-15)
    world=base@local
    np.testing.assert_allclose(world[:3,3],[2,2,4])
    np.testing.assert_allclose(world[:3,:3],np.eye(3),atol=1e-15)


@pytest.mark.parametrize('angle',[0,5,10])
def test_panel_normal_is_preserved_and_retract_distance_is_world_x(angle):
    base=root_transform([-.22,.28,.102],np.radians(angle))
    rotation=np.array([[0,0,1],[1,0,0],[0,1,0]])
    for offset in [-.065,.0018]:
        local=world_target_in_root(base,[.46+offset,0,1.2],rotation)
        world=base@local
        np.testing.assert_allclose(world[:3,:3],rotation,atol=1e-15)
        assert world[0,3]-.46==pytest.approx(offset)
    if angle==0:
        np.testing.assert_allclose(local[:3,3],[.4618+.22,-.28,1.2-.102])


def test_invalid_yaw_rejected():
    with pytest.raises(ValueError):root_transform([0,0,0],np.nan)
