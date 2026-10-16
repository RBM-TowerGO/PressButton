"""Check official wrist extrinsics and SAPIEN-to-USD camera axes on CPU.

PYTHONPATH=.cache/usd-inspect:src .conda/envs/pressb/bin/python -m pytest -q tests/test_astribot_camera_mounts.py
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pressb.astribot_kinematics import AstribotKinematics

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("side", ["left", "right"])
def test_official_wrist_mount_and_optical_axes(side):
    Gf = pytest.importorskip("pxr.Gf")
    cfg = json.loads((ROOT / "configs/astribot_sensors.json").read_text())
    mount = cfg["cameras"]["wrist_" + side]
    # Astribot's pinned ManiSkill CameraConfig uses the gripper base and
    # SAPIEN quaternion wxyz=(.5,-.5,-.5,-.5), not a USD camera quaternion.
    assert mount["link"] == f"astribot_gripper_{side}_base"
    actual = np.array(Gf.Matrix4d().SetLookAt(
        *[Gf.Vec3d(*mount[k]) for k in ("eye", "target", "up")]).GetInverse()).T
    sapien_rotation = Rotation.from_quat([-.5, -.5, -.5, .5]).as_matrix()
    usd_to_sapien = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    np.testing.assert_allclose(actual[:3, 3], [0., .04, .03], atol=1e-12)
    np.testing.assert_allclose(actual[:3, :3], sapien_rotation @ usd_to_sapien, atol=1e-12)
    # Include the actual fixed joint: gripper frame is rotated 90 degrees
    # about X and translated -45 mm along the wrist's Y axis.
    kin = AstribotKinematics()
    wrist = kin.fk(mount["link"], root=f"astribot_arm_{side}_link_7") @ actual
    np.testing.assert_allclose(wrist[:3, 3], [0, -.075, .04], atol=2e-7)
    np.testing.assert_allclose(-wrist[:3, 2], [0, -1, 0], atol=4e-6)


def test_nested_fixed_link_is_resolved_and_ambiguity_rejected():
    Usd = pytest.importorskip("pxr.Usd")
    from pressb.astribot_sensors import resolve_camera_mount

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot", "Xform")
    expected = "/Robot/wrist/gripper_base"
    stage.DefinePrim(expected, "Xform")
    stage.DefinePrim(expected + "/visuals/gripper_base", "Xform")
    assert resolve_camera_mount(stage, "/Robot", "gripper_base") == expected
    with pytest.raises(ValueError, match="uniquely"):
        resolve_camera_mount(stage, "/Robot", "missing")
    stage.DefinePrim("/Robot/another/gripper_base", "Xform")
    with pytest.raises(ValueError, match="uniquely"):
        resolve_camera_mount(stage, "/Robot", "gripper_base")


def test_all_camera_links_resolve_in_imported_robot():
    Usd = pytest.importorskip("pxr.Usd")
    from pressb.astribot_sensors import resolve_camera_mount

    asset = ROOT / "assets/generated/astribot/astribot_s1.usdc"
    if not asset.exists():
        pytest.skip("Astribot USD asset not generated")
    stage = Usd.Stage.Open(str(asset))
    cfg = json.loads((ROOT / "configs/astribot_sensors.json").read_text())
    for name, mount in cfg["cameras"].items():
        path = resolve_camera_mount(stage, "/World/astribot_whole_body", mount["link"])
        assert "/visuals/" not in path
        if name.startswith("wrist_"):
            side = name.removeprefix("wrist_")
            assert path.endswith(f"astribot_arm_{side}_link_7/astribot_gripper_{side}_base")


def test_official_head_monocular_pose_and_optics(monkeypatch):
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom
    from types import ModuleType, SimpleNamespace
    import sys
    from pressb.astribot_sensors import create_robot_cameras

    # Use real USD authoring while substituting only the runtime Camera handle.
    fake = ModuleType("isaacsim.sensors.camera")
    fake.Camera = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "isaacsim.sensors.camera", fake)
    cfg = json.loads((ROOT / "configs/astribot_sensors.json").read_text())
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot", "Xform")
    for mount in cfg["cameras"].values():
        stage.DefinePrim("/Robot/" + mount["link"], "Xform")
    cameras, paths = create_robot_cameras(stage, "/Robot", cfg)
    assert list(cameras) == ["head", "wrist_left", "wrist_right"]
    assert cameras["head"].resolution == (1280, 720)
    head = UsdGeom.Camera(stage.GetPrimAtPath(paths["head_camera"]))
    actual = np.asarray(head.GetLocalTransformation()).T
    reference = Rotation.from_quat([.7071, 0., 0., .7071]).as_matrix()
    usd_to_sapien = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    np.testing.assert_allclose(actual[:3, :3], reference @ usd_to_sapien, atol=1e-12)
    np.testing.assert_allclose(actual[:3, 3], [.0603, -.2139, -.0277], atol=1e-12)
    actual_fov = 2 * np.arctan(head.GetVerticalApertureAttr().Get() / (2 * head.GetFocalLengthAttr().Get()))
    assert actual_fov == pytest.approx(.8974046681441843, abs=1e-7)
    np.testing.assert_allclose(head.GetClippingRangeAttr().Get(), [.028338245138894207, 141.6912288615168], rtol=1e-7)
    for name in ("wrist_left", "wrist_right"):
        camera = UsdGeom.Camera(stage.GetPrimAtPath(paths[name + "_camera"]))
        assert cameras[name].resolution == (640, 480)
        assert camera.GetFocalLengthAttr().Get() == 18.

