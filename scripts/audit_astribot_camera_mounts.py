#!/usr/bin/env python3
"""Compare exported wrist cameras to Astribot's official ManiSkill extrinsics.

Run using PYTHONPATH=.cache/usd-inspect:src and the simulation Python.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

SOURCE = "https://github.com/Astribot-Dev/astribot_simulation/blob/1403f9623efaea2709731c74629f395869849e89/src/astribot_envs/astribot_maniskill_env.py"


def audit(directory, include_head=False):
    report = json.loads((directory / "report.json").read_text())
    stage = Usd.Stage.Open(str(directory / "scene.usda"))
    cache = UsdGeom.XformCache()
    sapien_rotation = Rotation.from_quat([-.5, -.5, -.5, .5]).as_matrix()
    usd_to_sapien = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    reference = np.eye(4)
    reference[:3, :3] = sapien_rotation @ usd_to_sapien
    reference[:3, 3] = [0., .04, .03]
    result = dict(source=SOURCE, source_position_m=[0, .04, .03],
                  source_quaternion_wxyz=[.5, -.5, -.5, -.5], cameras={})
    for side in ("left", "right"):
        name = "wrist_" + side
        camera = stage.GetPrimAtPath(report["camera_paths"][name])
        if not camera.IsA(UsdGeom.Camera):
            raise ValueError(f"Missing camera: {name}")
        mount = camera.GetParent()
        wrist = mount.GetParent()
        local = np.asarray(UsdGeom.Xformable(camera).GetLocalTransformation()).T
        world_camera = np.asarray(cache.GetLocalToWorldTransform(camera)).T
        world_mount = np.asarray(cache.GetLocalToWorldTransform(mount)).T
        world_wrist = np.asarray(cache.GetLocalToWorldTransform(wrist)).T
        reference_world = world_mount @ reference
        checks = dict(
            mount_name=mount.GetName() == f"astribot_gripper_{side}_base",
            wrist_parent=wrist.GetName() == f"astribot_arm_{side}_link_7",
            local_position=np.allclose(local[:3, 3], reference[:3, 3], atol=1e-7, rtol=0),
            local_rotation=np.allclose(local[:3, :3], reference[:3, :3], atol=1e-7, rtol=0),
            world_pose=np.allclose(world_camera, reference_world, atol=1e-6, rtol=0))
        result["cameras"][name] = dict(
            checks={k: bool(v) for k, v in checks.items()},
            path=str(camera.GetPath()), position_in_gripper_m=local[:3, 3].tolist(),
            pose_in_wrist=(np.linalg.inv(world_wrist) @ world_camera).tolist(),
            world_forward=(-world_camera[:3, 2]).tolist(),
            max_local_matrix_error=float(np.max(np.abs(local - reference))),
            max_world_matrix_error=float(np.max(np.abs(world_camera - reference_world))))
    if include_head:
        camera = stage.GetPrimAtPath(report["camera_paths"]["head"])
        if not camera.IsA(UsdGeom.Camera):
            raise ValueError("Missing head camera")
        mount = camera.GetParent()
        local = np.asarray(UsdGeom.Xformable(camera).GetLocalTransformation()).T
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_quat([.7071, 0, 0, .7071]).as_matrix() @ usd_to_sapien
        expected[:3, 3] = [.0603, -.2139, -.0277]
        world_camera = np.asarray(cache.GetLocalToWorldTransform(camera)).T
        world_expected = np.asarray(cache.GetLocalToWorldTransform(mount)).T @ expected
        prim = UsdGeom.Camera(camera)
        fov = float(2 * np.arctan(prim.GetVerticalApertureAttr().Get() / (2 * prim.GetFocalLengthAttr().Get())))
        from PIL import Image
        with Image.open(directory / "finished_head.png") as frame:
            resolution = frame.size
        checks = dict(
            mount_name=mount.GetName() == "astribot_head_link_2",
            local_pose=np.allclose(local, expected, atol=1e-7, rtol=0),
            world_pose=np.allclose(world_camera, world_expected, atol=1e-6, rtol=0),
            resolution=resolution == (1280, 720),
            vertical_fov=abs(fov - .8974046681441843) < 1e-7,
            clipping=np.allclose(prim.GetClippingRangeAttr().Get(),
                                [.028338245138894207, 141.6912288615168], rtol=1e-7))
        result["cameras"]["head"] = dict(
            checks={k: bool(v) for k, v in checks.items()}, path=str(camera.GetPath()),
            position_in_head_link_m=local[:3, 3].tolist(), resolution=list(resolution),
            vertical_fov_rad=fov, source_quaternion_wxyz=[.7071, .7071, 0, 0],
            max_local_matrix_error=float(np.max(np.abs(local - expected))),
            selected_view="head_rgbd single RGB viewpoint")
    result["status"] = "passed" if all(all(c["checks"].values())
                                                for c in result["cameras"].values()) else "failed"
    (directory / "camera_mount_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--include-head", action="store_true", help="Also verify the official monocular head view")
    args = parser.parse_args()
    result = audit(args.recording.resolve(), include_head=args.include_head)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
