#!/usr/bin/env python3
"""Load, exercise and render the fixed-base Astribot S1 in the existing Isaac env."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(app, args, report):
    import numpy as np
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.astribot_asset import add_astribot

    world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/10,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    # A local floor keeps the exported scene independent of NVIDIA's remote
    # default-environment USD and its textures.
    world.scene.add(FixedCuboid(prim_path="/World/Floor", name="floor",
        position=np.array([0., 0., -.025]), scale=np.array([200., 200., .05]),
        color=np.array([.16, .19, .23])))
    light = UsdLux.DomeLight.Define(world.stage, "/World/Light")
    light.CreateIntensityAttr(1000.)
    light.CreateColorAttr(Gf.Vec3f(.85, .90, 1.))
    metadata = add_astribot(world.stage)
    report["adapter"] = metadata
    probe = world.scene.add(DynamicCuboid(prim_path="/World/CollisionProbe", name="collision_probe",
        position=np.array([.22, .05, .60]), scale=np.array([.04, .04, .04]),
        mass=.05, color=np.array([.8, .12, .08])))
    contacts = []

    def on_contact(headers, data):
        for header in headers:
            actors = [str(PhysicsSchemaTools.intToSdfPath(a)) for a in (header.actor0, header.actor1)]
            if "/World/CollisionProbe" in actors and any(a.startswith(metadata["robot_path"] + "/") for a in actors):
                impulse = sum(float(np.linalg.norm(data[i].impulse)) for i in range(
                    header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
                if impulse > 0 and len(contacts) < 100:
                    contacts.append(dict(actors=actors, impulse_ns=impulse))

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    robot = world.scene.add(SingleArticulation(prim_path=metadata["robot_path"], name="astribot"))
    camera_path = "/World/PreviewCamera"
    camera_prim = UsdGeom.Camera.Define(world.stage, camera_path)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(2.7, -2.7, 2.0), Gf.Vec3d(0, 0, .87), Gf.Vec3d(0, 0, 1))
    camera_prim.AddTransformOp().Set(view.GetInverse())
    camera_prim.CreateFocalLengthAttr(28.)
    camera_prim.CreateHorizontalApertureAttr(20.955)
    camera_prim.CreateVerticalApertureAttr(20.955 * 720 / 960)
    camera_prim.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    camera = Camera(prim_path=camera_path, resolution=(960, 720), frequency=-1)
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(camera_path)
    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    names = list(robot.dof_names)
    if len(names) != 32 or set(names) != set(metadata["joint_paths"]):
        raise ValueError(f"Unexpected articulation DOFs: {names}")
    home = np.zeros(len(names), dtype=np.float32)
    robot.set_joints_default_state(positions=home, velocities=np.zeros_like(home))
    robot.set_joint_positions(home)
    robot.set_joint_velocities(np.zeros_like(home))
    indices = {side: [names.index(n) for n in joints] for side, joints in metadata["arm_joints"].items()}
    properties = robot.dof_properties
    limits = np.column_stack((properties["lower"], properties["upper"]))
    history, targets, base_positions = [], [], []
    controller = robot.get_articulation_controller()
    steps = 0

    def advance(target, count):
        nonlocal steps
        for _ in range(count):
            controller.apply_action(ArticulationAction(joint_positions=target))
            world.step(render=(steps % 12 == 0))
            actual = robot.get_joint_positions().copy()
            velocity = robot.get_joint_velocities().copy()
            base, orientation = robot.get_world_pose()
            if not all(np.isfinite(x).all() for x in (actual, velocity, base, orientation)):
                raise ValueError("Nonfinite physical state")
            if np.any(actual < limits[:, 0] - .02) or np.any(actual > limits[:, 1] + .02):
                raise ValueError("Joint exceeded its limits")
            history.append(actual)
            targets.append(np.array(target))
            base_positions.append(base.copy())
            steps += 1
        return robot.get_joint_positions().copy()

    def snapshot(name):
        for _ in range(12):
            world.render()
        rgb = camera.get_rgba()
        depth = camera.get_depth()
        if rgb is None or rgb.shape != (720, 960, 4) or np.std(rgb[:, :, :3]) < 5:
            raise ValueError("Invalid or blank RGB render")
        if depth is None or np.count_nonzero(np.isfinite(depth) & (depth > 0)) < 1000:
            raise ValueError("Invalid depth render")
        Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(args.output / f"{name}.png")
        np.save(args.output / f"{name}_depth.npy", depth)
        return dict(rgb_shape=list(rgb.shape), rgb_std=float(np.std(rgb[:, :, :3])),
                    finite_positive_depth_pixels=int(np.count_nonzero(np.isfinite(depth) & (depth > 0))))

    settled = advance(home, 360)
    report["hold_max_error_rad"] = float(np.max(np.abs(settled - home)))
    report["collision_probe"] = dict(contacts=contacts.copy(), settled_position_m=probe.get_world_pose()[0].tolist())
    report["render"] = snapshot("preview")
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Astribot: initial hold and RGB-D complete", flush=True)
    motions = []
    report["joint_motion"] = motions
    # Exercise all seven joints on both arms; avoid upper/lower-limit commands.
    for joint in range(7):
        target = home.copy()
        delta = -.08 if joint == 1 else .08
        for side in indices:
            target[indices[side][joint]] += delta
        for fraction in np.linspace(0., 1., 60):
            advance(home + fraction * (target - home), 1)
        actual = advance(target, 120)
        for side in indices:
            idx = indices[side][joint]
            motions.append(dict(joint=names[idx], target_rad=float(target[idx]),
                                actual_rad=float(actual[idx]),
                                error_rad=float(abs(actual[idx] - target[idx])),
                                displacement_rad=float(abs(actual[idx] - settled[idx]))))
        for fraction in np.linspace(1., 0., 60):
            advance(home + fraction * (target - home), 1)
        advance(home, 60)
        (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Astribot: both arms joint {joint + 1}/7 exercised", flush=True)
    report["joint_motion"] = motions
    report["motion_max_error_rad"] = max(x["error_rad"] for x in motions)
    report["motion_min_displacement_rad"] = min(x["displacement_rad"] for x in motions)
    report["return_home_max_error_rad"] = float(np.max(np.abs(advance(home, 240))))
    positions = np.asarray(base_positions)
    report["base_drift_m"] = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    world.reset()
    reset_actual = advance(home, 120)
    report["reset_max_error_rad"] = float(np.max(np.abs(reset_actual)))
    report["reset_render"] = snapshot("after_reset")
    report["physics_steps"] = steps
    np.savez_compressed(args.output / "joint_trace.npz", joint_names=names,
                        actual=np.asarray(history), target=np.asarray(targets),
                        base_positions=np.asarray(base_positions), physics_dt=1/120)
    report["simulated_seconds"] = steps / 120
    report["dof_names"] = names
    report["checks"] = dict(
        all_32_dofs=len(names) == 32,
        stable_hold=report["hold_max_error_rad"] < .03,
        both_arms_track=report["motion_max_error_rad"] < .03,
        all_arm_joints_move=report["motion_min_displacement_rad"] > .04,
        return_home=report["return_home_max_error_rad"] < .03,
        fixed_base=report["base_drift_m"] < .001,
        reset=report["reset_max_error_rad"] < .03,
        rgb_and_depth=True,
        robot_contact=bool(report["collision_probe"]["contacts"]),
        probe_supported_above_floor=report["collision_probe"]["settled_position_m"][2] > .08,
    )
    # Keep generated scene references relative so the repository can be relocated.
    scene_path = args.output / "scene.usda"
    world.stage.GetRootLayer().Export(str(scene_path))
    exported = Usd.Stage.Open(str(scene_path))
    prim = exported.GetPrimAtPath("/World/Astribot")
    prim.GetReferences().ClearReferences()
    from pressb.astribot_asset import ASSET
    prim.GetReferences().AddReference(os.path.relpath(ASSET, args.output))
    exported.GetRootLayer().Save()
    if not all(report["checks"].values()):
        raise RuntimeError(f"Failed checks: {report['checks']}")
    report["status"] = "passed"
    print("PASS: Astribot S1 load, both arms, hold, reset and RGB-D", flush=True)
    if args.hold:
        while app.is_running():
            controller.apply_action(ArticulationAction(joint_positions=home))
            world.step(render=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/astribot_asset_validation")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                  scope="Fixed-base asset and joint-space validation; no button task or VLA")
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=args.headless, create_new_stage=not args.headless,
                                 active_gpu=0, physics_gpu=0, multi_gpu=False,
                                 renderer="RayTracedLighting", width=960, height=720,
                                 extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                                             "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        run(app, args, report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        if app:
            app.app.post_quit(0 if report["status"] == "passed" else 1)
            app.close()


if __name__ == "__main__":
    main()
