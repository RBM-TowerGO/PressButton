#!/usr/bin/env python3
"""Preview Astribot and the physical 12-button PressB panel in one scene."""
import argparse
from datetime import datetime
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
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.astribot_asset import add_astribot, ASSET
    from pressb.scene import build_scene, set_button_light

    cfg = json.loads(args.config.read_text())
    world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/10,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    scene = build_scene(world.stage, cfg)
    metadata = add_astribot(world.stage)
    # Translate the fixed-base world anchor along with the robot's container.
    position = Gf.Vec3d(cfg["robot_base_x"], cfg["robot_base_y"], metadata["base_lift_m"])
    world.stage.GetPrimAtPath("/World/Astribot").GetAttribute("xformOp:translate").Set(position)
    UsdPhysics.FixedJoint(world.stage.GetPrimAtPath(metadata["articulation_path"])).CreateLocalPos0Attr(Gf.Vec3f(*position))
    robot = world.scene.add(SingleArticulation(prim_path=metadata["robot_path"], name="astribot"))
    sensors = None
    if args.sensors:
        from pressb.astribot_sensors import AstribotSensors
        sensors = AstribotSensors(world, robot, metadata)
        if args.lidar_points:
            world.stage.GetPrimAtPath(sensors.lidar_path).GetAttribute("drawPoints").Set(True)
    buttons = {floor: world.scene.add(SingleRigidPrim(prim_path=info["body_path"], name=f"button_{floor}"))
               for floor, info in scene["buttons"].items()}
    cameras = {}
    for name, path, resolution in [("overview", scene["camera_path"], (1600, 1000)),
                                    ("panel", scene["panel_camera_path"], (960, 720))]:
        prim = UsdGeom.Camera.Get(world.stage, path)
        prim.CreateFocalLengthAttr(30. if name == "overview" else 40.)
        prim.CreateVerticalApertureAttr(36. * resolution[1] / resolution[0])
        cameras[name] = Camera(prim_path=path, name=name, resolution=resolution, frequency=-1)
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(sensors.paths[args.view + "_camera"]
                                      if args.view != "overview" else scene["camera_path"])
    world.reset()
    for camera in cameras.values():
        camera.initialize()
    if sensors:
        sensors.initialize()
    home = np.zeros(robot.num_dof, dtype=np.float32)
    robot.set_joints_default_state(positions=home, velocities=np.zeros_like(home))
    robot.set_joint_positions(home)
    robot.set_joint_velocities(np.zeros_like(home))
    controller = robot.get_articulation_controller()

    def step(render=True):
        controller.apply_action(ArticulationAction(joint_positions=home))
        world.step(render=False)
        if render:
            world.render()
        for floor, info in scene["buttons"].items():
            x = buttons[floor].get_world_pose()[0][0]
            set_button_light(world.stage, info, x - info["rest_x"] >= .0015)

    for i in range(240):
        step(render=i % 12 == 0)
    for _ in range(12):
        world.render()
    for name, camera in cameras.items():
        rgba = camera.get_rgba()
        if rgba is None or np.std(rgba[:, :, :3]) < 5:
            raise RuntimeError(f"Blank {name} render")
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(args.output / f"{name}.png")
    q = robot.get_joint_positions()
    if not np.isfinite(q).all() or np.max(np.abs(q)) >= .03:
        raise RuntimeError("Robot failed to hold preview pose")
    report.update(robot_dofs=robot.num_dof, buttons=sorted(buttons),
                  hold_max_error_rad=float(np.max(np.abs(q))), physics_steps=240,
                  robot_position_m=list(position), config=cfg,
                  scope="Robot and physical button panel preview; automated pressing is not implemented")
    if sensors:
        sensors.save_snapshot(args.output / "sensors")
        report["sensors"] = sensors.calibration()
    # Export portable references, including generated panel label textures.
    output_scene = args.output / "scene.usda"
    world.stage.GetRootLayer().Export(str(output_scene))
    exported = Usd.Stage.Open(str(output_scene))
    from pxr import Sdf
    for prim in exported.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset:
                value = attribute.Get()
                if value and Path(value.path).is_absolute():
                    attribute.Set(Sdf.AssetPath(os.path.relpath(value.path, args.output)))
    exported.GetPrimAtPath("/World/Astribot").GetReferences().SetReferences(
        [Sdf.Reference(os.path.relpath(ASSET, args.output))])
    exported.GetRootLayer().Save()
    report["status"] = "passed"
    (args.output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: Astribot + 12 physical buttons. Preview: {args.output / 'overview.png'}", flush=True)
    if not args.headless:
        gui_steps = 0
        while app.is_running():
            gui_steps += 1
            step(render=gui_steps % 12 == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Save preview images and exit")
    parser.add_argument("--sensors", action="store_true", help="Attach and save head/wrist RGB-D, lidar, IMU and contact streams")
    parser.add_argument("--view", choices=["overview", "head", "wrist_left", "wrist_right"], default="overview")
    parser.add_argument("--lidar-points", action="store_true", help="Draw lidar returns in the GUI")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/astribot_panel_preview.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.view != "overview" or args.lidar_points:
        args.sensors = True
    args.output = (args.output or ROOT / "outputs" / ("astribot_panel_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running"}
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=args.headless, create_new_stage=not args.headless,
            active_gpu=0, physics_gpu=0, multi_gpu=False, renderer="RayTracedLighting",
            width=1600, height=1000, extra_args=["--/app/asyncRendering=false",
            "--/rtx/ecoMode/enabled=false", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        run(app, args, report)
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        traceback.print_exc()
        raise
    finally:
        (args.output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
        if app:
            app.app.post_quit(0 if report["status"] == "passed" else 1)
            app.close()


if __name__ == "__main__":
    main()
