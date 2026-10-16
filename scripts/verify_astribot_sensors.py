#!/usr/bin/env python3
"""Exercise robot-mounted RGB-D, lidar, IMU, contact and joint sensors."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(app, args, report):
    import numpy as np
    from PIL import Image, ImageDraw
    from pxr import Gf, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pressb.astribot_asset import add_astribot
    from pressb.astribot_sensors import AstribotSensors
    from pressb.scene import build_scene

    cfg = json.loads((ROOT / "configs/astribot_panel_preview.json").read_text())
    world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/10, backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(update_to_usd=True, update_velocities_to_usd=True)
    scene = build_scene(world.stage, cfg)
    adapter = add_astribot(world.stage)
    position = Gf.Vec3d(cfg["robot_base_x"], cfg["robot_base_y"], adapter["base_lift_m"])
    world.stage.GetPrimAtPath("/World/Astribot").GetAttribute("xformOp:translate").Set(position)
    UsdPhysics.FixedJoint(world.stage.GetPrimAtPath(adapter["articulation_path"])).CreateLocalPos0Attr(Gf.Vec3f(*position))
    robot = world.scene.add(SingleArticulation(prim_path=adapter["robot_path"], name="astribot"))
    rig = AstribotSensors(world, robot, adapter)
    # A side target is outside the button panel; its near Y face is exactly 1 m
    # from the lidar origin. Moving it tests freshness and range response.
    lidar_origin = np.array(position) + np.array(rig.config["lidar"]["translation"])
    target_position = lidar_origin + np.array([0., -1.05, 0.])
    target = world.scene.add(FixedCuboid(prim_path="/World/LidarCalibrationTarget", name="lidar_target",
        position=target_position, scale=np.array([.4, .1, .4]), color=np.array([.2, .6, .9])))
    probe = world.scene.add(DynamicCuboid(prim_path="/World/ContactProbe", name="contact_probe",
        position=np.array(position) + np.array([.22, .05, .5]), scale=np.full(3, .04), mass=.05,
        color=np.array([.9, .15, .1])))
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(scene["camera_path"])
    world.reset()
    rig.initialize()
    for _ in range(12):
        world.render()
    report["calibration"] = rig.calibration()
    home = np.zeros(robot.num_dof, dtype=np.float32)
    robot.set_joints_default_state(positions=home, velocities=home.copy())
    robot.set_joint_positions(home)
    robot.set_joint_velocities(home.copy())
    controller = robot.get_articulation_controller()
    captures, physics = [], []
    step_index = 0
    checks = report.setdefault("checks", {})

    def check(name, passed, details):
        checks[name] = dict(passed=bool(passed), details=details)
        print(f"{'PASS' if passed else 'FAIL'} {name}: {details}", flush=True)
        (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")

    def step(q):
        nonlocal step_index
        controller.apply_action(ArticulationAction(joint_positions=q))
        step_index += 1
        rendered = step_index % 12 == 0
        # step(render=True) at rendering_dt=.1 advances multiple physics ticks.
        # Keep exactly one 1/120 s tick per record, then render without stepping.
        world.step(render=False)
        if rendered:
            world.render()
        physics.append(rig.physics_frame())
        if rendered and step_index >= 36:
            captures.append(rig.capture())

    def settle(n=120):
        for _ in range(n):
            step(home)

    def save_capture(frame, prefix):
        arrays = {}
        for name, cam in frame["cameras"].items():
            Image.fromarray(cam["rgb"].astype(np.uint8)).save(args.output / f"{prefix}_{name}_rgb.png")
            arrays[name + "_depth_m"] = cam["depth_m"]
            arrays[name + "_intrinsics"] = cam["intrinsics"]
            depth = cam["depth_m"]
            shade = (255 * (1 - np.clip(np.nan_to_num(depth, nan=3., posinf=3.), 0, 3) / 3)).astype(np.uint8)
            Image.fromarray(shade).save(args.output / f"{prefix}_{name}_depth.png")
        for name, value in frame["lidar"].items():
            arrays["lidar_" + name] = value
        arrays.update({"world_from_" + n: t for n, t in frame["world_from_sensor"].items()})
        np.savez_compressed(args.output / f"{prefix}_sensors.npz", **arrays)

    settle(240)
    initial = captures[-1]
    save_capture(initial, "initial")
    for name, cam in initial["cameras"].items():
        depth = cam["depth_m"]
        finite = np.isfinite(depth) & (depth > .02) & (depth < 20.)
        width, height = rig.config["cameras"][name].get("resolution", rig.config["camera_resolution"])
        check(name + "_rgbd", cam["rgb"].shape == (height, width, 3) and depth.shape == (height, width)
              and np.std(cam["rgb"]) > 5 and np.mean(finite) > .5,
              dict(rgb_shape=list(cam["rgb"].shape), depth_valid_fraction=float(np.mean(finite)),
                   depth_median_m=float(np.median(depth[finite])), rgb_std=float(np.std(cam["rgb"]))))
        import omni.physx
        transform = initial["world_from_sensor"][name + "_camera"]
        hit = omni.physx.get_physx_scene_query_interface().raycast_closest(
            transform[:3, 3].tolist(), (-transform[:3, 2]).tolist(), 20.)
        measured = float(np.median(depth[height//2-2:height//2+2, width//2-2:width//2+2]))
        expected = float(hit.get("distance", -1))
        check(name + "_depth_scale", hit.get("hit", False) and abs(measured - expected) < .025,
              dict(central_depth_m=measured, physics_ray_m=expected, collider=hit.get("collision", "")))
    for name, imu in initial["imus"].items():
        gravity = float(np.linalg.norm(imu["lin_acc"]))
        check(name + "_imu_gravity", abs(gravity - 9.81) < .15 and imu["time"] > 0
              and abs(np.linalg.norm(imu["orientation"]) - 1) < .001,
              dict(acceleration_norm_m_s2=gravity, time=imu["time"]))

    def target_range(frame):
        points = frame["lidar"]["points_local_m"].reshape(-1, 3)
        valid = frame["lidar"]["hit"].reshape(-1)
        selected = points[valid & (np.abs(points[:, 0]) < .10) & (np.abs(points[:, 2]) < .08) & (points[:, 1] < -.5)]
        return float(np.median(-selected[:, 1])) if len(selected) else float("nan"), len(selected)

    before, count = target_range(initial)
    check("lidar_known_distance", count > 5 and abs(before - 1.) < .025,
          dict(expected_m=1., measured_m=before, target_hits=count, scan_shape=list(initial["lidar"]["range_m"].shape)))
    target.set_world_pose(position=target_position + np.array([0., -.3, 0.]))
    settle(60)
    after, count = target_range(captures[-1])
    check("lidar_moving_target", count > 5 and abs(after - 1.3) < .025 and abs(after - before - .3) < .025,
          dict(expected_m=1.3, measured_m=after, delta_m=after-before, target_hits=count))
    # Move the head and both wrists while recording at both acquisition rates.
    desired = home.copy()
    for name in ("astribot_head_joint_1", "astribot_arm_left_joint_7", "astribot_arm_right_joint_7"):
        desired[list(robot.dof_names).index(name)] = .15
    motion_start = len(physics)
    for i in range(120):
        step(desired * min(1., (i + 1) / 90))
    moved = captures[-1]
    save_capture(moved, "moved")
    for name in rig.cameras:
        key = name + "_camera"
        local_initial = initial["world_from_sensor"][key]
        local_moved = moved["world_from_sensor"][key]
        angle = float(np.arccos(np.clip((np.trace(local_initial[:3, :3].T @ local_moved[:3, :3]) - 1) / 2, -1, 1)))
        difference = float(np.mean(np.abs(initial["cameras"][name]["rgb"].astype(float) - moved["cameras"][name]["rgb"])))
        gyro = max(float(np.linalg.norm(p["imus"][name]["ang_vel"])) for p in physics[motion_start:])
        check(name + "_motion", angle > .08 and difference > 1 and gyro > .03,
              dict(camera_rotation_rad=angle, rgb_mean_change=difference, max_gyro_rad_s=gyro))
    base_contacts = [p["contacts"]["astribot_torso_base"] for p in physics]
    probe_contacts = [c for f in base_contacts for c in f.get("contacts", [])
                      if "/World/ContactProbe" in (c["body0"], c["body1"])]
    check("contact_probe", bool(probe_contacts) and max(f.get("force", 0) for f in base_contacts) > .1,
          dict(probe_contact_samples=len(probe_contacts), peak_base_force_n=max(f.get("force", 0) for f in base_contacts),
               contact_sensor_count=len(rig.contacts), probe_final_z_m=float(probe.get_world_pose()[0][2])))
    q_all = np.array([p["joints"]["position"] for p in physics])
    velocity = np.array([p["joints"]["velocity"] for p in physics])
    effort = np.array([p["joints"]["effort"] for p in physics])
    wrench = np.array([p["joints"]["reaction_wrench"] for p in physics])
    check("joint_state", q_all.shape[1] == 32 and all(np.isfinite(v).all() for v in (q_all, velocity, effort))
          and np.max(np.abs(effort)) > .01 and np.max(np.abs(robot.get_joint_positions() - desired)) < .03,
          dict(dofs=robot.num_dof, samples=len(q_all), max_effort_nm=float(np.max(np.abs(effort)))))
    check("joint_reaction_wrench", wrench.shape[1:] == (33, 6) and np.isfinite(wrench).all()
          and np.max(np.abs(wrench[:, :, :3])) > 1.,
          dict(shape=list(wrench.shape), max_force_n=float(np.max(np.abs(wrench[:, :, :3])))))
    for name in rig.imus:
        times = np.array([p["imus"][name]["time"] for p in physics[36:]])
        values = np.array([np.r_[p["imus"][name]["lin_acc"], p["imus"][name]["ang_vel"]] for p in physics[36:]])
        check(name + "_imu_freshness", np.allclose(np.diff(times), 1/120, atol=1e-5) and np.isfinite(values).all(),
              dict(samples=len(times), min_interval_s=float(np.min(np.diff(times))), max_interval_s=float(np.max(np.diff(times)))))
    check("contact_streams", all(f.get("time", 0) > 0 and np.isfinite(f.get("force", np.nan))
          for f in moved["contacts"].values()), dict(sensor_count=len(rig.contacts)))
    # Native camera timestamps must advance; physics timestamps stay independent.
    for name in rig.cameras:
        times = np.array([c["cameras"][name]["time"] for c in captures])
        lag = np.array([c["time"] - c["cameras"][name]["time"] for c in captures])
        check(name + "_freshness", np.allclose(np.diff(times), .1, atol=1e-5) and np.max(np.abs(lag)) <= .011,
              dict(frames=len(times), min_interval_s=float(np.min(np.diff(times))), max_abs_lag_s=float(np.max(np.abs(lag)))))
    # Retain pre-reset traces with native per-stream timestamps and measurements.
    trace = dict(time=np.array([p["time"] for p in physics]), q=q_all, velocity=velocity, effort=effort, reaction_wrench=wrench,
                 joint_names=np.array(robot.dof_names), capture_time=np.array([c["time"] for c in captures]))
    for name in rig.imus:
        for field in ("time", "lin_acc", "ang_vel", "orientation"):
            trace[f"imu_{name}_{field}"] = np.array([p["imus"][name][field] for p in physics])
    for name in rig.contacts:
        trace["contact_" + name + "_force_n"] = np.array([p["contacts"][name].get("force", np.nan) for p in physics])
        trace["contact_" + name + "_time"] = np.array([p["contacts"][name].get("time", 0) for p in physics])
    for name in rig.cameras:
        trace["camera_" + name + "_time"] = np.array([c["cameras"][name]["time"] for c in captures])
    trace["lidar_ranges_m"] = np.array([c["lidar"]["range_m"] for c in captures])
    np.savez_compressed(args.output / "timeseries.npz", **trace)
    world.reset()
    rig.initialize()
    for _ in range(12):
        world.render()
    controller = robot.get_articulation_controller()
    settle(120)
    reset = captures[-1]
    save_capture(reset, "reset")
    check("reset_recovery", rig.epoch == 2 and np.max(np.abs(robot.get_joint_positions())) < .03
          and all(c["time"] > 0 and np.std(c["rgb"]) > 5 for c in reset["cameras"].values())
          and all(v["time"] > 0 for v in reset["imus"].values()) and reset["lidar"]["hit"].sum() > 100,
          dict(epoch=rig.epoch, simulation_time=reset["time"], camera_times={n:c["time"] for n,c in reset["cameras"].items()}))
    # A single contact sheet is useful in the VS Code image viewer.
    sheet = Image.new("RGB", (1280, 3 * 510), "#171d28")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(rig.cameras):
        for col, suffix in enumerate(("rgb", "depth")):
            from PIL import ImageOps
            tile = ImageOps.contain(Image.open(args.output / f"initial_{name}_{suffix}.png").convert("RGB"), (640, 480))
            sheet.paste(tile, (col*640+(640-tile.width)//2, row*510+30+(480-tile.height)//2))
            draw.text((col*640+12, row*510+8), name + " / " + suffix, fill="white")
    sheet.save(args.output / "camera_overview.png")
    (args.output / "calibration.json").write_text(json.dumps(rig.calibration(), indent=2) + "\n")
    report.update(physics_steps=step_index, status="passed" if all(c["passed"] for c in checks.values()) else "failed")
    print(f"Sensor validation {report['status']}: {args.output}", flush=True)
    if report["status"] != "passed":
        raise RuntimeError("One or more sensor validation checks failed; see validation.json")
    if args.hold and not args.headless:
        while app.is_running():
            controller.apply_action(ArticulationAction(joint_positions=home))
            world.step(render=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold", action="store_true", help="Keep GUI open after validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = (args.output or ROOT / "outputs" / ("astribot_sensors_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running"}
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=args.headless, create_new_stage=not args.headless, active_gpu=0, physics_gpu=0,
            multi_gpu=False, renderer="RayTracedLighting", extra_args=["--/app/asyncRendering=false",
            "--/rtx/ecoMode/enabled=false", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        run(app, args, report)
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        traceback.print_exc()
        raise
    finally:
        (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        if app:
            app.app.post_quit(0 if report["status"] == "passed" else 1)
            app.close()


if __name__ == "__main__":
    main()
