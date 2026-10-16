#!/usr/bin/env python3
"""Press physical elevator buttons using S1's closed original right gripper."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(app, args, report):
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from pxr import Gf, Usd, UsdPhysics, PhysicsSchemaTools, UsdGeom, Sdf
    import omni.physx
    import omni.client
    if args.headless:
        # Frozen local assets do not need background directory subscriptions.
        omni.client.bypass_list_cache(True)
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.astribot_asset import add_astribot, ASSET
    from pressb.astribot_kinematics import AstribotKinematics, RIGHT_ARM, gripper_targets
    from pressb.base_pose import root_transform, world_target_in_root
    from pressb.astribot_press import prepare_hand, complete_collision_geometry, smooth_fraction
    from pressb.scene import build_scene, set_button_light

    cfg = json.loads(args.config.read_text())
    if cfg.get("scene_type") == "vib820":
        from pressb.vib820_scene import build_scene, set_button_light
    sequence = args.floors or cfg["sequence"]
    if not sequence or args.repeats < 1:
        raise ValueError("Button sequence and positive repeats are required")
    dt = 1 / cfg["physics_hz"]
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/cfg["capture_hz"],
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    scene = build_scene(world.stage, cfg)
    button_ids = sorted(scene["buttons"])
    button_index = {f: i for i, f in enumerate(button_ids)}
    if any(f not in button_index for f in sequence):
        raise ValueError(f"Requested button is absent; valid IDs: {button_ids}")
    latched = cfg.get("button_light_mode") == "latched"
    report.update(button_ids=button_ids, task_labels=cfg.get("task_labels", {}),
                  scene_dimensions=scene.get("dimensions"), geometry_provenance=cfg.get("provenance"))
    meta = add_astribot(world.stage)
    base = np.array([cfg["robot_base_x"], cfg["robot_base_y"], meta["base_lift_m"]])
    yaw = float(cfg.get("robot_base_yaw_rad", 0.))
    world_from_root = root_transform(base, yaw)
    root_prim = world.stage.GetPrimAtPath("/World/Astribot")
    root_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*base))
    UsdGeom.Xformable(root_prim).AddRotateZOp().Set(float(np.degrees(yaw)))
    root_joint = UsdPhysics.FixedJoint(world.stage.GetPrimAtPath(meta["articulation_path"]))
    root_joint.CreateLocalPos0Attr(Gf.Vec3f(*base))
    yaw_quat = Gf.Quatf(float(np.cos(yaw/2)), Gf.Vec3f(0, 0, float(np.sin(yaw/2))))
    root_joint.CreateLocalRot0Attr(yaw_quat)
    hand = prepare_hand(world.stage, meta)
    report.update(config=cfg, hand=hand, collision=complete_collision_geometry(world.stage, meta),
                  robot_base_m=base.tolist(), arm="right", sequence=sequence, repeats=args.repeats)
    report["source_sha256"] = {os.path.relpath(p,ROOT): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), args.config, ROOT/"src/pressb/astribot_press.py",
                  ROOT/"src/pressb/astribot_kinematics.py", ROOT/"src/pressb/astribot_asset.py",
                  ROOT/"src/pressb/base_pose.py",
                  ROOT/"src/pressb/scene.py"]}
    if cfg.get("scene_type") == "vib820":
        for rel in ["src/pressb/vib820_scene.py", "src/pressb/vib820_materials.py",
                    "assets/materials/vib820/manifest.json"]:
            p = ROOT/rel
            report["source_sha256"][rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    kin = AstribotKinematics()
    head_pitch = np.radians(float(cfg.get("head_pitch_deg", 0.)))
    if not np.isfinite(head_pitch) or not kin.limits["astribot_head_joint_2"][0] <= head_pitch <= kin.limits["astribot_head_joint_2"][1]:
        raise ValueError("head_pitch_deg must be finite and within head joint limits")
    initial_arm_positions = cfg.get("initial_arm_positions", {})
    if initial_arm_positions:
        expected = {f"astribot_arm_{side}_joint_{i}" for side in ("left", "right") for i in range(1, 8)}
        if set(initial_arm_positions) != expected:
            raise ValueError("initial_arm_positions must specify all 14 named arm joints")
        for name, value in initial_arm_positions.items():
            if not np.isfinite(value) or not kin.limits[name][0] <= value <= kin.limits[name][1]:
                raise ValueError(f"Invalid initial arm position: {name}={value}")
    if cfg.get("return_to_initial_pose") and not initial_arm_positions:
        raise ValueError("return_to_initial_pose requires initial_arm_positions")
    tcp_offset = np.array(hand["tcp_in_wrist"])
    # Gripper's broad opening axis is world Y, approach axis is world +X.
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

    def pose(x, y, z):
        return world_target_in_root(world_from_root, [x, y, z], rotation)

    def target_for(floor, distance):
        x,y,z = scene["buttons"][floor]["center"]
        return pose(x + distance, y, z)

    # A reproducible bent-elbow work posture, solved before any robot movement.
    home_pose = pose(*cfg.get("home_tip", [cfg["button_face_x"]-.18, -.045, 1.065]))
    report["key_pose_ik"] = {}
    ik_failed = False
    def precheck(name, goal, seed):
        nonlocal ik_failed
        detail = report["key_pose_ik"][name] = {"status": "unknown"}
        try:
            q = kin.solve(goal, seed, tcp_offset, diagnostics=detail)
            detail["status"] = "pass"
            return q
        except ValueError as exc:
            detail.update(status="fail", error=str(exc))
            ik_failed = True
            return np.asarray(seed).copy()
    home_seed = (np.array([initial_arm_positions[n] for n in RIGHT_ARM]) if initial_arm_positions
                 else np.array([.1, -.5, 1., 1.4, .8, .5, -.2]))
    home_arm = precheck("home", home_pose, home_seed)
    planned = {}
    for floor in sorted(set(sequence)):
        pre = precheck(f"approach_{floor}", target_for(floor, -cfg["approach_distance"]), home_arm)
        touch = precheck(f"touch_{floor}", target_for(floor, cfg["press_depth"]), pre)
        planned[floor] = pre
    report["reachability"] = {str(f): q.tolist() for f, q in planned.items()}
    report["home_arm_rad"] = home_arm.tolist()
    print("Key-pose IK:", json.dumps(report["key_pose_ik"]), flush=True)
    robot = world.scene.add(SingleArticulation(prim_path=meta["robot_path"], name="astribot"))
    wrist = world.scene.add(SingleRigidPrim(prim_path=meta["wrist_links"]["right"], name="right_wrist"))
    base_body = world.scene.add(SingleRigidPrim(prim_path=meta["base_link"], name="base_monitor"))
    buttons = {f: world.scene.add(SingleRigidPrim(prim_path=b["body_path"], name=f"button_{f}"))
               for f, b in scene["buttons"].items()}
    finger_bodies = {suffix: world.scene.add(SingleRigidPrim(
        prim_path=meta["robot_path"] + "/astribot_gripper_right_Link_" + suffix, name="finger_" + suffix))
        for suffix in ("L11", "R11")}
    cameras = {}
    camera_specs = [("overview", scene["camera_path"], (960, 600)),
                    ("panel", scene["panel_camera_path"], (640, 480))]
    if args.all_cameras:
        for name in ("front", "lobby"):
            if name + "_camera_path" in scene:
                camera_specs.append((name, scene[name + "_camera_path"], (640, 480)))
    camera_paths = {name: path for name, path, _ in camera_specs}
    for name, path, size in camera_specs:
        cameras[name] = Camera(prim_path=path, name=name, resolution=size, frequency=-1)
    if args.all_cameras:
        from pressb.astribot_sensors import create_robot_cameras, DEFAULT_CONFIG
        sensor_cfg = json.loads(DEFAULT_CONFIG.read_text())
        robot_cameras, robot_paths = create_robot_cameras(world.stage, meta["robot_path"], sensor_cfg)
        cameras.update(robot_cameras)
        camera_paths.update({name: robot_paths[name + "_camera"] for name in robot_cameras})
        report["robot_camera_config"] = sensor_cfg
    report["camera_paths"] = camera_paths
    world.reset()
    # Physics tensor reads remain live. Avoid rebuilding the display scene on
    # every 120-Hz step; explicitly publish transforms at capture boundaries.
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=False, update_velocities_to_usd=False)
    for camera in cameras.values():
        camera.initialize()
    warmup_time = world.current_time
    for _ in range(6):
        world.render()
    if world.current_time != warmup_time:
        raise RuntimeError("Camera warmup advanced physics unexpectedly")
    names = list(robot.dof_names)
    arm_idx = np.array([names.index(n) for n in RIGHT_ARM])
    grip_idx = np.array([names.index(n) for n in gripper_targets(hand["closed_angle_rad"])])
    limits = np.array([[robot.dof_properties["lower"][i], robot.dof_properties["upper"][i]] for i in range(len(names))])
    target = np.zeros(len(names), dtype=np.float32)
    target[names.index("astribot_head_joint_2")] = head_pitch
    for name, value in initial_arm_positions.items():
        target[names.index(name)] = value
    initial_target = target.copy()
    robot.set_joints_default_state(positions=target.copy(), velocities=np.zeros_like(target))
    robot.set_joint_positions(target.copy())
    robot.set_joint_velocities(np.zeros_like(target))
    control = robot.get_articulation_controller()
    control.apply_action(ArticulationAction(joint_positions=target.copy()))
    world.physics_sim_view.update_articulations_kinematic()
    omni.physx.get_physx_interface().update_transformations(False, True, True)
    report["initial_joint_positions_rad"] = dict(zip(names, initial_target.tolist()))
    actual_root = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root_prim)).T
    anchor_quat = root_joint.GetLocalRot0Attr().Get()
    report["initial_poses"] = dict(requested_root_xyz_m=base.tolist(), requested_yaw_rad=yaw,
        actual_root_world_matrix=actual_root.tolist(),
        actual_root_yaw_rad=float(np.arctan2(actual_root[1,0], actual_root[0,0])),
        fixed_joint_world_quat_wxyz=[float(anchor_quat.GetReal()), *map(float, anchor_quat.GetImaginary())],
        actual_root_xyz_m=list(UsdGeom.XformCache().GetLocalToWorldTransform(world.stage.GetPrimAtPath("/World/Astribot")).ExtractTranslation()),
        fixed_joint_world_anchor_m=list(UsdPhysics.FixedJoint(world.stage.GetPrimAtPath(meta["articulation_path"])).GetLocalPos0Attr().Get()),
        base_link_path=meta["base_link"], actual_base_link_xyz_m=base_body.get_world_pose()[0].tolist(),
        actual_base_link_quat_wxyz=base_body.get_world_pose()[1].tolist(), frame="world; root and base link have distinct fixed transforms")
    from pressb.astribot_validation import initial_overlap_check
    report["initial_overlap"] = initial_overlap_check(world.stage, meta)
    report["motion_started"] = False
    report["dynamic_monitor"] = dict(threshold_n=.1, coverage_complete=False,
        allowed="active button with right L11/R11; opposing right L11/R11; authored USD filtered pairs and directly joint-connected bodies")
    forces = np.zeros(len(button_ids))
    unexpected = []
    body_to_floor = {b["body_path"]: f for f, b in scene["buttons"].items()}
    tip_paths = {meta["robot_path"] + "/astribot_gripper_right_Link_" + s for s in ("L11", "R11")}
    active_floor = -1
    phase = "settle"
    step_index = 0
    records = []
    events = []
    episodes = []
    writers = {}
    video_frames = {}
    camera_timing = []
    base_initial = None
    max_fk_error = 0.
    light_states = np.zeros(len(button_ids), dtype=bool)
    cartesian_cache = {}
    epoch = 0
    inactive_idx = np.array([i for i,n in enumerate(names) if "arm_right" not in n and "gripper_right" not in n])

    def contacts(headers, data):
        for h in headers:
            actors = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.actor0, h.actor1)]
            if not h.num_contact_data:
                continue
            force = sum(float(np.linalg.norm(data[i].impulse)) / dt for i in range(
                h.contact_data_offset, h.contact_data_offset+h.num_contact_data))
            floor = next((body_to_floor[p] for p in actors if p in body_to_floor), None)
            if floor is not None and any(p in tip_paths for p in actors):
                forces[button_index[floor]] += force
                if floor == active_floor:
                    continue
            if force <= .1 or not any(p.startswith(meta["robot_path"] + "/") for p in actors):
                continue
            # Intended opposing-finger contact during closure/holding is allowed.
            if all(p in tip_paths for p in actors):
                continue
            if len(unexpected) < 500:
                unexpected.append(dict(step=step_index, phase=phase, floor=active_floor, simulation_time=world.current_time,
                                       actors=actors, force_n=force))

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)

    def matrix(body):
        p, q = body.get_world_pose()
        t = np.eye(4)
        t[:3, 3] = p
        t[:3, :3] = Rotation.from_quat(np.roll(q, -1)).as_matrix()
        return t

    def capture():
        world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        capture_time = float(world.current_time)
        # Render callbacks can occasionally leave the previous RGB buffer in
        # place. Wait for every camera at this physics time before writing any
        # stream; rendering again must not advance physics or duplicate frames.
        for _ in range(12):
            world.render()
            if abs(float(world.current_time) - capture_time) > 1e-7:
                raise RuntimeError("Camera synchronization advanced physics unexpectedly")
            if all(abs(float(camera.get_current_frame().get("rendering_time", -1)) - capture_time) < 1e-6
                   for camera in cameras.values()):
                break
        else:
            raise RuntimeError(f"Cameras did not produce fresh images at {capture_time}")
        timing = dict(epoch=epoch, step=step_index, simulation_time=float(world.current_time), cameras={})
        for name, camera in cameras.items():
            rgb = camera.get_rgba()
            if rgb is None or not rgb.size:
                raise RuntimeError(f"Missing {name} image after camera warmup")
            rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
            if args.video:
                if name not in writers:
                    writers[name] = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                        "-pixel_format", "rgb24", "-video_size", f"{rgb.shape[1]}x{rgb.shape[0]}",
                        "-framerate", str(cfg["capture_hz"]), "-i", "-", "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", "-threads", "2", str(args.output / f"{name}.mp4")], stdin=subprocess.PIPE)
                writers[name].stdin.write(rgb.tobytes())
                video_frames[name] = video_frames.get(name,0)+1
                timing["cameras"][name] = float(camera.get_current_frame().get("rendering_time", -1))
        if args.video:
            camera_timing.append(timing)

    def snapshot(label):
        world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        for _ in range(6):
            world.render()
        for name, camera in cameras.items():
            Image.fromarray(camera.get_rgba()[:, :, :3].astype(np.uint8)).save(args.output / f"{label}_{name}.png")

    def step():
        nonlocal step_index, base_initial, max_fk_error
        report["motion_started"] = True
        report["termination_phase"] = phase
        if phase == "press":
            report["press_phase_started"] = True
        forces.fill(0.)
        control.apply_action(ArticulationAction(joint_positions=target))
        world.step(render=False)
        q = robot.get_joint_positions().copy()
        if not np.isfinite(q).all() or np.any(q < limits[:, 0]-.02) or np.any(q > limits[:, 1]+.02):
            raise RuntimeError("Nonfinite or out-of-limit joint state")
        tip = matrix(wrist) @ tcp_offset
        fk = world_from_root @ kin.tcp(q[arm_idx], tcp_offset)
        max_fk_error = max(max_fk_error, float(np.linalg.norm(tip[:3, 3]-fk[:3, 3])))
        travel = np.array([buttons[f].get_world_pose()[0][0]-scene["buttons"][f]["rest_x"] for f in button_ids])
        new_lights = travel >= cfg["press_threshold"]
        if latched:
            new_lights = light_states | (new_lights & (forces > cfg["contact_threshold_n"]))
        for j in np.flatnonzero(new_lights != light_states):
            set_button_light(world.stage, scene["buttons"][button_ids[int(j)]], bool(new_lights[j]))
        light_states[:] = new_lights
        bp, bq = base_body.get_world_pose()
        if base_initial is None:
            base_initial = bp.copy()
        records.append(dict(q=q, qd=robot.get_joint_velocities().copy(), target=target.copy(), tip=tip[:3, 3].copy(), travel=travel,
                            force=forces.copy(), light=light_states.copy(), base=bp.copy(), base_quat=bq.copy(),
                            floor=active_floor, phase=phase, epoch=epoch, simulation_time=world.current_time,
                            time=(step_index+1)*dt))
        if (args.video or not args.headless) and step_index % round(cfg["physics_hz"]/cfg["capture_hz"]) == 0:
            capture()
        step_index += 1
        if np.linalg.norm(bp-base_initial)>.001:
            raise RuntimeError("Fixed base drift exceeded 1 mm")
        if max_fk_error>.004:
            raise RuntimeError(f"URDF/PhysX TCP mismatch: {max_fk_error:.6f} m")
        if unexpected:
            raise RuntimeError(f"Unexpected physical contact: {unexpected[-1]}")
        if np.max(np.abs(q[inactive_idx]-initial_target[inactive_idx])) > .02:
            raise RuntimeError("Stationary torso, head or left arm drifted over 0.02 rad")
        if phase not in ("settle", "close_gripper") and np.max(np.abs(q[grip_idx]-target[grip_idx]))>.02:
            raise RuntimeError("Right gripper failed to stay closed")
        return travel

    def hold(seconds):
        for _ in range(round(seconds/dt)):
            step()

    def move_arm(goal, minimum=1.):
        start = target[arm_idx].copy()
        duration = max(minimum, 1.875*float(np.max(np.abs(goal-start)))/cfg["joint_speed"])
        count = int(np.ceil(duration/dt))
        for i in range(1,count+1):
            target[arm_idx] = start+smooth_fraction(i/count)*(goal-start)
            step()

    def cartesian(distance_start, distance_end, seconds, stop_on_press=False):
        seed = target[arm_idx].copy()
        count = max(1, int(np.ceil(seconds/dt)))
        key = (active_floor, float(distance_start), float(distance_end), count)
        path = cartesian_cache.setdefault(key, {})
        for i in range(1,count+1):
            if (i-1)%4 == 0 or i == count:
                if i not in path:
                    distance = distance_start+(distance_end-distance_start)*smooth_fraction(i/count)
                    try:
                        path[i] = kin.solve(target_for(active_floor, distance), seed, tcp_offset)
                    except ValueError:
                        report["termination_reason"] = "path_ik_failed"
                        raise
                seed = path[i]
            target[arm_idx] = seed
            travel = step()
            if stop_on_press and travel[button_index[active_floor]]>=cfg["press_threshold"] and forces[button_index[active_floor]]>cfg["contact_threshold_n"]:
                return True
        return False

    try:
        if report["initial_overlap"]["status"] != "clear" or ik_failed:
            report["termination_reason"] = ("initial_collision" if report["initial_overlap"]["status"] == "collision"
                else "runtime_error" if report["initial_overlap"]["status"] == "unknown" else "target_ik_failed")
            snapshot("precheck_failed")
            raise RuntimeError(f"Precheck blocked motion: {report['termination_reason']}")
        hold(1.5 if initial_arm_positions else .5)
        if initial_arm_positions:
            snapshot("initial_ready")
        phase = "close_gripper"
        closed_targets = gripper_targets(hand["closed_angle_rad"])
        for i in range(240):
            for name, value in closed_targets.items():
                target[names.index(name)] = value*smooth_fraction((i+1)/240)
            step()
        hold(.5)
        report["closure_tracking_error_rad"] = float(np.max(np.abs(robot.get_joint_positions()[grip_idx]-target[grip_idx])))
        # Measure the opposing surfaces in the actual wrist/gripper frame.
        gripper_world = matrix(wrist) @ kin.fk("astribot_gripper_right_base", root="astribot_arm_right_link_7")
        actual_points = []
        for suffix, body in finger_bodies.items():
            tf = np.linalg.inv(gripper_world) @ matrix(body)
            p = np.array(hand["finger_vertices"]["astribot_gripper_right_Link_"+suffix])
            actual_points.append(p@tf[:3,:3].T+tf[:3,3])
        report["measured_closed_gap_m"] = float(actual_points[0][:,0].min()-actual_points[1][:,0].max())
        if not -.0001 <= report["measured_closed_gap_m"] <= .001 or report["closure_tracking_error_rad"]>.02:
            raise RuntimeError("Original fingers did not reach the validated closed pose")
        snapshot("closed")
        print("Closed gripper:", report["measured_closed_gap_m"], "m gap", flush=True)
        phase = "move_to_work_pose"
        move_arm(home_arm, 3.)
        hold(.6)
        startup_error = float(np.max(np.abs(robot.get_joint_positions()[arm_idx]-home_arm)))
        report["startup"] = dict(status="pass" if startup_error<.025 else "fail",
            completed_step=step_index-1, simulation_time=world.current_time,
            work_pose_error_rad=startup_error, unexpected_contacts_before_completion=len(unexpected))
        snapshot("work_pose")
        for repeat in range(args.repeats):
            for floor in sequence:
                active_floor = floor
                if latched:
                    light_states[:] = False
                    for info in scene["buttons"].values():
                        set_button_light(world.stage, info, False)
                start_step = step_index
                collision_start = len(unexpected)
                phase = "approach"
                move_arm(planned[floor], 1.)
                hold(.3)
                phase = "press"
                pressed = cartesian(-cfg["approach_distance"], cfg["press_depth"],
                    1.875*(cfg["approach_distance"]+cfg["press_depth"])/cfg["press_speed_m_s"], True)
                # Allow a short settling interval at the final target.
                if not pressed:
                    for _ in range(60):
                        travel=step()
                        if travel[button_index[floor]]>=cfg["press_threshold"] and forces[button_index[floor]]>cfg["contact_threshold_n"]:
                            pressed=True
                            break
                if pressed:
                    events.append(dict(type="pressed", floor=floor, step=step_index-1, repeat=repeat))
                    if repeat == 0:
                        snapshot(f"press_{floor}")
                phase = "retract"
                # Retrace Cartesian targets from the current commanded contact position.
                current_distance = (world_from_root @ kin.tcp(target[arm_idx],tcp_offset))[0,3]-scene["buttons"][floor]["center"][0]
                cartesian(current_distance, -cfg["approach_distance"], 1.5)
                hold(.4)
                released = records[-1]["travel"][button_index[floor]]<cfg["release_threshold"]
                phase = "return_home"
                move_arm(home_arm, 1.)
                hold(.3)
                rows = records[start_step:]
                max_travel = np.max([r["travel"] for r in rows],axis=0)
                wrong = [f for f in button_ids if f != floor and max_travel[button_index[f]]>=cfg["press_threshold"]]
                home_error = float(np.max(np.abs(robot.get_joint_positions()[arm_idx]-home_arm)))
                success = bool(pressed and released and not wrong and len(unexpected)==collision_start and home_error<.025)
                episodes.append(dict(floor=floor, repeat=repeat, success=success, pressed=pressed, released=bool(released),
                    wrong_buttons=wrong, unexpected_contacts=len(unexpected)-collision_start,
                    max_travel_m=float(max_travel[button_index[floor]]), home_error_rad=home_error,
                    start_step=start_step, end_step=step_index))
                print("EPISODE",json.dumps(episodes[-1]),flush=True)
                report["episodes"] = episodes
                (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        active_floor=-1
        if cfg.get("return_to_initial_pose"):
            phase = "return_to_ready"
            move_arm(initial_target[arm_idx], 2.)
            phase = "ready_hold"
            hold(1.5)
            ready_error = float(np.max(np.abs(robot.get_joint_positions()[arm_idx]-initial_target[arm_idx])))
            report["ready_return"] = dict(passed=ready_error<.025, max_joint_error_rad=ready_error)
            if ready_error >= .025:
                raise RuntimeError("Failed to return to initial raised-arm posture")
        phase="finished"
        snapshot("finished")
        # Verify reset recovery to the same ready posture without driving the
        # robot through a discontinuous target trajectory during an episode.
        robot.set_joints_default_state(positions=target.copy(), velocities=np.zeros_like(target))
        world.reset()
        epoch += 1
        light_states[:] = False if latched else True
        if latched:
            for info in scene["buttons"].values():
                set_button_light(world.stage, info, False)
        phase = "reset_check"
        hold(.6)
        snapshot("reset")
        reset_error = float(np.max(np.abs(robot.get_joint_positions()-target)))
        report["reset"] = dict(passed=reset_error<.025, max_joint_error_rad=reset_error,
                               buttons_released=bool(np.max(records[-1]["travel"])<cfg["release_threshold"]))
        report["status"]="passed" if all(e["success"] for e in episodes) and not unexpected else "failed"
        if not all([report["reset"]["passed"], report["reset"]["buttons_released"]]):
            report["status"] = "failed"
        report["dynamic_monitor"]["coverage_complete"] = True
    except BaseException:
        report["termination_phase"] = phase
        try:
            snapshot("failure")
        except Exception as exc:
            report["failure_snapshot_error"] = str(exc)
        raise
    finally:
        for writer in writers.values():
            writer.stdin.close()
            if writer.wait()!=0:
                report["video_error"] = True
                report["status"] = "failed"
        report.update(episodes=episodes, unexpected_contacts=unexpected, events=events, physics_steps=step_index,
                      max_fk_error_m=max_fk_error, joint_names=names,
                      success_count=sum(e["success"] for e in episodes),video_frames=video_frames)
        if camera_timing:
            (args.output / "camera_timing.json").write_text(json.dumps(camera_timing, indent=2) + "\n")
        if records:
            np.savez_compressed(args.output/"trajectory.npz", **{key:np.asarray([r[key] for r in records]) for key in records[0]})
            report["base_drift_m"]=float(np.max(np.linalg.norm(np.array([r["base"] for r in records])-base_initial,axis=1)))
            report["inactive_joint_max_abs_rad"] = float(np.max(np.abs(np.array([r["q"] for r in records])[:,inactive_idx])))
            report["inactive_joint_max_error_rad"] = float(np.max(np.abs(
                np.array([r["q"] for r in records])[:,inactive_idx]-initial_target[inactive_idx])))
        scene_path=args.output/"scene.usda"
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        world.stage.GetRootLayer().Export(str(scene_path))
        exported=Usd.Stage.Open(str(scene_path))
        for prim in exported.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName()==Sdf.ValueTypeNames.Asset:
                    value=attr.Get()
                    if value and Path(value.path).is_absolute():
                        attr.Set(Sdf.AssetPath(os.path.relpath(value.path,args.output)))
        exported.GetPrimAtPath("/World/Astribot").GetReferences().SetReferences([Sdf.Reference(os.path.relpath(ASSET,args.output))])
        exported.GetRootLayer().Save()
    if report["status"]!="passed":
        raise RuntimeError("Button task failed physical acceptance; see report.json")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/astribot_press.json")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--video",action="store_true")
    parser.add_argument("--all-cameras", action="store_true",
                        help="Capture all scene views plus head and both wrist cameras")
    parser.add_argument("--cpu-threads",type=int,default=32, help="SimulationApp CPU worker limit")
    parser.add_argument("--gpu",type=int,default=0,
                        help="RTX/physics device index; include it in CUDA_VISIBLE_DEVICES")
    parser.add_argument("--floors", "--buttons", dest="floors", type=int, nargs="+",
                        help="Button IDs; VIB-820 down call is 0, legacy floors are 24..35")
    parser.add_argument("--repeats",type=int,default=1)
    args=parser.parse_args()
    if not 1 <= args.cpu_threads <= (os.cpu_count() or 1):
        parser.error("--cpu-threads must be between 1 and CPU count")
    args.output=(args.output or ROOT/"outputs"/("astribot_press_"+datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    report={"status":"running","scope":"Fixed base, stationary torso/left arm; original closed right gripper; known button positions",
            "gpu":args.gpu,"cpu_threads":args.cpu_threads}
    app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=args.headless,create_new_stage=not args.headless,
            active_gpu=args.gpu,physics_gpu=args.gpu,multi_gpu=False,limit_cpu_threads=args.cpu_threads,renderer="RayTracedLighting",width=960,height=600,
            extra_args=["--/app/asyncRendering=false","--/rtx/ecoMode/enabled=false",
                        "--/validate/p2p/enabled=false","--/validate/iommu/enabled=false"] +
                       (["--/app/extensions/fsWatcherEnabled=false"] if args.headless else [])))
        run(app,args,report)
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        if app:
            app.app.post_quit(0 if report["status"]=="passed" else 1)
            app.close()


if __name__=="__main__":
    main()
