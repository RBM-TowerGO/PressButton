"""Cooperative version of the validated Astribot motion sequence.

The generator yields physics/capture/reset barriers to one resident GPU loop.
Control interpolation, IK seeds, closure geometry and contact thresholds follow
scripts/run_astribot_press.py. It never advances physics or drops camera frames.
"""
import json
import numpy as np

def episode(env):
    from scipy.spatial.transform import Rotation
    from pxr import UsdGeom, UsdPhysics, PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.utils.types import ArticulationAction
    from pressb.astribot_kinematics import AstribotKinematics, RIGHT_ARM, gripper_targets
    from pressb.astribot_press import smooth_fraction
    from pressb.base_pose import root_transform, world_target_in_root
    from pressb.astribot_collection.lights import set_button_light
    from types import SimpleNamespace
    world, cfg, meta, scene, report = env.world, env.config, env.meta, env.scene, env.report
    args = SimpleNamespace(output=env.output, repeats=1, headless=True, video=True)
    hand, robot, wrist, base_body, buttons, finger_bodies = (
        env.hand,env.robot,env.wrist,env.base_body,env.buttons,env.finger_bodies)
    dt = 1 / cfg["physics_hz"]
    sequence = [0]
    button_ids = [0]; button_index = {0:0}; latched = True
    base = np.array([cfg["robot_base_x"],cfg["robot_base_y"],meta["base_lift_m"]]) + env.offset
    yaw = float(cfg["robot_base_yaw_rad"])
    world_from_root = root_transform(base,yaw)
    root_prim = world.stage.GetPrimAtPath(env.prefix + "/Astribot")
    root_joint = UsdPhysics.FixedJoint(world.stage.GetPrimAtPath(meta["articulation_path"]))
    kin = AstribotKinematics()
    head_pitch = np.radians(cfg["head_pitch_deg"])
    initial_arm_positions = cfg["initial_arm_positions"]
    tcp_offset = np.array(hand["tcp_in_wrist"])
    rotation = np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=float)
    def pose(x,y,z): return world_target_in_root(world_from_root,[x,y,z],rotation)
    def target_for(floor,distance):
        x,y,z=scene["buttons"][floor]["center"]
        return pose(x+distance,y,z)
    report.update(config=cfg,hand={k:v for k,v in hand.items() if k!='finger_vertices'},
                  arm="right",sequence=sequence,repeats=1,button_ids=button_ids,
                  environment_offset_m=env.offset.tolist(),scene_dimensions=scene.get("dimensions"))
    report["key_pose_ik"]={}; ik_failed=False
    def precheck(name,goal,seed):
        nonlocal ik_failed
        detail=report["key_pose_ik"][name]={"status":"unknown"}
        try:
            q=kin.solve(goal,seed,tcp_offset,diagnostics=detail);detail["status"]="pass";return q
        except ValueError as exc:
            detail.update(status="fail",error=str(exc));ik_failed=True;return seed.copy()
    home_arm=precheck("home",pose(*(np.array(cfg["home_tip"])+env.offset)),
                      np.array([initial_arm_positions[n] for n in RIGHT_ARM]))
    pre=precheck("approach_0",target_for(0,-cfg["approach_distance"]),home_arm)
    precheck("touch_0",target_for(0,cfg["press_depth"]),pre)
    planned={0:pre};report["home_arm_rad"]=home_arm.tolist()
    report["reachability"]={"0":pre.tolist()}
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
        actual_root_xyz_m=list(UsdGeom.XformCache().GetLocalToWorldTransform(world.stage.GetPrimAtPath(env.prefix + "/Astribot")).ExtractTranslation()),
        fixed_joint_world_anchor_m=list(UsdPhysics.FixedJoint(world.stage.GetPrimAtPath(meta["articulation_path"])).GetLocalPos0Attr().Get()),
        base_link_path=meta["base_link"], actual_base_link_xyz_m=base_body.get_world_pose()[0].tolist(),
        actual_base_link_quat_wxyz=base_body.get_world_pose()[1].tolist(), frame="world; root and base link have distinct fixed transforms")
    from pressb.astribot_validation import initial_overlap_check
    report["initial_overlap"] = initial_overlap_check(world.stage, meta, floor_path=env.prefix + "/Environment/Floor")
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
    cartesian_futures = {}
    epoch = 0
    inactive_idx = np.array([i for i,n in enumerate(names) if "arm_right" not in n and "gripper_right" not in n])

    def contacts(headers, data, actor_paths):
        for h in headers:
            actors = [actor_paths[x] for x in (h.actor0, h.actor1)]
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

    subscription = env.subscribe_contacts(contacts)

    def matrix(body):
        p, q = body.get_world_pose()
        t = np.eye(4)
        t[:3, 3] = p
        t[:3, :3] = Rotation.from_quat(np.roll(q, -1)).as_matrix()
        return t

    def snapshot(label):
        yield ("snapshot", label)

    def step():
        nonlocal step_index, base_initial, max_fk_error
        report["motion_started"] = True
        report["termination_phase"] = phase
        if phase == "press":
            report["press_phase_started"] = True
        forces.fill(0.)
        yield ("step", target.copy())
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
        records.append(dict(q=q, qd=robot.get_joint_velocities().copy(), target=target.copy(), tip=tip[:3, 3].copy(), tip_pose=tip.copy(), travel=travel,
                            force=forces.copy(), light=light_states.copy(), base=bp.copy(), base_quat=bq.copy(),
                            floor=active_floor, phase=phase, epoch=epoch, simulation_time=world.current_time,
                            time=(step_index+1)*dt))
        if step_index % 4 == 0 and not getattr(env,'physics_only',False):
            yield ("capture", dict(epoch=epoch, step=step_index, simulation_time=float(world.current_time)))
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
            yield from step()

    def move_arm(goal, minimum=1.):
        start = target[arm_idx].copy()
        duration = max(minimum, 1.875*float(np.max(np.abs(goal-start)))/cfg["joint_speed"])
        count = int(np.ceil(duration/dt))
        for i in range(1,count+1):
            target[arm_idx] = start+smooth_fraction(i/count)*(goal-start)
            yield from step()

    def cartesian(distance_start, distance_end, seconds, stop_on_press=False):
        seed = target[arm_idx].copy()
        count = max(1, int(np.ceil(seconds/dt)))
        key = (active_floor, float(distance_start), float(distance_end), count)
        if key not in cartesian_cache:
            future = cartesian_futures.get(key)
            if future is None:
                future = env.ik_pool.submit(solve_path,active_floor,distance_start,distance_end,count,seed.copy())
            yield ("ik", future)
            try:
                cartesian_cache[key] = future.result()
            except ValueError:
                report["termination_reason"] = "path_ik_failed"
                raise
        path = cartesian_cache[key]
        for i in range(1,count+1):
            if (i-1)%4 == 0 or i == count:
                seed = path[i]
            target[arm_idx] = seed
            travel = yield from step()
            if stop_on_press and travel[button_index[active_floor]]>=cfg["press_threshold"] and forces[button_index[active_floor]]>cfg["contact_threshold_n"]:
                return True
        return False

    def solve_path(floor,distance_start,distance_end,count,previous):
        path={}
        for k in range(1,count+1):
            if (k-1)%4 == 0 or k == count:
                distance=distance_start+(distance_end-distance_start)*smooth_fraction(k/count)
                previous=kin.solve(target_for(floor,distance),previous,tcp_offset)
                path[k]=previous.copy()
        return path

    def replay():
        nonlocal phase,active_floor,epoch
        from pressb.astribot_collection.contract import validate_commit
        from pressb.astribot_collection.replay import reconstruct_targets
        source=env.replay_source;validate_commit(source)
        original=json.loads((source/'report.json').read_text())
        if original['joint_names']!=names:raise ValueError('Replay joint ordering mismatch')
        with np.load(source/'trajectory.npz',allow_pickle=False) as archive:
            reference={k:archive[k] for k in archive.files}
        with np.load(source/'training.npz',allow_pickle=False) as archive:
            training={k:archive[k] for k in archive.files}
        commands=reconstruct_targets(reference['target'],training)
        last_phase=None
        for index,command in enumerate(commands):
            next_phase=str(reference['phase'][index]);next_epoch=int(reference['epoch'][index])
            if next_epoch!=epoch:
                robot.set_joints_default_state(positions=command.copy(),velocities=np.zeros_like(command))
                while not (yield ('reset',command.copy())):
                    phase='wait_for_wave_reset'
                    yield from step()
                epoch=next_epoch;light_states[:]=False
                for info in scene['buttons'].values():set_button_light(world.stage,info,False)
            if next_phase=='move_to_work_pose' and last_phase!='move_to_work_pose':
                gripper_world=matrix(wrist)@kin.fk('astribot_gripper_right_base',root='astribot_arm_right_link_7')
                surfaces=[]
                for suffix,body in finger_bodies.items():
                    tf=np.linalg.inv(gripper_world)@matrix(body)
                    points=np.asarray(hand['finger_vertices']['astribot_gripper_right_Link_'+suffix])
                    surfaces.append(points@tf[:3,:3].T+tf[:3,3])
                report['measured_closed_gap_m']=float(surfaces[0][:,0].min()-surfaces[1][:,0].max())
                report['closure_tracking_error_rad']=float(np.max(np.abs(robot.get_joint_positions()[grip_idx]-command[grip_idx])))
            phase=next_phase;active_floor=int(reference['floor'][index]);target[:]=command
            yield from step()
            last_phase=phase
        start_info=original['startup'];index=start_info['completed_step']
        startup_error=float(np.max(np.abs(records[index]['q'][arm_idx]-home_arm)))
        report['startup']=dict(status='pass' if startup_error<.025 else 'fail',completed_step=index,
            work_pose_error_rad=startup_error,unexpected_contacts_before_completion=0)
        for source_episode in original['episodes']:
            start,end=source_episode['start_step'],source_episode['end_step'];rows=records[start:end]
            hits=[start+i for i,r in enumerate(rows) if r['travel'][0]>=cfg['press_threshold'] and r['force'][0]>cfg['contact_threshold_n']]
            released=bool(rows[-1]['travel'][0]<cfg['release_threshold'])
            home_error=float(np.max(np.abs(rows[-1]['q'][arm_idx]-home_arm)))
            episodes.append(dict(floor=0,repeat=0,start_step=start,end_step=end,pressed=bool(hits),released=released,
                wrong_buttons=[],unexpected_contacts=len(unexpected),home_error_rad=home_error,
                max_travel_m=float(max(r['travel'][0] for r in rows)),
                success=bool(hits and released and home_error<.025 and not unexpected)))
            if hits:events.append(dict(type='pressed',floor=0,step=hits[0],repeat=0))
        ready=[r for r in records if r['phase']=='ready_hold'][-1]
        ready_error=float(np.max(np.abs(ready['q'][arm_idx]-initial_target[arm_idx])))
        report['ready_return']=dict(passed=ready_error<.025,max_joint_error_rad=ready_error)
        reset_error=float(np.max(np.abs(records[-1]['q']-target)))
        report['reset']=dict(passed=reset_error<.025,max_joint_error_rad=reset_error,
                             buttons_released=bool(records[-1]['travel'][0]<cfg['release_threshold']))
        end=int(training['physical_steps'][-1])
        prefix_success=any(r['phase']=='press' and r['travel'][0]>=cfg['press_threshold'] and r['force'][0]>cfg['contact_threshold_n']
                           for r in records[:end+1])
        report['replay']=dict(source=str(source),method='30Hz_next_sample_targets_linear_4tick_interpolation_then_raw_return',
            training_terminal_step=end,training_prefix_success=bool(prefix_success))
        report['dynamic_monitor']['coverage_complete']=True
        report['status']='passed' if prefix_success and all(e['success'] for e in episodes) and ready_error<.025 and reset_error<.025 else 'failed'

    try:
        if report["initial_overlap"]["status"] != "clear" or ik_failed:
            report["termination_reason"] = ("initial_collision" if report["initial_overlap"]["status"] == "collision"
                else "runtime_error" if report["initial_overlap"]["status"] == "unknown" else "target_ik_failed")
            yield from snapshot("precheck_failed")
            raise RuntimeError(f"Precheck blocked motion: {report['termination_reason']}")
        if getattr(env,'replay_source',None) is not None:
            yield from replay()
            return
        count=int(np.ceil((1.875*(cfg['approach_distance']+cfg['press_depth'])/cfg['press_speed_m_s'])/dt))
        key=(0,float(-cfg['approach_distance']),float(cfg['press_depth']),count)
        cartesian_futures[key]=env.ik_pool.submit(solve_path,0,-cfg['approach_distance'],cfg['press_depth'],count,
                                                np.asarray(planned[0],dtype=np.float32))
        yield from hold(1.5 if initial_arm_positions else .5)
        if initial_arm_positions:
            yield from snapshot("initial_ready")
        phase = "close_gripper"
        closed_targets = gripper_targets(hand["closed_angle_rad"])
        for i in range(240):
            for name, value in closed_targets.items():
                target[names.index(name)] = value*smooth_fraction((i+1)/240)
            yield from step()
        yield from hold(.5)
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
        yield from snapshot("closed")
        print("Closed gripper:", report["measured_closed_gap_m"], "m gap", flush=True)
        phase = "move_to_work_pose"
        yield from move_arm(home_arm, 3.)
        yield from hold(.6)
        startup_error = float(np.max(np.abs(robot.get_joint_positions()[arm_idx]-home_arm)))
        report["startup"] = dict(status="pass" if startup_error<.025 else "fail",
            completed_step=step_index-1, simulation_time=world.current_time,
            work_pose_error_rad=startup_error, unexpected_contacts_before_completion=len(unexpected))
        yield from snapshot("work_pose")
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
                yield from move_arm(planned[floor], 1.)
                yield from hold(.3)
                phase = "press"
                pressed = yield from cartesian(-cfg["approach_distance"], cfg["press_depth"],
                    1.875*(cfg["approach_distance"]+cfg["press_depth"])/cfg["press_speed_m_s"], True)
                # Allow a short settling interval at the final target.
                if not pressed:
                    for _ in range(60):
                        travel=yield from step()
                        if travel[button_index[floor]]>=cfg["press_threshold"] and forces[button_index[floor]]>cfg["contact_threshold_n"]:
                            pressed=True
                            break
                if pressed:
                    events.append(dict(type="pressed", floor=floor, step=step_index-1, repeat=repeat))
                    # Keep the contact command through the first 30-Hz success
                    # sample, so neither the terminal observation nor the prior
                    # next-sample action can contain a retract command.
                    held=0
                    while (step_index-1)%4:
                        yield from step();held+=1
                    report['success_sampling_hold_steps']=held
                    if repeat == 0:
                        yield from snapshot(f"press_{floor}")
                phase = "retract"
                # Retrace Cartesian targets from the current commanded contact position.
                current_distance = (world_from_root @ kin.tcp(target[arm_idx],tcp_offset))[0,3]-scene["buttons"][floor]["center"][0]
                yield from cartesian(current_distance, -cfg["approach_distance"], 1.5)
                yield from hold(.4)
                released = records[-1]["travel"][button_index[floor]]<cfg["release_threshold"]
                phase = "return_home"
                yield from move_arm(home_arm, 1.)
                yield from hold(.3)
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
            yield from move_arm(initial_target[arm_idx], 2.)
            phase = "ready_hold"
            yield from hold(1.5)
            ready_error = float(np.max(np.abs(robot.get_joint_positions()[arm_idx]-initial_target[arm_idx])))
            report["ready_return"] = dict(passed=ready_error<.025, max_joint_error_rad=ready_error)
            if ready_error >= .025:
                raise RuntimeError("Failed to return to initial raised-arm posture")
        phase="finished"
        yield from snapshot("finished")
        # Verify reset recovery to the same ready posture without driving the
        # robot through a discontinuous target trajectory during an episode.
        robot.set_joints_default_state(positions=target.copy(), velocities=np.zeros_like(target))
        while not (yield ("reset", target.copy())):
            phase='wait_for_wave_reset'
            yield from step()
        epoch += 1
        light_states[:] = False if latched else True
        if latched:
            for info in scene["buttons"].values():
                set_button_light(world.stage, info, False)
        phase = "reset_check"
        yield from hold(.6)
        yield from snapshot("reset")
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
            yield from snapshot("failure")
        except Exception as exc:
            report["failure_snapshot_error"] = str(exc)
        raise
    finally:
        report.update(episodes=episodes, unexpected_contacts=unexpected, events=events, physics_steps=step_index,
                      max_fk_error_m=max_fk_error, joint_names=names,
                      success_count=sum(e["success"] for e in episodes),video_frames=env.video_frames)
        (args.output / "camera_timing.json").write_text(json.dumps(env.camera_timing) + "\n")
        if records:
            np.savez_compressed(args.output/"trajectory.npz", **{key:np.asarray([r[key] for r in records]) for key in records[0]})
            report["base_drift_m"]=float(np.max(np.linalg.norm(np.array([r["base"] for r in records])-base_initial,axis=1)))
            report["inactive_joint_max_abs_rad"] = float(np.max(np.abs(np.array([r["q"] for r in records])[:,inactive_idx])))
            report["inactive_joint_max_error_rad"] = float(np.max(np.abs(
                np.array([r["q"] for r in records])[:,inactive_idx]-initial_target[inactive_idx])))
    if report["status"] != "passed":
        raise RuntimeError("Physical episode acceptance failed")
