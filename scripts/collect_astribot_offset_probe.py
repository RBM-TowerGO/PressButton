"""Collect synchronized privileged labels without modifying the residual policy."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_astribot_online_rl import (Recorder, EpisodeStop, check_health, load_contract,
                                    predict, atomic_json, sha256)
from astribot_closedloop_control import ARM, interpolate_target, validate_prediction
from pressb.online_rl.astribot_features import normalize_latent
from pressb.online_rl.offset_probe import PANEL_AXES, offsets_mm, pose_matrix, group_split, near_mask


class ProbeRecorder(Recorder):
    def __init__(self, *args):
        super().__init__(*args)
        self.contact_points = []
        self.contact_positions_supported = None
        self.ever_contact = False
        self.contacts_this_tick = 0

    def contact(self, headers, data, actor_paths):
        super().contact(headers, data, actor_paths)
        for h in headers:
            actors = [actor_paths[x] for x in (h.actor0, h.actor1)]
            if self.button['body_path'] not in actors or not any(p in self.tips for p in actors):
                continue
            self.ever_contact = True
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                self.contacts_this_tick += 1
                value = getattr(data[i], 'position', None)
                self.contact_positions_supported = value is not None
                if value is not None:
                    self.contact_points.append(np.asarray(value, dtype=float).tolist())

    def tick(self, target):
        self.contact_points = []
        self.contacts_this_tick = 0
        return super().tick(target)

    def geometry(self):
        env = self.env
        wrist = pose_matrix(*env.wrist.get_world_pose())
        tcp = wrist @ np.asarray(env.hand['tcp_in_wrist'])
        button_pose = pose_matrix(*env.buttons[0].get_world_pose())
        # Authored cap collision is a 6-mm cylinder along local X.
        face = (button_pose @ np.array([-.003, 0., 0., 1.]))[:3]
        axes = button_pose[:3, :3] @ PANEL_AXES
        vertices = []
        for name, points in env.hand['finger_vertices'].items():
            suffix = name.rsplit('_', 1)[-1]
            transform = pose_matrix(*env.finger_bodies[suffix].get_world_pose())
            vertices.append(np.asarray(points) @ transform[:3, :3].T + transform[:3, 3])
        vertices = np.concatenate(vertices)
        normal_gap = (vertices - face) @ axes[:, 2] * 1000.
        tip_gap = float(offsets_mm(tcp[:3, 3], face, axes)[2])
        return dict(label_mm=offsets_mm(tcp[:3, 3], face, axes).tolist(),
                    tcp_world=tcp[:3, 3].tolist(), face_world=face.tolist(), panel_axes=axes.tolist(),
                    finger_min_gap_mm=float(normal_gap.min()),
                    tcp_to_front_normal_mm=float(tip_gap - normal_gap.min()),
                    travel_m=float(button_pose[0, 3] - self.button['rest_x']),
                    force_n=float(self.forces[0]), contact=bool(self.contacts_this_tick),
                    ever_contact=self.ever_contact, light=self.light,
                    contact_positions_supported=self.contact_positions_supported,
                    contact_points_world=self.contact_points,
                    contact_offsets_mm=[offsets_mm(p, face, axes).tolist() for p in self.contact_points])


class Samples:
    def __init__(self, path, scene, split, args):
        self.path, self.scene, self.split, self.args = path, scene, split, args
        self.rows, self.arrays = [], []

    def capture(self, runtime, recorder, index, *, requested=None, record=True, images=None):
        if images is None:
            images = runtime.render()[0]
        state, velocity = runtime.q[0].copy(), runtime.qd[0].copy()
        stamp = float(runtime.world.current_time)
        geometry = recorder.geometry() if record else None
        seed = self.args.seed + self.scene * 100003 + index
        result = predict(self.args.endpoint, images, state, seed)
        if (stamp != float(runtime.world.current_time) or not np.array_equal(state, runtime.q[0])
                or not np.array_equal(velocity, runtime.qd[0])):
            raise RuntimeError('Physics advanced during inference')
        action = np.asarray(result['action8'], dtype=np.float32)
        if action.shape != (7, 8) or not np.isfinite(action).all():
            raise ValueError('Invalid base action')
        latent = normalize_latent(result['vlm_latent'])
        if record:
            row = dict(scene=self.scene, split=self.split, request=index, seed=seed,
                       sim_time=stamp, physics_step=recorder.physics_step, requested_mm=requested,
                       image_sha256={k:hashlib.sha256(np.ascontiguousarray(images[k]).tobytes()).hexdigest()
                                     for k in ('head', 'wrist_right')}, **geometry)
            self.rows.append(row)
            self.arrays.append((state, velocity, action, latent, np.asarray(geometry['label_mm'])))
            with (self.path / 'observations.jsonl').open('a') as out:
                out.write(json.dumps(row, allow_nan=False) + '\n')
        return result

    def save(self):
        if not self.arrays:
            return
        keys = ('q', 'qd', 'base', 'latent', 'label_mm')
        data = {k: np.stack([a[i] for a in self.arrays]) for i, k in enumerate(keys)}
        data.update(scene=np.full(len(self.rows), self.scene), split=np.full(len(self.rows), self.split),
                    contact=np.array([r['contact'] for r in self.rows]),
                    ever_contact=np.array([r['ever_contact'] for r in self.rows]),
                    light=np.array([r['light'] for r in self.rows]))
        temp = self.path / 'samples.tmp.npz'
        np.savez_compressed(temp, **data)
        temp.replace(self.path / 'samples.npz')


def move(recorder, target, speed=.4, settle=24):
    start = recorder.command.copy()
    steps = max(4, int(np.ceil(np.max(np.abs(target - start)) / speed * 120)))
    for i in range(1, steps + 1):
        recorder.tick(start + (target-start) * (i / steps))
    for _ in range(settle):
        recorder.tick(target)
    # Physics records are not needed by the controlled sampler after each move.
    recorder.records.clear()


def controlled(runtime, cfg, contract, scene, split, path, args):
    from pressb.astribot_kinematics import AstribotKinematics, gripper_targets
    from pressb.base_pose import world_target_in_root
    runtime.prepare_wave([cfg], [path])
    recorder = ProbeRecorder(runtime, cfg, contract)
    samples = Samples(path, scene, split, args)
    recorder.start()
    failures = []
    try:
        env = recorder.env
        kin = AstribotKinematics()
        root = pose_matrix(*env.base_body.get_world_pose())
        rotation = np.array([[0,0,1],[1,0,0],[0,1,0]], float)
        tcp_offset = np.asarray(env.hand['tcp_in_wrist'])
        target = np.asarray(contract['fixed_action32'], dtype=np.float32).copy()
        for name, angle in gripper_targets(env.hand['closed_angle_rad']).items():
            target[list(env.robot.dof_names).index(name)] = angle
        target[ARM] = recorder.command[ARM]
        move(recorder, target)
        face = np.asarray(env.scene['buttons'][0]['center'])
        seed = target[ARM].copy()

        def go(position):
            nonlocal seed, target
            goal = world_target_in_root(root, position, rotation)
            seed = kin.solve(goal, seed, tcp_offset)
            target[ARM] = seed
            move(recorder, target)

        go(np.asarray(cfg['home_tip']) + env.offset)
        go(face + PANEL_AXES[:, 2] * .065)
        grid = list(itertools.product(args.grid, args.grid))
        # Far-to-near order; each lateral change starts after retreating.
        index = 0
        for horizontal, vertical in grid:
            lateral = PANEL_AXES[:, :2] @ (np.array([horizontal, vertical]) / 1000.)
            try:
                go(face + lateral + PANEL_AXES[:, 2] * .030)
            except ValueError as exc:
                for depth in args.depths:
                    failures.append(dict(request=index, requested_mm=[horizontal, vertical, depth], reason='ik_unreachable', detail=str(exc)))
                    index += 1
                continue
            for depth in sorted(args.depths, reverse=True):
                requested = [horizontal, vertical, depth]
                try:
                    go(face + lateral + PANEL_AXES[:, 2] * (depth / 1000.))
                except ValueError as exc:
                    failures.append(dict(request=index, requested_mm=requested, reason='ik_unreachable', detail=str(exc)))
                    index += 1
                    continue
                if recorder.ever_contact:
                    raise EpisodeStop('premature_target_contact')
                # Inference/label errors are infrastructure failures, never IK failures.
                samples.capture(runtime, recorder, index, requested=requested)
                index += 1
            go(face + lateral + PANEL_AXES[:, 2] * .030)
            samples.save()
            print(json.dumps(dict(kind='controlled_progress', scene=scene, valid=len(samples.rows), attempted=index)), flush=True)
        status = 'completed'
    except EpisodeStop as exc:
        failures.append(dict(reason=exc.reason))
        status = 'physical_failure'
    finally:
        samples.save()
        recorder.close()
    result = dict(scene=scene, split=split, status=status, valid=len(samples.rows), failures=failures,
                  planned=len(args.grid)**2 * len(args.depths))
    atomic_json(path / 'summary.json', result)
    return result


def closedloop(runtime, cfg, contract, scene, path, args, record=True, shadow_audit=False):
    from pressb.astribot_validation import initial_overlap_check
    runtime.prepare_wave([cfg], [path])
    recorder = ProbeRecorder(runtime, cfg, contract)
    samples = Samples(path, scene, 3, args)
    recorder.start()
    reason, success, control = 'timeout', False, 0
    trace = hashlib.sha256()
    try:
        overlap = initial_overlap_check(runtime.world.stage, recorder.env.meta,
                                        floor_path=recorder.env.prefix + '/Environment/Floor')
        if overlap['status'] != 'clear':
            raise EpisodeStop('initial_overlap_' + overlap['status'])
        recorder.tick(recorder.command)
        images = runtime.render()[0]
        index = 0
        while control < round(args.max_seconds * 30):
            if shadow_audit:
                baseline = samples.capture(runtime, recorder, index, record=False, images=images)
            result = samples.capture(runtime, recorder, index, record=record, images=images)
            if shadow_audit:
                for field in ('action8', 'action32', 'vlm_latent'):
                    np.testing.assert_array_equal(baseline[field], result[field],
                                                  err_msg='Read-only capture changed frozen prediction')
            targets = validate_prediction(result['action8'], result['action32'], contract['fixed_action32'], recorder.limits)
            trace.update(np.ascontiguousarray(targets).tobytes())
            for target in targets:
                if control >= round(args.max_seconds * 30):
                    break
                for step in interpolate_target(recorder.command, target):
                    success = recorder.tick(step) or success
                    trace.update(np.ascontiguousarray(runtime.q[0]).tobytes())
                control += 1
                if success:
                    break
            images = runtime.render()[0]
            index += 1
            if index % 25 == 0:
                samples.save()
            if success:
                reason = 'physical_press_and_light'
                break
        if record:
            # Terminal observation is labelled separately and never executed.
            samples.capture(runtime, recorder, index, record=True, images=images)
    except EpisodeStop as exc:
        reason = exc.reason
    finally:
        samples.save()
        recorder.close()
    labels = np.array([r['label_mm'] for r in samples.rows]).reshape(-1, 3)
    result = dict(scene=scene, status=reason, success=success and reason=='physical_press_and_light',
                  observations=len(labels), near_observations=int(near_mask(labels).sum()),
                  physics_steps=recorder.physics_step, trace_sha256=trace.hexdigest(), record=record,
                  paired_frozen_requests=index if shadow_audit else 0)
    atomic_json(path / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['controlled', 'closedloop', 'audit'], required=True)
    p.add_argument('--model-run', type=Path, default=ROOT/'models/astribot_head_right_joint8_ft_20261018')
    p.add_argument('--config', type=Path, default=ROOT/'docs/vib820_experiment_spec/config_v1_2.json')
    p.add_argument('--endpoint', default='http://127.0.0.1:8801')
    p.add_argument('--scenes', type=int, default=100)
    p.add_argument('--closedloop-min', type=int, default=40)
    p.add_argument('--closedloop-max', type=int, default=100)
    p.add_argument('--grid', type=float, nargs='+', default=[-8,-4,0,4,8])
    p.add_argument('--depths', type=float, nargs='+', default=[5,15,30])
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-seconds', type=float, default=25.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, health = check_health(args.endpoint, args.model_run, 8000, 'state_latent')
    contract, train_ids = load_contract(args.model_run, 'train')
    # Independent scene IDs: first 100 train IDs for controlled, separate train
    # IDs for closedloop. Existing VLA training membership is identical in both.
    ids = np.random.default_rng(args.seed).permutation(train_ids).tolist()
    selected = ids[:args.scenes] if args.mode == 'controlled' else ids[100:100+args.closedloop_max]
    split = group_split(np.array(ids[:args.scenes]), args.seed) if args.mode == 'controlled' else [3]*len(selected)
    protocol = dict(mode=args.mode, args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    model_sha256=manifest['model_sha256'], policy_health=health, scene_ids=selected,
                    split=list(map(int,split)), axes=PANEL_AXES.tolist(), units='mm',
                    source_sha256={str(f.relative_to(ROOT)):sha256(f) for f in
                                   [Path(__file__), ROOT/'src/pressb/online_rl/offset_probe.py']})
    previous = args.output / 'protocol.json'
    if previous.exists() and json.loads(previous.read_text()) != protocol:
        raise RuntimeError('Output already belongs to a different protocol')
    atomic_json(previous, protocol)
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, create_new_stage=False, active_gpu=args.gpu,
        physics_gpu=args.gpu, multi_gpu=False, limit_cpu_threads=8, renderer='RayTracedLighting',
        anti_aliasing=3, width=960, height=600, disable_viewport_updates=True,
        extra_args=['--/app/asyncRendering=false','--/rtx/ecoMode/enabled=false',
                    '--/validate/p2p/enabled=false','--/validate/iommu/enabled=false',
                    '--/app/extensions/fsWatcherEnabled=false']))
    runtime = None
    summaries = []
    try:
        from astribot_closedloop_runtime import LiveRuntime
        config = json.loads(args.config.read_text())
        shared = args.output / ('shared_' + str(time.time_ns()))
        runtime = LiveRuntime(app, config, shared, 1, args.gpu, 'gpu', 1, 'individual',
                              render_cameras=['head','wrist_right'])
        for scene, part in zip(selected, split):
            cfg = deepcopy(config)
            condition = contract['scene_conditions'][str(scene)]
            cfg.update(robot_base_x=condition['x_m'], robot_base_y=condition['y_m'], robot_base_yaw_rad=condition['yaw_rad'])
            path = args.output / f'scene_{scene:06d}'
            if path.exists() and not (path/'summary.json').exists():
                path.rename(args.output / f'interrupted_{scene:06d}_{time.time_ns()}')
            if (path/'summary.json').exists():
                result = json.loads((path/'summary.json').read_text())
            elif args.mode == 'controlled':
                result = controlled(runtime, cfg, contract, scene, int(part), path, args)
            elif args.mode == 'closedloop':
                result = closedloop(runtime, cfg, contract, scene, path, args)
            else:
                off = closedloop(runtime, cfg, contract, scene, path/'off', args, record=False)
                repeat = closedloop(runtime, cfg, contract, scene, path/'off_repeat', args, record=False)
                on = closedloop(runtime, cfg, contract, scene, path/'on', args, record=True, shadow_audit=True)
                reset_deterministic = off['trace_sha256'] == repeat['trace_sha256']
                exact_trace = off['trace_sha256'] == on['trace_sha256']
                passed = on['paired_frozen_requests'] > 0
                result = dict(scene=scene, passed=passed, off=off, off_repeat=repeat, on=on,
                              independent_reset_deterministic=reset_deterministic,
                              independent_reset_instrumentation_trace_equal=exact_trace,
                              acceptance='Every frozen request has identical action8/action32/latent with and without reads; no physics advance',
                              limitation='Independent-reset equivalence is only established when trace hashes match')
                atomic_json(path/'summary.json', result)
                if not passed:
                    raise RuntimeError('Read-only instrumentation changed closedloop trace')
            summaries.append(result)
            atomic_json(args.output/'summary.json', dict(mode=args.mode, scenes=summaries))
            print(json.dumps(result), flush=True)
            if args.mode == 'audit':
                break
            if args.mode == 'closedloop' and len(summaries)>=args.closedloop_min and sum(s['near_observations']>0 for s in summaries)>=20:
                break
    except BaseException:
        atomic_json(args.output/'error.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        if runtime is not None:
            runtime.close()
        app.close()


if __name__ == '__main__':
    main()
