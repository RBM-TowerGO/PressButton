"""Paired validation report with measured rigid-body fingertip diagnostics."""
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path('/home/js/robot/VLA/PressButton/PressB')
sys.path.insert(0, str(ROOT / 'src'))
from pressb.astribot_kinematics import AstribotKinematics, gripper_targets
from astribot_closedloop_control import ARM, GRIP, SIGNS

THRESHOLDS = dict(stagnation_max_joint_rad=.1, near_button_m=.02,
                  meaningful_approach_m=.05, retreat_after_closest_m=.03,
                  final_5s_approach_m=.01, initial_state_max_difference_rad=1e-5,
                  initial_image_mean_absolute_difference=0.02)

def read(path):
    return json.loads(Path(path).read_text())

def classify(success, reason, max_joint, max_force, initial, minimum, final, last5_progress):
    if success:
        return '成功'
    if reason != 'timeout':
        return '其他终止：' + reason
    if max_force > .02:
        return '接触但未完成按压'
    if max_joint < THRESHOLDS['stagnation_max_joint_rad']:
        return '初始姿态附近停滞'
    if minimum <= THRESHOLDS['near_button_m']:
        return '进入按钮近区但未有效接触'
    if initial - minimum >= THRESHOLDS['meaningful_approach_m'] and final - minimum >= THRESHOLDS['retreat_after_closest_m']:
        return '接近后偏离'
    if last5_progress >= THRESHOLDS['final_5s_approach_m']:
        return '超时时仍在接近'
    return '移动但未进入按钮近区'

def transform_points(positions, quaternions_wxyz, local):
    return Rotation.from_quat(np.roll(quaternions_wxyz, -1, axis=-1)).apply(
        np.broadcast_to(local, positions.shape)) + positions

def reference_points(meta):
    kin = AstribotKinematics()
    points = {}
    for suffix in ('L11', 'R11'):
        name = 'astribot_gripper_right_Link_' + suffix
        vertices = np.unique(np.asarray(meta['hand']['finger_vertices'][name]), axis=0)
        closed = kin.fk(name, gripper_targets(meta['hand']['closed_angle_rad']), root='astribot_gripper_right_base')
        z = (vertices @ closed[:3, :3].T + closed[:3, 3])[:, 2]
        points[suffix] = vertices[z >= z.max() - .001].mean(0)
    return points

def diagnose(path, audit_row, meta, local_points):
    report = read(path / 'report.json')
    with np.load(path / 'trajectory.npz', allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    assert trace['finger_positions'].shape == (len(trace['q']), 2, 3)
    assert trace['finger_orientations'].shape == (len(trace['q']), 2, 4)
    assert np.isfinite(trace['finger_positions']).all() and np.isfinite(trace['finger_orientations']).all()
    face_offset = np.asarray(meta['scene']['buttons']['0']['center']) - np.asarray(meta['scene']['buttons']['0']['body_center'])
    face = trace['button_position'] + face_offset
    tip = np.stack([transform_points(trace['finger_positions'][:, i], trace['finger_orientations'][:, i], local_points[s])
                    for i, s in enumerate(('L11', 'R11'))], axis=1)
    distance_each = np.linalg.norm(tip - face[:, None, :], axis=2)
    distance = distance_each.min(axis=1)
    closest = int(distance.argmin()); finger = int(distance_each[closest].argmin())
    relative_time = trace['time'] - trace['time'][0]
    last5 = int(np.searchsorted(relative_time, max(0., relative_time[-1] - 5)))
    initial, minimum, final = map(float, (distance[0], distance[closest], distance[-1]))
    last5_progress = float(distance[last5] - final)
    max_joint = float(np.abs(trace['q'][:, ARM] - trace['q'][0, ARM]).max())
    g = (trace['q'][:, GRIP] * SIGNS).mean(1)
    requests = read(path / 'policy_requests.json')
    predicted_step = [np.abs(np.asarray(r['action8'])[0, :7] - np.asarray(r['state32'])[ARM]).mean() for r in requests]
    mode = classify(report['success'], report['termination_reason'], max_joint, report['max_force_n'], initial, minimum, final, last5_progress)
    near_indices = np.flatnonzero(distance <= THRESHOLDS['near_button_m'])
    return dict(candidate_id=report['candidate_id'], success=report['success'], failure_mode=mode,
                termination_reason=report['termination_reason'], simulated_seconds=report['simulated_seconds'],
                initial_distance_mm=initial*1000, minimum_distance_mm=minimum*1000, final_distance_mm=final*1000,
                closest_time_seconds=float(relative_time[closest]), closest_finger=('L11', 'R11')[finger],
                closest_offset_xyz_mm=((tip[closest, finger] - face[closest])*1000).tolist(),
                final_5s_distance_reduction_mm=last5_progress*1000,
                first_near_time_seconds=float(relative_time[near_indices[0]]) if len(near_indices) else None,
                max_arm_change_rad=max_joint, max_force_n=report['max_force_n'], max_travel_mm=report['max_travel_m']*1000,
                gripper_closed_fraction=float((np.abs(g-meta['hand']['closed_angle_rad']) <= .1).mean()),
                first_action_mean_delta_rad=float(np.mean(predicted_step)) if predicted_step else None,
                initial_q32=report['initial_q32'], inference_latency_seconds=report.get('inference_latency_seconds'),
                kinematic_reference_diagnostic=audit_row['kinematic_tip_diagnostic'],
                diagnostic_note='Distance is between measured rigid-body distal mesh reference points and moving button-face center, not exact collision-surface clearance; labels describe behavior, not proven causes.')

def summarize(root):
    manifest = read(root / 'experiment.json')
    if (root / 'parallel_transition.json').exists():
        transition = read(root / 'parallel_transition.json')
        for step in (8000,17000):
            path = root / f'step_{step:06d}/protocol.json'
            protocol = read(path)
            protocol['execution'] = f"Concurrent independent environments, limit {transition['concurrent_environments']}; completed serial 8000 prefix retained. See component protocols and parallel_transition.json."
            path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    groups = {}
    protocols = {}
    for step in (8000, 17000):
        run = root / f'step_{step:06d}'
        protocol, summary, audit = [read(run / filename) for filename in ('protocol.json', 'summary.json', 'audit.json')]
        assert protocol['split'] == 'val' and protocol['checkpoint_step'] == step
        assert protocol['candidate_ids'] == manifest['candidate_ids']
        assert protocol['model_sha256'] == manifest['models'][str(step)]['sha256']
        assert summary['complete'] and audit['passed'] and audit['complete'] and audit['all_video_frames_decoded']
        assert summary['completed'] == audit['audited'] == 40
        assert summary['successes'] == audit['successes']
        assert [r['candidate_id'] for r in audit['results']] == manifest['candidate_ids']
        meta = read(run / 'shared/metadata.json'); local_points = reference_points(meta)
        rows = [diagnose(run / f'candidate_{r["candidate_id"]:06d}', r, meta, local_points) for r in audit['results']]
        groups[str(step)] = dict(successes=sum(r['success'] for r in rows), count=40,
                                failure_modes=dict(Counter(r['failure_mode'] for r in rows if not r['success'])),
                                reference_points_local={k:v.tolist() for k,v in local_points.items()}, results=rows)
        protocols[str(step)] = protocol
    for key in ('candidate_ids', 'execute_horizon', 'max_seconds', 'scene_config_sha256', 'data_contract_sha256', 'source_sha256'):
        assert protocols['8000'][key] == protocols['17000'][key], f'Unmatched protocol: {key}'
    pairs = []
    for old, new in zip(groups['8000']['results'], groups['17000']['results']):
        cid = old['candidate_id']; assert cid == new['candidate_id']
        state_error = float(np.max(np.abs(np.asarray(old['initial_q32']) - new['initial_q32'])))
        image_errors = {}
        first_requests = []
        for step in (8000, 17000):
            requests = read(root / f'step_{step:06d}/candidate_{cid:06d}/policy_requests.json')
            assert all(r['seed'] == 420000 + cid*100003 + r['request_index'] for r in requests)
            first_requests.append(requests[0])
        for camera in ('head', 'wrist_right'):
            images = [np.asarray(Image.open(root/f'step_{s:06d}/candidate_{cid:06d}/initial_{camera}.png').convert('RGB'), dtype=float) for s in (8000,17000)]
            image_errors[camera] = float(np.abs(images[0] - images[1]).mean()/255)
        initial_matched = state_error <= THRESHOLDS['initial_state_max_difference_rad'] and max(image_errors.values()) <= THRESHOLDS['initial_image_mean_absolute_difference']
        outcome = '均成功' if old['success'] and new['success'] else '新模型改善' if new['success'] else '新模型退步' if old['success'] else '均失败'
        pairs.append(dict(candidate_id=cid, outcome=outcome, initial_state_difference_rad=state_error,
                          initial_image_mae=image_errors, initial_conditions_matched=initial_matched,
                          old_mode=old['failure_mode'], new_mode=new['failure_mode'],
                          old_time_s=old['simulated_seconds'],new_time_s=new['simulated_seconds'],
                          old_min_distance_mm=old['minimum_distance_mm'],new_min_distance_mm=new['minimum_distance_mm']))
    counts = dict(Counter(p['outcome'] for p in pairs))
    gained, lost = counts.get('新模型改善',0), counts.get('新模型退步',0)
    discordant = gained+lost
    exact_p = min(1.,2*sum(math.comb(discordant,k) for k in range(min(gained,lost)+1))/2**discordant) if discordant else 1.
    both = [p for p in pairs if p['outcome']=='均成功']
    result = dict(complete=True, split='validation', trials_per_model=40, repeats_per_scene=1,
                  paired_initial_conditions_verified=all(p['initial_conditions_matched'] for p in pairs),
                  groups=groups, pairs=pairs, paired_outcomes=counts, exact_mcnemar_p=exact_p,
                  common_success_time_difference_median_s=float(np.median([p['new_time_s']-p['old_time_s'] for p in both])) if both else None,
                  thresholds=THRESHOLDS, video_count=320, all_videos_fully_decoded=True,
                  limits=['Validation-set comparison; not a fresh test-set generalization estimate.',
                          'One matched rollout per scene and model; no across-seed stability claim.',
                          'Distance categories are descriptive diagnostics, not proof of the failure cause.',
                          'Synchronous simulation; no real-time robot claim.'])
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (root/'paired_scenes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=[k for k in pairs[0] if k!='initial_image_mae']);writer.writeheader()
        writer.writerows({k:v for k,v in p.items() if k!='initial_image_mae'} for p in pairs)
    lines=['# Astribot 8,000／17,000 步闭环配对对照','',
           '40个冻结验证场景，每个模型各一次；相同场景集合、逐场景逐请求随机种子、7步执行、40秒上限。','',
           '| 模型 | 成功 | 成功率 |','|---|---:|---:|']
    for step in ('8000','17000'):
        n=groups[step]['successes'];lines.append(f'| {step}步 | {n}/40 | {n/40:.1%} |')
    lines += ['',f'配对变化：{counts}。精确McNemar双侧p值={exact_p:.4f}。',
              f'初始状态与图像一致性门槛全部通过：{result["paired_initial_conditions_verified"]}。',
              '若该项为false，须先复核不匹配场景，不能作严格配对优劣结论。','',
              '## 失败细分','', '| 类型 | 8000步 | 17000步 |','|---|---:|---:|']
    modes=sorted(set(groups['8000']['failure_modes']) | set(groups['17000']['failure_modes']))
    for mode in modes:
        lines.append(f"| {mode} | {groups['8000']['failure_modes'].get(mode,0)} | {groups['17000']['failure_modes'].get(mode,0)} |")
    lines += ['','## 逐场景对照','', '| 场景 | 8000步 | 17000步 | 原/新最近距离mm |','|---:|---|---|---:|']
    for p in pairs:
        lines.append(f"| {p['candidate_id']} | {p['old_mode']} | {p['new_mode']} | {p['old_min_distance_mm']:.1f} / {p['new_min_distance_mm']:.1f} |")
    lines += ['','## 解释边界','',
              '近区定义为两指尖远端网格参考点之一距动态按钮面中心不超过20mm；参考点不是精确碰撞表面。停滞定义为右臂各关节相对初始姿态变化均小于0.1rad。接近后偏离定义为曾缩短距离至少50mm，最终又比最近点远至少30mm。末5秒距离缩短至少10mm归为超时时仍在接近。',
              '分类按成功、其他终止、接触未成功、停滞、近区、偏离、仍接近、其余移动的顺序互斥划分。',
              '这是验证集上的单次配对仿真，不代表多种子稳定性或真机实时能力。测试集未参与本轮，模型权重与控制参数未调优。',
              '全部320段视频已逐帧解码，原独立审计核验状态输入、动作插值、接触评分、相机位姿和时间戳。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','pairs')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    summarize(parser.parse_args().root)
