"""Independent trajectory, observation, video-clock and scoring audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image
from astribot_closedloop_control import ARM,GRIP,SIGNS,interpolate_target
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.astribot_kinematics import AstribotKinematics,RIGHT_ARM
from pressb.base_pose import root_transform


def _audit_trial(path,decode=True):
    r=json.loads((path/'report.json').read_text())
    trace=np.load(path/'trajectory.npz',allow_pickle=False)
    requests=json.loads((path/'policy_requests.json').read_text())
    cameras=json.loads((path/'camera_timing.json').read_text())
    hits=(trace['travel']>=.001)&(trace['force']>.02)
    assert np.array_equal(hits,trace['physical_hit'])
    if r['success']:
        assert hits.any() and trace['light'].any() and not r['unexpected_contacts']
        assert r['termination_reason']=='physical_press_and_light'
    assert len(trace['q'])==r['physics_steps']
    np.testing.assert_allclose(np.diff(trace['time']),1/120,rtol=0,atol=2e-6)
    for index,request in enumerate(requests):
        start=request['physics_step']
        np.testing.assert_array_equal(request['state32'],trace['q'][start-1])
        assert abs(request['observation_time']-trace['time'][start-1])<1e-6
        end=requests[index+1]['physics_step'] if index+1<len(requests) else len(trace['q'])
        previous=trace['target'][start-1];expected=[]
        for following in np.asarray(request['action32'],dtype=np.float32)[:r['execute_horizon']]:
            ticks=interpolate_target(previous,following);expected.extend(ticks);previous=ticks[-1]
        np.testing.assert_allclose(trace['target'][start:end],np.asarray(expected)[:end-start],atol=1e-7,rtol=0)
        if index==0:
            for name,key in [('head','head_sha256'),('wrist_right','wrist_right_sha256')]:
                pixels=np.asarray(Image.open(path/f'initial_{name}.png').convert('RGB'))
                assert hashlib.sha256(pixels.tobytes()).hexdigest()==request[key]
    for row in cameras:
        assert all(abs(value-row['time'])<1e-6 for value in row['cameras'].values())
        assert set(row['pose_sync_errors'])=={'native_head','native_wrist_left','native_wrist_right'}
        assert max(row['pose_sync_errors'].values())<1e-4
    stamps=sorted({(x['physics_step']-1)/120 for x in cameras})
    if decode:
        for name,count in r['frames'].items():
            with av.open(str(path/f'{name}.mp4')) as reader:
                actual=[float(frame.pts*frame.time_base) for frame in reader.decode(video=0)]
            assert len(actual)==count
            np.testing.assert_allclose(actual,stamps,atol=1e-6,rtol=0)
    state8=np.c_[trace['q'][:,ARM],(trace['q'][:,GRIP]*SIGNS).mean(1)]
    meta=json.loads((path.parent/'shared/metadata.json').read_text())
    condition=r['scene_condition']
    world=root_transform([condition['x_m'],condition['y_m'],meta['meta']['base_lift_m']],condition['yaw_rad'])
    kin=AstribotKinematics();tcp=np.asarray(meta['hand']['tcp_in_wrist']);tip=[];wrist=[]
    sampled=trace['q'][::4]
    for q in sampled:
        pose=world@kin.fk('astribot_arm_right_link_7',dict(zip(RIGHT_ARM,q[ARM])))
        wrist.append(pose[:3,3]);tip.append((pose@tcp)[:3,3])
    tip=np.asarray(tip);button=np.asarray(meta['scene']['buttons']['0']['center'])
    fk_error=float(np.max(np.linalg.norm(np.asarray(wrist)-trace['wrist_position'][::4],axis=1)))
    return dict(candidate_id=r['candidate_id'],success=r['success'],reason=r['termination_reason'],
        simulated_seconds=r['simulated_seconds'],wall_seconds=r['wall_seconds'],requests=len(requests),
        first_physical_hit_step=int(np.flatnonzero(hits)[0]+1) if hits.any() else None,
        max_travel_m=float(trace['travel'].max()),max_force_n=float(trace['force'].max()),
        initial_state8=state8[0].tolist(),final_state8=state8[-1].tolist(),
        max_arm_change_rad=float(np.max(np.abs(state8[:,:7]-state8[0,:7]))),
        gripper_range_rad=[float(state8[:,7].min()),float(state8[:,7].max())],
        final_wrist_displacement_m=float(np.linalg.norm(trace['wrist_position'][-1]-trace['wrist_position'][0])),
        kinematic_tip_diagnostic=dict(final_offset_from_button_mm=((tip[-1]-button)*1000).tolist(),
            minimum_distance_mm=float(np.linalg.norm(tip-button,axis=1).min()*1000),
            max_kinematic_wrist_error_vs_physics_mm=fk_error*1000,
            note='Kinematic reference TCP diagnostic only; task success is scored from actual contact and button travel.'),
        max_camera_pose_error=max(max(x['pose_sync_errors'].values()) for x in cameras),
        inference_latency_seconds=r.get('inference_latency_seconds'),
        video_frames_each=len(stamps),videos_fully_decoded=decode,unexpected_contacts=r['unexpected_contacts'])


def audit_trial(path,decode=True):
    paths=[path/name for name in ('report.json','trajectory.npz','policy_requests.json','camera_timing.json',
        'initial_head.png','initial_wrist_right.png','head.mp4','wrist_right.mp4','panel.mp4','overview.mp4')]
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache=path/'independent_audit.json'
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved.get('audit_code_sha256')==code and saved.get('file_sha256')==fingerprints and (not decode or saved.get('videos_fully_decoded')):
            return saved
    result=_audit_trial(path,decode)
    result.update(audit_code_sha256=code,file_sha256=fingerprints)
    temporary=cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(cache)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--partial',action='store_true');p.add_argument('--skip-decode',action='store_true')
    a=p.parse_args();protocol=json.loads((a.root/'protocol.json').read_text())
    rows=[]
    for candidate in protocol['candidate_ids']:
        path=a.root/f'candidate_{candidate:06d}'
        if not (path/'report.json').exists():
            if a.partial:continue
            raise ValueError(f'Missing trial {candidate}')
        rows.append(audit_trial(path,not a.skip_decode))
        print(json.dumps({k:rows[-1][k] for k in ('candidate_id','success','reason','max_arm_change_rad','gripper_range_rad')}),flush=True)
    result=dict(passed=True,complete=len(rows)==len(protocol['candidate_ids']),requested=len(protocol['candidate_ids']),
        audited=len(rows),successes=sum(x['success'] for x in rows),reasons=dict(Counter(x['reason'] for x in rows)),
        model_sha256=protocol['model_sha256'],all_video_frames_decoded=not a.skip_decode,results=rows)
    name='partial_audit.json' if a.partial else 'audit.json'
    (a.root/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
