#!/usr/bin/env python3
"""Recompute batch identity and outcome evidence without using the classifier."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def audit(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    csv_rows=list(csv.DictReader((directory/'summary.csv').open()))
    nominal=manifest['nominal_xy_m'];bounds=manifest.get('delta_bounds_xy_m',[manifest.get('delta_bounds_m',[-.2,.2])]*2)
    rng=np.random.Generator(np.random.PCG64(manifest['seed']))
    raw=rng.uniform(
        [b[0] for b in bounds],[b[1] for b in bounds],(manifest['N_planned'],2))
    checks={};details=[];counts=dict(startup=0,press=0,full=0,collision=0,initial_collision=0,ik_pass=0)
    checks['sampling_reconstructs_exactly']=bool(np.array_equal(raw,[[s['dx_m'],s['dy_m']] for s in manifest['samples']]))
    if 'yaw_bounds_deg' in manifest:
        angles=rng.uniform(*manifest['yaw_bounds_deg'],size=manifest['N_planned'])
        checks['yaw_sampling_reconstructs_exactly']=bool(np.array_equal(angles,[s['yaw_deg'] for s in manifest['samples']]) and
            np.array_equal(np.radians(angles),[s['yaw_rad'] for s in manifest['samples']]))
    checks['unique_csv_matches_manifest']=len(csv_rows)==manifest['N_planned'] and {r['sample_id'] for r in csv_rows}=={s['sample_id'] for s in manifest['samples']}
    checks['all_source_snapshots_match_hashes']=all(hashlib.sha256((directory/'sources'/rel).read_bytes()).hexdigest()==value for rel,value in manifest['source_sha256'].items())
    checks['live_execution_sources_match_snapshot']=all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==value for rel,value in manifest['source_sha256'].items() if rel.startswith(('scripts/','src/','configs/','assets/','vendor/')) and rel!='scripts/audit_vib820_random_batch.py')
    checks['manifest_precedes_attempts']=True
    for s in manifest['samples']:
        home=directory/'episodes'/s['sample_id'];attempt=home/'attempts/0001';run=attempt/'run'
        result=json.loads((attempt/'result.json').read_text());report=json.loads((run/'report.json').read_text())
        config=json.loads((home/'config.json').read_text());started=json.loads((attempt/'started.json').read_text())
        check={}
        checks['manifest_precedes_attempts'] &= manifest['created_utc']<=started['started_utc']
        expected_config=dict(manifest['config'],robot_base_x=s['x_m'],robot_base_y=s['y_m'])
        if 'yaw_bounds_deg' in manifest:expected_config['robot_base_yaw_rad']=s['yaw_rad']
        check['effective_config']=config==expected_config and config==report['config']
        check['initial_joint_targets']=all(abs(report['initial_joint_positions_rad'][name]-value)<1e-6 for name,value in config.get('initial_arm_positions',{}).items())
        if 'head_pitch_deg' in config:
            check['head_pitch_target']=bool(abs(report['initial_joint_positions_rad']['astribot_head_joint_2']-np.radians(config['head_pitch_deg']))<1e-6)
        if manifest.get('all_cameras'):
            expected_cameras={'head','wrist_left','wrist_right','overview','panel','front','lobby'}
            check['all_seven_cameras']=set(report.get('camera_paths',{}))==expected_cameras and '--all-cameras' in started['command']
            from PIL import Image
            snapshots=list(run.glob('*_head.png'))
            camera_ok=bool(snapshots)
            for snapshot in snapshots:
                label=snapshot.stem[:-5]
                for name in expected_cameras:
                    path=run/f'{label}_{name}.png'
                    try:
                        with Image.open(path) as im:
                            im.load()
                            size=(1280,720) if name=='head' else (960,600) if name=='overview' else (640,480)
                            camera_ok &= im.size==size and any(lo!=hi for lo,hi in im.convert('RGB').getextrema())
                    except Exception:camera_ok=False
            check['valid_camera_snapshots']=camera_ok
        row=next(r for r in csv_rows if r['sample_id']==s['sample_id'])
        check['csv_matches_result']=all(row[key]==str(result[key]) for key in ('startup_status','primary_outcome','full_success','ik_status','initial_overlap','dynamic_collision'))
        check['identity']=result['sample_id']==s['sample_id'] and result['attempt_id']=='0001' and result['batch_id']==manifest['batch_id']
        check['manifest_hash']=result['manifest_sha256']==hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()
        check['config_hash']=result['config_sha256']==hashlib.sha256((home/'config.json').read_bytes()).hexdigest()
        pose=report['initial_poses']
        check['base_and_anchor']=bool(np.allclose(pose['actual_root_xyz_m'][:2],[s['x_m'],s['y_m']],atol=1e-6) and
            np.allclose(pose['fixed_joint_world_anchor_m'],pose['requested_root_xyz_m'],atol=1e-6) and abs(pose['requested_yaw_rad']-s['yaw_rad'])<1e-7)
        if 'yaw_bounds_deg' in manifest:
            expected_quat=np.array([np.cos(s['yaw_rad']/2),0,0,np.sin(s['yaw_rad']/2)])
            check['actual_yaw_and_anchor']=bool(abs(pose['actual_root_yaw_rad']-s['yaw_rad'])<1e-6 and
                abs(np.dot(pose['actual_base_link_quat_wxyz'],expected_quat))>1.-1e-6 and
                abs(np.dot(pose['fixed_joint_world_quat_wxyz'],expected_quat))>1.-1e-6)
        start_event=report.get('startup');startup=False;press=False
        if (run/'trajectory.npz').exists():
            with np.load(run/'trajectory.npz',allow_pickle=False) as z:
                n=len(z['time']);names=report['joint_names'];right=[i for i,name in enumerate(names) if 'arm_right' in name]
                arms=[i for i,name in enumerate(names) if 'arm_left' in name or 'arm_right' in name]
                expected=np.array([config.get('initial_arm_positions',{}).get(name,0.) for name in names])
                check['initial_ready_angles']=bool(np.max(np.abs(z['q'][0,arms]-expected[arms]))<.02)
                check['physical_timing']=bool(n==report['physics_steps'] and np.allclose(np.diff(z['time']),1/config['physics_hz'],atol=1e-10))
                check['simulator_timing']=bool(all(np.allclose(np.diff(z['simulation_time'][z['epoch']==e]),1/config['physics_hz'],atol=1e-7) for e in np.unique(z['epoch'])))
                if 'yaw_bounds_deg' in manifest:
                    check['yaw_held_in_physics']=bool(np.min(np.abs(z['base_quat']@expected_quat))>1.-1e-6)
                if 'head_pitch_deg' in config:
                    index=names.index('astribot_head_joint_2')
                    check['head_pitch_held']=bool(np.max(np.abs(z['q'][:,index]-np.radians(config['head_pitch_deg'])))<.02)
                if start_event:
                    i=start_event['completed_step'];error=float(np.max(np.abs(z['q'][i,right]-report['home_arm_rad'])))
                    startup=bool(z['phase'][i]=='move_to_work_pose' and error<.025 and
                        not any(e['step']<=i for e in report.get('unexpected_contacts',[])) and
                        report['initial_overlap']['status']=='clear' and all(v['status']=='pass' for v in report['key_pose_ik'].values()))
                    check['startup_error_matches_arrays']=abs(error-start_event['work_pose_error_rad'])<1e-7
                press=bool(((z['floor']==0)&(z['travel'][:,0]>=config['press_threshold'])&(z['force'][:,0]>config['contact_threshold_n'])).any())
                check['press_matches_arrays']=press==result['press_confirmed']
        else:
            check['early_failure_has_explicit_reason']=result['primary_outcome'] in ('initial_collision','target_ik_failed','runtime_error')
        check['startup_matches_arrays']=startup==(result.get('startup_status')=='pass')
        collision=bool(report.get('unexpected_contacts'));initial_collision=report['initial_overlap']['status']=='collision'
        full=result['full_success']
        if full:
            independent=json.loads((run/'independent_audit.json').read_text())
            check['full_success_audit']=independent['status']=='passed' and all(independent['checks'].values()) and startup and press and not collision and not initial_collision
            check['ready_return_checked']=all(independent['checks'].get(k) for k in ('initial_arm_posture','returned_to_initial_arm_posture','reset_preserves_initial_arm_posture','startup_work_pose'))
        for key,value in [('startup',startup),('press',press),('full',full),('collision',collision),('initial_collision',initial_collision),('ik_pass',all(v['status']=='pass' for v in report['key_pose_ik'].values()))]:counts[key]+=int(value)
        details.append(dict(sample_id=s['sample_id'],passed=all(check.values()),checks=check))
    checks['all_episode_evidence']=all(d['passed'] for d in details)
    checks['all_counts_match']=all(counts[k]==summary[field] for k,field in [('startup','N_startup_pass'),('press','N_press_confirmed'),('full','N_full_success'),('collision','N_collision'),('initial_collision','N_initial_collision'),('ik_pass','N_ik_pass')])
    checks['all_terminal']=summary['N_terminal']==manifest['N_planned']==sum(summary['outcomes'].values())
    if manifest['N_planned']>=20:
        middle=json.loads((directory/'checkpoints/20/interim_20.json').read_text())
        checkpoint=json.loads((directory/'checkpoints/20/checkpoint.json').read_text())
        checks['interim_at_twentieth_terminal']=middle['N_terminal']==20 and checkpoint['N_terminal']==20 and all(hashlib.sha256((directory/'checkpoints/20'/name).read_bytes()).hexdigest()==value for name,value in checkpoint['sha256'].items())
    output=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,recomputed_counts=counts,episodes=details,
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory/'independent_batch_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    result=audit(args.directory);print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
    raise SystemExit(result['status']!='passed')
