#!/usr/bin/env python3
"""Conservative first-attempt accounting for frozen VIB-820 manifests."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np


def ratio(n,d):
    return dict(numerator=n, denominator=d, fraction=n/d if d else None)


def wilson(n, d):
    if not d: return None
    z=1.959963984540054; p=n/d; scale=1+z*z/d
    center=(p+z*z/(2*d))/scale
    half=z*((p*(1-p)/d+z*z/(4*d*d))**.5)/scale
    return [max(0.,center-half),min(1.,center+half)]


def startup_status(report):
    event=report.get('startup')
    if event:
        if (event.get('status')=='pass' and event.get('work_pose_error_rad',1.)<.025
            and event.get('unexpected_contacts_before_completion')==0
            and event.get('completed_step') is not None): return 'pass'
        return 'fail'
    if (report.get('termination_reason') in ('initial_collision','target_ik_failed') or
        report.get('unexpected_contacts') and report.get('termination_phase') in
            ('settle','close_gripper','move_to_work_pose')): return 'fail'
    return 'unknown'



def summarize_records(records):
    n=len(records)
    count=lambda key,val: sum(r.get(key)==val for r in records)
    ikp,ikf=count('ik_status','pass'),count('ik_status','fail')
    attempted=count('motion_started',True)
    confirmed=count('press_confirmed',True)
    collision,clear=count('dynamic_collision','collision'),count('dynamic_collision','clear')
    initial,initial_clear=count('initial_overlap','collision'),count('initial_overlap','clear')
    startup=count('startup_status','pass')
    startup_fail=count('startup_status','fail')
    return dict(N_startup_pass=startup,N_startup_fail=startup_fail,N_startup_unknown=n-startup-startup_fail,
        confidence_intervals_95=dict(startup=wilson(startup,n),full_success=wilson(count('full_success',True),n)),
        N_planned=n,N_ik_pass=ikp,N_ik_fail=ikf,N_ik_unknown=n-ikp-ikf,
        N_attempted=attempted,N_motion_unknown=count('motion_started',None),N_press_confirmed=confirmed,
        N_press_unknown_started=sum(r.get('motion_started') is True and r.get('press_confirmed') is None for r in records),
        N_full_success=count('full_success',True),N_collision=collision,N_clear=clear,N_dynamic_unknown=n-collision-clear,
        N_initial_collision=initial,N_initial_clear=initial_clear,N_initial_unknown=n-initial-initial_clear,
        N_dynamic_unknown_started=sum(r.get('motion_started') is True and r.get('dynamic_collision')=='unknown' for r in records),
        contact_event_entries=sum(r.get('contact_event_entries',0) for r in records),
        outcomes=dict(Counter(r['primary_outcome'] for r in records)),
        failure_flags=dict(Counter(flag for r in records for flag in r.get('failure_flags',[]))),
        metrics=dict(startup=ratio(startup,n),key_pose_reachability=ratio(ikp,ikp+ikf),ik_coverage=ratio(ikp+ikf,n),confirmed_reachable_all=ratio(ikp,n),
            attempted_press=ratio(confirmed,attempted),all_press=ratio(confirmed,n),full_success=ratio(count('full_success',True),n),
            initial_overlap=ratio(initial,initial+initial_clear),initial_coverage=ratio(initial+initial_clear,n),
            dynamic_collision=ratio(collision,collision+clear),dynamic_coverage=ratio(collision+clear,n)))


def classify(run, exit_code=None, timed_out=False):
    run=Path(run)
    report=json.loads((run/'report.json').read_text()) if (run/'report.json').exists() else {}
    key=report.get('key_pose_ik',{})
    ik='fail' if any(v.get('status')=='fail' for v in key.values()) else 'pass' if len(key)>=3 and all(v.get('status')=='pass' for v in key.values()) else 'unknown'
    events=report.get('unexpected_contacts',[])
    dynamic='collision' if events else 'clear' if report.get('dynamic_monitor',{}).get('coverage_complete') else 'unknown'
    initial=report.get('initial_overlap',{}).get('status','unknown')
    result=dict(startup_status=startup_status(report),startup=report.get('startup'),ik_status=ik,initial_overlap=initial,dynamic_collision=dynamic,motion_started=report.get('motion_started'),
        press_phase_started=report.get('press_phase_started',False),press_confirmed=None,full_success=False,
        contact_event_entries=len(events),collision_events=events,exit_code=exit_code,timed_out=timed_out,
        error=report.get('error'),termination_phase=report.get('termination_phase'),physics_steps=report.get('physics_steps'),
        initial_poses=report.get('initial_poses'),key_pose_ik=key,missing_fields=[],failure_flags=[],
        trajectory_path=str(run/'trajectory.npz') if (run/'trajectory.npz').exists() else None)
    audit=dict(status='unavailable',reason='trajectory or required report fields unavailable')
    if (run/'trajectory.npz').exists():
        with np.load(run/'trajectory.npz',allow_pickle=False) as z:
            travel,force=z['travel'][:,0],z['force'][:,0]
            valid=np.isfinite(travel)&np.isfinite(force)
            phase=z['phase']
            press=valid&(z['floor']==0)&(travel>=report['config']['press_threshold'])&(force>report['config']['contact_threshold_n'])
            result.update(press_confirmed=bool(press.any()) if valid.all() else True if press.any() else None,
                          max_travel_m=float(travel.max()),max_force_n=float(force.max()),physical_samples=len(travel),
                          phase_duration_s={str(p):float(np.sum(phase==p)/report['config']['physics_hz']) for p in np.unique(phase)})
        try:
            from audit_astribot_press import audit as independent_audit
            audit=independent_audit(run)
        except Exception as exc:
            audit=dict(status='unavailable',reason=f'{type(exc).__name__}: {exc}')
    result['audit']=audit
    if report.get('camera_paths') and 'head' in report['camera_paths']:
        from PIL import Image
        expected = {'overview':(960,600), 'head':(1280,720),
                    **{n:(640,480) for n in ('panel','front','lobby','wrist_left','wrist_right')}}
        camera_checks = {'all_seven_paths': set(report['camera_paths']) == set(expected)}
        labels = {p.stem.rsplit('_head',1)[0] for p in run.glob('*_head.png')}
        camera_checks['snapshots_present'] = bool(labels)
        if report.get('status') == 'passed':
            camera_checks['all_success_phases'] = {'initial_ready','closed','work_pose','press_0','finished','reset'} <= labels
        for label in labels:
            for name,size in expected.items():
                try:
                    with Image.open(run/f'{label}_{name}.png') as im:
                        im.load()
                        good = im.size == size and any(lo != hi for lo,hi in im.convert('RGB').getextrema())
                    camera_checks[f'{label}_{name}'] = good
                except Exception:
                    camera_checks[f'{label}_{name}'] = False
        result['camera_audit'] = dict(status='passed' if all(camera_checks.values()) else 'failed',checks=camera_checks)
        (run/'camera_snapshot_audit.json').write_text(json.dumps(result['camera_audit'],indent=2)+'\n')
    (run/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n') if run.exists() else None
    reason=report.get('termination_reason')
    if reason in ('initial_collision','target_ik_failed','path_ik_failed','runtime_error'):
        outcome=reason
    elif events:
        outcome='unexpected_contact'
    elif timed_out or not report or report.get('error') and not report.get('episodes'):
        outcome='runtime_error'
    elif audit['status']=='passed' and initial=='clear' and ik=='pass' and dynamic=='clear' and exit_code==0:
        outcome='success'
        result['full_success']=True
    elif report.get('episodes') and not report['episodes'][0].get('pressed'):
        outcome='press_not_triggered'
    elif report.get('episodes') and not report['episodes'][0].get('success'):
        outcome='release_or_return_failed'
    elif report.get('ready_return') and not report['ready_return'].get('passed'):
        outcome='release_or_return_failed'
    elif report.get('reset') and not report['reset'].get('passed'):
        outcome='reset_failed'
    else:
        outcome='audit_failed'
    result['primary_outcome']=outcome
    if result.get('camera_audit',{}).get('status') == 'failed' and result['full_success']:
        outcome = result['primary_outcome'] = 'camera_failed'
        result['full_success'] = False
    if outcome!='success': result['failure_flags'].append(outcome)
    if ik=='fail' and outcome!='target_ik_failed': result['failure_flags'].append('target_ik_failed')
    if dynamic=='collision' and outcome!='unexpected_contact': result['failure_flags'].append('unexpected_contact')
    if audit['status']=='unavailable': result['missing_fields'].append('independent_audit')
    if result['trajectory_path'] is None: result['missing_fields'].append('trajectory: no recorded physical steps or process failure')
    return result


def summarize(directory, prefix='summary'):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    records=[]
    for sample in manifest['samples']:
        p=directory/'episodes'/sample['sample_id']/'attempts'/'0001'/'result.json'
        r=json.loads(p.read_text()) if p.exists() else dict(primary_outcome='pending',ik_status='unknown',initial_overlap='unknown',dynamic_collision='unknown',motion_started=False,press_confirmed=None,full_success=False)
        records.append(dict(sample,**r))
    summary=summarize_records(records)
    summary.update(batch_id=manifest['batch_id'],N_terminal=sum(r['primary_outcome'] not in ('pending','running') for r in records),
                   first_attempt_only=True, retries=sum(max(0,len(list((directory/'episodes'/r['sample_id']/'attempts').glob('*/result.json')))-1) for r in records))
    (directory/(prefix+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    fields=['sample_id','dx_m','dy_m','x_m','y_m','yaw_deg','yaw_rad','startup_status','primary_outcome','ik_status','initial_overlap','motion_started','press_confirmed','full_success','dynamic_collision','contact_event_entries','termination_phase','exit_code','timed_out','physics_steps','max_travel_m','max_force_n']
    with (directory/(prefix+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    from plot_vib820_random_base import plot
    plot(directory/(prefix+'.csv'),
         directory/(('station_results' if prefix=='summary' else prefix+'_stations')+'.png'),
         manifest['batch_id'], manifest=manifest)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.directory),indent=2))
