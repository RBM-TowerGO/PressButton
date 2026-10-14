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


def summarize_records(records):
    n=len(records)
    count=lambda key,val: sum(r.get(key)==val for r in records)
    ikp,ikf=count('ik_status','pass'),count('ik_status','fail')
    attempted=count('motion_started',True)
    confirmed=count('press_confirmed',True)
    collision,clear=count('dynamic_collision','collision'),count('dynamic_collision','clear')
    initial,initial_clear=count('initial_overlap','collision'),count('initial_overlap','clear')
    return dict(N_planned=n,N_ik_pass=ikp,N_ik_fail=ikf,N_ik_unknown=n-ikp-ikf,
        N_attempted=attempted,N_motion_unknown=count('motion_started',None),N_press_confirmed=confirmed,
        N_press_unknown_started=sum(r.get('motion_started') is True and r.get('press_confirmed') is None for r in records),
        N_full_success=count('full_success',True),N_collision=collision,N_clear=clear,N_dynamic_unknown=n-collision-clear,
        N_initial_collision=initial,N_initial_clear=initial_clear,N_initial_unknown=n-initial-initial_clear,
        N_dynamic_unknown_started=sum(r.get('motion_started') is True and r.get('dynamic_collision')=='unknown' for r in records),
        contact_event_entries=sum(r.get('contact_event_entries',0) for r in records),
        outcomes=dict(Counter(r['primary_outcome'] for r in records)),
        failure_flags=dict(Counter(flag for r in records for flag in r.get('failure_flags',[]))),
        metrics=dict(key_pose_reachability=ratio(ikp,ikp+ikf),ik_coverage=ratio(ikp+ikf,n),confirmed_reachable_all=ratio(ikp,n),
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
    result=dict(ik_status=ik,initial_overlap=initial,dynamic_collision=dynamic,motion_started=report.get('motion_started'),
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
    elif report.get('reset') and not report['reset'].get('passed'):
        outcome='reset_failed'
    else:
        outcome='audit_failed'
    result['primary_outcome']=outcome
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
    fields=['sample_id','dx_m','dy_m','x_m','y_m','primary_outcome','ik_status','initial_overlap','motion_started','press_confirmed','full_success','dynamic_collision','contact_event_entries','termination_phase','exit_code','timed_out','physics_steps','max_travel_m','max_force_n']
    with (directory/(prefix+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7))
    for outcome in sorted(set(r['primary_outcome'] for r in records)):
        rows=[r for r in records if r['primary_outcome']==outcome]
        ax.scatter([r['x_m'] for r in rows],[r['y_m'] for r in rows],label=f'{outcome} ({len(rows)})',s=65)
    for r in records: ax.annotate(r['sample_id'][-2:],(r['x_m'],r['y_m']),fontsize=7,xytext=(4,4),textcoords='offset points')
    ax.scatter(-.12,.28,marker='*',s=180,color='black',label='nominal smoke (excluded)')
    ax.add_patch(plt.Rectangle((-.32,.08),.4,.4,fill=False,linestyle='--',color='black'))
    ax.set(xlabel='World X (m, +X toward panel)',ylabel='World Y (m)',title=manifest['batch_id'],xlim=(-.35,.11),ylim=(.05,.51))
    ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend(loc='center left',bbox_to_anchor=(1, .5));fig.tight_layout()
    fig.savefig(directory/(('station_results' if prefix=='summary' else prefix+'_stations')+'.png'),dpi=180);plt.close(fig)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.directory),indent=2))
