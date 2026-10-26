#!/usr/bin/env python3
"""Audit real service/tunnel fault injection and fail-closed shadow recovery."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.contract import sha

def pct(values):return np.percentile(values,[50,95,99,100]).tolist() if values else []

def analyze(directory,markers):
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text())
    ticks=[json.loads(line) for line in (directory/'ticks.jsonl').open()]
    events=[json.loads(line) for line in (directory/'events.jsonl').open()]
    markers=json.loads(Path(markers).read_text());marks={r['event']:r['monotonic_ns'] for r in markers}
    reconnects=[r['session_id'] for r in events if r['event']=='reconnected_shadow']
    segments=[];last_covered_session=None;active=None;violations=[]
    for row in ticks:
        if row['state']=='EXECUTING':violations.append('unexpected_EXECUTING')
        if row['state']=='FAULT':
            if active is None:active=dict(first_fault_tick_ns=row['tick_ns'],last_fault_tick_ns=row['tick_ns'],previous_session_id=last_covered_session,ticks=0)
            active['last_fault_tick_ns']=row['tick_ns'];active['ticks']+=1
            if row['covered']:violations.append('covered_in_FAULT')
        elif active is not None:
            active['recovery_state']=row['state'];segments.append(active);active=None
        if row['covered']:
            if row['deadline_ns']<row['tick_ns'] or row['sdk_target'] is not None:violations.append('expired_or_real_send')
            if segments and 'next_covered_session_id' not in segments[-1]:segments[-1]['next_covered_session_id']=row['session_id']
            last_covered_session=row['session_id']
    if active is not None:active['recovery_state']=None;segments.append(active)
    windows={}
    for name,begin,end in [('server_pause','server_pause_begin','server_pause_end'),('tunnel_interrupt','tunnel_interrupt_begin','tunnel_interrupt_end'),('server_crash_model_change','server_crash_and_model_switch_begin',None)]:
        start=marks[begin];stop=marks[end] if end else ticks[-1]['tick_ns'];rows=[r for r in ticks if start<=r['tick_ns']<=stop]
        faults=[r for r in ticks if r['tick_ns']>=start and r['state']=='FAULT']
        empty=[r for r in rows if not r['covered']]
        windows[name]=dict(duration_ns=stop-start,ticks=len(rows),covered_ticks=sum(r['covered'] for r in rows),inactive_or_empty_ticks=sum(not r['covered'] for r in rows),first_fault_after_trigger_ns=faults[0]['tick_ns']-start if faults else None,first_would_stop_after_trigger_ns=empty[0]['tick_ns']-start if empty else None,max_tick_gap_ns=max(np.diff([r['tick_ns'] for r in rows]),default=0).item() if len(rows)>1 else 0)
    checks=dict(three_fault_segments=len(segments)>=3,all_recovery_to_shadow=all(s['recovery_state'] in (None,'SHADOW') for s in segments),fresh_session_after_recovery=all(s.get('next_covered_session_id')!=s['previous_session_id'] for s in segments if s.get('next_covered_session_id')),new_session_ids_unique=len(set(reconnects))==len(reconnects) and len(reconnects)>=2,model_change_rejected=any('health_changed_requires_new_session' in s for s in summary['fatal']),no_remaining_queue=summary['scheduler']['queue_length']==0,normal_execution_never_enabled=not summary['execution_enabled'] and not summary['sdk_imported'],ticks_continued_during_each_fault=all(w['ticks']>=2 for w in windows.values()),no_consume_after_fault_or_expiry=not violations,no_shadow_thread_failure=summary['shadow_thread_error'] is None)
    return dict(checks=checks,passed=all(checks.values()),summary=summary,markers=markers,windows=windows,fault_segments=segments,recovered_session_ids=reconnects,violations=violations,tick_gap_p50_p95_p99_max_ns=pct(np.diff([r['tick_ns'] for r in ticks]).tolist()),tick_jitter_p50_p95_p99_max_ns=pct([r['tick_ns']-r['planned_tick_ns'] for r in ticks]),artifact_sha256={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.npz')})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--markers',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.run,a.markers);Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(passed=result['passed'],checks=result['checks'],windows=result['windows'])))
if __name__=='__main__':main()
