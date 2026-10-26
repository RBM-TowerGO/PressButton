#!/usr/bin/env python3
"""P3 shadow timing, tick coverage, raw target changes and gate evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pressb.s1b_deployment.contract import sha
from analyze_s1b_p2 import analyze as analyze_p2,percentiles

def analyze(directory,include_sources=True):
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    all_requests=[json.loads(line) for line in (directory/'requests.jsonl').open()]
    requests=[r for r in all_requests if r['valid']]
    accepted=[r for r in requests if r.get('scheduler_accepted')]
    ticks=[json.loads(line) for line in (directory/'ticks.jsonl').open()]
    if not accepted:raise ValueError('No accepted prediction')
    # Exclude only discovery/cold start; qualification includes every tick slot
    # between first accepted prediction and last successful response.
    start=accepted[0]['scheduler_accept_ns'];end=requests[-1]['received_ns']
    eligible=[r for r in ticks if start<=r['tick_ns']<=end]
    if not eligible:raise ValueError('No qualified tick')
    expected_slots=eligible[-1]['tick_ordinal']-eligible[0]['tick_ordinal']+1
    covered=[r for r in eligible if r['covered']]
    skipped=expected_slots-len(eligible)
    wrong=[r for r in covered if r['deadline_ns']<r['simulated_send_ns'] or not 0<=r['deadline_ns']-r['tick_ns']<=33_333_334 or r['sdk_target'] is not None]
    request_by_identity={(r['session_id'],r['observation_id']):r for r in requests}
    if len(request_by_identity)!=len(requests):raise ValueError('Duplicate request identities')
    for row in covered:
        request=request_by_identity[(row['session_id'],row['observation_id'])]
        if row['deadline_ns']!=row['observation_monotonic_ns']+round((row['step_index']+1)*1e9/30):raise ValueError('Deadline rebase')
        if not np.array_equal(np.asarray(row['action_raw'],np.float32),np.asarray(request['action_raw'],np.float32)[row['step_index']]):raise ValueError('Raw action changed')
    raw=np.asarray([r['action_raw'] for r in requests],np.float32)
    state=np.asarray([r['state10'] for r in requests],np.float32)
    actions=np.asarray([r['action_raw'] for r in covered],np.float32)
    binary=actions[:,7]>=.5 if len(actions) else np.asarray([],bool)
    response_valid_fraction=len(accepted)/len(all_requests) if all_requests else 0.
    coverage=len(covered)/expected_slots
    latency_p99=float(np.percentile([r['observation_to_response_ns'] for r in requests],99))
    gates=dict(run_complete=bool(summary.get('complete')) and not summary.get('fatal'),known_record_time_domain=not summary.get('clock_jump',False) and not summary.get('callback_errors',{}).get('unmapped_source_time',0),no_sdk_execution=not summary.get('sdk_imported',False) and not summary.get('execution_enabled',False),response_valid_fraction_ge_99=response_valid_fraction>=.99,queue_coverage_ge_99=coverage>=.99,observation_response_p99_lt_200ms=latency_p99<200_000_000,no_expired_or_modified_send=not wrong,qualified_duration_ge_600=(end-start)>=600e9,no_shadow_thread_error=summary.get('shadow_thread_error') is None,never_executing=all(r['state']!='EXECUTING' for r in ticks))
    result=dict(step=summary['step'],summary=summary,qualified_seconds=(end-start)/1e9,qualified_first_ns=start,qualified_last_ns=end,qualified_tick_slots=expected_slots,logged_qualified_ticks=len(eligible),skipped_qualified_ticks=skipped,covered_ticks=len(covered),queue_coverage=coverage,response_valid_fraction=response_valid_fraction,invalid_simulated_sends=len(wrong),selected_step_1based_histogram=dict(Counter(str(r['step_index']+1) for r in covered)),uncovered_reasons=dict(Counter(r.get('reason','unknown') for r in eligible if not r['covered'])),tick_jitter_p50_p95_p99_max_ns=percentiles([r['tick_ns']-r['planned_tick_ns'] for r in eligible]),response_to_tick_p50_p95_p99_max_ns=percentiles([r['response_to_tick_ns'] for r in covered]),tick_to_deadline_p50_p95_p99_max_ns=percentiles([r['deadline_ns']-r['tick_ns'] for r in covered]),available_window_on_receive_min_ns=min(r['observation_monotonic_ns']+233_333_333-r['received_ns'] for r in requests),available_window_on_receive_p50_p95_p99_max_ns=percentiles([r['observation_monotonic_ns']+233_333_333-r['received_ns'] for r in requests]),capture_proxy_age_p50_p95_p99_max_ns=percentiles([r['observation_monotonic_ns']-r['source_mapped_ns']['head'] for r in requests]),receipt_to_send_p50_p95_p99_max_ns=percentiles([r['send_ns']-r['observation_monotonic_ns'] for r in requests]),network_rtt_p50_p95_p99_max_ns=percentiles([r['rtt_ns'] for r in requests]),server_preprocess_p50_p95_p99_max_seconds=percentiles([r['preprocess_seconds'] for r in requests]),server_inference_p50_p95_p99_max_seconds=percentiles([r['inference_seconds'] for r in requests]),rtt_minus_server_p50_p95_p99_max_ns=percentiles([r['rtt_ns']-1e9*(r['preprocess_seconds']+r['inference_seconds']) for r in requests]),observation_to_response_p50_p95_p99_max_ns=percentiles([r['observation_to_response_ns'] for r in requests]),raw_chunk_first_step_delta_abs_max_per_dimension=np.abs(np.diff(raw[:,0,:],axis=0)).max(0).tolist() if len(raw)>1 else [],simulated_selected_delta_abs_max_per_dimension=np.abs(np.diff(actions,axis=0)).max(0).tolist() if len(actions)>1 else [],predicted_head_minus_observed_abs_p50_p95_p99_max_rad=[percentiles(np.abs(raw[:,0,j]-state[:,j])) for j in (8,9)],simulated_gripper_switches=int(np.count_nonzero(np.diff(binary))),gates=gates,engineering_gate_pass=all(gates.values()),ticks_sha256=sha(directory/'ticks.jsonl'))
    if include_sources:result['p2_analysis']=analyze_p2(directory)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--no-sources',action='store_true');a=p.parse_args()
    result=analyze(a.run,not a.no_sources);path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('step','queue_coverage','response_valid_fraction','qualified_seconds','gates','engineering_gate_pass')}))
if __name__=='__main__':main()
