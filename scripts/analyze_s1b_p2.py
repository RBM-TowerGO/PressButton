#!/usr/bin/env python3
"""Summarize recorded read-only P2 evidence; no ROS or GPU required."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.contract import sha

def percentiles(values):
    return np.percentile(values,[50,95,99,100]).tolist() if len(values) else []

def analyze(directory):
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    ages=defaultdict(list);stamps=defaultdict(list);receipts=defaultdict(list);invalid=Counter();offsets=[]
    with (directory/'sources.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);key=row['stream']
            offsets.append(row['receipt_ns']-row['wall_ns'])
            if not row['valid']:
                invalid[row['error']]+=1;continue
            ages[key].append(row['source_to_receipt_ns'])
            stamps[key].append(row['header']['sec']*10**9+row['header']['nanosec'])
            receipts[key].append(row['receipt_ns'])
    rows=[];request_invalid=Counter()
    with (directory/'requests.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['valid']:rows.append(row)
            else:request_invalid[row['error']]+=1
    if not rows:raise ValueError('No valid requests')
    source_total=sum(map(len,stamps.values()))+sum(invalid.values())
    if source_total!=sum(summary['callbacks'].values()):raise ValueError('Source log coverage differs from callback counts')
    if len(rows)!=summary['valid_requests']:raise ValueError('Request log coverage differs from summary')
    steps_remaining=[int(np.count_nonzero(r['observation_monotonic_ns']+np.rint(np.arange(1,8)*1e9/30).astype(np.int64)>r['received_ns'])) for r in rows]
    source_stats={}
    for key,values in stamps.items():
        gaps=np.diff(values);receive_gaps=np.diff(receipts[key]);period=1e9/(30 if key in ('head','right') else 250)
        source_stats[key]=dict(count=len(values),observed_span_seconds=(values[-1]-values[0])/1e9,header_strictly_monotonic=bool(np.all(gaps>0)),receipt_strictly_monotonic=bool(np.all(receive_gaps>0)),source_to_receipt_p50_p95_p99_max_ns=percentiles(ages[key]),header_gap_p50_p95_p99_max_ns=percentiles(gaps),receipt_gap_p50_p95_p99_max_ns=percentiles(receive_gaps),estimated_missing_by_nominal_header_period=int(np.maximum(0,np.rint(gaps/period).astype(np.int64)-1).sum()))
    rtt=[r['rtt_ns'] for r in rows]
    return dict(summary=summary,sources=source_stats,invalid_sources=dict(invalid),invalid_requests=dict(request_invalid),valid_request_span_seconds=(rows[-1]['received_ns']-rows[0]['received_ns'])/1e9,first_request_receipt_ns=rows[0]['received_ns'],last_request_receipt_ns=rows[-1]['received_ns'],request_gap_p50_p95_p99_max_ns=percentiles(np.diff([r['received_ns'] for r in rows])),rtt_p50_p95_p99_max_ns=percentiles(rtt),header_source_to_response_p50_p95_p99_max_ns=percentiles([r['source_to_response_ns'] for r in rows]),encoding_and_selection_p50_p95_p99_max_ns=percentiles([r['send_ns']-r['observation_monotonic_ns'] for r in rows]),inference_p50_p95_p99_max_seconds=percentiles([r['inference_seconds'] for r in rows]),preprocessing_p50_p95_p99_max_seconds=percentiles([r['preprocess_seconds'] for r in rows]),robot_wall_monotonic_offset_range_ns=max(offsets)-min(offsets),responses_with_any_future_point_fraction=float(np.mean(np.asarray(steps_remaining)>0)),future_point_count_histogram=dict(Counter(map(str,steps_remaining))),source_log_records=source_total,source_log_matches_callback_counts=True,artifact_sha256={p.name:sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in ('.json','.jsonl','.npz')})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(a.run);path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('valid_request_span_seconds','rtt_p50_p95_p99_max_ns','responses_with_any_future_point_fraction')}))
if __name__=='__main__':main()
