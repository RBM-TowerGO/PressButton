#!/usr/bin/env python3
"""Read-only reviewed-data and official-converter probe audit; writes JSON evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = Path(__file__).resolve().parents[1]

def summary(a):
    a = np.asarray(a)
    return dict(zip(['min', 'p50', 'p95', 'max'], np.percentile(a, [0, 50, 95, 100]).tolist()))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/astribot_real_conversion_audit_20261025/report.json')
    args = parser.parse_args()
    data = ROOT/'data/astribot_reviewed_v1'
    rows, commands, states, gaps, mismatches = [], [], [], [], []
    intermediate_runs = []
    for record in json.loads((data/'episode_mapping.json').read_text()):
        p = data/'episodes'/record['output_file']
        if hashlib.sha256(p.read_bytes()).hexdigest() != record['output_sha256']:
            mismatches.append(p.name)
        with h5py.File(p) as f:
            c = f['joints_dict/joints_position_command'][:,22]
            s = f['joints_dict/joints_position_state'][:,22]
            t = f['time'][:]
            mask = (c != 0) & (c != 100)
            edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            for start, end in zip(edges[::2], edges[1::2]):
                segment = c[max(0,start-1):min(len(c),end+1)]
                intermediate_runs.append(dict(file=p.name,start=int(start),end_exclusive=int(end),
                    frames=int(end-start),duration_seconds=float(t[end-1]-t[start]),
                    previous=float(c[start-1]) if start else None,
                    following=float(c[end]) if end<len(c) else None,
                    monotonic=bool(np.all(np.diff(segment)>=0) or np.all(np.diff(segment)<=0))))
            values, counts = np.unique(c, return_counts=True)
            rows.append(dict(file=p.name, original_id=record['original_id'], frames=len(t),
                             command_values=values.tolist(), counts=counts.tolist(),
                             binary_switches=int(np.count_nonzero(np.diff(c >= 50))),
                             non_endpoint_targets=int(np.count_nonzero((c != 0) & (c != 100))),
                             state=summary(s)))
            commands.extend(c.tolist()); states.extend(s.tolist()); gaps.extend(np.diff(t)*1000)
    raw_dir = ROOT/'data/astribot_s1b_20261025/storage/sequential_archive/elevator1/mcap/elevator1_2026_10_25-17_32_50_2'
    raw = {}
    for p in sorted(raw_dir.glob('*.mcap')):
        with p.open('rb') as stream:
            for _, channel, message, decoded in make_reader(stream, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
                header = getattr(decoded, 'header', None)
                stamp = header.stamp.sec + header.stamp.nanosec/1e9 if header else None
                row = dict(log=message.log_time/1e9, publish=message.publish_time/1e9, header=stamp)
                if channel.topic.endswith('joint_space_command'):
                    row['value'] = list(decoded.command)
                elif channel.topic.endswith('joint_space_states'):
                    row['value'] = list(decoded.position)
                raw.setdefault(channel.topic, []).append(row)
    comparisons = []
    probe = ROOT/'outputs/astribot_real_head_audit_20261025/evidence/probe.hdf5'
    with h5py.File(probe) as f:
        fields = ['time', 'joints_dict/command_timestamp', 'joints_dict/state_timestamp',
                  'images_dict/head/rgb_timestamp', 'images_dict/right/rgb_timestamp']
        for field in fields:
            target = f[field][:]
            candidates = []
            for topic, messages in raw.items():
                for clock in ['log', 'publish', 'header']:
                    times = np.array([m[clock] for m in messages if m[clock] is not None])
                    if not len(times): continue
                    error = np.min(np.abs(target[:,None]-times[None,:]), axis=1)*1000
                    candidates.append(dict(topic=topic, clock=clock, max_error_ms=float(error.max()),
                                           median_error_ms=float(np.median(error)), exact_rows=int(np.sum(error==0))))
            comparisons.append(dict(field=field, frames=len(target), best_candidates=sorted(candidates,key=lambda r:r['max_error_ms'])[:8]))
        matches = []
        for part, start, n in [('astribot_arm_right',15,7),('astribot_gripper_right',22,1),('astribot_head',23,2)]:
            topic = '/'+part+'/joint_space_command'
            messages = raw[topic]
            times = np.array([m['log'] for m in messages])
            values = np.array([m['value'][:n] for m in messages])
            targets = f['joints_dict/joints_position_command'][:,start:start+n]
            master = f['time'][:]
            ix = np.argmin(np.abs(master[:,None]-times[None,:]),axis=1)
            err = np.max(np.abs(values[ix]-targets),axis=1)
            matches.append(dict(topic=topic, nearest_log_to_head_exact_rows=int(np.sum(err==0)), frames=len(master),
                                max_value_error=float(err.max()), nearest_log_minus_head_ms=summary((times[ix]-master)*1000)))
    values, counts = np.unique(commands, return_counts=True)
    report = dict(episodes=len(rows),frames=len(commands),sha256_mismatches=mismatches,
                  command_values=values.tolist(),command_counts=counts.tolist(),state=summary(states),
                  intermediate_runs=intermediate_runs,
                  total_binary_switches=sum(r['binary_switches'] for r in rows),
                  non_endpoint_targets=sum(r['non_endpoint_targets'] for r in rows),
                  master_interval_ms=summary(gaps),rows=rows,probe_timestamp_comparisons=comparisons,
                  probe_command_selection=matches,
                  limitations=['Probe is not one of the reviewed 96 episodes.',
                               'Nearest timestamp/value matches alone do not prove converter algorithm or physical execution time.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','command_values','command_counts','intermediate_runs','probe_timestamp_comparisons']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
