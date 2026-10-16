#!/usr/bin/env python3
"""Summarize completed seven-video benchmarks without promoting incomplete trials."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pressb.astribot_collection.contract import atomic_json
from pressb.astribot_collection.verification import summarize_run,choose_concurrency,concurrent_throughput

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep',type=Path,nargs='+',required=True)
    p.add_argument('--dual',type=Path,nargs=2)
    p.add_argument('--baseline-wall-seconds',type=float,required=True)
    p.add_argument('--baseline-successes',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    summaries=[summarize_run(d) for d in args.sweep];selected=choose_concurrency(summaries)
    report=dict(status='preflight_incomplete',sweep=summaries,selected_envs_per_gpu=selected['envs_per_gpu'],
        baseline=dict(successes=args.baseline_successes,wall_s=args.baseline_wall_seconds,
                      accepted_per_minute=60*args.baseline_successes/args.baseline_wall_seconds))
    if args.dual:
        dual=[summarize_run(d) for d in args.dual]
        if any(r['envs_per_gpu']!=selected['envs_per_gpu'] for r in dual):raise ValueError('Dual run differs from selected concurrency')
        report['dual_runs']=dual;report['dual']=concurrent_throughput(dual)
        report['throughput_ratio']=report['dual']['accepted_per_minute']/report['baseline']['accepted_per_minute']
        report['performance_gate_passed']=report['throughput_ratio']>=2 and all(
            r['max_vram_fraction']<=.8 and r['min_available_ram_gib']>=32 for r in dual)
        report['estimated_400_collection_minutes']=400/report['dual']['accepted_per_minute']
        report['estimate_scope']='Measured collection plus full raw acceptance; excludes development, export and final replay.'
    atomic_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sweep','dual_runs')},indent=2))
if __name__=='__main__':main()
