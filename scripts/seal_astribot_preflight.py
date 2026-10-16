#!/usr/bin/env python3
"""Build a production gate from verified artifacts; never accept a manual pass flag."""
import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from pressb.astribot_collection.contract import atomic_json,sha256,validate_commit,CAMERAS
from pressb.astribot_collection.dispatch import source_files
from pressb.astribot_collection.verification import (read,summarize_run,choose_concurrency,
                                                    concurrent_throughput,compare_initial_images)


def historical(directory,full_video):
    directory=Path(directory);result=read(directory/'worker_result.json')
    if result['status']!='finished':raise ValueError('Historical comparison unfinished')
    cases={}
    for row in result['results']:
        match=re.search(r'(?:reference|sample_)(\d{3})',row['id'])
        if not match:continue
        sample=int(match.group(1));report=read(Path(row['output'])/'report.json')
        if sample==50:
            passed=report.get('termination_reason')=='initial_collision' and not report.get('motion_started')
        else:
            passed=row.get('physical_audit',{}).get('status')=='passed'
            if full_video:
                passed=passed and row['status']=='accepted'
                if passed:validate_commit(row['output'])
        if not passed:raise ValueError(f'Historical case {sample} failed in {directory}')
        cases[sample]=dict(output=row['output'],passed=True)
    required={1,2,3,5,10,20,24,35,40,50}
    if not required<=cases.keys():raise ValueError(f'Missing historical cases: {required-cases.keys()}')
    return cases


def junit(path,required_prefixes):
    root=ET.parse(path).getroot();cases=root.findall('.//testcase')
    if root.findall('.//failure') or root.findall('.//error'):raise ValueError(f'Failed tests: {path}')
    passed=[c.attrib['name'] for c in cases if c.find('skipped') is None]
    if not all(any(n.startswith(prefix) for n in passed) for prefix in required_prefixes):
        raise ValueError(f'Missing fault injection evidence: {path}')
    return dict(path=str(Path(path).resolve()),sha256=sha256(path),passed=passed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--single-history',type=Path,required=True)
    p.add_argument('--multi-history',type=Path,required=True)
    p.add_argument('--sweep',type=Path,nargs=4,required=True)
    p.add_argument('--dual',type=Path,nargs=2,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--gpu-probe',type=Path,required=True)
    p.add_argument('--cpu-probe',type=Path,required=True)
    p.add_argument('--reset-run',type=Path,required=True)
    p.add_argument('--brightness-reference',type=Path,required=True)
    p.add_argument('--brightness-episodes',type=Path,nargs='+',required=True)
    p.add_argument('--official-export',type=Path,required=True)
    p.add_argument('--unit-junit',type=Path,required=True)
    p.add_argument('--nvenc-junit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();checks={};evidence_dir=args.output.parent/'gate_evidence'
    def evidence(name,payload):
        file=evidence_dir/(name+'.json');atomic_json(file,dict(status='passed',**payload))
        checks[name]=dict(status='passed',path=str(file.resolve()),sha256=sha256(file))
    frozen=source_files(ROOT)
    # Packaging/journal changes do not change images or physical control. All
    # capture dependencies must match the currently frozen source bytes.
    non_capture={'src/pressb/astribot_collection/dispatch.py','src/pressb/astribot_collection/verification.py',
                 'scripts/export_astribot_lerobot.py','scripts/finalize_astribot_collection.py',
                 'scripts/run_astribot_collection_pipeline.py','scripts/seal_astribot_preflight.py',
                 'scripts/summarize_astribot_benchmarks.py'}
    layouts=set();backends=set()
    for directory in set(args.sweep+args.dual+[args.multi_history,args.reset_run]):
        recorded=read(directory/'run_configuration.json')['source_sha256']
        for name,digest in frozen.items():
            if name not in non_capture and recorded.get(name)!=digest:raise ValueError(f'Stale capture dependency: {directory}: {name}')
        layout=read(directory/'run_configuration.json')['arguments']['render_layout'];layouts.add(layout)
        backends.add(read(directory/'run_configuration.json')['arguments']['backend'])
        if layout not in ('individual','homogeneous','isolated'):
            raise ValueError('Unverified tiled projection cannot be promoted')
    if len(layouts)!=1:raise ValueError('Preflight mixes render layouts')
    if len(backends)!=1:raise ValueError('Preflight mixes physical backends')
    layout=layouts.pop()
    backend=backends.pop()
    evidence('historical_cases',dict(single=historical(args.single_history,False),multi=historical(args.multi_history,True)))
    reset=read(args.reset_run/'reset_probe/reset_audit.json')
    if reset['status']!='passed' or len(reset['rounds'])<20 or not all(r['passed'] for r in reset['rounds']):
        raise ValueError('Twenty-reset acceptance incomplete')
    evidence('twenty_resets',dict(run=str(args.reset_run.resolve()),audit=reset))
    brightness=compare_initial_images(args.brightness_reference,args.brightness_episodes)
    if brightness['status']!='passed':raise ValueError('Environment brightness exceeds 5%')
    evidence('environment_isolation',dict(brightness=brightness,
        collision_evidence=checks['historical_cases'],projection='native CameraParams checked at each reset; each committed camera verified'))
    evidence('fault_injection',dict(unit=junit(args.unit_junit,[
        'test_out_of_order_selection','test_atomic_commit','test_killed_process_recovery',
        'test_schema_maps_names','test_production_gate_rejects']),
        nvenc=junit(args.nvenc_junit,['test_bounded_queue_and_full_decode_reject_truncation'])))
    gpu=read(args.gpu_probe/'shared/backend.json');cpu=read(args.cpu_probe/'shared/backend.json')
    if not (gpu['gpu_dynamics'] and gpu['broadphase']=='GPU' and gpu['tensor_backend']=='torch'):
        raise ValueError('GPU probe did not actually use GPU physics')
    cpu_result=read(args.cpu_probe/'worker_result.json');gpu_result=read(args.gpu_probe/'worker_result.json')
    if cpu['gpu_dynamics'] or not any(r.get('physical_audit',{}).get('status')=='passed' for r in cpu_result['results']):
        raise ValueError('CPU physics correctness was not established')
    gpu_passed=any(r.get('physical_audit',{}).get('status')=='passed' for r in gpu_result['results'])
    if backend=='gpu' and not gpu_passed:raise ValueError('Selected GPU backend failed contact/physical acceptance')
    evidence('backend_comparison',dict(selected=backend,cpu=cpu,gpu=gpu,cpu_result=cpu_result,gpu_result=gpu_result,
        gpu_contact_acceptance=gpu_passed,
        reason='Selected the accepted backend after measured comparison; GPU solver may use CPU contact/state readback.'))
    sweep=[summarize_run(d) for d in args.sweep];selected=choose_concurrency(sweep)
    selected_count=selected['envs_per_gpu']
    for directory in (args.reset_run,args.multi_history):
        if read(directory/'run_configuration.json')['arguments']['envs_per_gpu']!=selected_count:
            raise ValueError('Reset/historical evidence differs from production concurrency')
    if selected_count>1:
        # A small mean-brightness change alone cannot prove that no neighbouring
        # room is visible. Require the explicit per-room visibility mechanism.
        for directory in [args.multi_history,*args.dual]:
            for row in read(directory/'worker_result.json')['results']:
                if row['status']!='accepted':continue
                config=read(Path(row['output'])/'report.json').get('render_configuration',{})
                if config.get('environments')!=selected_count or not config.get('visibility_isolated'):
                    raise ValueError('Multiple visible rooms have not passed camera isolation')
    if read(args.single_history/'run_configuration.json')['arguments']['envs_per_gpu']!=1:
        raise ValueError('Single-environment historical comparison must use one environment')
    offsets={tuple(read(d/'report.json')['environment_offset_m']) for d in args.brightness_episodes}
    if len(offsets)<selected_count:raise ValueError('Brightness comparison does not cover every production environment')
    dual=[summarize_run(d) for d in args.dual]
    if {r['gpu'] for r in dual}!={0,1} or any(r['envs_per_gpu']!=selected['envs_per_gpu'] for r in dual):
        raise ValueError('Dual benchmark does not match selected concurrency on both GPUs')
    speed=concurrent_throughput(dual);baseline=read(args.baseline/'result.json');decode=read(args.baseline/'video_audit.json')
    if decode['status']!='passed' or len(baseline['rows'])!=4 or any(r['exit_code'] for r in baseline['rows']):
        raise ValueError('Old seven-video baseline incomplete')
    for row in baseline['rows']:
        if json.loads(row['audit']['stdout'])['status']!='passed':raise ValueError('Baseline physical audit failed')
    baseline_s=baseline['elapsed_s']+decode['elapsed_s'];ratio=speed['accepted_per_minute']/(240/baseline_s)
    max_vram=max(r['max_vram_fraction'] for r in dual);min_ram=min(r['min_available_ram_gib'] for r in dual)
    if ratio<2 or max_vram>.8 or min_ram<32:raise ValueError(f'Performance gate failed: ratio={ratio:.3f}, VRAM={max_vram:.3f}, RAM={min_ram:.1f} GiB')
    evidence('concurrency_sweep',dict(sweep=sweep,dual=dual,speed=speed,baseline_total_s=baseline_s,throughput_ratio=ratio))
    export=read(args.official_export/'meta/astribot_reader_audit.json')
    encoder=read(args.official_export/'meta/astribot_encoder_config.json')
    if encoder['options'].get('rc')!='1' or encoder['options'].get('cq')!='18' or encoder['options'].get('b')!='0':
        raise ValueError('Official export did not use the frozen NVENC CQ quality profile')
    if encoder['export_script_sha256']!=sha256(ROOT/'scripts/export_astribot_lerobot.py'):
        raise ValueError('Official export used a stale exporter')
    sources=read(args.official_export/'meta/astribot_sources.json');info=read(args.official_export/'meta/info.json')
    if export['status']!='passed' or export['episodes']!=len(sources) or not sources or info['robot_type']!='astribot' or info['fps']!=30:
        raise ValueError('Official export/reader validation incomplete')
    for source in sources:
        validate_commit(source['directory'])
        if source['commit_sha256']!=sha256(Path(source['directory'])/'commit.json'):raise ValueError('Export source commit changed')
    if info['features']['action']['shape']!=[32] or info['features']['observation.state']['shape']!=[32]:
        raise ValueError('Wrong full-joint export schema')
    for name,(w,h) in CAMERAS.items():
        if info['features']['observation.images.'+name]['shape']!=[h,w,3]:raise ValueError('Export camera size changed')
    evidence('official_export',dict(directory=str(args.official_export.resolve()),reader=export,sources=sources))
    atomic_json(args.output,dict(status='passed',source_sha256=frozen,checks=checks,
        configuration=dict(gpus=[0,1],envs_per_gpu=selected['envs_per_gpu'],backend=backend,render_subframes=1,render_layout=layout),
        throughput_ratio=ratio,max_vram_fraction=max_vram,min_available_ram_gib=min_ram,
        measured_accepted_per_minute=speed['accepted_per_minute']))
    print(f'Passed gate saved: {args.output}')
if __name__=='__main__':main()
