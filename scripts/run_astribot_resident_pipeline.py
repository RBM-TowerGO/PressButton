#!/usr/bin/env python3
"""Authorized resident single-scene process sweep, gated collection and release."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from run_astribot_collection_pipeline import Pipeline, read
from seal_astribot_preflight import historical, junit
from pressb.astribot_collection.contract import atomic_json, sha256, validate_commit, CAMERAS
from pressb.astribot_collection.dispatch import latest_resource, resource_violation, source_files
from pressb.astribot_collection.verification import (
    summarize_run, resident_throughput, choose_resident_concurrency, compare_initial_images)


NON_CAPTURE = {'src/pressb/astribot_collection/dispatch.py',
    'src/pressb/astribot_collection/verification.py', 'scripts/run_astribot_resident_pipeline.py',
    'scripts/run_astribot_collection_pipeline.py', 'scripts/seal_astribot_preflight.py',
    'scripts/summarize_astribot_benchmarks.py', 'scripts/export_astribot_lerobot.py',
    'scripts/finalize_astribot_collection.py'}


def check_capture_sources(directory, frozen):
    recorded = read(Path(directory)/'run_configuration.json')['source_sha256']
    for name,digest in frozen.items():
        if name not in NON_CAPTURE and recorded.get(name) != digest:
            raise ValueError(f'Stale capture dependency: {directory}: {name}')


def select_backend(cpu, gpu):
    historical(cpu, False)
    rejection = None
    try: historical(gpu, False)
    except ValueError as exc: rejection = str(exc)
    cpu_s = read(cpu/'worker_result.json')['timings']['physics_s']
    gpu_s = read(gpu/'worker_result.json')['timings']['physics_s']
    return dict(selected='gpu' if rejection is None and gpu_s < cpu_s else 'cpu',
                cpu_physics_s=cpu_s,gpu_physics_s=gpu_s,gpu_historical_acceptance=rejection is None,
                gpu_rejection=rejection,
                scope='same ten historical cases; physical stepping, not video throughput')


class ResidentPipeline(Pipeline):
    def group(self, name, count, backend, *, resets=0, gpus=(0, 1)):
        calls = []
        paths = []
        for gpu in gpus:
            for slot in range(count):
                worker_name = f'{name}_gpu{gpu}_slot{slot}'
                jobs = self.jobs([24, 24, 24], 1, worker_name)
                paths.append(self.output/worker_name)
                calls.append(lambda n=worker_name, j=jobs, g=gpu:
                             self.worker(n, j, g, 1, backend, resets=resets))
        violation = None
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures = [pool.submit(call) for call in calls]
            while not all(f.done() for f in futures):
                for path in paths:
                    sample = latest_resource(path/'resources.jsonl')
                    reason = resource_violation(sample, list(gpus)) if sample else None
                    if reason and violation is None:
                        violation = dict(reason=reason, sample=sample, unix_s=time.time())
                        atomic_json(self.output/(name+'_resource_stop.json'), violation)
                if violation or any(f.done() and f.exception() for f in futures):
                    for path in paths:
                        if path.exists(): (path/'STOP').touch()
                time.sleep(1)
            # Inspect every worker; do not silently drop failed/incomplete evidence.
            errors = [str(f.exception()) for f in futures if f.exception()]
            if errors: raise RuntimeError(f'Resident worker failure: {errors}')
        if violation:
            return dict(status='resource_rejected', processes_per_gpu=count,
                        gpus=list(gpus), violation=violation, directories=list(map(str, paths)))
        summaries = [summarize_run(path) for path in paths]
        return dict(status='passed', **resident_throughput(summaries, count, list(gpus)))

    def seal(self, sweeps, dual, backend, cpu, gpu, history, export):
        checks = {}
        evidence_dir = self.output/'gate_evidence'
        def evidence(name, payload):
            file = evidence_dir/(name+'.json')
            atomic_json(file, dict(status='passed', **payload))
            checks[name] = dict(status='passed', path=str(file.resolve()), sha256=sha256(file))

        count = dual['processes_per_gpu']
        frozen = source_files(ROOT)
        workers = [Path(r['directory']) for group in sweeps+[dual]
                   for r in group.get('workers', [])]
        for path in workers+[cpu, gpu, history]:
            cfg = read(path/'run_configuration.json')
            check_capture_sources(path, frozen)
            if cfg['arguments']['envs_per_gpu'] != 1: raise ValueError('Worker has multiple scenes')
        cases_cpu = historical(cpu, False)
        try: cases_gpu = dict(status='passed',cases=historical(gpu, False))
        except ValueError as exc:
            cases_gpu = dict(status='failed',error=str(exc),result=read(gpu/'worker_result.json'))
            if backend == 'gpu': raise ValueError('Selected GPU backend failed historical acceptance')
        cases_video = historical(history, True)
        evidence('historical_cases', dict(cpu=cases_cpu, gpu=cases_gpu, full_video=cases_video))

        resets = []
        brightness = []
        for row in dual['workers']:
            path = Path(row['directory'])
            cfg = read(path/'run_configuration.json')['arguments']
            if cfg['backend'] != backend or cfg['render_layout'] != self.args.render_layout:
                raise ValueError('Dual worker differs from frozen capture configuration')
            reset = read(path/'reset_probe/reset_audit.json')
            if reset['status'] != 'passed' or len(reset['rounds']) < 20 or not all(r['passed'] for r in reset['rounds']):
                raise ValueError(f'Twenty resets failed: {path}')
            resets.append(dict(worker=str(path), audit=reset))
            rows = read(path/'worker_result.json')['results']
            for result in rows:
                if result['status'] != 'accepted': raise ValueError('Dual episode failed full video acceptance')
                validate_commit(result['output'])
                config = read(Path(result['output'])/'report.json')['render_configuration']
                if config['environments'] != 1 or not config['visibility_isolated']:
                    raise ValueError('Resident process did not isolate its scene')
                if 'wave0_' in result['id']: brightness.append(Path(result['output']))
        evidence('twenty_resets', dict(workers=resets))
        reference = Path(read(history/'worker_result.json')['results'][0]['output'])
        comparison = compare_initial_images(reference, brightness)
        if comparison['status'] != 'passed': raise ValueError('Resident brightness comparison failed')
        evidence('environment_isolation', dict(brightness=comparison,
                 architecture='one complete scene and seven persistent cameras per OS process',
                 full_frame_audits=True, native_projection_verified_at_each_reset=True))
        legacy = ROOT/'outputs/vib820_gpu_collection_400_preflight'
        evidence('fault_injection', dict(unit=junit(self.output/'unit_fault_checks.xml', [
            'test_out_of_order_selection', 'test_atomic_commit', 'test_killed_process_recovery',
            'test_schema_maps_names', 'test_production_gate_rejects', 'test_resident']),
            nvenc=junit(self.output/'nvenc_fault_checks.xml', ['test_bounded_queue_and_full_decode_reject_truncation'])))
        cpu_actual = read(cpu/'shared/backend.json')
        gpu_actual = read(gpu/'shared/backend.json')
        if cpu_actual['gpu_dynamics'] or not (gpu_actual['gpu_dynamics'] and
                gpu_actual['broadphase'] == 'GPU' and gpu_actual['tensor_backend'] == 'torch'):
            raise ValueError('Actual physics backend differs from CPU/GPU comparison')
        evidence('backend_comparison', dict(selected=backend, cpu=cpu_actual, gpu=gpu_actual,
                 cpu_cases=cases_cpu, gpu_cases=cases_gpu, measurement=read(self.output/'backend_choice.json')))

        choice = choose_resident_concurrency(sweeps)
        if choice['processes_per_gpu'] != count: raise ValueError('Dual concurrency differs from selected sweep')
        baseline = read(legacy/'baseline/result.json')
        decoded = read(legacy/'baseline/video_audit.json')
        if decoded['status'] != 'passed' or len(baseline['rows']) != 4 or any(r['exit_code'] for r in baseline['rows']):
            raise ValueError('Baseline seven-video evidence incomplete')
        if any(json.loads(r['audit']['stdout'])['status'] != 'passed' for r in baseline['rows']):
            raise ValueError('Baseline physical audit failed')
        baseline_s = baseline['elapsed_s'] + decoded['elapsed_s']
        ratio = dual['accepted_per_minute'] / (240/baseline_s)
        if ratio < 2 or dual['max_vram_fraction'] > .8 or dual['min_available_ram_gib'] < 32:
            raise ValueError(f'Production performance gate failed: speedup={ratio:.3f}')
        evidence('concurrency_sweep', dict(sweep=sweeps, dual=dual, baseline_total_s=baseline_s,
                 throughput_ratio=ratio, requested_process_counts=[1, 2, 4]))

        reader = read(export/'meta/astribot_reader_audit.json')
        encoder = read(export/'meta/astribot_encoder_config.json')
        sources = read(export/'meta/astribot_sources.json')
        info = read(export/'meta/info.json')
        if encoder['export_script_sha256'] != sha256(ROOT/'scripts/export_astribot_lerobot.py'):
            raise ValueError('Stale official exporter')
        if any(encoder['options'].get(k) != v for k,v in dict(rc='1',cq='18',b='0').items()):
            raise ValueError('Official NVENC quality configuration differs')
        if reader['status'] != 'passed' or reader['episodes'] != len(sources) or not sources:
            raise ValueError('Official reader evidence incomplete')
        if info['robot_type'] != 'astribot' or info['fps'] != 30:
            raise ValueError('Wrong official robot/frequency')
        if any(info['features'][name]['shape'] != [32] for name in ('action','observation.state')):
            raise ValueError('Wrong joint schema')
        for name,(w,h) in CAMERAS.items():
            if info['features']['observation.images.'+name]['shape'] != [h,w,3]:
                raise ValueError('Wrong camera resolution')
        for source in sources:
            validate_commit(source['directory'])
            if source['commit_sha256'] != sha256(Path(source['directory'])/'commit.json'):
                raise ValueError('Pilot source modified')
        evidence('official_export', dict(directory=str(export), reader=reader, sources=sources))
        gate = self.output/'production_gate.json'
        atomic_json(gate, dict(status='passed', source_sha256=frozen, checks=checks,
            configuration=dict(gpus=[0,1],envs_per_gpu=1,processes_per_gpu=count,
                               backend=backend,render_subframes=1,render_layout=self.args.render_layout),
            throughput_ratio=ratio,max_vram_fraction=dual['max_vram_fraction'],
            min_available_ram_gib=dual['min_available_ram_gib'],
            measured_accepted_per_minute=dual['accepted_per_minute']))
        return gate

    def run(self):
        ids = [24,1,2,3,5,10,20,35,40,50]
        self.stage('resident_cpu_gpu_historical_comparison')
        if self.args.cpu_probe:
            cpu,gpu = self.args.cpu_probe.resolve(),self.args.gpu_probe.resolve()
            for path,expected in ((cpu,'cpu'),(gpu,'gpu')):
                check_capture_sources(path,self.source)
                cfg = read(path/'run_configuration.json')['arguments']
                if cfg['backend'] != expected or cfg['envs_per_gpu'] != 1 or not cfg['physics_only_preflight']:
                    raise ValueError('Backend comparison configuration differs')
                if read(path/'worker_result.json')['status'] != 'finished': raise ValueError('Unfinished backend evidence')
        else:
            jobs = self.jobs(ids, 1, 'backend_comparison')
            cpu, gpu = self.parallel([
                lambda:self.worker('backend_cpu',jobs,0,1,'cpu',physics_only=True,resets=20),
                lambda:self.worker('backend_gpu',jobs,1,1,'gpu',physics_only=True,resets=20)])
        choice = select_backend(cpu,gpu)
        backend = choice['selected']
        atomic_json(self.output/'backend_choice.json',choice)
        sweeps = []
        for count in (1,2,4):
            self.stage('resident_full_video_sweep',processes_per_gpu=count,backend=backend)
            result = self.group(f'sweep_p{count}',count,backend,gpus=(0,))
            sweeps.append(result)
            atomic_json(self.output/'resident_sweep.json',sweeps)
        choice = choose_resident_concurrency(sweeps)
        count = choice['processes_per_gpu']
        self.stage('resident_dual_gpu_full_video_and_twenty_resets',processes_per_gpu=count)
        dual = self.group('dual',count,backend,resets=20)
        atomic_json(self.output/'resident_dual.json',dual)
        if dual['status'] != 'passed': raise ValueError('Dual benchmark resource rejection')
        baseline = read(ROOT/'outputs/vib820_gpu_collection_400_preflight/baseline/result.json')
        decoded = read(ROOT/'outputs/vib820_gpu_collection_400_preflight/baseline/video_audit.json')
        ratio = dual['accepted_per_minute']/(240/(baseline['elapsed_s']+decoded['elapsed_s']))
        atomic_json(self.output/'resident_performance_gate.json',dict(throughput_ratio=ratio,
                    status='passed' if ratio>=2 else 'failed',dual=dual))
        if ratio < 2: raise ValueError(f'Resident dual-GPU throughput below required 2x: {ratio:.3f}')
        self.stage('resident_full_video_history',processes_per_gpu=count,backend=backend)
        history = self.worker('full_video_history',self.jobs(ids,1,'full_video_history'),0,1,backend,resets=20)
        historical(history, True)
        self.stage('resident_seal_preflight')
        gate = self.seal(sweeps,dual,backend,cpu,gpu,history,self.args.official_export.resolve())
        if not self.args.execute_production:
            self.state.update(status='preflight_passed',gate=str(gate))
            atomic_json(self.output/'pipeline_state.json',self.state)
            return
        raw = self.args.raw.resolve()
        exported,replays,release = (raw.parent/name for name in ('lerobot','replays','release'))
        self.stage('production_collection',processes_per_gpu=count,backend=backend,
                   estimated_minutes=400/read(gate)['measured_accepted_per_minute'])
        command = [sys.executable,ROOT/'scripts/collect_astribot_dataset.py','--output',raw,
                   '--gpus','0','1','--envs-per-gpu','1','--processes-per-gpu',count,
                   '--backend',backend,'--render-layout',self.args.render_layout,'--preflight-report',gate]
        if (raw/'attempts.json').exists(): command += ['--resume']
        self.command(command,self.output/'production.log')
        self.stage('official_export')
        lerobot = ROOT/'.conda/envs/lerobot/bin/python'
        self.command([lerobot,ROOT/'scripts/export_astribot_lerobot.py','--selection',raw/'selection.json',
                      '--output',exported,'--gpu','0'],self.output/'export.log')
        self.stage('twenty_stratified_action_replays')
        self.command([sys.executable,ROOT/'scripts/finalize_astribot_collection.py','--raw',raw,'--output',replays,
                      '--prepare-replays'],self.output/'prepare_replays.log')
        replay_runs = self.parallel([
            lambda:self.worker('final_replays_gpu0',replays/'jobs_gpu0.json',0,1,backend,physics_only=True),
            lambda:self.worker('final_replays_gpu1',replays/'jobs_gpu1.json',1,1,backend,physics_only=True)])
        self.stage('final_release_verification')
        self.command([lerobot,ROOT/'scripts/finalize_astribot_collection.py','--raw',raw,'--export',exported,
                      '--replay-runs',*replay_runs,'--output',release],self.output/'finalize.log')
        if read(release/'release_report.json')['status'] != 'complete': raise ValueError('Release incomplete')
        self.state.update(status='complete',release=str(release),completed_unix_s=time.time())
        atomic_json(self.output/'pipeline_state.json',self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw',type=Path,default=ROOT/'outputs/vib820_gpu_collection_400')
    parser.add_argument('--official-export',type=Path,required=True)
    parser.add_argument('--execute-production',action='store_true')
    parser.add_argument('--cpu-probe',type=Path,help='Reuse finished capture-source-matched backend evidence')
    parser.add_argument('--gpu-probe',type=Path)
    parser.add_argument('--render-layout',choices=['individual','homogeneous'],default='homogeneous')
    args = parser.parse_args()
    if bool(args.cpu_probe) != bool(args.gpu_probe): parser.error('Reuse requires both CPU and GPU evidence')
    pipeline = ResidentPipeline(args)
    try: pipeline.run()
    except BaseException as exc:
        pipeline.state.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        atomic_json(pipeline.output/'pipeline_state.json',pipeline.state)
        raise


if __name__ == '__main__': main()
