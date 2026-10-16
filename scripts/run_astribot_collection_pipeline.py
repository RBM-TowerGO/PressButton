#!/usr/bin/env python3
"""Execute evidence-gated preflight, collection, export and final action replays.

The pipeline stops on any failed acceptance; it never edits a gate to bypass it.
Completed runs can be reused only with identical source hashes and arguments.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from pressb.astribot_collection.contract import atomic_json,sha256
from pressb.astribot_collection.dispatch import source_files
from pressb.astribot_collection.verification import summarize_run,choose_concurrency,compare_initial_images
from seal_astribot_preflight import historical


def read(path):return json.loads(Path(path).read_text())


class Pipeline:
    def __init__(self,args):
        self.args=args;self.output=args.output.resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.env=os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES='0,1',OMNICLIENT_HUB_MODE='disabled',OMP_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES',PYTHONNOUSERSITE='1')
        self.source=source_files(ROOT)
        frozen=self.output/'pipeline_source.json'
        if frozen.exists() and read(frozen)!=self.source:raise ValueError('Pipeline source changed; use a fresh output directory')
        atomic_json(frozen,self.source)
        self.state=read(self.output/'pipeline_state.json') if (self.output/'pipeline_state.json').exists() else dict(started_unix_s=time.time())

    def stage(self,name,**details):
        self.state.update(status='running',stage=name,updated_unix_s=time.time(),**details)
        atomic_json(self.output/'pipeline_state.json',self.state)
        print(name,json.dumps(details),flush=True)

    def command(self,command,log):
        env=self.env.copy();env['TMPDIR']=tempfile.mkdtemp(prefix='ap_',dir=ROOT/'.cache/sim_tmp')
        with Path(log).open('ab') as file:
            subprocess.run([str(x) for x in command],cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT,check=True)

    def jobs(self,ids,envs,label):
        source=ROOT/'outputs/vib820_yaw_20261014/validated_n50_w8/episodes'
        rows=[dict(id=f'wave{i//envs}_reference{sample:03d}_{i%envs}',
                   config=str(source/f'sample_{sample:03d}/config.json')) for i,sample in enumerate(ids)]
        path=self.output/(label+'_jobs.json');atomic_json(path,rows);return path

    def worker(self,name,jobs,gpu,envs,backend,*,physics_only=False,resets=0):
        destination=self.output/name
        if (destination/'worker_result.json').exists() and read(destination/'worker_result.json')['status']=='finished':
            config=read(destination/'run_configuration.json')
            expected=dict(gpu=gpu,envs_per_gpu=envs,backend=backend,physics_only_preflight=physics_only,
                          reset_probe=resets,render_layout=self.args.render_layout,audit_commit=not physics_only)
            if config['source_sha256']!=self.source or any(config['arguments'][k]!=v for k,v in expected.items()):
                raise ValueError(f'Stale completed run: {destination}')
            return destination
        if destination.exists():raise ValueError(f'Interrupted preflight retained: {destination}; use a fresh pipeline directory')
        command=[sys.executable,ROOT/'scripts/collect_astribot_dataset.py','--output',destination,'--jobs',jobs,
                 '--gpu',gpu,'--envs-per-gpu',envs,'--backend',backend,'--render-layout',self.args.render_layout]
        command+=['--physics-only-preflight'] if physics_only else ['--audit-commit']
        if resets:command+=['--reset-probe',resets]
        self.command(command,self.output/(name+'.log'))
        # SimulationApp shutdown may turn an exception into exit status zero.
        # A successful subprocess exit therefore cannot establish worker success.
        error_file = destination/'worker_error.json'
        result_file = destination/'worker_result.json'
        if error_file.exists():
            raise RuntimeError(f'Worker {name} failed: {read(error_file).get("error", "unknown error")}')
        if not result_file.exists() or read(result_file).get('status') != 'finished':
            raise RuntimeError(f'Worker {name} exited without a finished result: {destination}')
        return destination

    def parallel(self,calls):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fn) for fn in calls]
            return [future.result() for future in futures]

    def image_probe(self,count,card,backend):
        destination=self.output/f'image_probe_k{count}'
        if (destination/'result.json').exists():
            cfg=read(destination/'configuration.json')
            if (cfg['source_sha256']!=self.source or cfg['envs_per_gpu']!=count or
                cfg['gpu']!=card or cfg['backend']!=backend or cfg['render_layout']!=self.args.render_layout):raise ValueError('Stale image probe')
            return destination
        if destination.exists():raise ValueError(f'Interrupted image probe: {destination}; use a fresh pipeline directory')
        config=ROOT/'outputs/vib820_yaw_20261014/validated_n50_w8/episodes/sample_024/config.json'
        self.command([sys.executable,ROOT/'scripts/probe_astribot_images.py','--output',destination,
                      '--config',config,'--gpu',card,'--envs-per-gpu',count,'--backend',backend,
                      '--render-layout',self.args.render_layout],
                     self.output/f'image_probe_k{count}.log')
        return destination

    def run(self):
        import psutil
        self.stage('waiting_for_previous_diagnostics')
        processes=[]
        for directory in self.args.wait_runs:
            try:processes.append(psutil.Process(read(directory/'run_configuration.json')['pid']))
            except psutil.NoSuchProcess:pass
        while any(p.is_running() and p.status()!=psutil.STATUS_ZOMBIE for p in processes):time.sleep(2)
        ids=[1,2,3,5,10,20,24,35,40,50]
        compare_jobs=self.jobs(ids,4,'backend_comparison')
        self.stage('cpu_gpu_physical_comparison')
        cpu,gpu=self.parallel([
            lambda:self.worker('backend_cpu',compare_jobs,0,4,'cpu',physics_only=True),
            lambda:self.worker('backend_gpu',compare_jobs,1,4,'gpu',physics_only=True)])
        historical(cpu,False)
        try:historical(gpu,False);gpu_correct=True
        except ValueError:gpu_correct=False
        cpu_time=read(cpu/'worker_result.json')['timings']['physics_s']
        gpu_time=read(gpu/'worker_result.json')['timings']['physics_s']
        backend='gpu' if gpu_correct and gpu_time<cpu_time else 'cpu'
        atomic_json(self.output/'backend_choice.json',dict(selected=backend,cpu_physics_s=cpu_time,
            gpu_physics_s=gpu_time,gpu_historical_acceptance=gpu_correct,
            comparison_scope='Same ten historical cases in four environments; physical stepping, not a video-throughput baseline.'))
        self.stage('matched_image_isolation_before_video_sweep')
        def image_queue(card,counts):return [self.image_probe(count,card,backend) for count in counts]
        image_runs=self.parallel([lambda:image_queue(0,(1,8)),lambda:image_queue(1,(2,4))])
        reference=self.output/'image_probe_k1/env0'
        comparisons=[path/f'env{i}' for queue in image_runs for path in queue
                     for i in range(read(path/'configuration.json')['envs_per_gpu'])]
        comparison=compare_initial_images(reference,comparisons)
        atomic_json(self.output/'early_brightness.json',comparison)
        if comparison['status']!='passed':raise ValueError('Matched image isolation failed before video sweep')
        self.stage('full_video_concurrency_sweep',backend=backend)
        good=[1,2,3,5,10,20,35,40]
        def sweep(count,card):
            if count==1:cases=[24,24,24]
            elif count==2:cases=[24,24,1,20,24,35]
            elif count==4:cases=[24,24,24,50]+good
            else:cases=[24]*7+[50]+good+good
            return self.worker(f'sweep_k{count}',self.jobs(cases,count,f'sweep_k{count}'),card,count,backend)
        # Balance the two independent queues; no GPU waits for the other card
        # between concurrency points. Each card still runs exactly one app.
        def queue(card,counts):return [sweep(count,card) for count in counts]
        queues=self.parallel([lambda:queue(0,(1,8)),lambda:queue(1,(2,4))])
        sweeps=sorted([path for queue in queues for path in queue],
                      key=lambda path:read(path/'run_configuration.json')['arguments']['envs_per_gpu'])
        choice=choose_concurrency([summarize_run(path) for path in sweeps]);count=choice['envs_per_gpu']
        self.stage('historical_reuse_and_single_environment',envs_per_gpu=count)
        multi=self.output/f'sweep_k{count}'
        if count in (4,8):
            historical(multi,True)
            single=self.worker('single_history',self.jobs(ids,1,'single_history'),0,1,backend,physics_only=True)
        else:
            single,multi=self.parallel([
                lambda:self.worker('single_history',self.jobs(ids,1,'single_history'),0,1,backend,physics_only=True),
                lambda:self.worker('multi_history',self.jobs(ids,count,'multi_history'),1,count,backend)])
        self.stage('dual_gpu_benchmark_and_twenty_resets')
        cases=[24]*count+[good[i%len(good)] for i in range(count*2)]
        dual_jobs=self.jobs(cases,count,'dual')
        dual=self.parallel([lambda:self.worker('dual_gpu0',dual_jobs,0,count,backend,resets=20),
                            lambda:self.worker('dual_gpu1',dual_jobs,1,count,backend,resets=20)])
        brightness=[directory/'episodes'/f'wave0_reference024_{index}' for directory in dual for index in range(count)]
        comparison=compare_initial_images(reference,brightness)
        atomic_json(self.output/'brightness.json',comparison)
        if comparison['status']!='passed':raise ValueError('Environment brightness acceptance failed')
        self.stage('seal_preflight')
        gate=self.output/'production_gate.json'
        legacy=ROOT/'outputs/vib820_gpu_collection_400_preflight'
        command=[sys.executable,ROOT/'scripts/seal_astribot_preflight.py','--single-history',single,'--multi-history',multi,
                 '--sweep',*sweeps,'--dual',*dual,'--baseline',legacy/'baseline','--gpu-probe',gpu,'--cpu-probe',cpu,
                 '--reset-run',dual[0],'--brightness-reference',reference,'--brightness-episodes',*brightness,
                 '--official-export',self.args.official_export,'--unit-junit',legacy/'unit_fault_checks.xml',
                 '--nvenc-junit',legacy/'nvenc_fault_checks.xml','--output',gate]
        self.command(command,self.output/'seal.log')
        if not self.args.execute_production:
            self.state.update(status='preflight_passed',gate=str(gate));atomic_json(self.output/'pipeline_state.json',self.state);return
        self.stage('production_collection',estimated_minutes=400/read(gate)['measured_accepted_per_minute'])
        raw=self.args.raw.resolve();exported=raw.parent/'lerobot';replays=raw.parent/'replays';release=raw.parent/'release'
        command=[sys.executable,ROOT/'scripts/collect_astribot_dataset.py','--output',raw,'--gpus','0','1',
                 '--envs-per-gpu',count,'--backend',backend,'--render-layout',self.args.render_layout,'--preflight-report',gate]
        if (raw/'attempts.json').exists():command+=['--resume']
        self.command(command,self.output/'production.log')
        self.stage('official_export')
        lerobot=ROOT/'.conda/envs/lerobot/bin/python'
        self.command([lerobot,ROOT/'scripts/export_astribot_lerobot.py','--selection',raw/'selection.json',
                      '--output',exported,'--gpu','0'],self.output/'export.log')
        self.stage('twenty_stratified_action_replays')
        self.command([sys.executable,ROOT/'scripts/finalize_astribot_collection.py','--raw',raw,'--output',replays,
                      '--prepare-replays'],self.output/'prepare_replays.log')
        replay_runs=self.parallel([
            lambda:self.worker('final_replays_gpu0',replays/'jobs_gpu0.json',0,count,backend,physics_only=True),
            lambda:self.worker('final_replays_gpu1',replays/'jobs_gpu1.json',1,count,backend,physics_only=True)])
        self.stage('final_release_verification')
        self.command([lerobot,ROOT/'scripts/finalize_astribot_collection.py','--raw',raw,'--export',exported,
                      '--replay-runs',*replay_runs,'--output',release],self.output/'finalize.log')
        if read(release/'release_report.json')['status']!='complete':raise ValueError('Release verification incomplete')
        self.state.update(status='complete',release=str(release),completed_unix_s=time.time())
        atomic_json(self.output/'pipeline_state.json',self.state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw',type=Path,default=ROOT/'outputs/vib820_gpu_collection_400')
    parser.add_argument('--official-export',type=Path,required=True)
    parser.add_argument('--wait-runs',type=Path,nargs='*',default=[])
    parser.add_argument('--execute-production',action='store_true')
    parser.add_argument('--render-layout',choices=['individual','homogeneous','isolated'],required=True)
    args=parser.parse_args();pipeline=Pipeline(args)
    try:pipeline.run()
    except BaseException as exc:
        pipeline.state.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        atomic_json(pipeline.output/'pipeline_state.json',pipeline.state)
        raise


if __name__=='__main__':main()
