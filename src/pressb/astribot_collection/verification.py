"""Evidence-based benchmark summaries; failed/incomplete runs are never throughput."""
from pathlib import Path
import json
import numpy as np
from .contract import CAMERAS, validate_commit


def read(path):
    return json.loads(Path(path).read_text())


def summarize_run(directory):
    directory=Path(directory)
    if (directory/'diagnostic_overlap.json').exists():
        raise ValueError(f'Performance interval overlaps a diagnostic workload: {directory}')
    config=read(directory/'run_configuration.json')
    result=read(directory/'worker_result.json')
    if result['status']!='finished':raise ValueError(f'Unfinished benchmark: {directory}')
    if config['arguments']['physics_only_preflight']:raise ValueError('Physics-only run is not a video benchmark')
    if not config['arguments']['audit_commit']:raise ValueError('Benchmark did not enforce full acceptance')
    if len(result['waves'])<3:raise ValueError('Need warmup and at least two measurement waves')
    rows={r['id']:r for r in result['results']}
    for row in rows.values():
        if row['status']=='accepted':validate_commit(row['output'])
        elif row['status']!='failed' or read(Path(row['output'])/'report.json').get('termination_reason')!='initial_collision':
            raise ValueError(f'Unaccepted benchmark episode: {row["id"]}: {row["status"]}')
    measured=result['waves'][1:]
    accepted=sum(r['status']=='accepted' for w in measured for r in w['results'])
    elapsed=sum(w['elapsed_s'] for w in measured)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    start=min(e['unix_s'] for e in events if e['kind']=='wave_start' and e['wave']>=1)
    end=max(e['unix_s'] for e in events if e['kind']=='wave_done' and e['wave']>=1)
    resources=[json.loads(line) for line in (directory/'resources.jsonl').read_text().splitlines()]
    resources=[r for r in resources if start<=r['unix_s']<=end]
    gpu=config['arguments']['gpu']
    gpu_rows=[g for r in resources for g in r.get('gpus',[]) if g['index']==gpu]
    if not resources or not gpu_rows:raise ValueError('Missing performance resource evidence')
    return dict(directory=str(directory.resolve()),gpu=gpu,envs_per_gpu=config['arguments']['envs_per_gpu'],
        accepted=accepted,measurement_waves=len(measured),measurement_s=elapsed,measurement_start_unix_s=start,
        measurement_end_unix_s=end,accepted_per_minute=60*accepted/elapsed,
        startup_s=result['timings'].get('simulation_app_startup_s',0)+result['timings'].get('startup_s',0),
        max_vram_fraction=max(g['memory_used_mib']/g['memory_total_mib'] for g in gpu_rows),
        min_available_ram_gib=min(r['available_ram_bytes'] for r in resources)/1024**3,
        mean_gpu_compute_percent=float(np.mean([g['compute_percent'] for g in gpu_rows])),
        mean_gpu_encoder_percent=float(np.mean([g['encoder_percent'] for g in gpu_rows])),
        mean_host_cpu_percent=float(np.mean([r['cpu_percent'] for r in resources])),
        stages_s={k:sum(w['stages'].get(k,0) for w in measured) for k in result['timings']},
        source_sha256=config['source_sha256'])


def concurrent_throughput(summaries):
    """Use actual overlapping measurement intervals, never extrapolate two GPUs."""
    if len({r['gpu'] for r in summaries})!=len(summaries):raise ValueError('Duplicate GPU in concurrent benchmark')
    start=min(r['measurement_start_unix_s'] for r in summaries)
    end=max(r['measurement_end_unix_s'] for r in summaries)
    if max(r['measurement_start_unix_s'] for r in summaries)>=min(r['measurement_end_unix_s'] for r in summaries):
        raise ValueError('GPU measurements did not run concurrently')
    accepted=sum(r['accepted'] for r in summaries)
    return dict(accepted=accepted,wall_s=end-start,accepted_per_minute=60*accepted/(end-start))


def choose_concurrency(summaries):
    if {r['envs_per_gpu'] for r in summaries}!={1,2,4,8}:raise ValueError('Incomplete 1/2/4/8 sweep')
    eligible=[r for r in summaries if r['max_vram_fraction']<=.8 and r['min_available_ram_gib']>=32]
    if not eligible:raise ValueError('No concurrency meets memory constraints')
    best=max(r['accepted_per_minute'] for r in eligible)
    return min((r for r in eligible if r['accepted_per_minute']>=.95*best),key=lambda r:r['envs_per_gpu'])


def compare_initial_images(single_directory,multi_directories):
    """Same historical configuration, same phase, all seven original resolutions."""
    from PIL import Image
    single=Path(single_directory);config=read(single/'report.json')['config'];rows=[]
    reference={name:np.asarray(Image.open(single/f'initial_ready_{name}.png'),dtype=float) for name in CAMERAS}
    for directory in map(Path,multi_directories):
        if read(directory/'report.json')['config']!=config:raise ValueError('Brightness reference configuration differs')
        differences={}
        for name,(width,height) in CAMERAS.items():
            image=np.asarray(Image.open(directory/f'initial_ready_{name}.png'),dtype=float)
            if image.shape!=(height,width,3):raise ValueError('Image dimensions changed')
            differences[name]=float(abs(image.mean()-reference[name].mean())/max(reference[name].mean(),1))
        rows.append(dict(directory=str(directory),relative_mean_differences=differences,
            passed=all(v<=.05 for v in differences.values())))
    return dict(status='passed' if rows and all(r['passed'] for r in rows) else 'failed',
                reference=str(single),maximum_relative_mean_difference=.05,comparisons=rows)


def resident_throughput(summaries, processes_per_gpu, gpus):
    """Count distinct resident workers over their actual overlapping intervals."""
    if len(summaries) != len(gpus)*processes_per_gpu:raise ValueError('Incomplete resident worker group')
    directories=[r['directory'] for r in summaries]
    if len(set(directories)) != len(directories):raise ValueError('Duplicate resident worker')
    if any(r['envs_per_gpu'] != 1 for r in summaries):raise ValueError('Resident worker must own one environment')
    if {g:sum(r['gpu']==g for r in summaries) for g in gpus} != {g:processes_per_gpu for g in gpus}:raise ValueError('Resident GPU/worker count mismatch')
    if any(r['gpu'] not in gpus for r in summaries):raise ValueError('Unexpected GPU')
    start=min(r['measurement_start_unix_s'] for r in summaries)
    end=max(r['measurement_end_unix_s'] for r in summaries)
    if len(summaries)>1 and max(r['measurement_start_unix_s'] for r in summaries)>=min(r['measurement_end_unix_s'] for r in summaries):raise ValueError('Resident measurements did not run concurrently')
    if end<=start:raise ValueError('Invalid measurement interval')
    accepted=sum(r['accepted'] for r in summaries)
    return dict(processes_per_gpu=processes_per_gpu,gpus=gpus,accepted=accepted,wall_s=end-start,
        accepted_per_minute=60*accepted/(end-start),max_vram_fraction=max(r['max_vram_fraction'] for r in summaries),
        min_available_ram_gib=min(r['min_available_ram_gib'] for r in summaries),workers=summaries)


def choose_resident_concurrency(groups):
    if {r['processes_per_gpu'] for r in groups}!={1,2,4}:raise ValueError('Incomplete resident 1/2/4 sweep')
    eligible=[r for r in groups if r.get('status') != 'resource_rejected' and r.get('max_vram_fraction',1)<=.8 and r.get('min_available_ram_gib',0)>=32]
    if not eligible:raise ValueError('No resident concurrency meets memory constraints')
    best=max(r['accepted_per_minute'] for r in eligible)
    return min((r for r in eligible if r['accepted_per_minute']>=.95*best),key=lambda r:r['processes_per_gpu'])
