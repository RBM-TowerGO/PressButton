#!/usr/bin/env python3
"""Prepare stratified action replays and verify a complete Astribot release.

Run verification in the isolated LeRobot environment. This writes derived
reports/previews only; committed raw episodes are never modified.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from pressb.astribot_collection.contract import CAMERAS, atomic_json, sha256, validate_commit
from pressb.astribot_collection.replay import stratified_replays


def read(path):
    return json.loads(Path(path).read_text())


def prepare_replays(raw, output):
    manifest = read(raw/'manifest.json')
    selection = read(raw/'selection.json')
    if len(selection) != 400:
        raise ValueError('Final replay selection requires 400 accepted episodes')
    selected = stratified_replays(selection, manifest['candidates'])
    jobs = []
    for entry in selected:
        source = Path(entry['directory'])
        validate_commit(source)
        config_path = output/f'config_{entry["candidate_id"]:06d}.json'
        atomic_json(config_path, read(source/'report.json')['config'])
        jobs.append(dict(id=f'replay_{entry["candidate_id"]:06d}',
                         candidate_id=entry['candidate_id'], config=str(config_path.resolve()),
                         replay_source=str(source.resolve())))
    atomic_json(output/'strata.json', selected)
    atomic_json(output/'jobs.json', jobs)
    for gpu in (0, 1):
        atomic_json(output/f'jobs_gpu{gpu}.json', jobs[gpu::2])
    return jobs


def preview(source, destination, gpu):
    """A derived three-view preview; original streams retain their resolution."""
    command = ['ffmpeg', '-nostdin', '-n', '-v', 'error', '-filter_complex_threads', '1']
    names = ('head', 'wrist_left', 'wrist_right')
    filters = []
    for i, name in enumerate(names):
        command += ['-threads', '2', '-i', str(source/f'{name}.mp4')]
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,scale=640:480:force_original_aspect_ratio=decrease,'
                       f'pad=640:520:(ow-iw)/2:40+(480-ih)/2:color=0x101824,setsar=1,'
                       f'drawtext=text={name}:x=12:y=8:fontsize=24:fontcolor=white[v{i}]')
    filters.append('[v0][v1][v2]hstack=inputs=3[out]')
    command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an',
                '-c:v', 'h264_nvenc', '-gpu', str(gpu), '-preset', 'p4', '-cq', '18',
                '-b:v', '0', '-bf', '0', '-g', '30', '-pix_fmt', 'yuv420p',
                '-threads', '1', '-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True)


def verify(raw, exported, replay_runs, output, gpu):
    from export_astribot_lerobot import reader_audit
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from pressb.astribot_collection.video import decode_audit
    started = time.time()
    manifest = read(raw/'manifest.json')
    selection = read(raw/'selection.json')
    summary = read(raw/'summary.json')
    if summary['status'] != 'collected' or len(selection) != 400:
        raise ValueError('Collection is incomplete')
    ids = [r['candidate_id'] for r in selection]
    if ids != sorted(set(ids)):
        raise ValueError('Selection is not unique and ordered')
    sources = read(exported/'meta/astribot_sources.json')
    expected = []
    raw_bytes = 0
    for entry in selection:
        source = Path(entry['directory'])
        commit = validate_commit(source)
        if commit['audit']['status'] != 'passed':
            raise ValueError('Raw acceptance is incomplete')
        expected.append({**entry, 'commit_sha256': sha256(source/'commit.json')})
        raw_bytes += sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    if sources != expected:
        raise ValueError('Export does not match the exact selected commits')
    reader = reader_audit(exported, expected)
    ds = LeRobotDataset('local/astribot_vib820', root=exported, video_backend='pyav')
    videos = defaultdict(lambda: dict(frames=0, size=None))
    for i in range(400):
        episode = ds.meta.episodes[i]
        length = int(episode['dataset_to_index'])-int(episode['dataset_from_index'])
        for name, size in CAMERAS.items():
            path = exported/ds.meta.get_video_file_path(i, 'observation.images.'+name)
            videos[path]['frames'] += length
            videos[path]['size'] = size
    def decode(item):
        path, spec = item
        return dict(path=str(path), **decode_audit(path, spec['frames'], spec['size']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        decoded = list(pool.map(decode, videos.items()))
    atomic_json(output/'training_video_audit.json', dict(status='passed', videos=decoded))
    strata = stratified_replays(selection, manifest['candidates'])
    required = {r['candidate_id']: r for r in strata}
    replay_evidence = {}
    for directory in replay_runs:
        result = read(directory/'worker_result.json')
        if result['status'] != 'finished':
            raise ValueError('Replay process did not finish')
        for row in result['results']:
            cid = row['job']['candidate_id']
            if cid not in required or cid in replay_evidence:
                raise ValueError('Unexpected or duplicate replay')
            report = read(Path(row['output'])/'report.json')
            if (row['status'] != 'passed' or row.get('physical_audit', {}).get('status') != 'passed'
                    or not report.get('replay', {}).get('training_prefix_success')
                    or Path(report['replay']['source']).resolve() != Path(required[cid]['directory']).resolve()):
                raise ValueError(f'Action replay failed: {cid}')
            replay_evidence[cid] = dict(directory=row['output'], report_sha256=sha256(Path(row['output'])/'report.json'))
    if replay_evidence.keys() != required.keys():
        raise ValueError('Not all 20 stratified action replays passed')
    candidates = {r['candidate_id']: r for r in manifest['candidates']}
    representatives = [min(selection, key=lambda r:candidates[r['candidate_id']]['yaw_deg']),
                       min(selection, key=lambda r:abs(candidates[r['candidate_id']]['yaw_deg']-5)),
                       max(selection, key=lambda r:candidates[r['candidate_id']]['yaw_deg'])]
    previews = []
    for entry in representatives:
        destination = output/f'three_views_candidate_{entry["candidate_id"]:06d}.mp4'
        if not destination.exists():
            preview(Path(entry['directory']), destination, gpu)
        frames = len(read(Path(entry['directory'])/'camera_timing.json'))
        decoded_preview = decode_audit(destination, frames, (1920, 520))
        previews.append(dict(candidate=candidates[entry['candidate_id']], path=str(destination.resolve()),
                             sha256=sha256(destination), audit=decoded_preview))
    distribution = {}
    for field, edges in [('x_m', np.linspace(-.30, -.14, 9)), ('y_m', np.linspace(0, .4, 9)),
                         ('yaw_deg', np.linspace(0, 10, 11))]:
        values = [candidates[cid][field] for cid in ids]
        distribution[field] = dict(min=min(values), max=max(values), mean=float(np.mean(values)),
                                   bin_edges=edges.tolist(), counts=np.histogram(values, edges)[0].tolist())
    freeze = read(raw/'production_freeze.json')
    journal = read(raw/'attempts.json')
    worker_intervals = []
    for worker in sorted({r['worker'] for r in journal if r['outcome'] != 'queued'}):
        config = read(Path(worker)/'run_configuration.json')
        result_path = Path(worker)/'worker_result.json'
        if result_path.exists():
            ended = config['started_unix_s']+read(result_path)['elapsed_s']
        else:
            # Failed startup attempts remain in the journal and time accounting.
            resource_path = Path(worker)/'resources.jsonl'
            samples = resource_path.read_text().splitlines() if resource_path.exists() else []
            ended = max([config['started_unix_s']]+[json.loads(line)['unix_s'] for line in samples])
        worker_intervals.append((config['started_unix_s'], ended))
    report = dict(status='complete', successful_episodes=400, raw_bytes=raw_bytes,
                  training_video_bytes=sum(r['bytes'] for r in decoded), selection_sha256=sha256(raw/'selection.json'),
                  finalizer_sha256=sha256(Path(__file__)), reader=reader, replay_count=20,
                  replay_evidence=replay_evidence, previews=previews, attempts=summary['attempts'],
                  outcomes=summary['outcomes'], failure_reasons=summary['failure_reasons'],
                  success_filtered_distribution=distribution, preflight=freeze['gate'],
                  collection_wall_s=max(x[1] for x in worker_intervals)-min(x[0] for x in worker_intervals),
                  export_timing=read(exported/'meta/astribot_export_timing.json'),
                  final_verification_s=time.time()-started, completed_unix_s=time.time())
    atomic_json(output/'release_report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-replays', action='store_true')
    p.add_argument('--export', dest='exported', type=Path)
    p.add_argument('--replay-runs', type=Path, nargs='+')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_replays:
        prepare_replays(args.raw, args.output)
    else:
        if not args.exported or not args.replay_runs:
            p.error('Verification requires --export and --replay-runs')
        verify(args.raw, args.exported, args.replay_runs, args.output, args.gpu)


if __name__ == '__main__':
    main()
