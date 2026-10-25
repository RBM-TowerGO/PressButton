#!/usr/bin/env python3
"""Read-only Astribot HDF5 inspection, Rerun playback, review and copy export."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'data/astribot_s1b_20261025/storage/trans/hdf5_output_elevator1'
DEFAULT_REVIEW = ROOT / 'data/astribot_s1b_20261025_review/review.json'
CAMERAS = ('head', 'left', 'right')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def datasets(f):
    result = {}
    f.visititems(lambda name, obj: result.update({name: obj})
                if isinstance(obj, h5py.Dataset) else None)
    return result


def image_offsets(group, n):
    sizes = group['rgb_size'][:]
    if (sizes.shape != (n,) or not np.isfinite(sizes).all()
            or (sizes <= 0).any() or (sizes != np.floor(sizes)).any()):
        raise ValueError(f'{group.name}: invalid image sizes')
    offsets = np.r_[0, np.cumsum(sizes.astype(np.int64))]
    if offsets[-1] != group['rgb'].size:
        raise ValueError(f'{group.name}: image byte length mismatch')
    return offsets


def inspect_file(path, full_images=False):
    report = {'errors': [], 'warnings': [], 'timestamp_stats': {}, 'image_samples': 0}
    with h5py.File(path, 'r') as f:
        time = f['time'][:]
        if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
            raise ValueError('time must contain at least two finite samples')
        n = len(time)
        report.update(frames=n, duration_seconds=float(time[-1] - time[0]))
        for name, d in datasets(f).items():
            if name.endswith('/rgb'):
                continue
            a = d[:]
            if a.ndim == 0 or len(a) != n:
                report['errors'].append(f'{name}: length mismatch')
                continue
            if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
                report['errors'].append(f'{name}: NaN/Inf')
            if name == 'time' or 'timestamp' in name:
                dt = np.diff(a)
                if (dt < 0).any() or (name == 'time' and (dt <= 0).any()):
                    report['errors'].append(f'{name}: nonmonotonic timestamps')
                offset = float(np.max(np.abs(a - time)))
                report['timestamp_stats'][name] = {
                    'max_abs_offset_seconds': offset,
                    'repeated': int((dt == 0).sum()),
                }
                if offset > .1:
                    report['warnings'].append(f'{name}: offset {offset:.6f}s > 0.1s')
        for name in ('joints_position_state', 'joints_position_command'):
            if f['joints_dict/' + name].shape != (n, 25):
                report['errors'].append(f'{name}: expected (N,25)')
        for cam in CAMERAS:
            group = f['images_dict/' + cam]
            offsets = image_offsets(group, n)
            blob = group['rgb'][:]
            indices = range(n) if full_images else sorted({0, n // 2, n - 1})
            for i in indices:
                try:
                    with Image.open(io.BytesIO(blob[offsets[i]:offsets[i + 1]].tobytes())) as im:
                        im.load()
                    report['image_samples'] += 1
                except Exception as e:
                    report['errors'].append(f'{cam} frame {i}: {e}')
    return report


def load_review(path):
    return json.loads(Path(path).read_text())


def get_entry(review, episode):
    return next(e for e in review['episodes'] if e['original_id'] == episode)


def checked_source(review, entry):
    path = Path(review['source']) / entry['file']
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Source changed since inspection: {path}')
    return path


def scan(args):
    if args.review.exists():
        raise ValueError('Review file exists; use a different --review to avoid losing decisions')
    files = list(args.source.glob('*.hdf5'))
    if not files:
        raise ValueError('No HDF5 files found')
    numbered = []
    for p in files:
        match = re.search(r'_episode_(\d+)\.hdf5$', p.name)
        if not match:
            raise ValueError(f'Unrecognized filename: {p.name}')
        numbered.append((int(match[1]), p))
    if len({i for i, _ in numbered}) != len(numbered):
        raise ValueError('Duplicate episode IDs')
    review = {'schema_version': 1, 'source': str(args.source.resolve()),
              'end_frame_convention': 'exclusive', 'episodes': []}
    seen = {}
    for original_id, p in sorted(numbered):
        digest = sha256(p)
        try:
            audit = inspect_file(p, args.full_images)
        except Exception as e:
            audit = {'errors': [str(e)], 'warnings': [], 'frames': 0}
        if digest in seen:
            audit['warnings'].append(f'Identical file to episode {seen[digest]}')
        seen[digest] = original_id
        review['episodes'].append(dict(original_id=original_id, file=p.name, sha256=digest,
            status='pending', start_frame=0, end_frame=audit['frames'], note='', audit=audit))
        print(f'{original_id}: {audit["frames"]} frames; '
              f'{len(audit["errors"])} errors, {len(audit["warnings"])} warnings', flush=True)
    write_json(args.review, review)
    print(f'Saved {len(numbered)} episodes: {args.review}')


def make_rrd(path, output, entry):
    import rerun as rr
    import rerun.blueprint as rrb
    blueprint = rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(*[rrb.Spatial2DView(origin=f'/images/{cam}', name=cam) for cam in CAMERAS]),
        rrb.Horizontal(
            rrb.TimeSeriesView(origin='/joints', name='Joint state / command (column indices)'),
            rrb.TimeSeriesView(origin='/timing', name='Timestamp minus master time (seconds)'),
            rrb.TimeSeriesView(origin='/wrench', name='Right endpoint wrench (raw 6 columns)')),
        rrb.TextDocumentView(origin='/info', name='Episode / review'), row_shares=[3, 2, 1]),
        rrb.TimePanel(timeline='seconds'), collapse_panels=True)
    rr.init('astribot_hdf5_review', recording_id=entry['sha256'])
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.save(output, default_blueprint=blueprint)
    rr.log('info', rr.TextDocument(json.dumps({k: entry[k] for k in
        ('original_id', 'file', 'status', 'start_frame', 'end_frame', 'note', 'audit')},
        ensure_ascii=False, indent=2)), static=True)
    with h5py.File(path, 'r') as f:
        t = f['time'][:]
        numeric = {k: d[:] for k, d in datasets(f).items() if not k.startswith('images_dict/')}
        images = {cam: (f[f'images_dict/{cam}/rgb'][:],
                       image_offsets(f[f'images_dict/{cam}'], len(t))) for cam in CAMERAS}
        timestamps = {k: d[:] for k, d in datasets(f).items() if 'timestamp' in k}
        for i in range(len(t)):
            rr.set_time('frame', sequence=i)
            rr.set_time('seconds', duration=float(t[i] - t[0]))
            for cam, (blob, offsets) in images.items():
                rr.log(f'images/{cam}', rr.EncodedImage(
                    contents=blob[offsets[i]:offsets[i + 1]], media_type='image/jpeg'))
            for kind in ('state', 'command'):
                a = numeric[f'joints_dict/joints_position_{kind}'][i]
                for j, value in enumerate(a):
                    rr.log(f'joints/j{j:02d}/{kind}', rr.Scalars(float(value)))
            for k, a in timestamps.items():
                rr.log('timing/' + k, rr.Scalars(float(a[i] - t[i])))
            for j, value in enumerate(numeric['endpoint_wrench_dict/astribot_arm_right'][i]):
                rr.log(f'wrench/column_{j}', rr.Scalars(float(value)))
    rr.get_global_data_recording().flush()
    rr.disconnect()
    print(f'Rerun recording: {output}', flush=True)


def mark_entry(review, episode, status, start=None, end=None, note=None):
    e = get_entry(review, episode)
    start = e['start_frame'] if start is None else start
    end = e['end_frame'] if end is None else end
    if not 0 <= start < end <= e['audit']['frames']:
        raise ValueError('Require 0 <= start < end <= frame count (end is exclusive)')
    if status == 'keep' and e['audit']['errors']:
        raise ValueError('Cannot approve a structurally invalid episode')
    e.update(status=status, start_frame=start, end_frame=end)
    if note is not None:
        e['note'] = note


def crop_copy(source, destination, start, end):
    """Preserve attrs and timestamps; rebuild JPEG byte ranges, not just frame arrays."""
    with h5py.File(source, 'r') as src, h5py.File(destination, 'x') as dst:
        n = len(src['time'])
        if not 0 <= start < end <= n:
            raise ValueError('Invalid crop bounds')
        dst.attrs.update(src.attrs)
        def copy(name, obj):
            if isinstance(obj, h5py.Group):
                dst.require_group(name).attrs.update(obj.attrs)
                return
            if name.endswith('/rgb'):
                offsets = image_offsets(obj.parent, n)
                data = obj[int(offsets[start]):int(offsets[end])]
            else:
                if obj.ndim == 0 or len(obj) != n:
                    raise ValueError(f'Unsupported non-frame dataset: {name}')
                data = obj[start:end]
            out = dst.create_dataset(name, data=data, dtype=obj.dtype)
            out.attrs.update(obj.attrs)
        src.visititems(copy)


def export(review, output, progress=None):
    output = output.resolve()
    source = Path(review['source']).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Export must be separate from the source directory')
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    # The persisted list is the user's export order (initially numeric ID order).
    selected = [e for e in review['episodes'] if e['status'] == 'keep']
    if not selected:
        raise ValueError('No episodes marked keep')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    try:
        (staging / 'episodes').mkdir()
        mapping = []
        for new_id, entry in enumerate(selected):
            path = checked_source(review, entry)
            name = f'episode_{new_id:06d}.hdf5'
            dest = staging / 'episodes' / name
            crop_copy(path, dest, entry['start_frame'], entry['end_frame'])
            audit = inspect_file(dest, full_images=True)
            if audit['errors']:
                raise ValueError(f'{name}: {audit["errors"]}')
            mapping.append(dict(new_id=new_id, output_file=name, source_file=str(path),
                source_sha256=entry['sha256'], output_sha256=sha256(dest),
                original_id=entry['original_id'], start_frame=entry['start_frame'],
                end_frame=entry['end_frame'], note=entry['note'], audit=audit))
            if progress:
                progress(new_id + 1, len(selected))
        write_json(staging / 'episode_mapping.json', mapping)
        write_json(staging / 'review_snapshot.json', review)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f'Exported {len(mapping)} reviewed HDF5 episodes to {output}; timestamps preserved')


def review_interactively(review, args):
    """Rerun is the viewer; explicit decisions are entered in this terminal."""
    for entry in review['episodes']:
        episode = entry['original_id']
        if entry['status'] != 'pending' or episode < args.start_at:
            continue
        output = args.review.parent / 'rerun' / f'episode_{episode:06d}.rrd'
        make_rrd(checked_source(review, entry), output, entry)
        command = [str(Path(sys.executable).parent / 'rerun'), str(output),
                   '--serve-web', '--bind', '127.0.0.1',
                   '--web-viewer-port', '9090', '--port', '9876']
        viewer = subprocess.Popen(command)
        try:
            print(f'\n原编号 {episode} | {entry["audit"]["frames"]} 帧 | '
                  f'告警 {entry["audit"]["warnings"]}\n'
                  '浏览器 http://localhost:9090（每条切换后刷新）\n'
                  'k=保留；k 起始帧 结束帧=裁剪保留；r=剔除；h=待排查；s=跳过；q=退出。', flush=True)
            while True:
                tokens = input('审核决定 > ').strip().split()
                if viewer.poll() is not None:
                    raise ValueError('Viewer exited; check whether ports 9090/9876 are already occupied')
                if tokens == ['q']:
                    return
                if tokens == ['s']:
                    break
                try:
                    if not tokens or tokens[0] not in ('k', 'r', 'h'):
                        raise ValueError('请输入 k / k start end / r / h / s / q')
                    if len(tokens) != 1 and not (tokens[0] == 'k' and len(tokens) == 3):
                        raise ValueError('裁剪格式：k start end，end 不包含')
                    start, end = (map(int, tokens[1:]) if len(tokens) == 3 else (None, None))
                    note = input('备注（可留空） > ')
                    mark_entry(review, episode, {'k': 'keep', 'r': 'reject', 'h': 'hold'}[tokens[0]],
                               start, end, note)
                    write_json(args.review, review)
                    break
                except ValueError as e:
                    print(e)
        finally:
            viewer.terminate()
            try:
                viewer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                viewer.kill(); viewer.wait()
    print('本轮待审核轨迹已浏览完。使用 list 查看结果，export 导出保留项。')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, default=DEFAULT_REVIEW)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('scan'); s.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    s.add_argument('--full-images', action='store_true')
    sub.add_parser('list')
    sub.add_parser('gui', help='Desktop graphical review, crop, reorder and export')
    interactive = sub.add_parser('review', help='Sequential browser playback with terminal decisions')
    interactive.add_argument('--start-at', type=int, default=0)
    v = sub.add_parser('view'); v.add_argument('episode', type=int)
    v.add_argument('--open', action='store_true', help='Open native Rerun viewer (requires desktop)')
    v.add_argument('--web', action='store_true', help='Serve local browser viewer until Ctrl-C')
    m = sub.add_parser('mark'); m.add_argument('episode', type=int)
    m.add_argument('status', choices=['pending', 'keep', 'reject', 'hold'])
    m.add_argument('--start', type=int); m.add_argument('--end', type=int)
    m.add_argument('--note')
    e = sub.add_parser('export'); e.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.command == 'scan':
        scan(args); return
    review = load_review(args.review)
    if args.command == 'list':
        for e in review['episodes']:
            print(f'{e["original_id"]:3} {e["status"]:7} [{e["start_frame"]},{e["end_frame"]}) '
                  f'warnings={len(e["audit"]["warnings"])} errors={len(e["audit"]["errors"])} {e["note"]}')
    elif args.command == 'mark':
        mark_entry(review, args.episode, args.status, args.start, args.end, args.note)
        write_json(args.review, review)
        print(f'Episode {args.episode}: {args.status}; saved {args.review}')
    elif args.command == 'export':
        export(review, args.output)
    elif args.command == 'review':
        review_interactively(review, args)
    elif args.command == 'gui':
        from astribot_review_gui import launch
        launch(args.review)
    elif args.command == 'view':
        entry = get_entry(review, args.episode)
        output = args.review.parent / 'rerun' / f'episode_{args.episode:06d}.rrd'
        make_rrd(checked_source(review, entry), output, entry)
        command = [str(Path(sys.executable).parent / 'rerun'), str(output)]
        if args.web:
            command += ['--serve-web', '--bind', '127.0.0.1', '--web-viewer-port', '9090', '--port', '9876']
        if args.open or args.web:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, StopIteration, OSError) as e:
        raise SystemExit(f'Error: {e}')
    except (KeyboardInterrupt, EOFError):
        print('\n已退出；已保存的审核决定保留，原始数据未改动。')
