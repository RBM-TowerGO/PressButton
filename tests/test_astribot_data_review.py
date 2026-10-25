import io
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from astribot_data_review import crop_copy, export, image_offsets, inspect_file, mark_entry, sha256, make_rrd


def fixture_file(path):
    with h5py.File(path, 'w') as f:
        f.attrs['test'] = 'preserved'
        f['time'] = np.arange(5) / 30 + 1000
        for name in ['joints_position_state', 'joints_position_command']:
            f['joints_dict/' + name] = np.arange(125).reshape(5, 25)
        f['joints_dict/command_timestamp'] = f['time'][:]
        f['endpoint_wrench_dict/astribot_arm_right'] = np.zeros((5, 6))
        for cam in ['head', 'left', 'right']:
            frames = []
            for i in range(5):
                b = io.BytesIO()
                Image.new('RGB', (10, 8), color=(i * 40, 0, 0)).save(b, format='JPEG')
                frames.append(b.getvalue())
            f[f'images_dict/{cam}/rgb'] = np.frombuffer(b''.join(frames), dtype='uint8')
            f[f'images_dict/{cam}/rgb_size'] = np.array(list(map(len, frames)), dtype=float)
            f[f'images_dict/{cam}/rgb_timestamp'] = f['time'][:]
    return path


def review_fixture(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    p = fixture_file(source / 'task_episode_9.hdf5')
    return {'source': str(source), 'episodes': [dict(original_id=9, file=p.name,
        sha256=sha256(p), status='pending', start_frame=0, end_frame=5,
        note='', audit=inspect_file(p))]}, p


def test_crop_preserves_correspondence_and_source(tmp_path):
    src = fixture_file(tmp_path / 'source.hdf5'); before = sha256(src)
    dst = tmp_path / 'cropped.hdf5'
    crop_copy(src, dst, 1, 4)
    with h5py.File(src) as a, h5py.File(dst) as b:
        assert b.attrs['test'] == 'preserved'
        np.testing.assert_array_equal(b['time'][:], a['time'][1:4])
        np.testing.assert_array_equal(b['joints_dict/joints_position_state'][:],
                                      a['joints_dict/joints_position_state'][1:4])
        for cam in ['head', 'left', 'right']:
            ga, gb = a[f'images_dict/{cam}'], b[f'images_dict/{cam}']
            offsets = image_offsets(ga, 5)
            np.testing.assert_array_equal(gb['rgb'][:], ga['rgb'][offsets[1]:offsets[4]])
    assert sha256(src) == before
    audit = inspect_file(dst, full_images=True)
    assert audit['errors'] == [] and audit['image_samples'] == 9


def test_export_approved_only_and_refuses_overwrite(tmp_path):
    review, src = review_fixture(tmp_path)
    with pytest.raises(ValueError, match='No episodes'):
        export(review, tmp_path / 'out')
    mark_entry(review, 9, 'keep', 1, 4, 'reviewed')
    export(review, tmp_path / 'out')
    mapping = json.loads((tmp_path / 'out/episode_mapping.json').read_text())
    assert mapping[0]['original_id'] == 9 and mapping[0]['new_id'] == 0
    assert mapping[0]['start_frame'] == 1 and mapping[0]['end_frame'] == 4
    assert mapping[0]['audit']['frames'] == 3
    with pytest.raises(ValueError, match='already exists'):
        export(review, tmp_path / 'out')


def test_reject_changed_source_and_invalid_bounds(tmp_path):
    review, src = review_fixture(tmp_path)
    with pytest.raises(ValueError, match='Require'):
        mark_entry(review, 9, 'keep', 3, 2)
    mark_entry(review, 9, 'keep')
    with h5py.File(src, 'a') as f:
        f['time'][0] = 900
    with pytest.raises(ValueError, match='Source changed'):
        export(review, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('out.partial-*'))


def test_flags_timestamp_issue_and_bad_image_index(tmp_path):
    src = fixture_file(tmp_path / 'source.hdf5')
    with h5py.File(src, 'a') as f:
        f['joints_dict/command_timestamp'][0] += 1
    audit = inspect_file(src)
    assert audit['warnings'] and audit['errors']
    with h5py.File(src, 'a') as f:
        f['images_dict/head/rgb_size'][0] += 1
    with pytest.raises(ValueError, match='byte length mismatch'):
        inspect_file(src)


def test_rerun_recording_is_finalized_and_readable(tmp_path):
    review, src = review_fixture(tmp_path)
    output = tmp_path / 'episode.rrd'
    make_rrd(src, output, review['episodes'][0])
    subprocess.run([str(Path(sys.executable).parent / 'rerun'), 'rrd', 'verify', str(output)],
                   check=True, capture_output=True)
