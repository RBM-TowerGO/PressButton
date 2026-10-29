"""Read local LeRobot v3 using explicit S1B contracts, never implicit Hub splits."""
import json
from pathlib import Path
import numpy as np
from pressb.s1b_deployment.contract import JOINTS, MIXED_TASK, TASK, sha, canonical
from .transforms import require, fit_stats


def read_json(path):
    return json.loads(Path(path).read_text())


class Dataset:
    def __init__(self, root, split='train', *, splits=None, anchors=None, contract=None):
        import pyarrow.parquet as pq
        self.root = Path(root).resolve()
        self.split = split
        require(split in ('train', 'validation'), 'Unknown split')
        paths = dict(splits=Path(splits or self.root/'meta/splits.json'),
                     anchors=Path(anchors or self.root/'meta/training_anchors.json'),
                     contract=Path(contract or self.root/'meta/data_contract.json'))
        self.manifests = {k: read_json(v) for k, v in paths.items()}
        self.manifest_hashes = {k: sha(v) for k, v in paths.items()}
        info = self.info = read_json(self.root/'meta/info.json')
        c = self.manifests['contract']
        require(c['names'] == JOINTS and c['arm_head_units'] == 'rad', 'Joint order/units mismatch')
        require(c['action_gripper']['encoding'] == 'binary' and c['action_gripper']['open'] == 0
                and c['action_gripper']['closed'] == 1, 'Gripper contract mismatch')
        require(c['time']['fps'] == 30 and c['time']['horizon'] == 7
                and c['time']['target_offsets_steps'] == list(range(1, 8))
                and c['time']['single_step_storage'] == 'action[k] = command selected for grid[k+1]; omit last grid row without next target'
                and c['time']['training_window'] == 'action[k:k+7], no additional one-step shift; require all seven valid',
                'Explicit next-grid action contract required')
        require(info['codebase_version'] == 'v3.0' and info['fps'] == 30, 'Expected LeRobot v3 at 30 Hz')
        for key in ('action', 'observation.state'):
            require(info['features'][key]['shape'] == [10] and info['features'][key]['names'] == JOINTS,
                    'Feature order mismatch')
        parts = self.manifests['splits']
        require(all(type(i) is int for name in ('train', 'validation') for i in parts[name]),
                'Episode IDs must be integers')
        require(all(parts[k] and len(parts[k]) == len(set(parts[k])) for k in ('train', 'validation')),
                'Empty/duplicate episode split')
        require(not set(parts['train']) & set(parts['validation']), 'Split leakage')
        files = sorted((self.root/'data').rglob('*.parquet'))
        table = pq.read_table(files).sort_by([('index', 'ascending')])
        self.numeric_hashes = {str(p.relative_to(self.root)): sha(p) for p in files}
        require(np.array_equal(table['index'].to_numpy(), np.arange(info['total_frames'])), 'Noncontiguous indices')
        self.states = np.asarray(table['observation.state'].to_pylist(), np.float32)
        self.actions = np.asarray(table['action'].to_pylist(), np.float32)
        require(all(v.shape == (len(table), 10) and np.isfinite(v).all() for v in (self.states, self.actions)),
                'Invalid numeric payload')
        require(np.isin(self.actions[:, 7], [0, 1]).all(), 'Nonbinary actions')
        flat = lambda k: np.asarray(table[k].to_pylist()).reshape(-1)
        episode, segment, valid, grid_valid = (flat(k) for k in ('episode_index', 'segment_id', 'valid_anchor', 'valid_grid_row'))
        self.episode_ids = episode
        self.episodes = {e['episode_index']: e for p in sorted((self.root/'meta/episodes').rglob('*.parquet'))
                         for e in pq.read_table(p).to_pylist()}
        require(set(parts['train']) | set(parts['validation']) == set(self.episodes), 'Incomplete episode partition')
        grid = flat('observation_grid_time')
        target = flat('target_grid_time')
        for name in ('train', 'validation'):
            require(all(type(i) is int for i in self.manifests['anchors'][name]), 'Anchors must be integers')
            a = np.asarray(self.manifests['anchors'][name], dtype=np.int64)
            require(a.size > 0 and len(a) == len(set(a.tolist())) and np.all((a >= 0) & (a + 6 < len(table))),
                    'Invalid anchors')
            expected = np.flatnonzero((valid == 1) & np.isin(episode, parts[name]))
            require(np.array_equal(np.sort(a), expected), 'Anchor coverage/split mismatch')
            windows = a[:, None] + np.arange(7)
            require(np.all(episode[windows] == episode[a, None])
                    and np.all(segment[windows] == segment[a, None]) and np.all(grid_valid[windows] == 1),
                    'Window crosses episode/gap')
            require(np.allclose(grid[windows] - grid[a, None], np.arange(7)/30, atol=1e-6, rtol=0)
                    and np.allclose(target[windows] - grid[windows], 1/30, atol=1e-6, rtol=0),
                    'Action grid timing mismatch')
        self.anchors = np.asarray(self.manifests['anchors'][split], dtype=np.int64)
        for sid, e in self.episodes.items():
            start, end = e['dataset_from_index'], e['dataset_to_index']
            require(end - start == e['length'] and np.all(episode[start:end] == sid), 'Episode metadata mismatch')
            require(len(e['tasks']) == 1 and e['tasks'][0] in (TASK, MIXED_TASK), 'Unsupported task')

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        import av
        anchor = int(self.anchors[index])
        e = self.episodes[int(self.episode_ids[anchor])]
        result = dict(state=self.states[anchor].copy(), actions=self.actions[anchor:anchor+7].copy(), prompt=e['tasks'][0])
        for name in ('head', 'right'):
            key = 'observation.images.' + name
            prefix = 'videos/' + key + '/'
            path = self.root / self.info['video_path'].format(video_key=key, chunk_index=e[prefix+'chunk_index'],
                                                            file_index=e[prefix+'file_index'])
            frame_id = round(e[prefix+'from_timestamp'] * 30) + anchor - e['dataset_from_index']
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                require(abs(float(stream.average_rate)-30) < 1e-6, 'Video fps mismatch')
                stream.codec_context.thread_count = 1
                container.seek(int(np.floor(frame_id/30/float(stream.time_base))), stream=stream, backward=True)
                for frame in container.decode(stream):
                    require(frame.pts is not None, 'Missing frame timestamp')
                    found = round(float(frame.pts * frame.time_base) * 30)
                    if found == frame_id:
                        result[name] = frame.to_ndarray(format='rgb24')
                        break
                    if found > frame_id:
                        break
            require(name in result, 'Video frame not found')
        return result

    def statistics(self):
        require(self.split == 'train', 'Statistics require train split')
        action_rows = np.unique((self.anchors[:, None] + np.arange(7)).reshape(-1))
        return fit_stats(self.states[self.anchors], self.actions[action_rows])

    def provenance(self):
        source_files = [self.root/'meta/info.json', *sorted((self.root/'meta/episodes').rglob('*.parquet')),
                        *sorted((self.root/'videos').rglob('*.mp4'))]
        hashes = {str(p.relative_to(self.root)): sha(p) for p in source_files}
        return dict(manifests=self.manifest_hashes, numeric_files=self.numeric_hashes,
                    source_files=hashes,
                    fingerprint=canonical(dict(manifests=self.manifest_hashes, numeric=self.numeric_hashes, sources=hashes)))
