"""Versioned episode-balanced sampling and transition diagnostics; no video decoding."""
import hashlib
import math
import numpy as np
from torch.utils.data import Sampler

SAMPLING = {'version': 'gripper_episode_v1', 'probabilities': [0.25, 0.25, 0.5]}
DIAGNOSTICS = {'version': 'gripper_transition_v1', 'radius': 7,
               'false_close_tolerance': 0.01, 'motion_error_ratio': 1.1}
GROUPS = ('initiate_close', 'hold_open', 'other')


def validate_options(sampling=None, diagnostics=None):
    if sampling is not None and dict(sampling) != SAMPLING:
        raise ValueError('Unsupported gripper sampling configuration')
    if diagnostics is not None and dict(diagnostics) != DIAGNOSTICS:
        raise ValueError('Unsupported gripper diagnostics configuration')


def episode_arrays(dataset):
    for sid, frames in dataset._anchors:
        e = dataset._episodes[sid]
        start, length = e['dataset_from_index'], e['length']
        state = dataset.state_raw(dataset._states[start:start + length])[:, 7]
        action = dataset.action_raw(dataset._actions[start:start + length])[:, 7]
        frames = np.asarray(frames, dtype=np.int64)
        if np.any(frames < 0) or np.any(frames + 6 >= length):
            raise ValueError('Incomplete or cross-episode action window')
        if not np.isin(action, [0, 1]).all():
            raise ValueError('Expected binary gripper labels')
        yield int(sid), frames, state, action


def sampling_groups(dataset):
    groups = [{} for _ in GROUPS]
    offset = 0
    for sid, frames, state, action in episode_arrays(dataset):
        future_close = np.stack([action[frames + j] for j in range(7)]).max(0) >= .5
        open_state = state[frames] < 50
        labels = np.where(open_state, np.where(future_close, 0, 1), 2)
        for group in range(3):
            indices = offset + np.flatnonzero(labels == group)
            if len(indices):
                groups[group][sid] = indices
        offset += len(frames)
    if offset != len(dataset) or any(not group for group in groups):
        raise ValueError('Missing sampling group or inconsistent dataset anchors')
    return groups


class GripperDistributedSampler(Sampler):
    """Same global replacement stream on all ranks; cursor indexes the rank slice."""
    def __init__(self, dataset, num_replicas, rank, seed=42):
        if not 0 <= rank < num_replicas:
            raise ValueError('Invalid rank')
        self.groups = sampling_groups(dataset)
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.epoch, self.start_index = 0, 0
        self._cache = None

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._cache = None

    def set_start_index(self, start_index):
        if type(start_index) is not int or not 0 <= start_index <= self.num_samples:
            raise ValueError('Invalid sampler cursor')
        self.start_index = start_index

    def global_sequence(self):
        if self._cache is None:
            rng = np.random.default_rng(self.seed + self.epoch)
            labels = rng.choice(3, size=self.total_size, p=SAMPLING['probabilities'])
            sequence = np.empty(self.total_size, dtype=np.int64)
            for group, episodes in enumerate(self.groups):
                positions = np.flatnonzero(labels == group)
                ids = sorted(episodes)
                choices = rng.integers(len(ids), size=len(positions))
                for k, sid in enumerate(ids):
                    at = positions[choices == k]
                    sequence[at] = rng.choice(episodes[sid], size=len(at), replace=True)
            self._cache = sequence, labels
        return self._cache

    def __iter__(self):
        sequence, _ = self.global_sequence()
        return iter(sequence[self.rank::self.num_replicas][self.start_index:].tolist())

    def __len__(self):
        return self.num_samples - self.start_index

    def report(self):
        sequence, labels = self.global_sequence()
        remaining = np.concatenate([labels[r::self.num_replicas][self.start_index:]
                                    for r in range(self.num_replicas)])
        return dict(version=SAMPLING['version'], epoch=self.epoch, rank_start_index=self.start_index,
                    sequence_sha256=hashlib.sha256(sequence.astype('<i8').tobytes()).hexdigest(),
                    groups={name: dict(episodes=len(self.groups[g]),
                        anchors=sum(len(v) for v in self.groups[g].values()),
                        drawn=int((labels == g).sum()), remaining_drawn=int((remaining == g).sum()),
                        fraction=float((labels == g).mean())) for g, name in enumerate(GROUPS)})


def transition_anchors(dataset, radius=7):
    anchors, onsets = set(), set()
    for sid, frames, state, action in episode_arrays(dataset):
        transitions = np.flatnonzero((action[:-1] < .5) & (action[1:] >= .5)) + 1
        valid = set(frames.tolist())
        for t in transitions:
            anchors.update((sid, f) for f in range(max(0, t-radius), t+radius+1) if f in valid)
            if t in valid and state[t] < 50:
                onsets.add((sid, int(t)))
    return sorted(anchors), onsets


def gripper_record(predicted, target, open_state, onset=False):
    p, t = np.asarray(predicted) >= .5, np.asarray(target) >= .5
    actual = np.flatnonzero(t)
    guess = np.flatnonzero(p)
    # Delay is identifiable inside this window only when it contains a 0 -> 1.
    transition = bool(np.any((~t[:-1]) & t[1:])) and not t[0]
    delay = int(guess[0] - actual[0]) if transition and len(guess) else None
    return dict(predicted=p.tolist(), target=t.tolist(), open_state=bool(open_state),
                onset=bool(onset), transition=transition, delay_frames=delay)


def gripper_metrics(rows):
    if not rows:
        return {'anchors': 0}
    records = [r['gripper'] for r in rows]
    p = np.asarray([r['predicted'] for r in records]); t = np.asarray([r['target'] for r in records])
    opened = np.asarray([r['open_state'] for r in records]); onset = np.asarray([r['onset'] for r in records])
    first = opened & t[:, 0]; future = opened & t.any(1)
    delays = [r['delay_frames'] for r in records if r['delay_frames'] is not None]
    def ratio(n, d):
        return float(n / d) if d else None
    errors = np.mean([r['error'] for r in rows], axis=0)
    return dict(anchors=len(rows), first_close_count=int(first.sum()),
                first_close_recall=ratio(p[first, 0].sum(), first.sum()),
                future_close_count=int(future.sum()),
                future_close_recall=ratio((p[future] & t[future]).any(1).sum(), future.sum()),
                onset_count=int(onset.sum()), onset_first_close_recall=ratio(p[onset, 0].sum(), onset.sum()),
                open_targets=int((~t).sum()), false_close_rate=ratio((p & ~t).sum(), (~t).sum()),
                onset_predictions=[dict(episode=r['episode'], frame=r['frame'], **r['gripper'])
                                   for r in rows if r['gripper']['onset']],
                delay_frames=delays, mean_delay_frames=float(np.mean(delays)) if delays else None,
                missed_transitions=sum(r['transition'] and r['delay_frames'] is None for r in records),
                arm_mae_rad=float(errors[:, :7].mean()), head_mae_rad=errors[:, 8:].mean(0).tolist())


def diagnostics_report(rows):
    def summarize(selected):
        return dict(overall=gripper_metrics(selected), by_source_batch={
            source: gripper_metrics([r for r in selected if r['source_batch'] == source])
            for source in sorted({r['source_batch'] for r in selected})})
    return dict(base=summarize([r for r in rows if r['is_base']]),
                transitions=summarize([r for r in rows if r['is_transition']]))


def gripper_candidate(metrics, baseline):
    current = metrics['gripper']['base']['overall']; base = baseline['gripper']['base']['overall']
    recall = metrics['gripper']['transitions']['overall'].get('onset_first_close_recall')
    if recall is None or current['false_close_rate'] is None or base['false_close_rate'] is None:
        return None
    if current['false_close_rate'] > base['false_close_rate'] + DIAGNOSTICS['false_close_tolerance']:
        return None
    for c, b in zip([current['arm_mae_rad'], *current['head_mae_rad']],
                    [base['arm_mae_rad'], *base['head_mae_rad']]):
        if c > b * DIAGNOSTICS['motion_error_ratio']:
            return None
    return [recall, -metrics['overall']['selection_score_normalized']]
