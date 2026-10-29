"""Shared, dependency-light S1B input and reversible normalization transforms."""
from dataclasses import dataclass
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, shape):
    value = np.asarray(value, dtype=np.float32)
    require(value.shape == shape and np.isfinite(value).all(), f'Expected finite {shape}')
    return value


def validate_stats(stats):
    require(stats.get('schema') == 's1b-pi05-quantile-v1', 'Wrong normalization schema')
    for key in ('state', 'actions'):
        numeric(stats[key]['center'], (10,))
        require((numeric(stats[key]['scale'], (10,)) > 0).all(), 'Nonpositive normalization scale')
    require(stats['actions']['center'][7] == 0 and stats['actions']['scale'][7] == 1,
            'Gripper actions require identity normalization')


def fit_stats(states, actions):
    result = {'schema': 's1b-pi05-quantile-v1'}
    for key, values in [('state', states), ('actions', actions)]:
        values = np.asarray(values, dtype=np.float32)
        require(values.ndim == 2 and values.shape[1] == 10 and len(values) > 0
                and np.isfinite(values).all(), 'Invalid normalization samples')
        low, high = np.quantile(values, [.01, .99], axis=0)
        center, scale = (low + high) / 2, np.maximum((high - low) / 2, 1e-6)
        if key == 'actions':
            require(np.isin(values[:, 7], [0, 1]).all(), 'Nonbinary gripper action')
            center[7], scale[7] = 0, 1
        result[key] = dict(center=center.tolist(), scale=scale.tolist())
    validate_stats(result)
    return result


@dataclass(frozen=True)
class Inputs:
    stats: dict

    def __post_init__(self):
        validate_stats(self.stats)

    def __call__(self, data):
        head, right = (np.asarray(data[k]) for k in ('head', 'right'))
        for value, shape in [(head, (720, 1280, 3)), (right, (360, 640, 3))]:
            require(value.shape == shape and value.dtype == np.uint8, 'Expected native RGB uint8')
        state = numeric(data['state'], (10,))
        st = self.stats['state']
        result = dict(
            state=((state - np.asarray(st['center'])) / np.asarray(st['scale'])).astype(np.float32),
            image={'base_0_rgb': head, 'left_wrist_0_rgb': np.zeros_like(head), 'right_wrist_0_rgb': right},
            image_mask={'base_0_rgb': np.True_, 'left_wrist_0_rgb': np.False_, 'right_wrist_0_rgb': np.True_},
            prompt=data['prompt'])
        if 'actions' in data:
            actions = numeric(data['actions'], (7, 10))
            require(np.isin(actions[:, 7], [0, 1]).all(), 'Nonbinary gripper action')
            st = self.stats['actions']
            result['actions'] = ((actions - st['center']) / st['scale']).astype(np.float32)
        return result


@dataclass(frozen=True)
class Outputs:
    stats: dict

    def __post_init__(self):
        validate_stats(self.stats)

    def __call__(self, data):
        actions = numeric(data['actions'], (7, 32))[:, :10]
        st = self.stats['actions']
        raw = (actions * st['scale'] + st['center']).astype(np.float32)
        return {'actions': numeric(raw, (7, 10))}
