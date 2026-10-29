"""Explicit preparation/training/deployment manifests, independent of Qwen assets."""
import json
from pathlib import Path
import shutil
from pressb.s1b_deployment.contract import sha, canonical, JOINTS, MIXED_TASK, TASK
from .dataset import read_json
from .transforms import require, validate_stats

ROOT = Path(__file__).resolve().parents[3]
BUNDLE_FILES = ('config.json', 'normalization.json', 'split.json', 'anchors.json',
                'source_contract.json', 'provenance.json', 'upstream.json', 'adapter.json')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def adapter_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def prepare(dataset, output):
    output = Path(output)
    require(not output.exists(), 'Preparation output already exists')
    stats = dataset.statistics()
    output.mkdir(parents=True)
    write_json(output/'config.json', dict(name='pi05_s1b_joint10', pi05=True, action_dim=32,
        action_horizon=7, discrete_state_input=True, fps=30,
        cameras={'head': 'base_0_rgb', 'right': 'right_wrist_0_rgb'}))
    write_json(output/'normalization.json', stats)
    for key, file in [('splits', 'split.json'), ('anchors', 'anchors.json'), ('contract', 'source_contract.json')]:
        write_json(output/file, dataset.manifests[key])
    write_json(output/'provenance.json', dataset.provenance())
    write_json(output/'upstream.json', read_json(ROOT/'repos.lock.json')['repositories']['openpi'])
    write_json(output/'adapter.json', adapter_hashes())
    write_json(output/'bundle.json', {name: sha(output/name) for name in BUNDLE_FILES})


def validate_bundle(bundle):
    bundle = Path(bundle)
    expected = read_json(bundle/'bundle.json')
    require(set(expected) == set(BUNDLE_FILES), 'Incomplete preparation manifest')
    require({name: sha(bundle/name) for name in BUNDLE_FILES} == expected, 'Preparation hash mismatch')
    require(read_json(bundle/'adapter.json') == adapter_hashes(), 'Adapter code differs: prepare/train again')
    require(read_json(bundle/'upstream.json') == read_json(ROOT/'repos.lock.json')['repositories']['openpi'],
            'Upstream lock mismatch')
    validate_stats(read_json(bundle/'normalization.json'))
    return expected


def seal_training(checkpoint, bundle, step, training):
    """Called only after the official trainer has atomically completed a save."""
    checkpoint, bundle = Path(checkpoint), Path(bundle)
    validate_bundle(bundle)
    target = checkpoint/'s1b'
    target.mkdir(exist_ok=False)
    for name in (*BUNDLE_FILES, 'bundle.json'):
        shutil.copyfile(bundle/name, target/name)
    write_json(target/'training.json', dict(training, step=int(step),
        model_sha256=sha(checkpoint/'model.safetensors'), bundle_sha256=sha(bundle/'bundle.json')))


def export(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    bundle = checkpoint/'s1b'
    validate_bundle(bundle)
    training = read_json(bundle/'training.json')
    require(training['model_sha256'] == sha(checkpoint/'model.safetensors')
            and training['bundle_sha256'] == sha(bundle/'bundle.json'), 'Unsealed/changed training checkpoint')
    require(not output.exists(), 'Export output already exists')
    task = read_json(bundle/'source_contract.json').get('task', TASK)
    require(task in (TASK, MIXED_TASK), 'Unsupported deployment task')
    output.mkdir(parents=True)
    for name in (*BUNDLE_FILES, 'bundle.json', 'training.json'):
        shutil.copyfile(bundle/name, output/name)
    shutil.copyfile(checkpoint/'model.safetensors', output/'model.safetensors')
    files = {name: sha(output/name) for name in (*BUNDLE_FILES, 'bundle.json', 'training.json', 'model.safetensors')}
    # Contract hash includes all runtime assets and weights, so the existing wire
    # identity binds normalization/preprocessing as well as model parameters.
    write_json(output/'data_contract.json', dict(schema='s1b-openpi-pi05-v1', joint_names=JOINTS,
        state_dim=10, action_dim=10, action_horizon=7, fps=30, task=task, assets=files))
    files['data_contract.json'] = sha(output/'data_contract.json')
    identity = dict(backend='openpi_pi05', schema='s1b-openpi-artifacts-v1', task=task, files=files,
                    models={str(training['step']): files['model.safetensors']})
    write_json(output/'identity.json', identity)
    return identity


def verify(run, step, lock):
    from pressb.s1b_deployment.contract import validate_checkpoint
    run = Path(run).resolve()
    identity = read_json(lock)
    validate_checkpoint(identity, step, backend='openpi_pi05')
    required = set(BUNDLE_FILES) | {'bundle.json', 'training.json', 'model.safetensors', 'data_contract.json'}
    require(set(identity['files']) == required, 'Incomplete OpenPI artifact manifest')
    require({name: sha(run/name) for name in required} == identity['files'], 'OpenPI artifact hash mismatch')
    validate_bundle(run)
    contract = read_json(run/'data_contract.json')
    require(contract['schema'] == 's1b-openpi-pi05-v1' and contract['joint_names'] == JOINTS
            and (contract['state_dim'], contract['action_dim'], contract['action_horizon'], contract['fps']) == (10,10,7,30),
            'Wrong joint10 contract')
    require(contract['assets'] == {k:v for k,v in identity['files'].items() if k != 'data_contract.json'},
            'Unbound runtime assets')
    training = read_json(run/'training.json')
    require(training['step'] == step and training['model_sha256'] == identity['models'][str(step)]
            == identity['files']['model.safetensors']
            and training['bundle_sha256'] == sha(run/'bundle.json'), 'Training identity mismatch')
    require(identity['task'] == contract['task'] and identity['task'] in (TASK, MIXED_TASK), 'Task mismatch')
    return dict(backend='openpi_pi05', task=identity['task'], step=step,
        model_sha256=identity['models'][str(step)], data_contract_sha256=identity['files']['data_contract.json'])
