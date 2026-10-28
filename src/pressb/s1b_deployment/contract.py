"""Immutable asset identity and explicit deployment configuration."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import yaml

TASK = 'Press the down button while maintaining the demonstrated head posture.'
MIXED_TASK = 'Use the right gripper to press the circular button on the elevator panel.'
VERSION = 's1b-joint10-v1'
JOINTS = [f'astribot_arm_right_joint_{i}' for i in range(1, 8)] + ['astribot_gripper_right/closure', 'astribot_head_joint_1', 'astribot_head_joint_2']
OFFSETS = np.rint(np.arange(1, 8) * 1e9 / 30).astype(np.int64)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()

def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read_config(path, mode='shadow', execution_config=None):
    c = yaml.safe_load(Path(path).read_text())
    if c['protocol'] != VERSION or c['model']['state_dim'] != 10 or c['model']['action_dim'] != 10 or c['model']['horizon'] != 7 or c['model']['task'] not in (TASK, MIXED_TASK):
        raise ValueError('Wrong joint10 deployment contract')
    if mode not in ('shadow', 'execute'): raise ValueError('Unknown mode')
    if mode == 'execute' and execution_config is not None:
        from .live_execution import validate_execution_config
        validate_execution_config(execution_config)
        c = dict(c, mode='execute', execution=execution_config)
        return c
    if mode == 'execute':
        missing = []
        def visit(v, prefix):
            if isinstance(v, dict):
                for k, x in v.items(): visit(x, prefix + '.' + k)
            elif v is None: missing.append(prefix)
        for k in ('sdk', 'constraints', 'observation'): visit(c[k], k)
        if missing: raise ValueError('Unverified execute fields: ' + ', '.join(missing))
        raise ValueError('P4 execution is not implemented or validated')
    return c

def verify(run, step, lock, *, hash_checkpoint=True):
    run = Path(run)
    if step not in (2000, 4500, 8000): raise ValueError('Explicit checkpoint 2000, 4500 or 8000 required')
    identity = json.loads(Path(lock).read_text())
    actual = {k: sha(run / k) for k in identity['files']}
    if actual != identity['files'] or tree(run / 'processor') != identity['processor']:
        raise ValueError('Processor/data/config/runtime identity mismatch')
    m = json.loads((run / f'checkpoints/step_{step:06d}/manifest.json').read_text())
    checkpoint = run / f'checkpoints/step_{step:06d}/model.pt'
    if (not m['complete'] or m['step'] != step or not checkpoint.is_file()
            or (hash_checkpoint and sha(checkpoint) != m['model_sha256'])
            or m['model_sha256'] != identity['models'][str(step)]):
        raise ValueError('Checkpoint hash/manifest mismatch')
    contract = json.loads((run / 'data_contract.json').read_text())
    if m['data_contract_sha256'] != sha(run / 'data_contract.json'): raise ValueError('Contract mismatch')
    s = contract['schema']
    if (s['state_dim'], s['action_dim'], s['action_horizon']) != (10, 10, 7): raise ValueError('Wrong dimensions')
    for key in ('state', 'action'):
        for field in ('mean', 'scale'):
            x = np.asarray(contract['normalization'][key][field])
            if x.shape != (10,) or not np.isfinite(x).all() or (field == 'scale' and (x <= 0).any()): raise ValueError('Bad normalization')
    if contract['normalization']['action']['mean'][7] != 0 or contract['normalization']['action']['scale'][7] != 1: raise ValueError('Gripper action must use identity normalization')
    schema = json.loads((run / 'action_schema.json').read_text())
    training = json.loads((run / 'training_contract.json').read_text())
    if canonical(training['contract']) != training['fingerprint'] or training['fingerprint'] != m['training_fingerprint']:
        raise ValueError('Training fingerprint mismatch')
    if canonical(json.loads((run / 'split.json').read_text())) != m['split_sha256']:
        raise ValueError('Split identity mismatch')
    for key in ('dataset_sha256', 'data_contract_sha256', 'split_sha256', 'training_fingerprint'):
        if schema[key] != m[key]: raise ValueError('Schema provenance mismatch: ' + key)
    task = identity.get('task', TASK)
    if task not in (TASK, MIXED_TASK): raise ValueError('Unsupported task')
    return dict(task=task, step=step, model_sha256=m['model_sha256'], data_contract_sha256=m['data_contract_sha256'], processor=identity['processor'], normalization=contract['normalization'])
