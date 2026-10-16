"""Frozen sampling, full-joint schema, durable commits and training alignment."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import numpy as np

CAMERAS = {'head': (1280, 720), 'overview': (960, 600),
           **{n: (640, 480) for n in ('wrist_left','wrist_right','panel','front','lobby')}}
JOINT_NAMES = tuple('astribot_' + n for n in (
    'torso_joint_1','torso_joint_2','torso_joint_3','torso_joint_4',
    'arm_left_joint_1','arm_right_joint_1','head_joint_1','arm_left_joint_2',
    'arm_right_joint_2','head_joint_2','arm_left_joint_3','arm_right_joint_3',
    'arm_left_joint_4','arm_right_joint_4','arm_left_joint_5','arm_right_joint_5',
    'arm_left_joint_6','arm_right_joint_6','arm_left_joint_7','arm_right_joint_7',
    'gripper_left_joint_L1','gripper_left_joint_L2','gripper_left_joint_R1','gripper_left_joint_R2',
    'gripper_right_joint_L1','gripper_right_joint_L2','gripper_right_joint_R1','gripper_right_joint_R2',
    'gripper_left_joint_L11','gripper_left_joint_R11','gripper_right_joint_L11','gripper_right_joint_R11'))
TASK = 'Press the down hall-call button.'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

@contextmanager
def exclusive_lock(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)

def candidate_block(rng, start=1, count=512):
    # Retain historical offset arithmetic and all-XY-then-all-yaw ordering.
    xy = rng.uniform([-.18,-.28],[-.02,.12],(count,2)) + [-.12,.28]
    yaw = rng.uniform(0.,10.,count)
    return [dict(candidate_id=start+i, x_m=float(x), y_m=float(y),
                 yaw_deg=float(yaw[i]), yaw_rad=float(np.deg2rad(yaw[i])))
            for i,(x,y) in enumerate(xy)]

def create_manifest(root, config, seed=20261014, target=400, maximum=1024):
    if not 1 <= target <= maximum <= 1024 or maximum not in (512,1024):
        raise ValueError('Require 1 <= target <= maximum; maximum is 512 or 1024')
    path = Path(root)/'manifest.json'
    if path.exists(): raise FileExistsError(path)
    rng = np.random.Generator(np.random.PCG64(seed))
    data = dict(schema='astribot_vib820_raw_v1', seed=seed, rng='PCG64', target_successes=target,
                maximum_candidates=maximum, config=config, joint_names=list(JOINT_NAMES),
                cameras=CAMERAS, task=TASK, candidates=candidate_block(rng),
                sampling='512 XY offsets then 512 positive yaw; successful candidates selected in ID order')
    data['rng_state'] = rng.bit_generator.state
    atomic_json(path, data); return data

def extend_manifest(root):
    path = Path(root)/'manifest.json'; data = json.loads(path.read_text())
    if len(data['candidates']) + 512 > data['maximum_candidates']: raise ValueError('Candidate limit reached')
    rng = np.random.Generator(np.random.PCG64()); rng.bit_generator.state = data['rng_state']
    data['candidates'] += candidate_block(rng,len(data['candidates'])+1)
    data['rng_state'] = rng.bit_generator.state; atomic_json(path,data); return data

def ordered_successes(results, target):
    """Only terminal contiguous IDs may contribute: never skip pending early IDs."""
    selected=[]
    for cid in range(1, max(results,default=0)+1):
        row=results.get(cid)
        if row is None or row.get('outcome') not in ('success','physical_failure','infrastructure_exhausted'):
            break
        if row['outcome']=='success': selected.append(row)
        if len(selected)==target: break
    return selected

def training_arrays(trace, timing, joint_names):
    if len(joint_names)!=32 or set(joint_names)!=set(JOINT_NAMES): raise ValueError('Incorrect full-joint schema')
    order=[joint_names.index(n) for n in JOINT_NAMES]
    rows=np.asarray([t['step'] for t in timing],dtype=np.int64)
    if len(rows)<2 or np.any(np.diff(rows)!=4): raise ValueError('Missing/duplicate camera sampling steps')
    if rows.min()<0 or rows.max()>=len(trace['q']): raise ValueError('Camera step out of trace')
    if not all(t['epoch']==int(trace['epoch'][r]) and abs(t['simulation_time']-trace['simulation_time'][r])<1e-6
               and set(t['cameras'])==set(CAMERAS)
               and all(abs(s-t['simulation_time'])<1e-6 for s in t['cameras'].values()) for r,t in zip(rows,timing)):
        raise ValueError('Camera/physics timestamp mismatch')
    hits=np.flatnonzero((trace['phase']=='press') & (trace['travel'][:,0]>=.001) & (trace['force'][:,0]>.02))
    if not len(hits): raise ValueError('No physical press')
    ends=np.flatnonzero((rows>=hits[0]) & (trace['epoch'][rows]==0))
    if not len(ends): raise ValueError('No sampled physical success')
    end=int(ends[0]); selected=rows[:end+1]
    if len(selected)<2 or not trace['light'][selected[-1],0]: raise ValueError('Invalid training terminal light')
    if trace['phase'][selected[-1]]!='press': raise ValueError('Retract phase reached before sampled success')
    if not np.array_equal(trace['target'][selected[-1]],trace['target'][hits[0]]):
        raise ValueError('Contact command changed before terminal success sample')
    if np.any(trace['epoch'][selected]!=0): raise ValueError('Reset frame in training')
    states=trace['q'][selected][:,order].astype(np.float32)
    # Next sampling instant's actual command, terminal repeats the previous label.
    actions=np.empty_like(states)
    actions[:-1]=trace['target'][selected[1:]][:,order]
    actions[-1]=actions[-2]
    return dict(state=states,action=actions,velocity=trace['qd'][selected][:,order].astype(np.float32),
                physical_steps=selected, terminal_video_frame=end, trigger_step=int(hits[0]))

def commit_episode(staging, destination, audit):
    staging,destination=Path(staging),Path(destination)
    if audit.get('status')!='passed': raise ValueError('Cannot commit unaudited data')
    if destination.exists(): raise FileExistsError(destination)
    files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file() and p.name!='commit.json'}
    for relative in files:
        with (staging/relative).open('rb') as file: os.fsync(file.fileno())
    atomic_json(staging/'commit.json',dict(schema=1,audit=audit,sha256=files))
    destination.parent.mkdir(parents=True,exist_ok=True)
    os.rename(staging,destination)
    fd=os.open(destination.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def validate_commit(directory):
    directory=Path(directory); c=json.loads((directory/'commit.json').read_text())
    if c['audit']['status']!='passed' or not c['sha256']: raise ValueError('Invalid commit')
    for relative,digest in c['sha256'].items():
        p=(directory/relative).resolve()
        if not p.is_relative_to(directory.resolve()) or sha256(p)!=digest: raise ValueError(f'Corrupt committed file: {relative}')
    if 'report.json' in c['sha256']:
        from .assets import validate_assets
        validate_assets(json.loads((directory/'report.json').read_text()).get('shared_assets',{}))
    return c
