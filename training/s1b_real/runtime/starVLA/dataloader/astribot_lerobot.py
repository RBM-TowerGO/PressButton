"""Read-only Astribot v3 data: head/right wrist, normalized joint7+gripper1.

The prepared contract is written outside the published dataset. It binds source
physics checks, dataset hashes, held-out episodes, and train-only statistics.
"""
from __future__ import annotations
from collections import OrderedDict
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .piper_lerobot import PiperLeRobotDataset, _require, _sha256, _canonical_sha, collate_fn

CAMERA_KEYS = ('observation.images.head', 'observation.images.wrist_right')
ARM = np.array([5, 8, 11, 13, 15, 17, 19])
GRIP = np.array([24, 25, 26, 27, 30, 31])
SIGNS = np.array([-1, -1, 1, 1, 1, -1], dtype=np.float32)
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
SEMANTICS = dict(kind='astribot_joint8_v1', state_dim=8, action_dim=8,
    action_horizon=7, video_horizon=8, fps=30, physics_hz=120, units='radians',
    camera_keys=list(CAMERA_KEYS), arm_indices=ARM.tolist(), gripper_indices=GRIP.tolist(),
    gripper_signs=SIGNS.tolist(), joint_names=list(JOINT_NAMES),
    normalization='train_only_mean_std', state_gripper='signed_six_joint_mean',
    action_timing='next_sample_absolute_target', terminal_action='repeat_previous',
    controller_interpolation='four_physics_ticks_linear', task=TASK)

def compress_joints(value):
    x = np.asarray(value, dtype=np.float32)
    _require(x.shape[-1:] == (32,) and np.isfinite(x).all(), 'Expected finite (...,32) joints')
    grip = (x[..., GRIP].astype(np.float64) * SIGNS).mean(-1, keepdims=True).astype(np.float32)
    return np.concatenate((x[..., ARM], grip), axis=-1)

def expand_actions(value, fixed):
    x = np.asarray(value, dtype=np.float32)
    _require(x.shape[-1:] == (8,) and np.isfinite(x).all(), 'Expected finite (...,8) actions')
    f = np.asarray(fixed, dtype=np.float32)
    _require(f.shape == (32,) and np.isfinite(f).all(), 'Expected finite 32D fixed target template')
    result = np.broadcast_to(f, x.shape[:-1] + (32,)).copy()
    result[..., ARM] = x[..., :7]
    result[..., GRIP] = x[..., 7:8] * SIGNS
    return result

def normalize(values, stats):
    return ((np.asarray(values,dtype=np.float32) - np.asarray(stats['mean'],dtype=np.float32)) /
            np.asarray(stats['scale'],dtype=np.float32))

def denormalize(values, stats):
    return (np.asarray(values,dtype=np.float32) * np.asarray(stats['scale'],dtype=np.float32) +
            np.asarray(stats['mean'],dtype=np.float32))

def _atomic(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        _require(json.loads(path.read_text())==data, f'Existing contract differs: {path}')
        return
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    try:
        try: os.link(temporary,path)
        except FileExistsError: _require(json.loads(path.read_text())==data,'Concurrent contract differs')
    finally: temporary.unlink()

def metadata(root):
    root=Path(root); info=json.loads((root/'meta/info.json').read_text())
    _require(info['codebase_version']=='v3.0' and info['fps']==30 and info['robot_type']=='astribot','Wrong dataset type/rate')
    for key in ('action','observation.state'):
        spec=info['features'][key]
        _require(spec['shape']==[32] and tuple(spec['names'])==JOINT_NAMES and spec['dtype']=='float32','Joint schema mismatch')
    for key,shape in zip(CAMERA_KEYS,([720,1280,3],[480,640,3])):
        _require(info['features'][key]['shape']==shape,'Camera schema mismatch')
    columns=['episode_index','length','tasks','dataset_from_index','dataset_to_index','data/chunk_index','data/file_index']
    for k in CAMERA_KEYS: columns += [f'videos/{k}/{x}' for x in ('chunk_index','file_index','from_timestamp','to_timestamp')]
    episodes=[r for p in sorted((root/'meta/episodes').rglob('*.parquet')) for r in pq.read_table(p,columns=columns).to_pylist()]
    episodes.sort(key=lambda e:e['episode_index'])
    sources=json.loads((root/'meta/astribot_sources.json').read_text())
    _require(len(episodes)==len(sources)==info['total_episodes']==400,'Expected verified 400 episode release')
    end=0
    for i,(e,s) in enumerate(zip(episodes,sources)):
        _require(e['episode_index']==i and e['dataset_from_index']==end and e['dataset_to_index']-end==e['length'] and e['tasks']==[TASK],'Episode order/task mismatch')
        e.update(source_episode_id=int(s['candidate_id']),task=TASK,floor=0,videos={})
        for key in CAMERA_KEYS:
            prefix=f'videos/{key}/'; first=e[prefix+'from_timestamp']*30; last=e[prefix+'to_timestamp']*30
            _require(abs(first-round(first))<.002 and abs(last-first-e['length'])<.002,'Video interval mismatch')
            p=root/info['video_path'].format(video_key=key,chunk_index=e[prefix+'chunk_index'],file_index=e[prefix+'file_index'])
            _require(p.is_file(),f'Missing video {p}')
            e['videos'][key]=(str(p),int(round(first)))
        end=e['dataset_to_index']
    _require(end==info['total_frames'],'Frame count mismatch')
    return info,episodes,sources

def numeric(root,info,episodes):
    total=info['total_frames']; seen=np.zeros(total,dtype=bool)
    arrays={k:np.empty((total,32),dtype=np.float32) for k in ('state','action','velocity')}
    expected=np.empty((total,3),dtype=np.int64)
    steps=np.empty(total,dtype=np.int64)
    for e in episodes:
        a,b=e['dataset_from_index'],e['dataset_to_index']
        expected[a:b,0]=e['episode_index'];expected[a:b,1]=e['source_episode_id'];expected[a:b,2]=np.arange(b-a)
    for path in sorted((Path(root)/'data').rglob('*.parquet')):
        t=pq.read_table(path); idx=t['index'].to_numpy().reshape(-1)
        _require(len(idx)>0 and idx.min()>=0 and idx.max()<total and len(set(idx.tolist()))==len(idx) and not seen[idx].any(),'Duplicate/missing indices')
        for col,j in [('episode_index',0),('source_candidate_id',1),('frame_index',2)]:
            _require(np.array_equal(t[col].to_numpy().reshape(-1),expected[idx,j]),f'{col} mismatch')
        _require(np.allclose(t['timestamp'].to_numpy(),expected[idx,2]/30,atol=2e-6,rtol=0),'Timestamp mismatch')
        for column,key in [('observation.state','state'),('action','action'),('observation.velocity','velocity')]:
            a=t[column].combine_chunks().values.to_numpy().reshape(len(idx),-1)
            _require(a.shape==(len(idx),32) and np.isfinite(a).all(),f'Invalid {key}')
            arrays[key][idx]=a
        steps[idx]=t['source_physics_step'].to_numpy().reshape(-1);seen[idx]=True
    _require(seen.all(),'Missing numeric data')
    for e in episodes:
        a,b=e['dataset_from_index'],e['dataset_to_index']
        _require(np.all(np.diff(steps[a:b])==4),'Physics sampling mismatch')
        _require(np.array_equal(arrays['action'][b-1],arrays['action'][b-2]),'Terminal action mismatch')
    return arrays,steps

def prepare_contract(root,source_audit,output,seed=42):
    root=Path(root).resolve(); output=Path(output).resolve()
    _require(root not in output.parents,'Contract must live outside immutable dataset')
    info,episodes,sources=metadata(root); arrays,steps=numeric(root,info,episodes)
    evidence=json.loads(Path(source_audit).read_text())
    _require(evidence['status']=='passed' and evidence['action_semantics']=='next_sample_absolute_joint_targets_radians_terminal_repeat_previous','Unverified source action semantics')
    by_id={e['candidate_id']:e for e in evidence['episodes']}
    _require(len(by_id)==400,'Source audit count mismatch')
    for e,s in zip(episodes,sources):
        src=by_id[e['source_episode_id']]; a,b=e['dataset_from_index'],e['dataset_to_index']
        _require(src['commit_sha256']==s['commit_sha256'] and src['frames']==b-a,'Source commit mismatch')
        for k,x in {**arrays,'physical_steps':steps}.items():
            digest=hashlib.sha256(np.ascontiguousarray(x[a:b]).tobytes()).hexdigest()
            _require(digest==src['array_sha256'][k],f'Original source differs: candidate {src["candidate_id"]}, {k}')
    # Group identical initial conditions so repeated captures cannot leak.
    groups={}
    for e in episodes:
        j=by_id[e['source_episode_id']]['job']; key=tuple(j[k] for k in ('x_m','y_m','yaw_deg'))
        groups.setdefault(key,[]).append(e['source_episode_id'])
    _require(len(groups)==400,'Repeated scene conditions require a grouped split')
    ids=sorted(by_id); shuffled=np.random.default_rng(seed).permutation(ids).tolist()
    split={name:sorted(values) for name,values in [('test',shuffled[:40]),('val',shuffled[40:80]),('train',shuffled[80:])]}
    train=np.zeros(info['total_frames'],dtype=bool)
    for e in episodes:
        if e['source_episode_id'] in split['train']:train[e['dataset_from_index']:e['dataset_to_index']]=True
    fixed=arrays['action'][0].copy(); act8=compress_joints(arrays['action'])
    _require(np.array_equal(expand_actions(act8,fixed),arrays['action']),'Action compression must be lossless')
    stats={}
    for key in ('state','action'):
        a=compress_joints(arrays[key])[train].astype(np.float64); std=a.std(0)
        stats[key]=dict(mean=a.mean(0).tolist(),std=std.tolist(),scale=np.maximum(std,1e-6).tolist(),count=len(a))
    # Hash all payload files once during preparation, not repeatedly on GPU ranks.
    paths=sorted([*root.glob('meta/*.json'),*root.glob('meta/*.parquet'),*(root/'meta/episodes').rglob('*.parquet'),*(root/'data').rglob('*.parquet'),*(root/'videos').rglob('*.mp4')])
    hashes={str(p.relative_to(root)):_sha256(p) for p in paths}
    identity=_canonical_sha(hashes)
    partition=dict(kind='astribot_episode_split_v1',seed=seed,source_export_manifest_sha256=identity,camera_keys=list(CAMERA_KEYS),
        **{f'{k}_source_episode_ids':v for k,v in split.items()})
    data=dict(schema=SEMANTICS,dataset_sha256=identity,file_sha256=hashes,file_sizes={str(p.relative_to(root)):p.stat().st_size for p in paths},
        split=partition,normalization=stats,fixed_action32=fixed.tolist(),source_audit_sha256=_sha256(source_audit),
        source_contract_sha256=evidence['collection_contract_sha256'],verified_source_episodes=400,
        scene_conditions={str(k):by_id[k]['job'] for k in ids})
    _atomic(output,data)
    return data

class AstribotLeRobotDataset(PiperLeRobotDataset):
    camera_keys=CAMERA_KEYS
    state_dim=action_dim=8

    def __init__(self,root,contract_path,split='train',split_manifest=None,action_horizon=7,video_horizon=8,
                 resolution=224,seed=42,sample_stride=1,frame_cache_size=256,decoder_cache_size=4):
        self.root=Path(root).resolve();self.contract=json.loads(Path(contract_path).read_text())
        c=self.contract
        _require(c['schema']==SEMANTICS,'Unsupported Astribot data contract')
        _require(_canonical_sha(c['file_sha256'])==c['dataset_sha256'],'Dataset identity mismatch')
        _require(split in ('train','val','test') and action_horizon==7 and video_horizon==8,'Wrong split/horizon')
        _require(isinstance(sample_stride,int) and sample_stride>0 and decoder_cache_size>0 and frame_cache_size>=0,'Invalid reader settings')
        for relative,digest in c['file_sha256'].items():
            p=self.root/relative
            _require(p.is_file() and p.stat().st_size==c['file_sizes'][relative],f'Dataset file missing/changed: {relative}')
            if not relative.endswith('.mp4'): _require(_sha256(p)==digest,f'Dataset changed: {relative}')
        self.split_manifest=c['split'];_require(self.split_manifest['seed']==seed,'Split seed mismatch')
        if isinstance(split_manifest,dict):_require(split_manifest==self.split_manifest,'Split contract differs')
        elif split_manifest is not None: _atomic(split_manifest,self.split_manifest)
        sets=[set(self.split_manifest[f'{k}_source_episode_ids']) for k in ('train','val','test')]
        _require([len(x) for x in sets]==[320,40,40] and not any(sets[i]&sets[j] for i in range(3) for j in range(i)),'Split leakage')
        self.export_manifest_sha256=c['dataset_sha256'];self.split_manifest_sha256=_canonical_sha(self.split_manifest)
        self.info,episodes,_=metadata(self.root);arrays,_=numeric(self.root,self.info,episodes)
        _require(set.union(*sets)=={e['source_episode_id'] for e in episodes},'Split coverage mismatch')
        _require(np.array_equal(expand_actions(compress_joints(arrays['action']),c['fixed_action32']),arrays['action']),'Lossy action contract')
        self.normalization=c['normalization']
        self._states=normalize(compress_joints(arrays['state']),self.normalization['state'])
        self._actions=normalize(compress_joints(arrays['action']),self.normalization['action'])
        self._states.flags.writeable=self._actions.flags.writeable=False
        self.source_episode_ids=tuple(self.split_manifest[f'{split}_source_episode_ids']);selected=set(self.source_episode_ids)
        self._all_episodes=episodes;self._episodes={e['source_episode_id']:e for e in episodes if e['source_episode_id'] in selected}
        self.episode_lengths={sid:e['length'] for sid,e in self._episodes.items()}
        self._anchors=[];self._ends=[]
        for e in self._episodes.values():
            anchors=list(range(0,e['length'],sample_stride))
            if anchors[-1]!=e['length']-1:anchors.append(e['length']-1)
            self._anchors.append((e['source_episode_id'],np.asarray(anchors,dtype=np.int64)))
            self._ends.append((self._ends[-1] if self._ends else 0)+len(anchors))
        self.fps=30;self.action_horizon=7;self.video_horizon=8;self.split=split;self.sample_stride=sample_stride
        self.resolution=(resolution,resolution) if isinstance(resolution,int) else tuple(resolution)
        _require(len(self.resolution)==2 and min(self.resolution)>0,'Invalid resolution')
        self.frame_cache_size=frame_cache_size;self.decoder_cache_size=decoder_cache_size
        self._pid=os.getpid();self._frames=OrderedDict();self._decoders=OrderedDict()

    def action_raw(self,values): return denormalize(values,self.normalization['action'])
    def state_raw(self,values): return denormalize(values,self.normalization['state'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-audit',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();c=prepare_contract(args.root,args.source_audit,args.output)
    print(json.dumps(dict(prepared=True,dataset_sha256=c['dataset_sha256'],source_episodes=c['verified_source_episodes'],split_sizes={k:len(c['split'][f'{k}_source_episode_ids']) for k in ('train','val','test')})))
