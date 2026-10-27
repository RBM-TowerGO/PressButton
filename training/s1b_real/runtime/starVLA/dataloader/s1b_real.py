"""S1B joint10: immutable v3 numeric payload, train-only stats, approved anchors."""
from collections import OrderedDict
import json
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .piper_lerobot import PiperLeRobotDataset, _require, _sha256, _canonical_sha, collate_fn

SEMANTICS=dict(kind='s1b_real_joint10_record_time_v1',state_dim=10,action_dim=10,action_horizon=7,
    video_horizon=8,fps=30,camera_keys=['observation.images.head','observation.images.right'],
    units=['rad']*7+['binary_action_continuous_sdk_state']+['rad']*2,
    action_timing='stored_action_already_next_grid_target',normalization='train_only_mean_std_binary_action_identity')

class S1BRealDataset(PiperLeRobotDataset):
    camera_keys=tuple(SEMANTICS['camera_keys'])
    state_dim=action_dim=10
    def __init__(self,root,contract_path,split='train',split_manifest=None,action_horizon=7,video_horizon=8,
                 resolution=224,seed=42,sample_stride=1,frame_cache_size=256,decoder_cache_size=4,image_mode='resize'):
        _require(image_mode in ('resize','native'),'image_mode must be resize or native')
        _require(type(resolution) is int and resolution > 0,'resolution must be a positive integer')
        _require(type(frame_cache_size) is int and frame_cache_size >= 0,'Invalid frame cache size')
        self.image_mode=image_mode
        self.root=Path(root);self.contract=json.loads(Path(contract_path).read_text());c=self.contract
        _require(c['schema']==SEMANTICS,'Wrong real joint10 schema')
        _require(split in ('train','val') and (action_horizon,video_horizon)==(7,8),'Wrong split/horizons')
        _require(seed==42 and sample_stride==1,'v1 uses frozen anchors, seed 42 and stride 1')
        _require(_canonical_sha(c['file_sha256'])==c['dataset_sha256'],'Invalid dataset manifest')
        for name,digest in c['file_sha256'].items():
            p=self.root/name
            _require(p.is_file() and p.stat().st_size==c['file_sizes'][name],f'Missing/changed {name}')
            if not name.endswith('.mp4'):_require(_sha256(p)==digest,f'Hash mismatch {name}')
        self.info=json.loads((self.root/'meta/info.json').read_text())
        _require(self.info['codebase_version']=='v3.0' and self.info['fps']==30,'Wrong format')
        for key in ('action','observation.state'):_require(self.info['features'][key]['shape']==[10],'Wrong dimensions')
        partitions=json.loads((self.root/'meta/splits.json').read_text())
        _require(not set(partitions['train'])&set(partitions['validation']),'Split leakage')
        self.split_manifest=partitions
        if split_manifest is not None:
            from .astribot_lerobot import _atomic
            if isinstance(split_manifest,dict):_require(split_manifest==partitions,'Split differs')
            else:_atomic(split_manifest,partitions)
        self.export_manifest_sha256=c['dataset_sha256'];self.split_manifest_sha256=_canonical_sha(partitions)
        table=pq.read_table(sorted((self.root/'data').rglob('*.parquet'))).sort_by([('index','ascending')])
        _require(np.array_equal(table['index'].to_numpy(),np.arange(self.info['total_frames'])),'Bad indices')
        raw={k:np.asarray(table[col].to_pylist(),dtype=np.float32) for k,col in [('state','observation.state'),('action','action')]}
        _require(all(v.shape==(len(table),10) and np.isfinite(v).all() for v in raw.values()),'Bad numeric data')
        _require(np.isin(raw['action'][:,7],[0,1]).all(),'Nonbinary actions')
        self.normalization=c['normalization']
        self._states=self.normalize(raw['state'],'state');self._actions=self.normalize(raw['action'],'action')
        self._states.flags.writeable=self._actions.flags.writeable=False
        episodes=[e for p in sorted((self.root/'meta/episodes').rglob('*.parquet')) for e in pq.read_table(p).to_pylist()]
        episodes.sort(key=lambda e:e['episode_index'])
        selected=partitions['train' if split=='train' else 'validation']
        anchors=json.loads((self.root/'meta/training_anchors.json').read_text())['train' if split=='train' else 'validation']
        _require(len(anchors)==len(set(anchors)),'Duplicate anchors')
        valid=np.asarray(table['valid_anchor'].to_pylist()).reshape(-1)
        global_anchors=np.asarray(anchors,dtype=np.int64)
        _require(np.all((global_anchors>=0)&(global_anchors<len(table))),'Out of range anchors')
        _require(valid[global_anchors].all(),'Unapproved anchor')
        self._episodes={};self._anchors=[];self._ends=[];self.valid_frames={}
        for e in episodes:
            sid=e['episode_index'];a,b=e['dataset_from_index'],e['dataset_to_index']
            if sid not in selected:continue
            e.update(source_episode_id=sid,floor=0,task=e['tasks'][0],videos={})
            for key in self.camera_keys:
                prefix='videos/'+key+'/'
                path=self.root/self.info['video_path'].format(video_key=key,chunk_index=e[prefix+'chunk_index'],file_index=e[prefix+'file_index'])
                e['videos'][key]=(str(path),int(round(e[prefix+'from_timestamp']*30)))
            local=global_anchors[(global_anchors>=a)&(global_anchors<b)]-a
            _require(len(local)>0 and np.all(local+6<e['length']),'Invalid complete action windows')
            _require(np.array_equal(np.flatnonzero(valid[a:b]),local),'Anchor file coverage differs')
            self._episodes[sid]=e;self.valid_frames[sid]=local
            self._anchors.append((sid,local));self._ends.append((self._ends[-1] if self._ends else 0)+len(local))
        _require(self._ends[-1]==len(anchors),'Split/anchor coverage differs')
        self.source_episode_ids=tuple(selected);self.episode_lengths={sid:e['length'] for sid,e in self._episodes.items()}
        self.fps=30;self.action_horizon=7;self.video_horizon=8;self.split=split;self.sample_stride=1
        self.resolution=(resolution,resolution);self.frame_cache_size=frame_cache_size;self.decoder_cache_size=decoder_cache_size
        self._pid=os.getpid();self._frames=OrderedDict();self._decoders=OrderedDict()
    def normalize(self,value,key):
        s=self.normalization[key];return ((value-np.array(s['mean'],np.float32))/np.array(s['scale'],np.float32))
    def action_raw(self,value):
        s=self.normalization['action'];return np.asarray(value)*np.array(s['scale'],np.float32)+np.array(s['mean'],np.float32)
    def state_raw(self,value):
        s=self.normalization['state'];return np.asarray(value)*np.array(s['scale'],np.float32)+np.array(s['mean'],np.float32)
    def get_frame(self,source_episode_id,frame_index):
        if source_episode_id not in self.valid_frames or frame_index not in self.valid_frames[source_episode_id]:
            raise ValueError('Not an approved full action-window anchor')
        example=super().get_frame(source_episode_id,frame_index)
        _require(not example['action_is_pad'].any(),'Action window must not be padded')
        # Last accepted action anchor may lack the eighth exported image. The inherited
        # video mask excludes its final tubelet from the auxiliary world-model loss.
        return example
