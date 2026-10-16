#!/usr/bin/env python3
"""Export accepted 32-joint/seven-camera Astribot episodes with LeRobot 0.6.1."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/hf'));os.environ.setdefault('OMP_NUM_THREADS','2')
from pressb.astribot_collection.contract import CAMERAS,JOINT_NAMES,TASK,atomic_json,validate_commit,sha256

@contextmanager
def link_immutable_videos(aggregate_module,parts,staging):
    """Use the official aggregator while avoiding a second copy of shard MP4s.

    Only its immutable MP4 copy operation is replaced. Parquet/metadata rewriting
    and episode/video offset logic remain entirely in the official implementation.
    """
    import shutil
    original=aggregate_module.shutil
    def copy(source,destination,*args,**kwargs):
        src,dst=Path(source).resolve(),Path(destination).resolve()
        if src.suffix=='.mp4' and src.is_relative_to(Path(parts).resolve()) and dst.is_relative_to(Path(staging).resolve()):
            os.link(src,dst);return str(dst)
        return shutil.copy(source,destination,*args,**kwargs)
    aggregate_module.shutil=SimpleNamespace(**{name:getattr(original,name) for name in dir(original) if not name.startswith('__')})
    aggregate_module.shutil.copy=copy
    try:yield
    finally:aggregate_module.shutil=original

def features():
    return {**{f'observation.images.{n}':dict(dtype='video',shape=(h,w,3),names=['height','width','channels']) for n,(w,h) in CAMERAS.items()},
        'observation.state':dict(dtype='float32',shape=(32,),names=list(JOINT_NAMES)),
        'action':dict(dtype='float32',shape=(32,),names=list(JOINT_NAMES)),
        'observation.velocity':dict(dtype='float32',shape=(32,),names=list(JOINT_NAMES)),
        'source_candidate_id':dict(dtype='int64',shape=(1,),names=['candidate_id']),
        'source_physics_step':dict(dtype='int64',shape=(1,),names=['physics_step'])}

def frames(path):
    import av
    with av.open(str(path)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-i/30)>1e-6:raise ValueError('Nonconsecutive video PTS')
            yield frame.to_ndarray(format='rgb24')

def reader_audit(root,entries):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset('local/astribot_vib820',root=root,video_backend='pyav')
    if ds.num_episodes!=len(entries):raise ValueError('Official reader episode count differs')
    checked=[]
    for i,entry in enumerate(entries):
        src=Path(entry['directory'])
        with np.load(src/'training.npz',allow_pickle=False) as a:
            state=a['state'];action=a['action'];n=len(state)
        meta=ds.meta.episodes[i];start=int(meta['dataset_from_index']);end=int(meta['dataset_to_index'])
        if end-start!=n:raise ValueError('Official reader frame count differs')
        for local in sorted({0,n//2,n-1}):
            row=ds[start+local]
            np.testing.assert_array_equal(row['observation.state'].numpy(),state[local])
            np.testing.assert_array_equal(row['action'].numpy(),action[local])
            for name,(w,h) in CAMERAS.items():
                image=row[f'observation.images.{name}']
                if tuple(image.shape)!=(3,h,w) or not image.isfinite().all():raise ValueError('Official RGB shape/content invalid')
        np.testing.assert_array_equal(action[-1],action[-2])
        checked.append(dict(candidate_id=entry['candidate_id'],frames=n,sampled=[0,n//2,n-1]))
    result=dict(status='passed',official_reader='LeRobotDataset/0.6.1/pyav',episodes=len(entries),checks=checked)
    atomic_json(Path(root)/'meta/astribot_reader_audit.json',result);return result

def write_shard(destination,entries,gpu):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig
    class CollectionEncoderConfig(RGBEncoderConfig):
        def get_codec_options(self,encoder_threads=None,as_strings=False):
            options=super().get_codec_options(encoder_threads,as_strings=as_strings)
            # LeRobot 0.6.1 hard-codes NVENC rc=0 (constant QP), which silently
            # ignores CQ. Keep its official writer and explicitly select VBR/CQ.
            options['rc']='1' if as_strings else 1
            options['b']='0' if as_strings else 0
            return options
    destination=Path(destination)
    if destination.exists():
        saved=json.loads((destination/'meta/astribot_sources.json').read_text())
        if saved!=entries:raise ValueError('Shard source manifest mismatch')
        verify_encoder(destination)
        reader_audit(destination,entries);return
    staging=destination.with_name(destination.name+'.inprogress')
    preserve_interrupted(staging)
    counts=[]
    for e in entries:
        with np.load(Path(e['directory'])/'training.npz',allow_pickle=False) as a:counts.append(len(a['state']))
    encoder=CollectionEncoderConfig(vcodec='h264_nvenc',pix_fmt='yuv420p',crf=None,g=30,preset='15',
                                   extra_options={'cq':'18','bf':'0','gpu':str(gpu)})
    ds=LeRobotDataset.create(repo_id='local/astribot_'+destination.name,root=staging,fps=30,robot_type='astribot',
        features=features(),use_videos=True,video_backend='pyav',
        # LeRobot 0.6.1 validates NVENC's enum as numeric: FFmpeg p4 == 15.
        rgb_encoder=encoder,
        streaming_encoding=True,encoder_threads=2,encoder_queue_maxsize=max(counts)+1,metadata_buffer_size=1)
    try:
        for entry in entries:
            source=Path(entry['directory']);streams={n:frames(source/f'{n}.mp4') for n in CAMERAS}
            try:
                with np.load(source/'training.npz',allow_pickle=False) as a:
                    for i in range(len(a['state'])):
                        frame={f'observation.images.{name}':next(stream) for name,stream in streams.items()}
                        frame.update({'observation.state':a['state'][i],'action':a['action'][i],
                            'observation.velocity':a['velocity'][i],
                            'source_candidate_id':np.array([entry['candidate_id']],dtype=np.int64),
                            'source_physics_step':np.array([a['physical_steps'][i]],dtype=np.int64),'task':TASK})
                        ds.add_frame(frame)
                ds.save_episode()
            finally:
                for stream in streams.values():stream.close()
    finally:ds.finalize()
    atomic_json(staging/'meta/astribot_sources.json',entries)
    atomic_json(staging/'meta/astribot_encoder_config.json',dict(codec='h264_nvenc',pixel_format='yuv420p',
        options=encoder.get_codec_options(2,as_strings=True),export_script_sha256=sha256(Path(__file__))))
    reader_audit(staging,entries);os.rename(staging,destination)

def preserve_interrupted(staging):
    """Restart only the unfinished shard, retaining its previous evidence."""
    if staging.exists():
        preserved=staging.with_name(staging.name+f'.interrupted_{time.time_ns()}')
        os.rename(staging,preserved)

def verify_encoder(root):
    recorded=json.loads((Path(root)/'meta/astribot_encoder_config.json').read_text())
    options=recorded['options']
    required={'rc':'1','b':'0','cq':'18','preset':'15','bf':'0','g':'30'}
    if (recorded['codec']!='h264_nvenc' or recorded['pixel_format']!='yuv420p'
            or any(options.get(k)!=v for k,v in required.items())):
        raise ValueError('Existing export used a different encoding profile')
    if recorded['export_script_sha256']!=sha256(Path(__file__)):
        raise ValueError('Existing export used a different exporter version')

def main():
    started=time.time();stages={}
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True,help='Ordered JSON list of candidate_id/directory')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--preflight',action='store_true',help='Explicitly allow fewer than 400; not a final release')
    args=p.parse_args();entries=json.loads(args.selection.read_text())
    if not args.preflight and len(entries)!=400:raise ValueError('Final publication requires exactly 400 accepted episodes')
    ids=[e['candidate_id'] for e in entries]
    if ids!=sorted(set(ids)) or not ids:raise ValueError('Selection must be nonempty, unique and ordered')
    for e in entries:
        validate_commit(e['directory'])
        e['commit_sha256']=sha256(Path(e['directory'])/'commit.json')
    stages['source_validation_s']=time.time()-started
    parts=args.output.with_name(args.output.name+'_parts');parts.mkdir(parents=True,exist_ok=True)
    paths=[];shard_started=time.time()
    for start in range(0,len(entries),25):
        dest=parts/f'part_{start//25:04d}';write_shard(dest,entries[start:start+25],args.gpu);paths.append(dest)
    stages['shards_and_reader_audit_s']=time.time()-shard_started
    if args.output.exists():
        if json.loads((args.output/'meta/astribot_sources.json').read_text())!=entries:
            raise ValueError('Published dataset source manifest mismatch')
        verify_encoder(args.output)
        reader_audit(args.output,entries);return
    from lerobot.datasets import aggregate
    staging=args.output.with_name(args.output.name+'.inprogress')
    preserve_interrupted(staging)
    aggregate_started=time.time()
    with link_immutable_videos(aggregate,parts,staging):
        aggregate.aggregate_datasets(repo_ids=['local/astribot_'+d.name for d in paths],roots=paths,
            aggr_repo_id='local/astribot_vib820',aggr_root=staging,concatenate_videos=False,concatenate_data=False)
    atomic_json(staging/'meta/astribot_sources.json',entries)
    atomic_json(staging/'meta/astribot_encoder_config.json',json.loads((paths[0]/'meta/astribot_encoder_config.json').read_text()))
    reader_audit(staging,entries)
    (staging/'README.md').write_text('# Astribot VIB-820\n\nSeven RGB cameras, 32 actual joint angles and absolute commanded angles, radians. '
        '120 Hz raw physics, 30 Hz training. Training ends at first physical success with visible lamp; final action repeats preceding target. '
        'Successful-only selection from independent uniform X [-0.30,-0.14] m, Y [0,0.40] m, positive yaw [0,10] degrees. '
        'This is a success-filtered distribution. Raw evidence includes release, return and reset.\n')
    os.rename(staging,args.output)
    stages['aggregation_and_reader_audit_s']=time.time()-aggregate_started
    atomic_json(args.output/'meta/astribot_export_timing.json',dict(started_unix_s=started,
        finished_unix_s=time.time(),elapsed_s=time.time()-started,stages=stages))
if __name__=='__main__':main()
