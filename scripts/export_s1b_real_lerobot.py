#!/usr/bin/env python3
"""Export reviewed S1B HDF5 to LeRobot v3 with binary gripper and future joint10 targets."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT/'docs/plan/astribot_s1b_real_finetune/data_contract.json'
REPO_ID = 'local/astribot_s1b_real_joint10_v1'
CAMERAS = {'head': (720, 1280, 3), 'right': (360, 640, 3)}

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def binary_gripper(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError('Invalid gripper command; expected finite SDK [0,100]')
    return (values >= 50).astype(np.float32)

def build_arrays(f):
    t = f['time'][:]
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Source time must be finite and strictly increasing')
    relative = t-t[0]
    grid = np.arange(int(np.floor(relative[-1]*30))+1, dtype=np.float64)/30
    ix = np.searchsorted(relative, grid, side='right')-1
    valid = (grid-relative[ix] <= 2/30)
    for field, limit in [('joints_dict/state_timestamp',10), ('joints_dict/command_timestamp',20),
                         ('images_dict/right/rgb_timestamp',40), ('images_dict/head/rgb_timestamp',.001)]:
        timestamps = f[field][:]
        valid &= np.isfinite(timestamps[ix]) & (np.abs(timestamps[ix]-t[ix])*1000 <= limit)
    segments = np.cumsum(np.r_[False, np.diff(t)>2/30])[ix]
    state = f['joints_dict/joints_position_state'][:,15:25]
    command = f['joints_dict/joints_position_command'][:,15:25]
    if not np.isfinite(state).all() or not np.isfinite(command).all():
        raise ValueError('Nonfinite joint10 values')
    raw_gripper = command[:,7].copy()
    command[:,7] = binary_gripper(raw_gripper)
    n = len(grid)-1
    if n < 7: raise ValueError('Episode too short for horizon 7')
    anchors = np.zeros(n, dtype=np.int64)
    good = np.ones(len(grid)-7, dtype=bool)
    for off in range(8):
        good &= valid[off:off+len(good)] & (segments[off:off+len(good)] == segments[:len(good)])
    anchors[:len(good)] = good
    # Retain every grid position except the no-next-target last row; invalid anchors cannot be sampled.
    return {
        'observation.state':state[ix[:-1]].astype(np.float32), 'action':command[ix[1:]].astype(np.float32),
        'source_observation_row':ix[:-1,None].astype(np.int64), 'source_action_row':ix[1:,None].astype(np.int64),
        'source_observation_time':t[ix[:-1],None], 'source_action_time':t[ix[1:],None],
        'target_grid_time':(t[0]+grid[1:])[:,None], 'observation_grid_time':(t[0]+grid[:-1])[:,None],
        'source_gripper_command':raw_gripper[ix[1:],None], 'valid_anchor':anchors[:,None],
        'valid_grid_row':valid[:-1,None].astype(np.int64), 'segment_id':segments[:-1,None].astype(np.int64),
    }

class Images:
    def __init__(self, f, camera):
        self.data = f[f'images_dict/{camera}/rgb'][:]
        sizes = f[f'images_dict/{camera}/rgb_size'][:]
        if np.any(sizes != sizes.astype(np.int64)) or np.any(sizes<=0): raise ValueError('Invalid JPEG sizes')
        self.offsets = np.r_[0,np.cumsum(sizes.astype(np.int64))]
        if self.offsets[-1] != self.data.size: raise ValueError('JPEG byte length mismatch')
        self.camera, self.cached = camera, (-1,None)
    def get(self, index):
        if self.cached[0] != index:
            encoded = self.data[self.offsets[index]:self.offsets[index+1]].tobytes()
            with Image.open(io.BytesIO(encoded)) as image: value=np.array(image.convert('RGB'))
            if value.shape != CAMERAS[self.camera]: raise ValueError('Unexpected image dimensions')
            self.cached=(index,value)
        return self.cached[1]

def features(names):
    result = {f'observation.images.{c}':dict(dtype='video',shape=shape,names=['height','width','channels']) for c,shape in CAMERAS.items()}
    for name in ['observation.state','action']: result[name]=dict(dtype='float32',shape=(10,),names=names)
    for name in ['source_observation_row','source_action_row','source_original_id','valid_anchor','valid_grid_row','segment_id']:
        result[name]=dict(dtype='int64',shape=(1,),names=[name])
    for name in ['source_observation_time','source_action_time','target_grid_time','observation_grid_time','source_gripper_command']:
        result[name]=dict(dtype='float64',shape=(1,),names=[name])
    return result

def split_episodes(count):
    # Contiguous acquisition blocks prevent immediate neighboring demonstrations being randomly split.
    if count != 96: raise ValueError('v1 release expects the locked 96-episode review')
    blocks = np.arange(12); np.random.default_rng(42).shuffle(blocks)
    val = sorted(i for block in blocks[:2] for i in range(int(block)*8,int(block)*8+8))
    return {'seed':42, 'method':'contiguous reviewed-order blocks of 8; two held-out blocks',
            'train': [i for i in range(count) if i not in val], 'validation':val,
            'limitation':'Temporal grouping only; scene/session identities unavailable; no claim of independent scenes.'}

def normalization(arrays):
    result={}
    for key in ['observation.state','action']:
        values=np.concatenate([a[key] for a in arrays]).astype(np.float64)
        low,high=values.min(0),values.max(0);constant=low==high
        mean=values.mean(0);mean[constant]=low[constant]
        scale=np.maximum(values.std(0),1e-6)
        if key=='action':mean[7]=0;scale[7]=1
        result[key]={'mean':mean.tolist(),'std':scale.tolist(),'constant':constant.tolist(),
                     'min':low.tolist(),'max':high.tolist()}
        restored=((values-mean)/scale)*scale+mean
        np.testing.assert_allclose(restored,values,rtol=0,atol=1e-10)
    return result

def audit(root, source, records, expected, splits):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset(REPO_ID,root=root,video_backend='pyav')
    if ds.num_episodes != len(records):raise ValueError('Episode count mismatch')
    offset=0;image_checks=[];anchor_indices={'train':[],'validation':[]}
    for episode,(record, arrays) in enumerate(zip(records,expected)):
        n=len(arrays['action'])
        # Official HF table: compare every numerical row, without decoding all videos twice.
        table=ds.hf_dataset.select(range(offset,offset+n)).with_format(None)
        for key,values in arrays.items():np.testing.assert_array_equal(np.asarray(table[key]).reshape(values.shape),values)
        np.testing.assert_array_equal(np.asarray(table['source_original_id']).reshape(-1),np.full(n,record['original_id']))
        np.testing.assert_array_equal(np.asarray(table['episode_index']),np.full(n,episode))
        np.testing.assert_array_equal(np.asarray(table['frame_index']),np.arange(n))
        np.testing.assert_allclose(np.asarray(table['timestamp']),np.arange(n)/30,atol=1e-5,rtol=0)
        split='train' if episode in splits['train'] else 'validation'
        valid=np.flatnonzero(arrays['valid_anchor'][:,0])
        anchor_indices[split].extend((offset+valid).tolist())
        if len(valid) and valid[-1]+6>=n:raise ValueError('Cross-episode action window')
        with h5py.File(source/'episodes'/record['output_file']) as f:
            streams={c:Images(f,c) for c in CAMERAS}
            for index in sorted({0,n//2,n-1}):
                row=ds[offset+index]
                for camera,stream in streams.items():
                    decoded=row[f'observation.images.{camera}'].numpy().transpose(1,2,0)*255
                    original=stream.get(int(arrays['source_observation_row'][index,0]))
                    if decoded.shape != original.shape or not np.isfinite(decoded).all():raise ValueError('Video invalid')
                    mae=float(np.abs(decoded-original.astype(np.float32)).mean())
                    if mae>15:raise ValueError(f'Video RGB/order/content mismatch: MAE {mae}')
                    image_checks.append(dict(episode=episode,frame=index,camera=camera,mae=mae))
        offset+=n
    if len(ds)!=offset:raise ValueError('Total frame count mismatch')
    write_json(root/'meta/training_anchors.json',anchor_indices)
    report=dict(status='passed',episodes=len(records),frames=offset,
                anchors={k:len(v) for k,v in anchor_indices.items()},numeric_rows_checked=offset,
                image_checks=image_checks,reader='LeRobotDataset 0.6.1 / pyav')
    write_json(root/'meta/reader_audit.json',report)
    return report

def publish_readme(staging):
    (staging/'README.md').write_text('# S1B real joint10 v1\n\n96 reviewed demonstrations. Record-time proxy, 30 Hz, binary gripper.\n'
        'Use meta/splits.json and meta/training_anchors.json; never sample all rows as 7-step anchors.\n'
        'action[k] already targets t[k]+1/30; read action[k:k+7], with no extra shift.\n'
        'Use meta/training_normalization.json, NOT all-data meta/stats.json, for training.\n'
        'Deployment timing and joint10 training adapter are not yet validated.\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--finish-staging',action='store_true',help='Revalidate a fully written staging dataset and publish it')
    p.add_argument('--source',type=Path,default=ROOT/'data/astribot_reviewed_v1')
    p.add_argument('--output',type=Path,default=ROOT/'data/astribot_s1b_real_joint10_v1')
    args=p.parse_args();source=args.source.resolve();output=args.output.resolve()
    staging=output.with_name(output.name+'.inprogress')
    if output.exists() or (staging.exists() and not args.finish_staging):
        raise FileExistsError('Output/staging exists; preserve it or validate complete staging with --finish-staging')
    contract=json.loads(CONTRACT.read_text())
    if contract['version']!='s1b-real-joint10-record-time-v1':raise ValueError('Unrecognized contract version')
    records=json.loads((source/'episode_mapping.json').read_text());splits=split_episodes(len(records))
    expected=[]
    for record in records:
        path=source/'episodes'/record['output_file']
        if sha256(path)!=record['output_sha256']:raise ValueError(f'Source SHA mismatch: {path}')
        with h5py.File(path) as f:expected.append(build_arrays(f))
    stats=normalization([expected[i] for i in splits['train']])
    if args.finish_staging:
        for name,value in [('source_mapping.json',records),('splits.json',splits),('training_normalization.json',stats)]:
            if json.loads((staging/'meta'/name).read_text())!=value:raise ValueError(f'Staging metadata differs: {name}')
        saved=json.loads((staging/'meta/data_contract.json').read_text())
        checked=dict(contract,status='converted_pending_model_training_adapter',model_ready=False)
        if saved!=checked:raise ValueError('Staging contract differs')
        report=audit(staging,source,records,expected,splits)
        for record in records:
            if sha256(source/'episodes'/record['output_file'])!=record['output_sha256']:
                raise ValueError('Source changed during validation')
        provenance=json.loads((staging/'meta/export_provenance.json').read_text())
        provenance['validation_script_sha256']=sha256(Path(__file__))
        write_json(staging/'meta/export_provenance.json',provenance)
        publish_readme(staging)
        os.rename(staging,output)
        print(json.dumps({k:v for k,v in report.items() if k!='image_checks'},indent=2))
        print(f'Published: {output}')
        return
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig
    encoder=RGBEncoderConfig(vcodec='h264',pix_fmt='yuv420p',g=30,crf=18,preset='fast')
    started=time.time()
    ds=LeRobotDataset.create(REPO_ID,30,features(contract['names']),root=staging,robot_type='astribot_s1b',
        use_videos=True,video_backend='pyav',rgb_encoder=encoder,streaming_encoding=True,
        encoder_threads=4,encoder_queue_maxsize=16,metadata_buffer_size=1)
    try:
        for episode,(record,arrays) in enumerate(zip(records,expected)):
            with h5py.File(source/'episodes'/record['output_file']) as f:
                streams={c:Images(f,c) for c in CAMERAS}
                for index,src in enumerate(arrays['source_observation_row'][:,0]):
                    frame={key:value[index] for key,value in arrays.items()}
                    frame.update({f'observation.images.{c}':s.get(int(src)) for c,s in streams.items()})
                    frame['source_original_id']=np.array([record['original_id']],dtype=np.int64)
                    frame['task']='Press the down button while maintaining the demonstrated head posture.'
                    ds.add_frame(frame)
                ds.save_episode()
            print(f'Exported {episode+1}/{len(records)}: {len(arrays["action"])} frames',flush=True)
    finally:ds.finalize()
    write_json(staging/'meta/source_mapping.json',records)
    write_json(staging/'meta/splits.json',splits)
    write_json(staging/'meta/training_normalization.json',stats)
    contract['status']='converted_pending_model_training_adapter';contract['model_ready']=False
    write_json(staging/'meta/data_contract.json',contract)
    write_json(staging/'meta/export_provenance.json',dict(script_sha256=sha256(Path(__file__)),contract_sha256=sha256(CONTRACT),
        source=str(source),lerobot='0.6.1',h5py=h5py.__version__,numpy=np.__version__,
        codec='h264',crf=18,preset='fast',source_hashes_verified=True))
    print('Running official reader audit...',flush=True)
    report=audit(staging,source,records,expected,splits)
    # Verify source files again before publishing.
    for record in records:
        if sha256(source/'episodes'/record['output_file'])!=record['output_sha256']:raise ValueError('Source changed during export')
    report['elapsed_seconds']=time.time()-started
    write_json(staging/'meta/reader_audit.json',report)
    publish_readme(staging)
    os.rename(staging,output)
    print(json.dumps({k:v for k,v in report.items() if k!='image_checks'},indent=2))
    print(f'Published: {output}')

if __name__=='__main__':main()
