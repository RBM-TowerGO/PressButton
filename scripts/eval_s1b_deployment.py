#!/usr/bin/env python3
"""Reproduce P0/P1 assets, held-out errors, HTTP equality and GPU timings."""
import argparse
import http.client
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.contract import read_config, canonical, VERSION, JOINTS, sha
from pressb.s1b_deployment.policy import Policy, sdk_gripper
from pressb.s1b_deployment.protocol import encode,decode
from pressb.s1b_deployment.server import make_server

def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['offline'],default='offline')
    p.add_argument('--config',default='configs/s1b_deployment_v1.yaml')
    p.add_argument('--checkpoint',type=int,choices=[2000,8000],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--iterations',type=int,default=200)
    a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    c=read_config(a.config);write(out/'resolved_config.json',dict(config=c,sha256=canonical(c)))
    import torch
    env=dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','Pillow','omegaconf','accelerate']},gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),sdk='not imported; P0/P1 offline only')
    env['source_sha256']={str(path):sha(path) for path in [Path(__file__),*sorted(Path('src/pressb/s1b_deployment').glob('*.py'))]}
    write(out/'environment.json',env)
    policy=Policy(c['model']['run'],a.checkpoint,c['model']['identity'],c['model']['device'])
    write(out/'identity.json',policy.identity)
    from starVLA.dataloader.s1b_real import S1BRealDataset
    from starVLA.training.train_s1b_real import validation_anchors
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(Path(c['model']['run'])/'config.local.yaml')
    ds=S1BRealDataset(root=cfg.datasets.vla_data.data_root_dir,contract_path=cfg.datasets.vla_data.contract_path,split='val',split_manifest=Path(c['model']['run'])/'split.json',
                     resolution=policy.resolution, image_mode=policy.image_mode,
                     frame_cache_size=int(cfg.datasets.vla_data.frame_cache_size))
    anchors=validation_anchors(ds,3)
    if len(anchors)!=48: raise ValueError('Expected exactly 48 anchors')
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    for ordinal,(sid,frame) in enumerate(anchors):
        sample=ds.get_frame(sid,frame)
        if sample['lang'] != c['model']['task']: raise ValueError('Dataset instruction mismatch')
        seed=42+500000+sid*100003+frame
        raw,elapsed=policy.predict_prepared(sample['image'],np.stack([sample['state']]),seed)
        target=ds.action_raw(sample['action'])
        rows.append(dict(ordinal=ordinal,episode=sid,frame=frame,seed=seed,seconds=elapsed,action_raw=raw.tolist(),target=target.tolist(),error=np.abs(raw-target).tolist(),sdk_gripper=sdk_gripper(raw).tolist()))
        print(json.dumps(dict(event='anchor',ordinal=ordinal,seconds=elapsed)),flush=True)
    write(out/'anchors.json',rows)
    # Decode independently and compare the saved training preprocessing bit-for-bit.
    sid,frame=anchors[0]; episode=ds._episodes[sid]
    import av
    images=[]
    for key in ds.camera_keys:
        path,start=episode['videos'][key]
        with av.open(path) as container:
            for i,f in enumerate(container.decode(video=0)):
                if i==start+frame: images.append(f.to_ndarray(format='rgb24'));break
    sample=ds.get_frame(sid,frame);state=ds.state_raw(sample['state'])[0].astype(np.float32)
    from PIL import Image
    for raw_image,prepared in zip(images,sample['image']):
        expected=Image.fromarray(raw_image)
        if policy.image_mode == 'resize':
            expected=expected.resize((policy.resolution,policy.resolution),Image.Resampling.BILINEAR)
        if not np.array_equal(np.asarray(expected),np.asarray(prepared)): raise ValueError('Image preprocessing differs from training')
    direct,_=policy.predict(*images,state,42)
    server=make_server(policy,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=60)
    fields=dict(protocol_version=VERSION,session_id='offline-http',observation_id=np.int64(0),observation_monotonic_ns=np.int64(1),seed=np.int64(42),head=images[0],right=images[1],state10=state,source_time_ns=np.zeros(3,np.int64),mapped_time_ns=np.zeros(3,np.int64),**{k:policy.identity[k] for k in ['model_sha256','data_contract_sha256']})
    request_started=time.perf_counter()
    conn.request('POST','/predict',body=encode(fields));response=conn.getresponse();body=response.read()
    http_rtt=time.perf_counter()-request_started
    if response.status!=200:raise ValueError(body)
    remote=decode(body);difference=float(np.max(np.abs(direct-remote['action_raw'])))
    if difference!=0:raise ValueError(f'HTTP equality failed: {difference}')
    conn.close();server.shutdown();server.server_close();thread.join()
    boundary=[]
    for gripper in [-1.,0.,100.,101.]:
        measured=state.copy();measured[7]=gripper
        raw,timing=policy.predict(*images,measured,42)
        boundary.append(dict(gripper_feedback=gripper,finite=bool(np.isfinite(raw).all()),**timing))
    timings=[]
    for i in range(a.warmup+a.iterations):
        ordinal=i%len(anchors);sid,frame=anchors[ordinal];sample=ds.get_frame(sid,frame)
        _,elapsed=policy.predict_prepared(sample['image'],np.stack([sample['state']]),42)
        timings.append(dict(iteration=i,warmup=i<a.warmup,anchor_ordinal=ordinal,episode=sid,frame=frame,seed=42,seconds=elapsed))
        if i%20==0:print(json.dumps(dict(event='benchmark',iteration=i,seconds=elapsed)),flush=True)
    write(out/'timings.json',timings)
    errors=np.mean([r['error'] for r in rows],axis=0)
    lat=np.array([r['seconds'] for r in timings if not r['warmup']])
    summary=dict(http_first_request_rtt_seconds=http_rtt, load_peak_allocated_gib=policy.load_peak_allocated_gib, http_first_request_inference_seconds=float(remote['inference_seconds']), step=a.checkpoint,joints=JOINTS,anchors=len(rows),load_seconds=policy.load_seconds,first_inference_seconds=rows[0]['seconds'],warmup=a.warmup,iterations=a.iterations,p50_p95_p99_seconds=np.percentile(lat,[50,95,99]).tolist(),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,per_horizon_per_dimension_mae=errors.tolist(),per_dimension_mae=errors.mean(0).tolist(),arm_mae_rad=float(errors[:,:7].mean()),gripper_accuracy=float(np.mean([(np.array(r['action_raw'])[:,7]>=.5)==(np.array(r['target'])[:,7]>=.5) for r in rows])),http_max_abs_error=difference,http_tolerance=0,boundary=boundary,complete=a.warmup>=20 and a.iterations>=200)
    write(out/'summary.json',summary);ds.close();print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
