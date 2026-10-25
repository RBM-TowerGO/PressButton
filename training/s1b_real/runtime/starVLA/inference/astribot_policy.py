"""Strict offline Astribot checkpoint inference; no robot commands are sent."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from starVLA.dataloader.astribot_lerobot import (SEMANTICS,CAMERA_KEYS,TASK,compress_joints,
    expand_actions,normalize,denormalize,AstribotLeRobotDataset,_sha256,_canonical_sha)
from starVLA.inference.piper_policy import (install_saved_processor,load_trained_state,
    disable_unused_qwen_cache,validate_config_contract)

def validate_checkpoint(checkpoint):
    checkpoint=Path(checkpoint).resolve(strict=True);run=checkpoint.parent.parent
    if checkpoint.parent.name!='checkpoints':raise ValueError('Expected run/checkpoints/step_*')
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    if manifest.get('complete') is not True or manifest.get('step',0)<1:raise ValueError('Incomplete checkpoint')
    if _sha256(checkpoint/'model.pt')!=manifest['model_sha256']:raise ValueError('Model checksum mismatch')
    schema=json.loads((run/'action_schema.json').read_text())
    if any(schema.get(k)!=v for k,v in SEMANTICS.items()):raise ValueError('Wrong action semantics')
    for k in ('dataset_sha256','split_sha256','training_fingerprint','data_contract_sha256'):
        if schema[k]!=manifest[k]:raise ValueError(f'Checkpoint schema mismatch: {k}')
    data=json.loads((run/'data_contract.json').read_text())
    if _sha256(run/'data_contract.json')!=schema['data_contract_sha256'] or data['schema']!=SEMANTICS:raise ValueError('Data contract mismatch')
    split=json.loads((run/'split.json').read_text())
    if split!=data['split'] or _canonical_sha(split)!=schema['split_sha256'] or data['dataset_sha256']!=schema['dataset_sha256']:raise ValueError('Split identity mismatch')
    training=json.loads((run/'training_contract.json').read_text())
    if training['fingerprint']!=schema['training_fingerprint'] or _canonical_sha(training['contract'])!=training['fingerprint']:raise ValueError('Training fingerprint mismatch')
    return dict(run=run,checkpoint=checkpoint,manifest=manifest,schema=schema,data=data,training=training['contract'])

class AstribotPolicy:
    def __init__(self,checkpoint,device='cuda:0'):
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoProcessor
        from starVLA.model.framework.VLA_JEPA import VLA_JEPA
        self.checked=validate_checkpoint(checkpoint);c=self.checked
        self.device=torch.device(device);torch.cuda.set_device(self.device)
        cfg=OmegaConf.load(c['run']/'config.yaml')
        validate_config_contract(OmegaConf.to_container(cfg,resolve=True),c['training'])
        if cfg.framework.action_model.action_dim!=8 or cfg.framework.action_model.state_dim!=8:raise ValueError('Expected joint8 model')
        cfg.framework.qwenvl.device_map=str(self.device)
        model=VLA_JEPA(config=cfg)
        install_saved_processor(model,AutoProcessor.from_pretrained(c['run']/'processor',local_files_only=True))
        self.loaded_tensors=load_trained_state(model,c['checkpoint']/'model.pt')
        disable_unused_qwen_cache(model);model.to(self.device).requires_grad_(False).eval()
        self.model=model;self.config=cfg;self.data=c['data'];self.resolution=int(cfg.datasets.vla_data.resolution_size)

    def predict(self,*,head_image,wrist_right_image,state,task=TASK,seed=42):
        import torch
        if task!=TASK:raise ValueError('Unsupported task')
        state=np.asarray(state,dtype=np.float32)
        if state.shape==(32,):state=compress_joints(state)
        if state.shape!=(8,) or not np.isfinite(state).all():raise ValueError('Expected one raw state8 or state32')
        images=[]
        for image in (head_image,wrist_right_image):
            if not isinstance(image,Image.Image):raise TypeError('Expected PIL RGB images in head/right-wrist order')
            images.append(image.convert('RGB').resize((self.resolution,self.resolution),Image.Resampling.BILINEAR))
        normalized_state=normalize(state,self.data['normalization']['state']).reshape(1,1,8)
        with torch.random.fork_rng(devices=[self.device.index]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.manual_seed(int(seed));torch.cuda.manual_seed(int(seed))
            result=self.model.predict_action([images],[task],state=normalized_state)['normalized_actions']
        normalized=np.asarray(result,dtype=np.float32)
        if normalized.shape!=(1,7,8) or not np.isfinite(normalized).all():raise ValueError('Invalid policy output')
        raw=denormalize(normalized[0],self.data['normalization']['action'])
        expanded=expand_actions(raw,self.data['fixed_action32'])
        return dict(action8=raw,action32=expanded,normalized_actions=normalized[0])

def verify(checkpoint,root,output,anchors=12):
    """Reload once and check predictions, inverse scaling, and controller expansion."""
    import torch
    from starVLA.training.train_astribot import validation_anchors
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    policy=AstribotPolicy(checkpoint);c=policy.checked
    ds=AstribotLeRobotDataset(root,c['run']/'data_contract.json',split='val',resolution=policy.resolution)
    selected=validation_anchors(ds,3,anchors);rows=[]
    for ordinal,(sid,frame) in enumerate(selected):
        sample=ds.get_frame(sid,frame);state=ds.state_raw(sample['state'])[0]
        kwargs=dict(head_image=sample['image'][0],wrist_right_image=sample['image'][1],state=state,seed=425000+sid*100003+frame)
        pred=policy.predict(**kwargs)
        if ordinal==0:
            second=policy.predict(**kwargs)
            np.testing.assert_array_equal(pred['action8'],second['action8'])
        # Independent coordinate checks (not only recompressing with the same helper).
        full=pred['action32'];raw=pred['action8']
        np.testing.assert_array_equal(full[:,[5,8,11,13,15,17,19]],raw[:,:7])
        for idx,sign in [(24,-1),(25,-1),(26,1),(27,1),(30,1),(31,-1)]:np.testing.assert_array_equal(full[:,idx],sign*raw[:,7])
        active={5,8,11,13,15,17,19,24,25,26,27,30,31}
        fixed=[i for i in range(32) if i not in active]
        np.testing.assert_array_equal(full[:,fixed],np.broadcast_to(np.asarray(policy.data['fixed_action32'],dtype=np.float32)[fixed],(7,len(fixed))))
        expected=pred['normalized_actions']*np.asarray(policy.data['normalization']['action']['scale'],dtype=np.float32)+np.asarray(policy.data['normalization']['action']['mean'],dtype=np.float32)
        np.testing.assert_array_equal(expected,raw)
        target=ds.action_raw(sample['action']);valid=~sample['action_is_pad'];error=np.abs(raw-target)
        rows.append(dict(candidate_id=sid,frame=frame,valid_actions=int(valid.sum()),arm_mae_rad=float(error[valid,:7].mean()),
            gripper_mae_rad=float(error[valid,7].mean()),action8=raw.tolist(),action32=full.tolist(),action_is_pad=sample['action_is_pad'].tolist()))
    report=dict(success=True,kind='astribot_independent_checkpoint_inference',checkpoint=str(c['checkpoint']),
        step=c['manifest']['step'],model_sha256=c['manifest']['model_sha256'],loaded_tensors=policy.loaded_tensors,
        data_contract_sha256=c['manifest']['data_contract_sha256'],camera_keys=list(CAMERA_KEYS),anchors=len(rows),
        same_seed_repeat_exact=True,normalization_and_expansion_verified=True,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
        limits=['Offline inference and structural checks; no claim of closed-loop success.','Predictions are not clipped or sent to a robot.'],samples=rows)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');ds.close()
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--dataset-root',required=True);p.add_argument('--output',required=True);p.add_argument('--anchors',type=int,default=12)
    a=p.parse_args();verify(a.checkpoint,a.dataset_root,a.output,a.anchors)
