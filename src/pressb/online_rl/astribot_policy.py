"""Experiment adapter; historical starVLA runtime and checkpoint remain intact.

The predict preprocessing and action postprocessing deliberately match the saved
wrapper. Only the full prediction return and pooled tokens are retained.
"""
import numpy as np
from PIL import Image
from starVLA.inference.astribot_policy import AstribotPolicy as FrozenAstribotPolicy
from starVLA.dataloader.astribot_lerobot import TASK, compress_joints, normalize, denormalize, expand_actions
from .astribot_features import pooled_tokens


class AstribotLatentPolicy(FrozenAstribotPolicy):
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
            prediction=self.model.predict_action([images],[task],state=normalized_state)
            result=prediction['normalized_actions']
        normalized=np.asarray(result,dtype=np.float32)
        if normalized.shape!=(1,7,8) or not np.isfinite(normalized).all():raise ValueError('Invalid policy output')
        raw=denormalize(normalized[0],self.data['normalization']['action'])
        expanded=expand_actions(raw,self.data['fixed_action32'])
        return dict(action8=raw,action32=expanded,normalized_actions=normalized[0],
                    vlm_latent=pooled_tokens(prediction["embodied_action_tokens"]))
