import os
from pathlib import Path
import numpy as np
import pytest
import torch
from starVLA.training.train_s1b_real import select_pretrained_state, validation_anchors
from starVLA.dataloader.s1b_real import S1BRealDataset

def test_transfer_only_approved_interfaces():
    source={'action_model.state_encoder.layer1.weight':torch.ones(4,8),
            'action_model.state_encoder.layer1.bias':torch.ones(4),
            'action_model.action_encoder.layer1.weight':torch.ones(4,8),
            'action_model.action_decoder.layer2.weight':torch.ones(8,4),
            'action_model.action_decoder.layer2.bias':torch.ones(8),'backbone':torch.ones(4,4)}
    target={k:torch.zeros_like(v) for k,v in source.items()}
    for k in list(target):
        if k.endswith('layer1.weight'):target[k]=torch.zeros(4,10)
        if k.endswith('layer2.weight'):target[k]=torch.zeros(10,4)
        if k.endswith('layer2.bias'):target[k]=torch.zeros(10)
    loaded,report=select_pretrained_state(source,target)
    assert set(loaded)=={'backbone'} and len(report['adapted_keys'])==5
    target['backbone']=torch.zeros(3,3)
    with pytest.raises(ValueError):select_pretrained_state(source,target)

def test_actual_data_and_masks():
    base=Path(os.environ.get('S1B_PROJECT',Path(__file__).resolve().parents[3]))
    root=base/'data/astribot_s1b_real_joint10_v1'
    c=base/'training/s1b_real/dataset_contract.json'
    train=S1BRealDataset(root,c,split='train');val=S1BRealDataset(root,c,split='val')
    assert len(train)==18541 and len(val)==3637
    assert not set(train.source_episode_ids)&set(val.source_episode_ids)
    assert train.normalization['action']['mean'][7]==0 and train.normalization['action']['scale'][7]==1
    for ds in [train,val]:
        for i in [0,len(ds)//2,len(ds)-1]:
            sample=ds[i]; assert sample['state'].shape==(1,10) and sample['action'].shape==(7,10)
            assert sample['video'].shape==(2,8,224,224,3) and not sample['action_is_pad'].any()
            assert np.isfinite(sample['action']).all() and np.isin(sample['action'][:,7],[0,1]).all()
            np.testing.assert_array_equal(sample['action'][:,8:],np.zeros((7,2)))
            a=sample['dataset_index'];np.testing.assert_array_equal(sample['action'],ds._actions[a:a+7])
        anchors=validation_anchors(ds,3)
        assert all(frame in ds.valid_frames[sid] for sid,frame in anchors)
        sid=ds.source_episode_ids[0]
        with pytest.raises(ValueError):ds.get_frame(sid,ds.episode_lengths[sid]-1)
        ds.close()
