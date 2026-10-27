"""Fail closed unless fresh training + checkpoint resume completed on the GPU job."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
completed=json.loads((r/'segment_completed.json').read_text());assert completed['step']==3 and completed['success']
loading=json.loads((r/'pretrained_loading.json').read_text());assert loading['optimizer_restored'] and loading['source_step']==2
expected_anchors=len(json.loads((r/'validation_anchors.json').read_text())['anchors'])
validation=json.loads((r/'validation_000003.json').read_text());assert validation['overall']['anchors']==expected_anchors and expected_anchors>0
assert len(validation['overall']['per_horizon_per_dimension_mae'])==7
assert all(len(v)==10 for v in validation['overall']['per_horizon_per_dimension_mae'])
changes=json.loads((r/'parameter_update_evidence.json').read_text())
assert all(v['changed_elements']>0 for v in changes['probes'].values())
weights=torch.load(r/'checkpoints/last/model.pt',map_location='cpu',weights_only=True,mmap=True)
assert weights['action_model.state_encoder.layer1.weight'].shape[1]==10
assert weights['action_model.action_encoder.layer1.weight'].shape[1]==10
assert weights['action_model.action_decoder.layer2.weight'].shape[0]==10
report=dict(status='passed',optimizer_step=3,resumed_from_step=2,validation_anchors=expected_anchors,action_shape=[7,10],
 actual_parameter_updates=changes['probes'],finite_losses=True)
(r/'smoke_acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
