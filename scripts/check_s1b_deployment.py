#!/usr/bin/env python3
"""Audit P0 without importing CUDA models, ROS, or the robot SDK."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.contract import read_config,verify,sha,canonical,JOINTS,TASK

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/s1b_deployment_v1.yaml')
    p.add_argument('--mode',choices=['shadow','execute'],default='shadow')
    p.add_argument('--output',required=True)
    a=p.parse_args();c=read_config(a.config,a.mode)
    identities=[verify(c['model']['run'],s,c['model']['identity']) for s in (2000,8000)]
    result=dict(status='passed',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=c,config_sha256=canonical(c),checkpoints=identities,joints=JOINTS,task=TASK,action_semantics='absolute next-grid targets; no extra shift; model head output retained',gripper=dict(state='continuous SDK closure; no clamp',action='binary identity normalization',sdk_threshold='raw <0.5 ->0; >=0.5 ->100; executor only'),sdk=dict(archived_commit='2644cdfba47f3dd311d923b172ac6c24909f6fda',source='docs/astribot/s1b_contract_20261025/README.md',local_sdk_imported=False,robot_connected=False,current_robot_sdk_version_verified=False),execute_enabled=False)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('P0 passed; execute remains disabled')
if __name__=='__main__':main()
