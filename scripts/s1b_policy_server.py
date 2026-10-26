#!/usr/bin/env python3
"""Serve the verified S1B joint10 model on loopback."""
import argparse
import sys
import time
import subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.contract import read_config
from pressb.s1b_deployment.policy import Policy
from pressb.s1b_deployment.server import make_server
from pressb.s1b_deployment.recorder import Recorder

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/s1b_deployment_v1.yaml')
    p.add_argument('--run')
    p.add_argument('--checkpoint',type=int,choices=(2000,8000),required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=8796)
    a=p.parse_args(); c=read_config(a.config)
    policy=Policy(a.run or c['model']['run'],a.checkpoint,c['model']['identity'],a.device)
    recorder=Recorder(Path(c['logging']['root'])/f'server_{a.checkpoint}_{time.time_ns()}',c,policy.identity)
    recorder.append('startup', code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), sdk='not imported', device=str(policy.device), load_seconds=policy.load_seconds)
    with make_server(policy,a.port,recorder,tcp_nodelay=c['network'].get('tcp_nodelay',True)) as server:
        print(f'ready: http://127.0.0.1:{server.server_port}',flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
if __name__=='__main__': main()
