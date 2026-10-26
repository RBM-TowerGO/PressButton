#!/usr/bin/env python3
"""Prepare a private head/right sensor runtime without changing system files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import yaml

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Output directory must not exist')
    config=source/'config/camera_config/camera_config.yaml'
    original=yaml.safe_load(config.read_text())
    selected=[row for row in original['camera_config_list'] if row['id'] in (6,9)]
    if {row['id'] for row in selected}!={6,9}:raise ValueError('Unrecognized vendor camera ids')
    if {row['location'] for row in selected}!={'head_rgbd','right_wrist_rgbd_color'}:raise ValueError('Unrecognized camera mapping')
    baseline={str(p.relative_to(source)):sha(p) for p in sorted((source/'config').rglob('*')) if p.is_file()}
    for name in ('astribot_camera_main','astribot_camera_sub'):
        p=source/'lib/astribot_camera'/name;baseline[str(p.relative_to(source))]=sha(p)
    (output/'lib/astribot_camera').mkdir(parents=True)
    shutil.copytree(source/'config',output/'config')
    for name in ('astribot_camera_main','astribot_camera_sub'):
        shutil.copy2(source/'lib/astribot_camera'/name,output/'lib/astribot_camera'/name)
    original['camera_config_list']=selected
    (output/'config/camera_config/camera_config.yaml').write_text(yaml.safe_dump(original,sort_keys=False))
    # Keep head RGBD CameraType=3 and vendor communication entries intact: the
    # installed driver expects its depth and publisher slots even for RGB use.
    result=dict(source=str(source),output=str(output),baseline_sha256=baseline,selected=selected,head_rgbd_and_communication_config_preserved=True,resolved_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'preparation_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='/opt/astribot_ros/software/astribot_camera');p.add_argument('--output',required=True);a=p.parse_args()
    result=prepare(a.source,a.output);print(json.dumps(dict(output=result['output'],selected=result['selected'])))
if __name__=='__main__':main()
