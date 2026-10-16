"""Raw acceptance: independent physics, all-camera decode, timing, visible light."""
from pathlib import Path
import json
import subprocess
import numpy as np
from .contract import CAMERAS, atomic_json, training_arrays
from .video import decode_audit

def frame_at(path,index,size):
    # Decode from the beginning to avoid keyframe/seek rounding at the cut.
    data=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-filter_threads','1','-i',str(path),'-vf',f'select=eq(n\\,{index})',
        '-frames:v','1','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-'])
    return np.frombuffer(data,np.uint8).reshape(size[1],size[0],3)

def visible_lamp(directory,arrays):
    directory=Path(directory)
    calibration=json.loads((directory/'calibration.json').read_text())
    uv=np.asarray(calibration['cameras']['panel']['light_ring_pixels'],int)
    if uv.shape!=(128,2) or (uv<0).any() or (uv[:,0]>=640).any() or (uv[:,1]>=480).any():
        raise ValueError('Panel validation ring lies outside camera')
    uv=np.unique(uv,axis=0)
    end=arrays['terminal_video_frame']
    before=frame_at(directory/'panel.mp4',max(0,end-1),CAMERAS['panel'])
    after=frame_at(directory/'panel.mp4',end,CAMERAS['panel'])
    # A visible arc is enough when the gripper occludes part of the ring.
    # Check decoded evidence (not the light-state boolean or USD material).
    dark=before[uv[:,1],uv[:,0]].astype(float).mean(axis=1)
    lit=after[uv[:,1],uv[:,0]].astype(float).mean(axis=1)
    changed=(lit-dark>=20)&(lit>=100)
    result=dict(status='passed' if changed.sum()>=8 else 'failed',method='projected_fixed_light_ring_decoded_RGB',
                previous_frame=max(0,end-1),terminal_frame=end,brightened_ring_samples=int(changed.sum()),
                threshold_delta_rgb=20,threshold_absolute_rgb=100,required_samples=8,
                p90_before=float(np.percentile(dark,90)),p90_terminal=float(np.percentile(lit,90)))
    atomic_json(directory/'lamp_validation.json',result)
    return result

def audit_raw(directory):
    from audit_astribot_press import audit as physics_audit
    directory=Path(directory);report=json.loads((directory/'report.json').read_text())
    from .assets import validate_assets
    validate_assets(report.get('shared_assets',{}))
    physics=physics_audit(directory)
    if physics['status']!='passed':raise ValueError('Independent physical audit failed')
    with np.load(directory/'trajectory.npz',allow_pickle=False) as src:trace={k:src[k] for k in src.files}
    timing=json.loads((directory/'camera_timing.json').read_text())
    arrays=training_arrays(trace,timing,report['joint_names'])
    calibration=json.loads((directory/'calibration.json').read_text())['cameras']
    if set(calibration)!=set(CAMERAS):raise ValueError('Incomplete camera calibration')
    for name,size in CAMERAS.items():
        c=calibration[name];k=np.asarray(c['intrinsics'])
        if tuple(c['resolution'])!=size or k.shape!=(3,3) or not np.isfinite(k).all():raise ValueError('Invalid intrinsics')
        if not c.get('native_projection_verified'):raise ValueError('Camera projection was not verified against renderer')
        if not np.isclose(k[0,0],k[1,1],rtol=1e-6):raise ValueError('Camera differs from verified square-pixel projection')
    if not all(np.isfinite(trace[k]).all() for k in ('q','qd','target','tip','base','base_quat','force')):
        raise ValueError('Nonfinite raw data')
    videos={n:decode_audit(directory/f'{n}.mp4',len(timing),size) for n,size in CAMERAS.items()}
    if set(report['video_frames'])!=set(CAMERAS) or any(c!=len(timing) for c in report['video_frames'].values()):
        raise ValueError('Camera count mismatch')
    lamp=visible_lamp(directory,arrays)
    result=dict(status='passed' if lamp['status']=='passed' else 'failed',physics=physics,videos=videos,lamp=lamp,
                training_frames=len(arrays['state']),terminal_video_frame=arrays['terminal_video_frame'],trigger_step=arrays['trigger_step'])
    atomic_json(directory/'collection_audit.json',result)
    if result['status']!='passed': raise ValueError('First success frame has no verified visible lamp')
    np.savez_compressed(directory/'training.npz',**arrays)
    return result
