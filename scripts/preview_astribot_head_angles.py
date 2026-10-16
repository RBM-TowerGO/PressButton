#!/usr/bin/env python3
"""Record independent held head-pitch poses, retaining all seven camera streams."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def run(args, report):
    import numpy as np
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.astribot_asset import add_astribot
    from pressb.astribot_kinematics import AstribotKinematics
    from pressb.astribot_press import prepare_hand, complete_collision_geometry
    from pressb.astribot_sensors import create_robot_cameras, DEFAULT_CONFIG
    from pressb.vib820_scene import build_scene

    cfg = json.loads(args.config.read_text())
    sensor_cfg = json.loads(DEFAULT_CONFIG.read_text())
    kin = AstribotKinematics()
    pitch_joint = 'astribot_head_joint_2'
    lo, hi, _ = kin.limits[pitch_joint]
    if any(not np.isfinite(a) or not lo <= np.radians(a) <= hi for a in args.angles):
        raise ValueError('Requested head pitch exceeds URDF joint limits')
    hz, fps = cfg['physics_hz'], cfg['capture_hz']
    assert hz % fps == 0
    world = World(stage_units_in_meters=1., physics_dt=1/hz, rendering_dt=1/fps,
                  backend='numpy', device='cpu')
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    scene = build_scene(world.stage, cfg)
    meta = add_astribot(world.stage)
    base = Gf.Vec3d(cfg['robot_base_x'],cfg['robot_base_y'],meta['base_lift_m'])
    world.stage.GetPrimAtPath('/World/Astribot').GetAttribute('xformOp:translate').Set(base)
    UsdPhysics.FixedJoint(world.stage.GetPrimAtPath(meta['articulation_path'])).CreateLocalPos0Attr(Gf.Vec3f(*base))
    prepare_hand(world.stage,meta)
    complete_collision_geometry(world.stage,meta)
    robot = world.scene.add(SingleArticulation(prim_path=meta['robot_path'],name='astribot'))
    cameras, paths = {}, {}
    for name,key,size in [('overview','camera_path',(960,600)),('panel','panel_camera_path',(640,480)),
                          ('front','front_camera_path',(640,480)),('lobby','lobby_camera_path',(640,480))]:
        paths[name]=scene[key]
        cameras[name]=Camera(prim_path=scene[key],name=name,resolution=size,frequency=-1)
    robot_cameras, robot_paths = create_robot_cameras(world.stage,meta['robot_path'],sensor_cfg)
    cameras.update(robot_cameras)
    paths.update({name:robot_paths[name+'_camera'] for name in robot_cameras})
    report.update(config=cfg,robot_camera_config=sensor_cfg,camera_paths=paths,
                  base_xyz_m=list(base),head_joint=pitch_joint,joint_limits_rad=[lo,hi],angles=[])
    world.reset()
    for camera in cameras.values():
        camera.initialize()
    names=list(robot.dof_names)
    pitch_index=names.index(pitch_joint)
    target=np.zeros(len(names),dtype=np.float32)
    for name,value in cfg['initial_arm_positions'].items():
        target[names.index(name)]=value
    controller=robot.get_articulation_controller()
    corners=np.array([[cfg['button_face_x'],cfg['panel_center_y']+y,z,1]
                      for y in [-cfg['panel_width']/2,cfg['panel_width']/2]
                      for z in [scene['dimensions']['panel_bottom'],scene['dimensions']['panel_top']]])

    def render_sync():
        now=float(world.current_time)
        for _ in range(16):
            world.render()
            if abs(float(world.current_time)-now)>1e-7:
                raise RuntimeError('Rendering advanced physics')
            if all(abs(float(c.get_current_frame().get('rendering_time',-1))-now)<1e-6 for c in cameras.values()):
                return now
        raise RuntimeError('Stale camera frame')

    for angle in args.angles:
        folder=args.output/f'pitch_{angle:02g}deg'
        folder.mkdir()
        target[pitch_index]=np.radians(angle)
        robot.set_joint_positions(target.copy())
        robot.set_joint_velocities(np.zeros_like(target))
        controller.apply_action(ArticulationAction(joint_positions=target.copy()))
        # Each angle is an independent held pose; initialization is not recorded.
        for _ in range(hz):
            controller.apply_action(ArticulationAction(joint_positions=target.copy()))
            world.step(render=False)
        render_sync()
        timing, states, times, writers=[],[],[],{}
        try:
            for step in range(round(args.seconds*hz)):
                controller.apply_action(ArticulationAction(joint_positions=target.copy()))
                world.step(render=False)
                q=robot.get_joint_positions().copy()
                if not np.isfinite(q).all() or np.max(np.abs(q-target))>=.025:
                    raise RuntimeError('Failed to hold requested posture')
                states.append(q);times.append(float(world.current_time))
                if (step+1)%(hz//fps):
                    continue
                now=render_sync()
                row=dict(simulation_time=now,physics_step=step+1,cameras={})
                for name,camera in cameras.items():
                    rgba=camera.get_rgba()
                    if rgba is None or not rgba.size:
                        raise RuntimeError(f'Missing image: {name}')
                    rgb=np.ascontiguousarray(rgba[:,:,:3],dtype=np.uint8)
                    if name not in writers:
                        Image.fromarray(rgb).save(folder/f'{name}.png')
                        writers[name]=subprocess.Popen(['ffmpeg','-n','-v','error','-f','rawvideo',
                            '-pixel_format','rgb24','-video_size',f'{rgb.shape[1]}x{rgb.shape[0]}',
                            '-framerate',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
                            '-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',
                            str(folder/f'{name}.mp4')],stdin=subprocess.PIPE)
                    writers[name].stdin.write(rgb.tobytes())
                    row['cameras'][name]=float(camera.get_current_frame()['rendering_time'])
                timing.append(row)
        finally:
            for writer in writers.values():
                writer.stdin.close()
            codes=[writer.wait() for writer in writers.values()]
            if any(codes):
                raise RuntimeError(f'Video encoder failure: {codes}')
        T=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(world.stage.GetPrimAtPath(paths['head']))).T
        prim=UsdGeom.Camera.Get(world.stage,paths['head'])
        fy=720*prim.GetFocalLengthAttr().Get()/prim.GetVerticalApertureAttr().Get()
        points=corners@np.linalg.inv(T).T
        uv=np.c_[640+fy*points[:,0]/(-points[:,2]),360-fy*points[:,1]/(-points[:,2])]
        inside=bool((points[:,2]<0).all() and (uv>=0).all() and (uv<[1280,720]).all())
        item=dict(requested_pitch_deg=angle,actual_pitch_deg=float(np.degrees(q[pitch_index])),
                  max_joint_error_rad=float(np.max(np.abs(np.array(states)-target))),
                  frames_per_camera=len(timing),fps=fps,duration_s=len(timing)/fps,
                  head_world_from_camera=T.tolist(),panel_corner_pixels=uv.tolist(),
                  panel_corners_inside_frame=inside,
                  coverage_scope='Projection of faceplate corners; inspect images for occlusion',
                  max_camera_physics_lag_s=max(abs(v-row['simulation_time']) for row in timing for v in row['cameras'].values()))
        np.savez_compressed(folder/'trajectory.npz',q=states,target=target,time=times,joint_names=names)
        (folder/'camera_timing.json').write_text(json.dumps(timing,indent=2)+'\n')
        (folder/'report.json').write_text(json.dumps(item,indent=2)+'\n')
        report['angles'].append(item)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('ANGLE',json.dumps(item),flush=True)
    report['status']='passed'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/astribot_vib820_ready.json')
    parser.add_argument('--angles',type=float,nargs='+',default=[0,15,30,45])
    parser.add_argument('--seconds',type=float,default=3.)
    parser.add_argument('--gpu',type=int,default=1)
    args=parser.parse_args()
    if args.seconds<=0 or len(args.angles)!=len(set(args.angles)):
        parser.error('Positive duration and unique angles are required')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='Independent held-pose camera previews, no pressing or full collision audit',
                gpu=args.gpu,physics_hz=120,capture_hz=30)
    app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,active_gpu=args.gpu,physics_gpu=args.gpu,multi_gpu=False,
            limit_cpu_threads=8,renderer='RayTracedLighting',extra_args=['--/app/asyncRendering=false',
            '--/rtx/ecoMode/enabled=false','--/validate/p2p/enabled=false','--/validate/iommu/enabled=false']))
        run(args,report)
    except BaseException as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}')
        traceback.print_exc()
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if app:
            app.app.post_quit(0 if report['status']=='passed' else 1);app.close()


if __name__=='__main__':
    main()
