from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from isaacsim import SimulationApp
app=SimulationApp(dict(headless=True,create_new_stage=False,active_gpu=0,physics_gpu=0,multi_gpu=False,
    limit_cpu_threads=8,renderer='RayTracedLighting',anti_aliasing=3,width=960,height=600,
    disable_viewport_updates=True,extra_args=['--/app/asyncRendering=false','--/rtx/ecoMode/enabled=false',
    '--/validate/p2p/enabled=false','--/validate/iommu/enabled=false','--/app/extensions/fsWatcherEnabled=false']))
runtime=None
try:
    from astribot_closedloop_runtime import LiveRuntime as Runtime
    from pressb.astribot_collection.contract import atomic_json
    from pressb.astribot_validation import initial_overlap_check
    from PIL import Image
    output=ROOT/'outputs/astribot_closedloop_20261018/adapter_preflight_v4'
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'docs/vib820_experiment_spec/config_v1_2.json').read_text())
    data=json.loads((ROOT/'models/astribot_head_right_joint8_ft_20261018/data_contract.json').read_text())
    condition=data['scene_conditions']['1']
    cfg.update(robot_base_x=condition['x_m'],robot_base_y=condition['y_m'],robot_base_yaw_rad=condition['yaw_rad'])
    runtime=Runtime(app,cfg,output/'shared',1,0,'gpu',1,'individual')
    runtime.prepare_wave([cfg],[output/'candidate_000001'])
    env=runtime.envs[0]
    import numpy as np
    import omni.physx
    from isaacsim.core.utils.types import ArticulationAction
    initial=runtime.targets[0].copy()
    env.robot.set_joints_default_state(initial.copy(),np.zeros_like(initial))
    env.robot.set_joint_positions(initial.copy())
    env.robot.set_joint_velocities(np.zeros_like(initial))
    env.robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=initial.copy()))
    runtime.world.physics_sim_view.update_articulations_kinematic()
    omni.physx.get_physx_interface().update_transformations(False,True,True)
    overlap=initial_overlap_check(runtime.world.stage,env.meta,floor_path=env.prefix+'/Environment/Floor')
    runtime.view.set_joint_position_targets(runtime.tensor(runtime.targets))
    runtime.world.step(render=False);runtime.refresh()
    from omni.physxfabric import get_physx_fabric_interface
    get_physx_fabric_interface().force_update(1/120,float(runtime.world.current_time))
    images=runtime.render()[0]
    for name,image in images.items():Image.fromarray(image).save(output/f'initial_{name}.png')
    result=dict(overlap=overlap,backend=runtime.actual_backend,q32=runtime.q[0].tolist(),
                images={k:list(v.shape) for k,v in images.items()},simulation_time=runtime.world.current_time)
    from pxr import UsdGeom
    result['physical_wrist']=env.wrist.get_world_pose()[0].tolist()
    result['usd_wrist']=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(runtime.world.stage.GetPrimAtPath(env.meta['wrist_links']['right']))).T.tolist()
    result['camera_params']=[]
    for (_,_,_,entries),ann in zip(runtime.products,runtime.camera_param_annotators):
        native=ann.get_data()
        result['camera_params'].append(dict(names=[n for e,n in entries],keys=list(native),
            view=np.asarray(native.get('cameraViewTransform')).tolist(),
            usd=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(runtime.world.stage.GetPrimAtPath(entries[0][0].camera_paths[entries[0][1]]))).T.tolist()))
    atomic_json(output/'result.json',result)
    print(json.dumps(result),flush=True)
finally:
    if runtime is not None:runtime.close()
    app.app.post_quit();app.close()
