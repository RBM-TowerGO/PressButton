"""Evaluate live Astribot observations in Isaac Sim; never load demo actions."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from astribot_closedloop_control import validate_prediction, interpolate_target, physical_hit, FIXED


class TrialStop(Exception):
    pass


def predict(endpoint, images, state, seed):
    payload=io.BytesIO()
    np.savez(payload,head=images['head'],wrist_right=images['wrist_right'],state=state,seed=np.array(seed))
    req=urllib.request.Request(endpoint+'/predict',data=payload.getvalue(),headers={'Content-Type':'application/octet-stream'})
    started=time.perf_counter()
    with urllib.request.urlopen(req,timeout=300) as response:
        with np.load(io.BytesIO(response.read()),allow_pickle=False) as result:
            output={key:result[key].copy() for key in result.files}
    output['rpc_seconds']=time.perf_counter()-started
    return output


def run_trial(runtime,cfg,out,candidate_id,args,contract):
    from PIL import Image
    from pressb.astribot_collection.contract import CAMERAS, atomic_json
    from astribot_eval_video import ObservationVideo
    from pressb.astribot_collection.lights import set_button_light
    from pressb.astribot_validation import initial_overlap_check
    started=time.perf_counter()
    runtime.prepare_wave([cfg],[out])
    env=runtime.envs[0];world=runtime.world
    names=env.robot.dof_names
    limits=np.c_[env.robot.dof_properties['lower'],env.robot.dof_properties['upper']]
    fixed=np.asarray(contract['fixed_action32'],dtype=np.float32)
    command=runtime.targets[0].copy()
    np.testing.assert_allclose(command[FIXED],fixed[FIXED],atol=1e-7,rtol=0)
    button=env.scene['buttons'][0]
    tips={env.meta['robot_path']+'/astribot_gripper_right_Link_'+s for s in ('L11','R11')}
    forces=[0.];unexpected=[];records=[];requests=[];writers={};camera_times=[]
    physics_step=0;light=False;hit_step=None;images=runtime.last_images[0]
    base_initial=env.base_body.get_world_pose()[0].copy()
    report=dict(candidate_id=candidate_id,status='running',success=False,scene_condition=contract['scene_conditions'][str(candidate_id)],
                model_sha256=args.model_sha256,execute_horizon=args.execute_horizon,
                max_seconds=args.max_seconds,expert_actions_used=False,initial_expert_motion=False,
                task=contract['schema']['task'],physics_hz=120,action_hz=30,
                policy_inputs=['head RGB','right wrist RGB','measured joint positions','task text'],
                environment_uses_button_geometry_only_for_physics_and_scoring=True)

    def contact(headers,data,actor_paths):
        for h in headers:
            actors=[actor_paths[x] for x in (h.actor0,h.actor1)]
            force=sum(float(np.linalg.norm(data[i].impulse))*120 for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data))
            if button['body_path'] in actors and any(p in tips for p in actors):
                forces[0]+=force
                continue
            if force>.1 and any(p.startswith(env.meta['robot_path']+'/') for p in actors) and not all(p in tips for p in actors):
                if len(unexpected)<500:unexpected.append(dict(step=physics_step,actors=actors,force_n=force))

    subscription=env.subscribe_contacts(contact)

    def capture(label=None):
        nonlocal images
        images=runtime.render()[0]
        camera_times.append(dict(physics_step=physics_step,time=float(world.current_time),cameras=runtime.camera_times[0].copy(),
                                 pose_sync_errors=runtime.last_pose_sync.copy()))
        for name in ('head','wrist_right','panel','overview'):
            if name not in writers: writers[name]=ObservationVideo(out/f'{name}.mp4',CAMERAS[name],args.execute_horizon)
            writers[name].write(images[name],physics_step)
            if label:Image.fromarray(images[name]).save(out/f'{label}_{name}.png')

    def tick(target):
        nonlocal command,physics_step,light,hit_step
        forces[0]=0.
        runtime.targets[0]=target
        runtime.view.set_joint_position_targets(runtime.tensor(runtime.targets))
        previous_time=float(world.current_time)
        world.step(render=False);runtime.refresh()
        if abs(float(world.current_time)-previous_time-1/120)>2e-6:
            raise RuntimeError('Physics timestep differs from 120 Hz contract')
        command=np.asarray(target,dtype=np.float32).copy();physics_step+=1
        q=runtime.q[0].copy()
        travel=float(env.buttons[0].get_world_pose()[0][0]-button['rest_x'])
        hit=physical_hit(travel,forces[0],cfg)
        if hit and hit_step is None:hit_step=physics_step
        if hit and not light:
            light=True;set_button_light(world.stage,button,True)
        actual_light=bool(world.stage.GetPrimAtPath(button['body_path']).GetAttribute('pressb:illuminated').Get())
        base=env.base_body.get_world_pose()[0].copy()
        records.append(dict(q=q,qd=runtime.qd[0].copy(),target=command.copy(),travel=travel,
                            force=forces[0],light=actual_light,physical_hit=hit,base=base,
                            wrist_position=env.wrist.get_world_pose()[0].copy(),
                            time=float(world.current_time),request_index=len(requests)-1))
        if not np.isfinite(q).all() or np.any(q<limits[:,0]-.02) or np.any(q>limits[:,1]+.02):
            raise TrialStop('nonfinite_or_out_of_limit_state')
        if unexpected:raise TrialStop('unexpected_contact')
        if float(np.linalg.norm(base-base_initial))>.001:raise TrialStop('base_drift_over_1mm')
        if np.max(np.abs(q[FIXED]-fixed[FIXED]))>.02:raise TrialStop('inactive_joint_drift_over_0.02rad')
        if light != actual_light:raise RuntimeError('Button light state mismatch')

    infrastructure_error=None
    try:
        overlap=initial_overlap_check(world.stage,env.meta,floor_path=env.prefix+'/Environment/Floor')
        report['initial_overlap']=overlap
        if overlap['status']!='clear':raise TrialStop('initial_overlap_'+overlap['status'])
        # Source frame zero was observed after one fixed initial physics tick.
        tick(command);capture('initial')
        report['initial_q32']=runtime.q[0].tolist()
        # Simulation time is frozen during model inference (synchronous control).
        action_index=0;request_index=0
        while action_index<round(args.max_seconds*30):
            state=runtime.q[0].copy();observation_time=float(world.current_time)
            seed=420000+candidate_id*100003+request_index
            result=predict(args.endpoint,images,state,seed)
            if world.current_time!=observation_time or not np.array_equal(runtime.q[0],state):
                raise RuntimeError('Physics advanced during inference')
            if args.smoke and request_index==0:
                repeated=predict(args.endpoint,images,state,seed)
                np.testing.assert_array_equal(result['action32'],repeated['action32'])
                report['same_input_seed_repeat_exact']=True
            try:
                chunk=validate_prediction(result['action8'],result['action32'],fixed,limits)
            except (ValueError,AssertionError) as exc:
                report['invalid_prediction']=dict(error=str(exc),action8=result['action8'].tolist(),action32=result['action32'].tolist())
                raise TrialStop('invalid_policy_command') from exc
            requests.append(dict(request_index=request_index,physics_step=physics_step,observation_time=observation_time,
                seed=seed,state32=state.tolist(),action8=result['action8'].tolist(),action32=chunk.tolist(),
                inference_seconds=float(result['inference_seconds']),rpc_seconds=float(result['rpc_seconds']),
                head_sha256=hashlib.sha256(images['head'].tobytes()).hexdigest(),
                wrist_right_sha256=hashlib.sha256(images['wrist_right'].tobytes()).hexdigest()))
            for target in chunk[:args.execute_horizon]:
                if action_index>=round(args.max_seconds*30):break
                for interpolated in interpolate_target(command,target):tick(interpolated)
                action_index+=1
                if hit_step is not None:
                    report.update(status='success',success=True,termination_reason='physical_press_and_light')
                    break
            capture()
            if report['success']:break
            request_index+=1
            if request_index%20==0:
                print(json.dumps(dict(candidate=candidate_id,actions=action_index,simulation_seconds=action_index/30,
                                      max_travel_m=max(r['travel'] for r in records))),flush=True)
        if not report['success']:report.update(status='failed',termination_reason='timeout')
    except TrialStop as exc:
        report.update(status='failed',termination_reason=str(exc))
    except BaseException:
        infrastructure_error=traceback.format_exc()
        report.update(status='infrastructure_error',termination_reason='infrastructure_error',traceback=infrastructure_error)
    finally:
        runtime.contact_router.callbacks.pop(0,None)
        del subscription
        try:
            capture('final')
            for writer in writers.values():writer.close()
        except Exception:
            report['video_error']=traceback.format_exc()
            infrastructure_error=infrastructure_error or report['video_error']
            report.update(status='infrastructure_error',success=False)
        report.update(physics_steps=physics_step,first_hit_step=hit_step,unexpected_contacts=unexpected,
            policy_requests=len(requests),wall_seconds=time.perf_counter()-started,
            simulated_seconds=max(0,physics_step-1)/120,
            frames={k:w.frames for k,w in writers.items()},
            max_travel_m=max((r['travel'] for r in records),default=0.),
            max_force_n=max((r['force'] for r in records),default=0.),
            final_light=bool(records[-1]['light']) if records else False)
        if requests:
            latency=[r['inference_seconds'] for r in requests]
            report['inference_latency_seconds']=dict(median=float(np.median(latency)),p95=float(np.percentile(latency,95)),max=max(latency))
        if records:
            np.savez_compressed(out/'trajectory.npz',**{k:np.asarray([r[k] for r in records]) for k in records[0]})
        atomic_json(out/'policy_requests.json',requests)
        atomic_json(out/'camera_timing.json',camera_times)
        atomic_json(out/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('initial_overlap','initial_q32','unexpected_contacts','traceback')}),flush=True)
    if infrastructure_error:raise RuntimeError(infrastructure_error)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'docs/vib820_experiment_spec/config_v1_2.json')
    p.add_argument('--endpoint',default='http://127.0.0.1:8794')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--execute-horizon',type=int,default=7,choices=range(1,8))
    p.add_argument('--max-seconds',type=float,default=40.)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    contract=json.loads((a.model_run/'data_contract.json').read_text())
    manifest=json.loads((a.model_run/'checkpoints/step_008000/manifest.json').read_text())
    a.model_sha256=manifest['model_sha256']
    with urllib.request.urlopen(a.endpoint+'/health',timeout=30) as response:health=json.load(response)
    if not health['ready'] or health['model_sha256']!=a.model_sha256 or health['loaded_tensors']!=1617:
        raise RuntimeError('Wrong model server identity')
    ids=contract['split']['val_source_episode_ids'][:1] if a.smoke else contract['split']['test_source_episode_ids']
    a.output.mkdir(parents=True,exist_ok=False)
    from pressb.astribot_collection.contract import atomic_json,sha256
    protocol=dict(kind='astribot_live_closedloop_v1',smoke=a.smoke,candidate_ids=ids,split='val' if a.smoke else 'test',
        execute_horizon=a.execute_horizon,action_hz=30,physics_hz=120,max_seconds=a.max_seconds,
        observation_hz=30/a.execute_horizon,video='policy observations with physical timestamps; no interpolated or repeated frames',
        inference=health,model_sha256=a.model_sha256,data_contract_sha256=sha256(a.model_run/'data_contract.json'),
        scene_config_sha256=sha256(a.config),source_sha256={str(Path(__file__).name):sha256(__file__),
            'astribot_closedloop_control.py':sha256(Path(__file__).with_name('astribot_closedloop_control.py')),
            'astribot_policy_server.py':sha256(Path(__file__).with_name('astribot_policy_server.py')),
            'astribot_closedloop_runtime.py':sha256(Path(__file__).with_name('astribot_closedloop_runtime.py')),
            'astribot_eval_video.py':sha256(Path(__file__).with_name('astribot_eval_video.py')),
            'astribot_collection/runtime.py':sha256(ROOT/'src/pressb/astribot_collection/runtime.py'),
            'astribot_collection/lights.py':sha256(ROOT/'src/pressb/astribot_collection/lights.py')},
        control='absolute joint8 expanded to joint32; 4-tick linear interpolation between commanded targets',
        initial_state='configured raised arms and open gripper; one fixed physics tick; no expert lead-in',
        success='simultaneous button travel >= 0.001 m and fingertip force > 0.02 N, latched lamp, no unexpected contact',
        synchronous_simulation=True,real_time_claim=False,no_demo_action_access=True,
        note='All 40 held-out conditions are predeclared; no policy weights or normalization fitted on test outcomes.')
    atomic_json(a.output/'protocol.json',protocol)
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,create_new_stage=False,active_gpu=a.gpu,physics_gpu=a.gpu,
        multi_gpu=False,limit_cpu_threads=8,renderer='RayTracedLighting',anti_aliasing=3,width=960,height=600,
        disable_viewport_updates=True,extra_args=['--/app/asyncRendering=false','--/rtx/ecoMode/enabled=false',
        '--/validate/p2p/enabled=false','--/validate/iommu/enabled=false','--/app/extensions/fsWatcherEnabled=false']))
    runtime=None;results=[]
    try:
        from astribot_closedloop_runtime import LiveRuntime as Runtime
        cfg=json.loads(a.config.read_text())
        runtime=Runtime(app,cfg,a.output/'shared',1,a.gpu,'gpu',1,'individual')
        for candidate in ids:
            c=deepcopy(cfg);condition=contract['scene_conditions'][str(candidate)]
            c.update(robot_base_x=condition['x_m'],robot_base_y=condition['y_m'],robot_base_yaw_rad=condition['yaw_rad'])
            report=run_trial(runtime,c,a.output/f'candidate_{candidate:06d}',candidate,a,contract)
            results.append(report)
            atomic_json(a.output/'summary.json',dict(complete=len(results)==len(ids),requested=len(ids),completed=len(results),
                successes=sum(r['success'] for r in results),success_rate=sum(r['success'] for r in results)/len(results),
                results=[{k:r[k] for k in ('candidate_id','status','success','termination_reason','simulated_seconds','policy_requests','max_travel_m','max_force_n','wall_seconds')} for r in results]))
    except BaseException:
        atomic_json(a.output/'infrastructure_error.json',dict(error=traceback.format_exc(),completed=len(results)))
        raise
    finally:
        if runtime is not None:runtime.close()
        app.app.post_quit();app.close()


if __name__=='__main__':
    main()
