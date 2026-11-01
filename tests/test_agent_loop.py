import base64
import copy
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pressb.agent_loop.client import build_request, parse_proposal, proposal_still_valid, RobotClient
from pressb.agent_loop.common import pose_error, request
from pressb.agent_loop.service import Service, make_server


class FakeBackend:
    error=None
    def __init__(self):
        self.pose=[.4,-.2,1.,0.,0.,0.,1.];self.calls=[];self.stopped=False;self.block=None
    def snapshot(self):
        return dict(pose_m_xyzw=list(self.pose),receipt_ns=time.monotonic_ns(),joints={})
    def move(self,kind,data,cancel,phase):
        self.calls.append((kind,data));phase('MOVING' if kind=='move' else 'EXTENDING')
        if self.block:
            while not self.block.is_set():
                if cancel.wait(.01):raise RuntimeError('stopped')
        if kind=='move':self.pose[1]-=data['right_mm']/1000;self.pose[2]+=data['up_mm']/1000
        else:phase('RETRACTING')
        return dict(returned_to_start=True if kind=='press' else None)
    def stop(self):self.stopped=True
    def close(self):self.stop()


class FakeCamera:
    def __init__(self):self.revision=0;self.fail=False
    def selection(self):return dict(revision=self.revision,point_normalized=None)
    def capture(self,after,cancel):
        if self.fail:raise ValueError('missing_depth')
        image=io.BytesIO();Image.new('RGB',(16,16),'red').save(image,format='JPEG')
        now=time.monotonic_ns()
        return dict(frames={k:dict(data_base64=base64.b64encode(v).decode(),source_ns=time.time_ns(),receipt_ns=now)
                    for k,v in dict(head=image.getvalue(),right=image.getvalue(),depth=np.full((2,2),1800,dtype='<u2').tobytes()).items()},
            depth=dict(median_m=.18,center_m=.18,valid_pixels=25),
            metadata=dict(depth=dict(height=2,width=2),scale_m=.0001),selection=self.selection())


@pytest.fixture
def setup(tmp_path):
    backend=FakeBackend();camera=FakeCamera();service=Service(backend,camera,tmp_path/'robot')
    yield service,backend,camera
    service.close()


def wait(service,receipt):
    service.worker.join(3)
    assert not service.worker.is_alive()
    return service.task(receipt['task_id'])


def observe(service):
    task=wait(service,service.submit(dict(session_id=service.session_id,command_id=str(time.time_ns())),True))
    assert task['status']=='completed',task
    return service.observations[task['capture']['observation_id']]


def action(service,obs,cid='cmd',**kwargs):
    return dict(session_id=service.session_id,observation_id=obs['observation_id'],command_id=cid,**kwargs)


def tool(name,args):
    return dict(choices=[dict(finish_reason='tool_calls',message=dict(content=None,tool_calls=[
        dict(function=dict(name=name,arguments=json.dumps(args)))]))])


def test_move_deduplication_and_fresh_observation(setup):
    service,backend,camera=setup;obs=observe(service)
    body=action(service,obs,type='move',right_mm=10,up_mm=-5)
    receipt=service.submit(body);result=wait(service,receipt)
    assert result['status']=='completed'
    assert backend.pose[:3]==pytest.approx([.4,-.21,.995])
    assert service.submit(body)==receipt and len(backend.calls)==1
    with pytest.raises(ValueError,match='conflict'):service.submit(dict(body,right_mm=2))
    with pytest.raises(ValueError,match='old_observation'):service.submit(dict(body,command_id='next'))


def test_press_formula_and_roundtrip(setup):
    service,backend,camera=setup;obs=observe(service)
    original=list(backend.pose)
    bad=action(service,obs,type='press',forward_mm=180)
    with pytest.raises(ValueError):service.submit(bad)
    with pytest.raises(ValueError,match='formula'):service.submit(dict(bad,forward_mm=20))
    receipt=service.submit(dict(bad,forward_mm=30))
    result=wait(service,receipt)
    assert result['motion']['returned_to_start'] and result['motion']['action_completed']
    assert result['motion']['button_trigger_evidence'] is None
    assert backend.pose==original


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,'3',100])
def test_invalid_move(setup,value):
    service,_,_=setup;obs=observe(service)
    with pytest.raises(ValueError):service.submit(action(service,obs,type='move',right_mm=value,up_mm=0))


def test_stop_priority_and_busy(setup):
    service,backend,_=setup;obs=observe(service);backend.block=threading.Event()
    receipt=service.submit(action(service,obs,type='move',right_mm=1,up_mm=0))
    with pytest.raises(ValueError,match='not_ready'):service.submit(action(service,obs,cid='other',type='move',right_mm=1,up_mm=0))
    service.stop();result=wait(service,receipt)
    assert backend.stopped and result['status']=='interrupted'
    with pytest.raises(ValueError):service.submit(action(service,obs,cid='after-stop',type='move',right_mm=1,up_mm=0))


def test_capture_failure_does_not_replay_motion(setup):
    service,backend,camera=setup;obs=observe(service);camera.fail=True
    body=action(service,obs,type='move',right_mm=1,up_mm=0)
    receipt=service.submit(body);result=wait(service,receipt)
    assert result['status']=='failed' and result['motion']['action_completed']
    assert service.phase=='NEEDS_OBSERVATION'
    assert service.submit(body)==receipt and len(backend.calls)==1
    camera.fail=False;observe(service)
    assert service.phase=='READY' and len(backend.calls)==1


def test_reject_changed_pose_point_and_session(setup):
    service,backend,camera=setup;obs=observe(service)
    body=action(service,obs,type='move',right_mm=1,up_mm=0)
    with pytest.raises(ValueError,match='session'):service.submit(dict(body,session_id='old'))
    backend.pose[1]+=.01
    camera.revision+=1
    with pytest.raises(ValueError,match='point_changed'):service.submit(body)
    camera.revision-=1
    assert wait(service,service.submit(body))['status']=='completed'


def test_http_client_downloads_and_hashes(setup,tmp_path):
    service,_,_=setup;server=make_server(service,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=RobotClient('http://127.0.0.1:'+str(server.server_port),tmp_path/'client')
        obs,folder=client.observe()
        assert np.load(folder/'right_depth_raw.npy').shape==(2,2)
        assert (folder/'head.jpg').exists() and (folder/'right.jpg').exists()
        proposal=dict(name='propose_move',arguments=dict(right_mm=2,up_mm=0,reason='test'))
        result,(new,_) = client.execute(proposal,obs)
        assert new['observation_id']!=obs['observation_id']
        assert result['motion']['action_completed']
        with pytest.raises(ValueError):request(client.base,'/v2/artifacts/../../session.json')
    finally:server.shutdown();server.server_close();thread.join()


def test_model_only_wrist_and_formula(setup):
    service,_,_=setup;obs=observe(service)
    wrist=next(p for p in service.artifacts.values() if p.name=='right.jpg').read_bytes()
    body=build_request('按3',obs,wrist,service.session,[],'English prompt')
    content=body['messages'][1]['content']
    assert len([v for v in content if v['type']=='image_url'])==1
    assert body['chat_template_kwargs']=={'enable_thinking':False}
    p=parse_proposal(tool('propose_press',dict(forward_mm=30,reason='aligned')),obs,service.session)
    assert p['arguments']['forward_mm']==30
    with pytest.raises(ValueError):parse_proposal(tool('propose_press',dict(forward_mm=180,reason='bad')),obs,service.session)
    with pytest.raises(ValueError):parse_proposal(tool('unknown',{}),obs,service.session)
    invalid=tool('propose_move',dict(right_mm=1,up_mm=0,reason='ok'))
    invalid['choices'][0]['message']['tool_calls']*=2
    with pytest.raises(ValueError):parse_proposal(invalid,obs,service.session)
    assert proposal_still_valid(obs,copy.deepcopy(obs),p)
    new=copy.deepcopy(obs);new['depth']['median_m']=.19
    assert not proposal_still_valid(obs,new,p)


def test_restart_journal_not_replayed(tmp_path):
    root=tmp_path/'sessions';old=root/'old'/'tasks';old.mkdir(parents=True)
    (old/'task.json').write_text(json.dumps(dict(status='running')))
    service=Service(FakeBackend(),FakeCamera(),root)
    try:assert json.loads((old/'task.json').read_text())['status']=='interrupted'
    finally:service.close()


def test_pose_quaternion_sign():
    assert pose_error([0,0,0,0,0,0,1],[0,0,0,0,0,0,-1])==(0,0)


@pytest.mark.parametrize('read_delay',[0.,.035])
def test_sdk_backend_real_interpolation_and_immediate_return(read_delay):
    from pressb.agent_loop.sdk_backend import SDKBackend
    class Robot:
        def __init__(self):
            self.pose=[.4,-.2,1.,0.,0.,0.,1.];self.torso=[0,0,0,0,0,0,1];self.commands=[];self.stopped=False
            self.cartesian_calls=0;self.hold_commands=[]
        def get_control_rights_status(self):return True
        def get_current_cartesian_pose(self,*args,**kwargs):
            assert threading.current_thread().name!='agent-cartesian-servo'
            time.sleep(read_delay)
            return [list(self.pose),list(self.torso)]
        def set_cartesian_pose(self,names,poses,**kwargs):
            assert names==['astribot_arm_right','astribot_torso']
            assert all(len(p)==7 for p in poses)
            assert poses[1]==self.torso
            assert kwargs==dict(control_way='filter',use_wbc=False,add_default_torso=False,remote_wbc_option=False)
            self.cartesian_calls+=1
            self.pose=list(poses[0]);self.commands.append((time.monotonic(),list(self.pose)))
        def set_different_type_command(self,names,types,commands,**kwargs):
            raise AssertionError('Agent motion must reuse keyboard public Cartesian API, not mixed commands')
        def set_joints_position(self,*args,**kwargs):
            raise AssertionError('Arm stream must never switch to joint hold')
        def stop_robot(self):self.stopped=True
    robot=Robot()
    class Monitor:
        error=driver_fault=None
        def snapshot(self):
            now=time.time_ns()
            return {k:dict(receipt_ns=time.monotonic_ns(),header=dict(sec=now//10**9,nanosec=now%10**9),
                position=[0]*(4 if k=='astribot_torso' else 7),velocity=[0]*7,torque=[0]*7) for k in ('astribot_arm_right','astribot_torso','astribot_gripper_right')}
    backend=SDKBackend(robot,Monitor(),100)
    try:
        assert robot.commands  # Startup has already started the Cartesian stream.
        assert robot.commands[0][1]==[.4,-.2,1.,0.,0.,0.,1.]
        time.sleep(.08)
        assert robot.cartesian_calls>1
        assert not robot.hold_commands
        phases=[];cancel=threading.Event()
        initial=list(robot.pose)
        result=backend.move('press',dict(forward_mm=2),cancel,phases.append)
        assert phases==['EXTENDING','RETRACTING','SETTLING']
        assert result['command_returned_to_start']
        assert result['returned_to_start'] is None
        assert pose_error(robot.pose,initial)[0]<.01
        assert max(p[0] for _,p in robot.commands)==pytest.approx(initial[0]+.002,abs=.0001)
        # Peak Cartesian speed, allowing scheduler sampling jitter.
        speeds=[abs(b[1][0]-a[1][0])/(b[0]-a[0]) for a,b in zip(robot.commands,robot.commands[1:])]
        assert max(speeds)<.025
        assert all(p[3:]==initial[3:] for _,p in robot.commands)
        time.sleep(.04)
        cartesian_count=robot.cartesian_calls
        time.sleep(.08)
        assert robot.cartesian_calls>cartesian_count
        start_index=len(robot.commands)
        planar=backend.move('move',dict(right_mm=0,up_mm=15),cancel,lambda p:None)
        assert planar['target_pose_m_xyzw'][:3]==pytest.approx([.4,-.2,1.015])
        assert robot.pose[:3]==pytest.approx([.4,-.2,1.015])
        if read_delay:
            # Slow state reads must not serialize command sends.
            assert len(robot.commands[start_index:])>60
        for _,p in robot.commands[start_index:]:
            assert p[0]==initial[0] and p[3:]==initial[3:]
    finally:backend.close()
    assert robot.stopped


def test_storage_failure_never_reexecutes(setup,monkeypatch):
    service,backend,_=setup;obs=observe(service)
    def broken(*args,**kwargs):raise OSError('disk_full')
    monkeypatch.setattr(service,'_save_observation',broken)
    body=action(service,obs,type='move',right_mm=1,up_mm=0)
    receipt=service.submit(body);result=wait(service,receipt)
    assert result['status']=='failed' and result['motion']['action_completed']
    assert service.submit(body)==receipt and len(backend.calls)==1


def test_model_wait_without_confirmation_cannot_execute(tmp_path,monkeypatch):
    import s1b_agent
    from types import SimpleNamespace
    backend=FakeBackend();camera=FakeCamera();service=Service(backend,camera,tmp_path/'robot')
    server=make_server(service,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def proposal(*args,**kwargs):return dict(name='propose_move',arguments=dict(right_mm=1,up_mm=0,reason='test'))
    monkeypatch.setattr(s1b_agent,'propose',proposal)
    monkeypatch.setattr('builtins.input',lambda *args:'')
    prompt=tmp_path/'prompt';prompt.write_text('test')
    args=SimpleNamespace(output=str(tmp_path/'client'),prompt=str(prompt),model_base='http://unused/v1',
        replay=None,thor_port=18000,thor_host='unused',robot_base='http://127.0.0.1:'+str(server.server_port),
        max_rounds=1,goal='test')
    try:
        s1b_agent.run(args)
        assert not backend.calls and backend.stopped
    finally:server.shutdown();server.server_close();service.close();thread.join()


def test_human_confirmed_full_loop_executes_once(tmp_path,monkeypatch):
    import s1b_agent
    from types import SimpleNamespace
    backend=FakeBackend();service=Service(backend,FakeCamera(),tmp_path/'robot')
    server=make_server(service,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setattr(s1b_agent,'propose',lambda *a,**k:dict(name='propose_move',arguments=dict(right_mm=2,up_mm=1,reason='test')))
    monkeypatch.setattr('builtins.input',lambda *args:'y')
    prompt=tmp_path/'prompt';prompt.write_text('test')
    args=SimpleNamespace(output=str(tmp_path/'client'),prompt=str(prompt),model_base='http://unused/v1',
        replay=None,thor_port=18000,thor_host='unused',robot_base='http://127.0.0.1:'+str(server.server_port),max_rounds=1,goal='test')
    try:
        s1b_agent.run(args)
        assert len(backend.calls)==1
        assert backend.pose[:3]==pytest.approx([.4,-.202,1.001])
        execution=json.loads((tmp_path/'client/rounds/0000/execution.json').read_text())
        assert execution['decision_observation_id']!=execution['execution_observation_id']
        assert len(list((tmp_path/'client/observations').glob('*/manifest.json')))==3
    finally:server.shutdown();server.server_close();service.close();thread.join()


def test_missing_robot_api_reports_address_without_creating_output(tmp_path,monkeypatch):
    import s1b_agent
    from types import SimpleNamespace
    def refused(*args,**kwargs):raise ConnectionRefusedError('refused')
    monkeypatch.setattr(s1b_agent,'request',refused)
    output=tmp_path/'not-created'
    args=SimpleNamespace(replay=None,robot_base='http://127.0.0.1:6210',output=str(output))
    with pytest.raises(RuntimeError,match='机器人 API http://127.0.0.1:6210'):
        s1b_agent.run(args)
    assert not output.exists()


def test_stop_response_finishes_before_server_close(setup,monkeypatch):
    service,_,_=setup
    gate=threading.Event();closed=threading.Event()
    original=service.stop
    def slow_ack():
        result=original();gate.wait(2);return result
    monkeypatch.setattr(service,'stop',slow_ack)
    server=make_server(service,0);server.timeout=.01
    def main():
        while not service.cancel.is_set():server.handle_request()
        server.server_close();closed.set()
    main_thread=threading.Thread(target=main,daemon=True);main_thread.start()
    responses=[]
    caller=threading.Thread(target=lambda:responses.append(request('http://127.0.0.1:'+str(server.server_port),'/v2/stop',{})),daemon=True)
    caller.start()
    try:
        assert service.cancel.wait(1)
        assert not closed.wait(.05)
        gate.set();caller.join(2);main_thread.join(2)
        assert responses[0]['phase']=='STOPPED' and closed.is_set()
    finally:gate.set();server.server_close()
