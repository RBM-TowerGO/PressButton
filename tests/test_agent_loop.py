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

from pressb.agent_loop.client import build_request, parse_proposal, RobotClient
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
        if kind=='move':self.pose[:3]=[v+d/1000 for v,d in zip(self.pose[:3],data['xyz_mm'])]
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
    body=action(service,obs,type='move',xyz_mm=[0,-10,-5])
    receipt=service.submit(body);result=wait(service,receipt)
    assert result['status']=='completed'
    assert backend.pose[:3]==pytest.approx([.4,-.21,.995])
    assert service.submit(body)==receipt and len(backend.calls)==1
    with pytest.raises(ValueError,match='conflict'):service.submit(dict(body,xyz_mm=[0,-2,-5]))
    with pytest.raises(ValueError,match='old_observation'):service.submit(dict(body,command_id='next'))


def test_xyz_persists_and_legacy_press_is_rejected(setup):
    service,backend,_=setup;obs=observe(service)
    with pytest.raises(ValueError,match='invalid_command'):
        service.submit(action(service,obs,type='press',forward_mm=15))
    result=wait(service,service.submit(action(service,obs,type='move',xyz_mm=[45,20,0])))
    assert result['motion']['action_completed']
    assert backend.pose[:3]==pytest.approx([.445,-.18,1.])
    assert result['motion']['button_trigger_evidence'] is None
    obs=service.observations[result['capture']['observation_id']]
    wait(service,service.submit(action(service,obs,cid='retract',type='move',xyz_mm=[-100,0,0])))
    assert backend.pose[0]==pytest.approx(.345)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,'3',[]])
def test_invalid_move(setup,value):
    service,_,_=setup;obs=observe(service)
    with pytest.raises(ValueError):service.submit(action(service,obs,type='move',xyz_mm=[value,0,0]))


def test_stop_priority_and_busy(setup):
    service,backend,_=setup;obs=observe(service);backend.block=threading.Event()
    receipt=service.submit(action(service,obs,type='move',xyz_mm=[0,-1,0]))
    with pytest.raises(ValueError,match='not_ready'):service.submit(action(service,obs,cid='other',type='move',xyz_mm=[0,-1,0]))
    service.stop();result=wait(service,receipt)
    assert backend.stopped and result['status']=='interrupted'
    with pytest.raises(ValueError):service.submit(action(service,obs,cid='after-stop',type='move',xyz_mm=[0,-1,0]))


def test_capture_failure_does_not_replay_motion(setup):
    service,backend,camera=setup;obs=observe(service);camera.fail=True
    body=action(service,obs,type='move',xyz_mm=[0,-1,0])
    receipt=service.submit(body);result=wait(service,receipt)
    assert result['status']=='failed' and result['motion']['action_completed']
    assert service.phase=='NEEDS_OBSERVATION'
    assert service.submit(body)==receipt and len(backend.calls)==1
    camera.fail=False;observe(service)
    assert service.phase=='READY' and len(backend.calls)==1


def test_session_observation_tokens_and_depth_is_not_motion_gate(setup):
    service,backend,camera=setup;obs=observe(service)
    body=action(service,obs,type='move',xyz_mm=[1,0,0])
    with pytest.raises(ValueError,match='session'):service.submit(dict(body,session_id='old'))
    with pytest.raises(ValueError,match='old_observation'):service.submit(dict(body,observation_id='old'))
    backend.pose[1]+=.01
    camera.revision+=1
    obs['depth']['median_m']=None
    assert wait(service,service.submit(body))['status']=='completed'


def test_http_client_downloads_and_hashes(setup,tmp_path):
    service,_,_=setup;server=make_server(service,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=RobotClient('http://127.0.0.1:'+str(server.server_port),tmp_path/'client')
        obs,folder=client.observe()
        assert np.load(folder/'right_depth_raw.npy').shape==(2,2)
        assert (folder/'head.jpg').exists() and (folder/'right.jpg').exists()
        proposal=dict(name='propose_move',arguments=dict(xyz_mm=[0,-2,0],reason='test'))
        result,(new,_) = client.execute(proposal,obs)
        assert new['observation_id']!=obs['observation_id']
        assert result['motion']['action_completed']
        with pytest.raises(ValueError):request(client.base,'/v2/artifacts/../../session.json')
    finally:server.shutdown();server.server_close();thread.join()


def test_model_dual_views_and_keyboard_tools(setup):
    service,_,_=setup;obs=observe(service)
    wrist=next(p for p in service.artifacts.values() if p.name=='right.jpg').read_bytes()
    head=next(p for p in service.artifacts.values() if p.name=='head.jpg').read_bytes()
    body=build_request('按下',obs,wrist,service.session,[],'English prompt',head)
    content=body['messages'][-1]['content']
    assert len([v for v in content if v['type']=='image_url'])==2
    assert body['chat_template_kwargs']=={'enable_thinking':False}
    for name,args in [('propose_move',dict(xyz_mm=[45,20,0],reason='align')),
                      ('propose_head',dict(head_deg=[0,30],reason='view')),
                      ('propose_gripper',dict(key='c',reason='prepare'))]:
        assert parse_proposal(tool(name,args),obs,service.session)['arguments']==args
    for name,args in [('propose_press',dict(forward_mm=15,reason='old')),
                      ('propose_move',dict(right_mm=1,up_mm=0,reason='old')),
                      ('propose_head',dict(head_deg=[0,True],reason='bad')),
                      ('propose_gripper',dict(key='x',reason='bad'))]:
        with pytest.raises(ValueError):parse_proposal(tool(name,args),obs,service.session)
    invalid=tool('propose_move',dict(xyz_mm=[1,0,0],reason='ok'))
    invalid['choices'][0]['message']['tool_calls']*=2
    with pytest.raises(ValueError):parse_proposal(invalid,obs,service.session)


def test_restart_journal_not_replayed(tmp_path):
    root=tmp_path/'sessions';old=root/'old'/'tasks';old.mkdir(parents=True)
    (old/'task.json').write_text(json.dumps(dict(status='running')))
    service=Service(FakeBackend(),FakeCamera(),root)
    try:assert json.loads((old/'task.json').read_text())['status']=='interrupted'
    finally:service.close()


def test_pose_quaternion_sign():
    assert pose_error([0,0,0,0,0,0,1],[0,0,0,0,0,0,-1])==(0,0)


def test_storage_failure_never_reexecutes(setup,monkeypatch):
    service,backend,_=setup;obs=observe(service)
    def broken(*args,**kwargs):raise OSError('disk_full')
    monkeypatch.setattr(service,'_save_observation',broken)
    body=action(service,obs,type='move',xyz_mm=[0,-1,0])
    receipt=service.submit(body);result=wait(service,receipt)
    assert result['status']=='failed' and result['motion']['action_completed']
    assert service.submit(body)==receipt and len(backend.calls)==1


def test_automatic_full_loop_executes_once(tmp_path,monkeypatch,approved_forward_view):
    import s1b_agent
    from types import SimpleNamespace
    backend=FakeBackend();service=Service(backend,FakeCamera(),tmp_path/'robot')
    server=make_server(service,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setattr('pressb.agent_loop.conversation.tokenize', lambda *a:1000)
    from pressb.agent_loop import view_acquisition
    from test_view_acquisition import reply, located
    monkeypatch.setattr(view_acquisition, 'inspect', lambda base, body, oid, directory, cancel=None:
                        view_acquisition.validate(reply(oid=oid, head=located()), oid))
    def fake_propose(*a, **k):
        args = dict(xyz_mm=[2,1,0],reason='test')
        progress = dict(observation_id=k['conversation'].current_id, subgoal='approach', view='wrist',
                        button_vertical='level', button_horizontal='center', last_effect='first')
        response = tool('propose_move', dict(args, progress=progress))
        response['choices'][0]['message']['tool_calls'][0]['id'] = 'test-call'
        k['conversation'].accept(response)
        proposal = dict(name='propose_move', arguments=args)
        k['conversation'].forward_guard_decision = approved_forward_view(proposal, k['conversation'].current_id)
        return proposal
    monkeypatch.setattr(s1b_agent,'propose',fake_propose)
    monkeypatch.setattr('builtins.input',lambda *args:pytest.fail('automatic loop must not ask for confirmation'))
    prompt=tmp_path/'prompt';prompt.write_text('test')
    args=SimpleNamespace(output=str(tmp_path/'client'),prompt=str(prompt),model_base='http://unused/v1',
        replay=None,thor_port=18000,thor_host='unused',robot_base='http://127.0.0.1:'+str(server.server_port),max_rounds=1,goal='test',decision_mode='direct')
    try:
        s1b_agent.run(args)
        assert len(backend.calls)==1 and backend.stopped
        assert backend.pose[:3]==pytest.approx([.402,-.199,1.])
        execution=json.loads((tmp_path/'client/rounds/0000/execution.json').read_text())
        assert execution['decision_observation_id']==execution['execution_observation_id']
        assert len(list((tmp_path/'client/observations').glob('*/manifest.json')))==2
        result=json.loads((tmp_path/'client/result.json').read_text())
        assert result['status']=='max_rounds' and result['human_verified_success'] is None
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
