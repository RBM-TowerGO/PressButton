"""Shared SDK routing, multimodal context, and automatic-stop regressions."""
import copy
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1b_keyboard_control import CommandExecutor, keyboard_command
from pressb.agent_loop.sdk_backend import SDKBackend
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.client import build_request, RobotClient
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.service import Service, make_server
from test_agent_loop import FakeBackend, FakeCamera, action, observe, wait


class Robot:
    def __init__(self):
        self.pose = [.4, -.2, 1., 0., 0., 0., 1.]
        self.torso = [0., 0., .5, 0., 0., 0., 1.]
        self.head = [0., 0.]
        self.calls = []
        self.stopped = False

    def get_control_rights_status(self):
        return True

    def get_current_cartesian_pose(self, *args, **kwargs):
        assert threading.current_thread().name != 'keyboard-command-servo'
        return [list(self.pose), list(self.torso)]

    def get_current_joints_position(self, names):
        assert threading.current_thread().name != 'keyboard-command-servo'
        return [list(self.head)]

    def set_cartesian_pose(self, names, poses, **kwargs):
        assert kwargs == dict(control_way='filter', use_wbc=False, add_default_torso=False, remote_wbc_option=False)
        self.calls.append(('cartesian', copy.deepcopy((names, poses, kwargs))))
        self.pose = list(poses[0])

    def set_different_type_command(self, names, types, values, **kwargs):
        assert names == ['astribot_arm_right', 'astribot_torso', 'astribot_gripper_right']
        assert types == ['cartesian', 'cartesian', 'joints']
        assert kwargs == dict(control_way='filter', use_wbc=False)
        self.calls.append(('mixed', copy.deepcopy((names, values, kwargs))))
        self.pose = list(values[0])

    def set_joints_position(self, names, values, **kwargs):
        assert names == ['astribot_head']
        assert kwargs == dict(control_way='filter', use_wbc=False, add_default_torso=False)
        self.calls.append(('head', copy.deepcopy((names, values, kwargs))))
        self.head = list(values[0])

    def stop_robot(self):
        self.stopped = True


def monitor_rows():
    now = time.time_ns()
    return {name: dict(header=dict(sec=now//10**9, nanosec=now % 10**9))
            for name in ('astribot_arm_right', 'astribot_torso', 'astribot_gripper_right')}


def test_keyboard_and_backend_same_commands_and_sdk_routes():
    robots = [Robot(), Robot()]
    keyboard = CommandExecutor(robots[0], 100, 1000, 1000)
    backend = SDKBackend(robots[1], SimpleNamespace(snapshot=monitor_rows), 100, 1000, 1000)
    inputs = ['c', {'head_deg': [0, 30]}, [0, 0, 300], [45, 20, -50],
              [15, 0, 0], [-100, 0, 0], 'o', [0, 0, 0]]
    try:
        assert all(call[0] == 'cartesian' for r in robots for call in r.calls)
        for value in inputs:
            command = keyboard_command(value)
            offsets = [len(r.calls) for r in robots]
            prepared = keyboard.prepare(command)
            task = keyboard.start(prepared)
            keyboard.wait(task, threading.Event())
            result = backend.move(command['type'], command, threading.Event(), lambda _: None)
            assert task['ended']['monotonic_ns'] >= task['started']['monotonic_ns']
            assert result['completed_monotonic_ns'] >= result['started_monotonic_ns']
            assert result['target_pose_m_xyzw'] == pytest.approx(task['goal'])
            assert robots[0].pose == pytest.approx(robots[1].pose)
            assert robots[0].head == pytest.approx(robots[1].head)
            route = 'head' if command['type'] == 'head' else 'mixed'
            final_calls = [[call for call in r.calls[offset:] if call[0] == route][-1]
                           for r, offset in zip(robots, offsets)]
            assert final_calls[0] == final_calls[1]
            assert result['completion_basis'] == ('first_successful_send' if route == 'mixed' and
                                                  command['type'] == 'gripper' else 'command_interpolation')
        assert robots[0].pose[:3] == pytest.approx([.36, -.18, 1.25])
        assert robots[0].pose[3:] == [0, 0, 0, 1]
    finally:
        keyboard.close()
        backend.close()


@pytest.mark.parametrize('command', [dict(type='head', head_deg=[0, 30]), dict(type='gripper', key='c')])
def test_http_head_and_gripper_share_task_protocol(tmp_path, command):
    service = Service(FakeBackend(), FakeCamera(), tmp_path/'robot')
    try:
        obs = observe(service)
        body = action(service, obs, **command)
        receipt = service.submit(body)
        result = wait(service, receipt)
        assert result['status'] == 'completed'
        assert service.submit(body) == receipt
        assert len(service.backend.calls) == 1
    finally:
        service.close()


def test_demo_loads_raw_boundaries_and_keeps_missing_final(tmp_path):
    camera = FakeCamera()
    import base64
    root = tmp_path/'demo'
    root.mkdir()
    session = dict(schema='pressb-keyboard-record-v1', settings=dict(frame='chassis', units='mm'),
                   recording_complete=False, final=dict(status='missing'))
    (root/'session.json').write_text(json.dumps(session))
    for i in range(10):
        folder = root/'operations'/f'{i+1:06d}'
        folder.mkdir(parents=True)
        op = dict(operation_id=f'{i+1:06d}', kind='move', command=dict(xyz_mm=[i,0,0]),
                  before=dict(status='complete'), after=dict(status='complete'))
        (folder/'operation.json').write_text(json.dumps(op))
        for phase in ('before', 'after'):
            p = folder/phase
            p.mkdir()
            bundle = camera.capture(0, threading.Event())
            (p/'metadata.json').write_text(json.dumps(dict(depth=bundle['depth'])))
            for view in ('head', 'right'):
                (p/(view+'.jpg')).write_bytes(base64.b64decode(bundle['frames'][view]['data_base64']))
    demo = Demonstration(root)
    assert len(demo.images) == 10
    assert demo.summary['session']['final']['status'] == 'missing'
    assert not demo.summary['session']['recording_complete']
    image = (root/'operations/000001/before/head.jpg').read_bytes()
    obs = dict(observation_id='live', state={}, selection={}, depth={})
    body = build_request('press', obs, image, dict(schema=SESSION_SCHEMA), [], 'prompt', image, demo)
    content = [item for message in body['messages'][1:] for item in message['content']]
    assert len([x for x in content if x['type'] == 'image_url']) == 12
    assert 'HISTORICAL_DEMONSTRATION_NOT_LIVE' in content[1]['text']
    assert 'LIVE HEAD' in content[-4]['text']
    assert 'LIVE RIGHT WRIST' in content[-2]['text']
    before = demo.content[2]['image_url']['url']
    assert before == build_request('press', obs, image, dict(schema=SESSION_SCHEMA), [], 'p', image, demo)['messages'][1]['content'][3]['image_url']['url']

    # A successful press can coexist with an interrupted withdrawal. Preserve
    # that boundary explicitly rather than turning its target into a success.
    interrupted = root/'operations/000011'
    interrupted.mkdir()
    operation = dict(operation_id='000011', kind='move', command=dict(xyz_mm=[-100,0,0]),
                     status='interrupted', before=dict(status='complete'), after=dict(status='missing'))
    (interrupted/'operation.json').write_text(json.dumps(operation))
    (interrupted/'before').mkdir()
    for view in ('head','right'):
        (interrupted/'before'/(view+'.jpg')).write_bytes(image)
    updated = Demonstration(root)
    assert len(updated.images) == 11
    assert 'status=interrupted' in updated.content[-2]['text']
    assert updated.summary['operations'][-1]['observations']['after']['missing_images'] == ['head','right']
    assert json.loads((interrupted/'operation.json').read_text()) == operation


@pytest.mark.parametrize('cancel_stage', ['action', 'view'])
def test_cancel_during_model_wait_discards_late_action(tmp_path, monkeypatch, cancel_stage):
    import s1b_agent
    service = Service(FakeBackend(), FakeCamera(), tmp_path/'robot')
    server = make_server(service, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    release = threading.Event()
    controls = []
    original = s1b_agent.StopControl
    def factory():
        control = original()
        controls.append(control)
        return control
    monkeypatch.setattr(s1b_agent, 'StopControl', factory)
    from pressb.agent_loop import view_acquisition
    from test_view_acquisition import reply, located
    def inspect_view(base, body, oid, directory, cancel=None):
        if cancel_stage == 'view':
            controls[0].cancel.set()
            release.wait(2)
        return view_acquisition.validate(reply(oid=oid, head=located()), oid)
    monkeypatch.setattr(view_acquisition, 'inspect', inspect_view)
    def propose(*args, **kwargs):
        assert cancel_stage == 'action', 'cancelled view must not start an action request'
        controls[0].cancel.set()
        release.wait(2)
        return dict(name='propose_move', arguments=dict(xyz_mm=[10,0,0], reason='late'))
    monkeypatch.setattr(s1b_agent, 'propose', propose)
    monkeypatch.setattr('pressb.agent_loop.conversation.tokenize', lambda *a:1000)
    prompt = tmp_path/'prompt'
    prompt.write_text('test')
    args = SimpleNamespace(output=str(tmp_path/'client'), prompt=str(prompt), model_base='http://unused/v1',
        replay=None, thor_port=18000, thor_host='unused', robot_base='http://127.0.0.1:'+str(server.server_port),
        max_rounds=20, goal='test',decision_mode='direct')
    try:
        with pytest.raises(InterruptedError):
            s1b_agent.run(args)
        release.set()
        assert not service.backend.calls and service.backend.stopped
        assert json.loads((tmp_path/'client/result.json').read_text())['status'] == 'operator_stopped'
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        service.close()
        thread.join()


def test_q_during_model_request_stops_without_waiting(monkeypatch):
    import os
    import s1b_agent
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(s1b_agent.sys, 'stdin', SimpleNamespace(isatty=lambda: True, fileno=lambda: read_fd))
    stopped = threading.Event()
    release = threading.Event()
    try:
        with s1b_agent.StopControl() as control:
            control.client = SimpleNamespace(stop=lambda: stopped.set())
            os.write(write_fd, b'q\n')
            with pytest.raises(InterruptedError):
                control.call(lambda: release.wait(2))
            assert stopped.wait(1)
    finally:
        release.set()
        os.close(read_fd)
        os.close(write_fd)


def test_old_server_schema_rejected_before_motion(tmp_path, monkeypatch):
    from pressb.agent_loop import client
    monkeypatch.setattr(client, 'request', lambda *a, **k: dict(schema='pressb-agent-v2'))
    with pytest.raises(ValueError, match='schema'):
        RobotClient('http://unused', tmp_path)


def test_no_motion_before_first_observation(tmp_path):
    service = Service(FakeBackend(), FakeCamera(), tmp_path/'robot')
    try:
        with pytest.raises(ValueError, match='old_observation'):
            service.submit(dict(session_id=service.session_id, command_id='early',
                                observation_id=None, type='move', xyz_mm=[1,0,0]))
        assert not service.backend.calls
    finally:
        service.close()


def test_model_network_failure_is_logged_without_retry(tmp_path, monkeypatch):
    from http.client import RemoteDisconnected
    from pressb.agent_loop import client
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise RemoteDisconnected('connection closed')
    monkeypatch.setattr(client, 'request', fail)
    with pytest.raises(RemoteDisconnected):
        client.propose('http://unused', {}, {}, dict(schema=SESSION_SCHEMA), tmp_path)
    error = json.loads((tmp_path/'error.json').read_text())
    assert error['stage'] == 'model_request' and error['error_type'] == 'RemoteDisconnected'
    assert len(calls) == 1


def test_invalid_model_output_keeps_repairing_until_valid(tmp_path, monkeypatch):
    from pressb.agent_loop import client
    from test_conversation import response
    calls = []
    def invalid(*args, **kwargs):
        calls.append(copy.deepcopy(args[2]))
        return (dict(choices=[]) if len(calls) <= 4 else
                response('propose_head', dict(head_deg=[0,30],reason='obtain view')))
    monkeypatch.setattr(client, 'request', invalid)
    proposal = client.propose('http://unused', dict(messages=[]), dict(observation_id='now'),
                              dict(schema=SESSION_SCHEMA), tmp_path)
    assert len(calls) == 5 and proposal['arguments']['head_deg'] == [0,30]
    assert all(len(body['messages']) == 1 for body in calls[1:])
    assert len(list(tmp_path.glob('response-*.json'))) == 5
    assert len(list(tmp_path.glob('repair-request-*.json'))) == 4
    assert len(list(tmp_path.glob('validation-*.json'))) == 4
    assert json.loads((tmp_path/'correction.json').read_text())['status'] == 'accepted'
    assert not (tmp_path/'error.json').exists()


def test_repeated_invalid_action_remains_cancellable(tmp_path, monkeypatch):
    from pressb.agent_loop import client
    cancel=threading.Event();calls=[]
    def invalid(*args, **kwargs):
        calls.append(True)
        if len(calls)==4:cancel.set()
        return dict(choices=[])
    monkeypatch.setattr(client,'request',invalid)
    with pytest.raises(InterruptedError,match='operator_stopped'):
        client.propose('fake',dict(messages=[]),dict(observation_id='now'),
                       dict(schema=SESSION_SCHEMA),tmp_path,cancel=cancel)
    assert len(calls)==4 and not (tmp_path/'proposal.json').exists()
