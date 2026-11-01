"""Local dashboard and launcher checks; no robot or model connection."""
import base64
import json
import io
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import s1b_agent_dashboard as dashboard
import s1b_agent_run as launcher
from pressb.agent_loop.common import atomic_json, SESSION_SCHEMA


def test_dashboard_rounds_live_view_and_stop(tmp_path, monkeypatch):
    stop = threading.Event()
    atomic_json(tmp_path/'launcher.json', dict(phase='running', goal='按下'))
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'proposal.json', dict(name='propose_move', arguments=dict(xyz_mm=[5,0,0], reason='<script>example</script>')))
    atomic_json(rd/'response-0.json', dict(elapsed_s=1.25, response=dict(choices=[])))
    atomic_json(rd/'execution.json', {})
    atomic_json(rd/'result.json', dict(status='completed'))
    (tmp_path/'terminal.log').write_text('SDK waiting\nAgent output\n')
    monkeypatch.setattr(dashboard, 'request', lambda *a, **k: dict(frames=dict(head=dict(jpeg='a',stale=False,age_ms=1)),depth={}))
    server, thread = dashboard.serve(dashboard.Dashboard(tmp_path, 'http://unused', stop), 0)
    base = 'http://127.0.0.1:'+str(server.server_port)
    try:
        with urlopen(base) as response:
            html = response.read().decode()
        assert '头部相机' in html and '右腕相机' in html and 'Agent 输出' in html
        assert '.textContent=' in html and 'innerHTML' not in html
        with urlopen(base+'/api/agent') as response:
            data = json.load(response)
        assert data['rounds'][0]['phase'] == '执行完成'
        assert data['rounds'][0]['elapsed_s'] == 1.25
        assert data['rounds'][0]['proposal']['arguments']['xyz_mm'] == [5,0,0]
        assert 'Agent output' in data['log']
        with urlopen(base+'/api/cameras') as response:
            assert not json.load(response)['archived']
        headers = {'Content-Type':'application/json', 'Origin':'http://untrusted.example'}
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base+'/api/stop', data=b'{}', headers=headers))
        assert exc.value.code == 403 and not stop.is_set()
        headers.pop('Origin')
        with urlopen(Request(base+'/api/stop', data=b'{}', headers=headers)) as response:
            assert json.load(response)['status'] == 'stop_requested'
        assert stop.is_set()
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_camera_disconnect_shows_last_saved_images_as_archived(tmp_path, monkeypatch):
    folder = tmp_path/'agent/observations/one'
    atomic_json(folder/'manifest.json', dict(observation_id='one',depth=dict(median_m=.1)))
    (folder/'head.jpg').write_bytes(b'head')
    (folder/'right.jpg').write_bytes(b'right')
    def disconnected(*args, **kwargs):
        raise ConnectionRefusedError()
    monkeypatch.setattr(dashboard, 'request', disconnected)
    result = dashboard.Dashboard(tmp_path, 'http://unused', threading.Event()).cameras()
    assert result['archived'] and result['depth'] == {}
    assert result['frames']['head']['stale']
    assert base64.b64decode(result['frames']['head']['jpeg']) == b'head'


def test_launcher_waits_for_new_schema_and_handles_stop(monkeypatch):
    calls = iter([ConnectionRefusedError(), dict(schema=SESSION_SCHEMA)])
    def request(*args, **kwargs):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(launcher, 'request', request)
    process = SimpleNamespace(poll=lambda:None)
    assert launcher.wait_for_service(process, 'http://unused', threading.Event())['schema'] == SESSION_SCHEMA
    stop = threading.Event();stop.set()
    with pytest.raises(InterruptedError):
        launcher.wait_for_service(process, 'http://unused', stop)
    with pytest.raises(RuntimeError, match='退出'):
        launcher.wait_for_service(SimpleNamespace(poll=lambda:1), 'http://unused', threading.Event())


@pytest.mark.parametrize('failure_context', [None, 'reviewed-case'])
def test_launcher_cancels_agent_when_service_exits(tmp_path, monkeypatch, failure_context):
    from pathlib import Path
    demo = tmp_path/'demo'
    atomic_json(demo/'session.json', {})
    events = []
    processes = []
    closed = []
    monkeypatch.setattr(launcher.sys, 'stdin', SimpleNamespace(isatty=lambda:True))
    monkeypatch.setattr(launcher, 'check_port_available', lambda port:None)
    reviews = []
    monkeypatch.setattr(launcher, 'FailureCase', lambda path:reviews.append(path))
    monkeypatch.setattr(launcher.signal, 'signal', lambda *a:None)
    class Event(threading.Event):
        def __init__(self):
            super().__init__();events.append(self)
    monkeypatch.setattr(launcher.threading, 'Event', Event)
    class Process:
        def __init__(self, argv, **kwargs):
            self.argv=argv;self.returncode=None;self.pid=999999;self.stdout=io.BytesIO()
            self.is_agent=any(Path(a).name=='s1b_agent.py' for a in argv)
            if self.is_agent:
                assert kwargs['stdin'] == subprocess.DEVNULL
                if failure_context:
                    assert argv[argv.index('--failure-case')+1] == str(Path(failure_context).resolve())
                else:
                    assert '--failure-case' not in argv
                processes[0].returncode=0
                events[1].set()  # finish dashboard review without blocking this test
            processes.append(self)
        def poll(self):return self.returncode
        def wait(self, timeout):return self.returncode
    monkeypatch.setattr(launcher.subprocess, 'Popen', Process)
    monkeypatch.setattr(launcher, 'wait_for_service', lambda *a:dict(schema=SESSION_SCHEMA,session_id='one'))
    monkeypatch.setattr(launcher, 'request', lambda *a, **k:{})
    def stop(process):
        if process is not None and process.returncode is None:
            closed.append(process);process.returncode=-2
    monkeypatch.setattr(launcher, 'stop_process', stop)
    server=SimpleNamespace(server_port=8766,shutdown=lambda:None,server_close=lambda:None)
    monkeypatch.setattr(launcher, 'serve', lambda *a:(server,SimpleNamespace(join=lambda *a:None)))
    args=SimpleNamespace(host='fake',demo=str(demo),goal='test',output=str(tmp_path/'run'),
        robot_port=6210,camera_port=8765,web_port=8766,thor_port=18000,thor_host='fake',max_rounds=20,
        failure_case=failure_context)
    launcher.launch(args)
    assert reviews == ([failure_context] if failure_context else [])
    assert len(processes) == 2 and processes[1] in closed
    assert json.loads((tmp_path/'run/launcher.json').read_text())['reason'] == 'robot_service_exited'


def test_sdk_prompt_forwarded_without_newline():
    read_fd, write_fd = os.pipe()
    seen = threading.Event()
    chunks = []
    def emit(text):
        chunks.append(text)
        if '接管? ' in ''.join(chunks):
            seen.set()
    with os.fdopen(read_fd, 'rb') as stream:
        reader = threading.Thread(target=launcher.relay_output, args=(stream, emit))
        reader.start()
        try:
            os.write(write_fd, '接管? '.encode())
            assert seen.wait(2), 'SDK prompt buffered until newline / EOF'
        finally:
            os.close(write_fd)
            reader.join(2)
