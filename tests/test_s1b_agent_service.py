"""Startup regressions without SDK initialization or robot commands."""
import subprocess
import sys
from pathlib import Path

import pytest


def test_output_is_closed_when_startup_fails(monkeypatch):
    import s1b_agent_service as service
    import s1b_keyboard_direct as keyboard
    events=[]
    class Output:
        def __init__(self):events.append('opened')
        def close(self):events.append('closed')
    def fail(args):
        events.append('run')
        raise RuntimeError('startup-sentinel')
    monkeypatch.setattr(keyboard,'SDKOutput',Output)
    monkeypatch.setattr(service,'run',fail)
    monkeypatch.setattr(sys,'argv',['s1b_agent_service.py'])
    with pytest.raises(RuntimeError,match='startup-sentinel'):service.main()
    assert events==['opened','run','closed']


def test_actual_output_filter_flushes_startup_and_error_before_native_exit():
    root=Path(__file__).resolve().parents[1]
    code='''
import os,sys
sys.path.insert(0,'scripts')
import s1b_agent_service as service
def fail(args):
    print('startup-before-sdk',flush=True)
    raise RuntimeError('startup-sentinel')
service.run=fail
sys.argv=['s1b_agent_service.py']
try:service.main()
except Exception as exc:
    print('service-error:'+str(exc),file=sys.stderr,flush=True)
    os._exit(1)
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,timeout=10)
    assert result.returncode==1
    assert 'startup-before-sdk' in result.stdout
    assert 'service-error:startup-sentinel' in result.stderr
    assert '__enter__' not in result.stderr


@pytest.mark.parametrize('auto_confirm', [False, True])
def test_deployment_archive_contains_grounded_loop_without_connecting(monkeypatch, auto_confirm):
    import tarfile
    from types import SimpleNamespace
    import s1b_agent_service as service
    import s1b_keyboard_direct as keyboard
    import start_s1b_inference as inference
    names=[]
    packaged={}
    required=('src/pressb/agent_loop/perception.py', 'src/pressb/agent_loop/grounding.py',
              'src/pressb/agent_loop/view_acquisition.py',
              'src/pressb/agent_loop/localization.py', 'src/pressb/agent_loop/control_policy.py',
              'src/pressb/agent_loop/short_loop.py', 'src/pressb/agent_loop/demonstration.py',
              'scripts/s1b_keyboard_control.py')
    def fake_run(argv,**kwargs):
        if argv[0]=='scp':
            with tarfile.open(argv[2]) as archive:
                names.extend(archive.getnames())
                for name in required:
                    packaged[name]=archive.extractfile(name).read()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(service.subprocess,'run',fake_run)
    ssh_calls=[]
    monkeypatch.setattr(keyboard,'run_ssh',lambda argv:ssh_calls.append(argv) or 0)
    monkeypatch.setattr(inference,'check_port_available',lambda *a:None)
    monkeypatch.setattr(inference,'sdk_environment',lambda *a:'true')
    args=SimpleNamespace(host='no-robot',port=6210,view_port=8765,output='/tmp/test',
                         control_hz=250,speed_mm_s=20,head_speed_deg_s=20,sdk_root='/unused',
                         auto_confirm=auto_confirm)
    assert service.launch(args)==0
    assert len(ssh_calls)==1
    assert ('--auto-confirm' in ssh_calls[0][-1]) is auto_confirm
    root=Path(__file__).resolve().parents[1]
    for name in required:
        assert name in names
        assert packaged[name]==(root/name).read_bytes()
        compile(packaged[name],name,'exec')
    assert not any('__pycache__' in n for n in names)
