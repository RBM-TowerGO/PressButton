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
