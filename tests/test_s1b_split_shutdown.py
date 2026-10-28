"""Launcher teardown allows slow SDK exit without hiding failures."""
import importlib
from pathlib import Path
import subprocess
import json
import sys

import pytest


@pytest.mark.parametrize('failure',['none','fatal','exception'])
def test_sdk_worker_exits_after_persisting_results_despite_stuck_finalizers(tmp_path,failure):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    code='''
import atexit,json,sys,threading
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from s1b_control_service import finish_service
out=Path(sys.argv[2]);failure=sys.argv[3]
# Model SDK background threads and finalizers that cannot finish at exit.
threading.Thread(target=threading.Event().wait,daemon=False).start()
atexit.register(threading.Event().wait)
class SDK:
    def __del__(self):threading.Event().wait()
sdk=SDK()
summary=dict(fatal=['cleanup_failure'] if failure=='fatal' else [],sdk_imported=True)
if failure=='exception':
    try:raise RuntimeError('original_control_failure')
    except RuntimeError:finish_service(out,summary,exit_sdk_process=True)
else:finish_service(out,summary,exit_sdk_process=True)
'''
    result=subprocess.run([sys.executable,'-c',code,str(scripts),str(tmp_path),failure],
                          capture_output=True,text=True,timeout=5)
    expected=int(failure!='none')
    assert result.returncode==expected
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['process_exit_code']==expected
    assert summary['process_exit_method']=='after_explicit_cleanup'
    assert json.loads(result.stdout)['event']=='finished'
    if failure=='exception':assert 'original_control_failure' in result.stderr
    if failure=='fatal':assert 'cleanup_failure' in result.stderr


@pytest.mark.parametrize('sdk_imported,direct',[(False,True),(True,False)])
def test_shadow_and_imported_callers_keep_normal_python_exit(monkeypatch,tmp_path,sdk_imported,direct):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    service=importlib.import_module('s1b_control_service')
    def unexpected_exit(code):raise AssertionError('must not terminate caller')
    monkeypatch.setattr(service.os,'_exit',unexpected_exit)
    service.finish_service(tmp_path,dict(fatal=[],sdk_imported=sdk_imported),exit_sdk_process=direct)
    assert json.loads((tmp_path/'summary.json').read_text())['process_exit_method']=='python'


@pytest.mark.parametrize('timeouts,returncode',[(0,0),(1,0),(2,0),(1,255)])
def test_control_exit_wait_preserves_status(monkeypatch,capsys,timeouts,returncode):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    launcher=importlib.import_module('s1b_split_launch')
    class Process:
        calls=0
        def wait(self,timeout):
            assert timeout==10
            self.calls+=1
            if self.calls<=timeouts:
                raise subprocess.TimeoutExpired('fake-control',timeout)
            return returncode
    proc=Process()
    assert launcher.wait_control_exit(proc)==returncode
    assert proc.calls==timeouts+1
    assert capsys.readouterr().out.count('等待控制服务')==timeouts


def test_control_exit_wait_has_finite_budget(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    launcher=importlib.import_module('s1b_split_launch')
    class Process:
        calls=0
        def wait(self,timeout):
            self.calls+=1
            raise subprocess.TimeoutExpired('fake-control',timeout)
    proc=Process()
    with pytest.raises(RuntimeError,match='control_shutdown_timeout_after_30s'):
        launcher.wait_control_exit(proc)
    assert proc.calls==3
