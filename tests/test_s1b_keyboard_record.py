"""Boundary persistence and failure handling without robot/camera hardware."""
import base64
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s1b_keyboard_record import Recorder, stamp


def make_bundle():
    wall=time.time_ns();mono=time.monotonic_ns()
    data={'head':b'head-jpeg','right':b'right-jpeg','depth':np.array([[123,456]],dtype='<u2').tobytes()}
    return dict(frames={k:dict(data_base64=base64.b64encode(v).decode(),source_ns=wall,receipt_ns=mono)
                        for k,v in data.items()},metadata={'depth':{'height':1,'width':2}},
                depth={'median_m':.123},selection={'revision':0})


@pytest.fixture
def recorder(tmp_path,monkeypatch):
    rec=Recorder(tmp_path/'session','http://unused',{},capture_timeout=.03)
    monkeypatch.setattr(rec,'_fetch',lambda timeout:make_bundle())
    return rec


def test_operation_capture_files_and_boundaries(recorder):
    rec=recorder;pose=[.4,-.2,1,0,0,0,1];goal=[.42,-.205,1.01,0,0,0,1]
    op=rec.begin('move',{'xyz_mm':[20,-5,10]},pose,goal,20,threading.Event())
    assert json.loads((rec.root/'operations/000001/operation.json').read_text())['status']=='prepared'
    started=stamp();rec.started(started);ended=stamp();rec.finish('completed',ended)
    rec.close()
    saved=json.loads((rec.root/'operations/000001/operation.json').read_text())
    assert saved['status']=='completed' and saved['completion_basis']=='command_interpolation'
    for phase in ('before','after'):
        folder=rec.root/'operations/000001'/phase
        assert (folder/'head.jpg').read_bytes()==b'head-jpeg'
        assert (folder/'right.jpg').read_bytes()==b'right-jpeg'
        assert np.load(folder/'right_depth_raw.npy').tolist()==[[123,456]]
        meta=json.loads((folder/'metadata.json').read_text())
        assert 'data_base64' not in meta['frames']['head']
        assert meta['frames']['head']['source_ns']>meta['boundary']['utc_ns']
    assert saved['before']['frames']['head']['source_ns']<started['utc_ns']
    assert saved['after']['frames']['head']['source_ns']>ended['utc_ns']
    assert (rec.root/'summary.md').is_file()
    session=json.loads((rec.root/'session.json').read_text())
    assert session['recording_complete'] and session['status']=='closed'


def test_stale_frames_are_retried_and_never_relabelled(recorder,monkeypatch):
    old=make_bundle();boundary=stamp();calls=[]
    def fetch(timeout):
        calls.append(timeout)
        return old if len(calls)==1 else make_bundle()
    rec=recorder;rec.capture_timeout=.2
    monkeypatch.setattr(rec,'_fetch',fetch)
    assert rec.capture('fresh',boundary)['status']=='complete'
    assert len(calls)==2
    monkeypatch.setattr(rec,'_fetch',lambda timeout:old)
    rec.capture_timeout=.01
    result=rec.capture('old',boundary)
    assert result['status']=='missing' and 'predate' in result['error']
    assert not (rec.root/'old/head.jpg').exists()


@pytest.mark.parametrize('failure',['timeout','depth'])
def test_capture_failure_is_persisted_and_does_not_raise(recorder,monkeypatch,failure):
    def fetch(timeout):
        if failure=='timeout':raise TimeoutError('camera timeout')
        bundle=make_bundle();del bundle['frames']['depth'];return bundle
    monkeypatch.setattr(recorder,'_fetch',fetch)
    result=recorder.capture('missing')
    assert result['status']=='missing'
    assert not recorder.session['recording_complete']
    assert json.loads((recorder.root/'missing/capture.json').read_text())['error']


def test_disk_failure_does_not_raise_and_recovery_preserves_error(recorder,monkeypatch,capsys):
    with monkeypatch.context() as patch:
        def fail(*a,**kw):raise OSError('disk full')
        patch.setattr(Path,'open',fail)
        recorder.event('input',raw='20\n')
        assert recorder.capture('disk')['status']=='missing'
    recorder.close()
    assert 'disk full' in capsys.readouterr().err
    assert not json.loads((recorder.root/'session.json').read_text())['recording_complete']


def test_interruption_and_unfinished_journal(recorder):
    rec=recorder;pose=[0,0,0,0,0,0,1]
    rec.begin('gripper',{'target':100},pose,pose,20,threading.Event())
    rec.started(stamp())
    assert json.loads((rec.root/'operations/000001/operation.json').read_text())['status']=='running'
    # Before finalization a killed process leaves this state, never completed.
    rec.close('SDK error')
    op=json.loads((rec.root/'operations/000001/operation.json').read_text())
    assert op['status']=='interrupted' and op['reason']=='SDK error'
    assert op['after']['status']=='complete'
    assert op['completion_basis']=='first_successful_send'


def test_cancelled_capture_does_not_wait_or_save_old_frame(recorder,monkeypatch):
    cancel=threading.Event();cancel.set()
    monkeypatch.setattr(recorder,'_fetch',lambda timeout:pytest.fail('must not fetch after cancellation'))
    assert recorder.capture('cancelled',cancel=cancel)['status']=='missing'


@pytest.mark.parametrize('copy_exit',[0,1])
def test_remote_record_auto_camera_and_copy_preserves_robot_data(tmp_path,monkeypatch,capsys,copy_exit):
    import s1b_keyboard_direct as module
    import start_s1b_inference
    calls=[]
    def run(command,**kw):
        calls.append(command)
        return SimpleNamespace(returncode=copy_exit if command[:2]==['scp','-rq'] else 0)
    monkeypatch.setattr(module.subprocess,'run',run)
    monkeypatch.setattr(module,'run_ssh',lambda command:calls.append(command) or 1)
    monkeypatch.setattr(start_s1b_inference,'check_port_available',lambda port:None)
    args=SimpleNamespace(host='S1B',sdk_root='/sdk',step_mm=2,speed_mm_s=20,control_hz=250,
                         view_cameras=False,view_port=8765,record=True,record_dir=str(tmp_path/'local'))
    assert module.launch_remote(args)==1
    assert args.view_cameras
    assert any('s1b_keyboard_record.py' in str(arg) for arg in calls[1])
    ssh=next(c for c in calls if c[:2]==['ssh','-t'])
    assert '--record --record-dir' in ssh[-1] and '--view-cameras' in ssh[-1]
    assert calls[-1][:2]==['scp','-rq'] and '/.local/share/pressb/keyboard_sessions/' in calls[-1][2]
    assert calls[-1][2].endswith('/*')  # '/.' is rejected by real scp.
    # The trap deletes only the deployment directory, not the recording directory.
    assert 'rm -rf ~/.local' not in ssh[-1]
    captured=capsys.readouterr()
    assert ('记录复制失败' in captured.err)==bool(copy_exit)
