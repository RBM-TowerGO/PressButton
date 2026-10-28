"""Torso control tests: fake actuators only, including live HTTP startup."""
import hashlib
import http.client
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pressb.s1b_deployment.torso import duration,interpolate
from pressb.s1b_deployment.executor import SDKBackend,ExecutionError
from pressb.s1b_deployment.live_execution import LiveExecution
from pressb.s1b_deployment.control_service import ControlSession,make_control_server
from pressb.s1b_deployment.client import control_exchange
from pressb.s1b_deployment.protocol import encode
from test_s1b_live_execution import Monitor,Backend
from test_s1b_startup import response

ROOT=Path(__file__).resolve().parents[1]
HOME_LEFT = np.array([.159, -.022, -1.42, 1.66, -.345, .115, .125])
HOME_RIGHT = np.array([-.159, -.022, 1.42, 1.66, .345, .115, -.125])

@pytest.fixture
def live():
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    tc=c['torso_alignment'];tc.update(minimum_duration_seconds=.04,max_velocity=100.,max_acceleration=10000.,stable_seconds=.02,timeout_seconds=.5)
    c['stop_stability_window_ns']=5_000_000
    q=np.r_[HOME_RIGHT,0.,0.,0.]
    class TorsoMonitor(Monitor):
        torso=np.array([.29,-.58,.29,0.]);left=HOME_LEFT.copy();frozen=False
        def torso_command(self):return self.torso.copy()
        def snapshot(self):
            s=super().snapshot()
            s['astribot_torso'].update(position=self.torso.tolist(),velocity=[0.]*4)
            s['astribot_arm_left'].update(position=self.left.tolist(),velocity=[0.]*7)
            return s
    m=TorsoMonitor(q)
    class TorsoBackend(Backend):
        torso_target=None
        def torso_limits(self):return [0.,-2.35,0.,-1.58],[1.5,0.,2.4,1.58],[7.87,7.87,8.94,8.94]
        def send(self,target):
            super().send(target)
            if not m.frozen:m.torso=self.torso_target.copy()
    b=TorsoBackend(q);events=[]
    obj=LiveExecution(c,events.append,operator_present=True,estop_ready=True,monitor=m,backend=b)
    obj.baseline=m.snapshot();obj.prepare=lambda:None
    yield obj,events
    obj.monitor.stale=False
    obj.close()


def test_reference_matches_reviewed_source():
    import h5py
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())['torso_alignment']
    p=ROOT/c['source']['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==c['source']['sha256']
    with h5py.File(p) as f:np.testing.assert_array_equal(c['target'],f['joints_dict/joints_position_command'][0,3:7])


def test_quintic_bounds():
    a=np.array([.29,-.58,.29,0]);b=np.array([.6,-1.2,.6,0]);v=np.array([.15]*4)
    t=duration(a,b,v,.1,8);dt=t/10000
    samples=np.array([interpolate(a,b,u) for u in np.linspace(0,1,10001)])
    assert np.max(abs(np.diff(samples,axis=0)/dt))<=.15+1e-6
    assert np.max(abs(np.diff(samples,n=2,axis=0)/dt**2))<=.1+1e-6
    np.testing.assert_allclose(samples[[0,-1]], [a,b])


def test_alignment_and_fixed_hold(live):
    obj,events=live;obj.acquire();phases=[];obj.align_torso(phases.append)
    assert 'ALIGNING_TORSO' in phases and 'SETTLING_TORSO' in phases
    assert obj.torso.ready and obj.model_commands==0
    np.testing.assert_allclose(obj.backend.torso_target,obj.torso.target)
    obj.hold();assert obj.backend.calls[-1].shape==(10,)
    assert any(r['event']=='torso_aligned' for r in events)
    obj.stop('operator_stop');before=len(obj.backend.calls)
    with pytest.raises(ExecutionError):obj.hold()
    assert len(obj.backend.calls)==before


@pytest.mark.parametrize('fault',['limits','target','initial','stale','nan'])
def test_prepare_failures_no_motion(live,fault):
    obj,_=live
    if fault=='limits':obj.backend.torso_limits=lambda:([0]*4,[1]*4,[0]*4)
    if fault=='target':obj.torso.target[0]=100
    if fault=='initial':obj.monitor.torso_command=lambda:np.array([.5,-.58,.29,0])
    if fault=='stale':obj.monitor.stale=True
    if fault=='nan':obj.monitor.torso=np.array([np.nan]*4)
    with pytest.raises(ExecutionError):obj.acquire()
    assert not obj.backend.calls


@pytest.mark.parametrize('fault',['stop','rights','timeout'])
def test_transition_faults(live,fault):
    obj,_=live;obj.acquire();obj.monitor.frozen=True
    if fault=='timeout':obj.torso.c['timeout_seconds']=.06
    def phase(_):
        if fault=='stop':obj.stop('operator_stop')
        if fault=='rights':obj.backend.acquired=False
    with pytest.raises(ExecutionError):obj.align_torso(phase)
    assert not obj.active
    obj.close();assert obj.backend.released
    before=len(obj.backend.calls);time.sleep(.01);assert len(obj.backend.calls)==before


def test_tracking_and_stop_feedback(live):
    obj,_=live;obj.acquire();obj.align_torso(lambda p:None)
    obj.monitor.torso=obj.torso.target+np.array([.04,0,0,0])
    obj.torso.check(obj.monitor.snapshot(),time.monotonic_ns())
    obj.torso.bad_since=time.monotonic_ns()-210_000_000
    with pytest.raises(ExecutionError,match='torso_tracking'):obj.hold()
    obj.stop('operator_stop');obj.observer.wait()
    assert len(obj.observer.diagnostics['joint_names'])==14


def test_sdk_command_shape_and_stop_latch():
    backend=SDKBackend('/unused',250,control_way='filter');calls=[]
    backend.client=SimpleNamespace(get_control_rights_status=lambda:True,
        set_joints_position=lambda *a,**k:calls.append((a,k)),stop_robot=lambda:None)
    backend.torso_target=np.array([.6,-1.2,.6,0]);backend.send(np.zeros(10))
    args,kwargs=calls[-1];assert args[0][-1]=='astribot_torso'
    assert [len(v) for v in args[1]]==[7,1,2,4]
    assert kwargs==dict(control_way='filter',use_wbc=False,add_default_torso=False)
    backend.stop()
    with pytest.raises(ExecutionError,match='backend_stopped'):backend.send(np.zeros(10))
    assert len(calls)==1


def test_http_alignment_boundary(live):
    obj,events=live;core=ControlSession('test','boot',2000,'model','contract',events.append,lambda r:None,obj,execution_strategy='continuous')
    server=make_control_server(core,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
    try:
        status,_=control_exchange(conn,'POST','/start',encode(response()))
        assert status==202 and core.start_requested.wait(1)
        old=time.monotonic_ns();core.start()
        status,data=control_exchange(conn,'GET','/health');health=json.loads(data)
        assert status==200 and health['phase']=='AWAITING_PREDICTION'
        assert core.accept_observations_after_ns>old and core.runner.started_ns>=core.accept_observations_after_ns
        r=response(2);r['observation_receipt_ns']=np.full(5,old,dtype=np.int64)
        with pytest.raises(ValueError,match='predates'):core.accept(r)
    finally:
        core.stop();conn.close();server.shutdown();server.server_close();thread.join();core.close()


def test_hold_only_never_executes_prediction(live):
    obj,events=live;obj.c['torso_only']=True
    core=ControlSession('test','boot',2000,'model','contract',events.append,lambda r:None,obj,execution_strategy='continuous')
    try:
        core.request_start(response());core.confirm_start();core.start()
        assert core.health()['torso_only']
        time.sleep(.02)
        assert obj.backend.calls and obj.model_commands==0
        with pytest.raises(ValueError,match='torso_only'):core.accept(response(2))
    finally:core.stop();core.close()


@pytest.mark.parametrize('strategy',['synchronous','continuous'])
def test_http_reports_alignment_and_accepts_only_fresh_sensors(live,strategy):
    obj,events=live;core=ControlSession('test','boot',2000,'model','contract',events.append,lambda r:None,obj,execution_strategy=strategy)
    server=make_control_server(core,0);server_thread=threading.Thread(target=server.serve_forever,daemon=True);server_thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
    entered=threading.Event();release=threading.Event();original=obj.torso.run;errors=[]
    def paused(phase):
        phase('ALIGNING_TORSO');entered.set()
        assert release.wait(.05)
        original(phase)
    obj.torso.run=paused
    def start():
        try:core.start()
        except Exception as e:errors.append(e)
    worker=threading.Thread(target=start)
    try:
        r=response();r['execution_strategy']=np.asarray(strategy)
        status,_=control_exchange(conn,'POST','/start',encode(r));assert status==202
        worker.start();assert entered.wait(1)
        status,data=control_exchange(conn,'GET','/health');assert json.loads(data)['phase']=='ALIGNING_TORSO'
        assert core.runner.started_ns is None
        release.set();worker.join(1);assert not errors and not worker.is_alive()
        r=response(2);r['execution_strategy']=np.asarray(strategy)
        r['observation_receipt_ns']=np.full(5,int(r['observation_monotonic_ns']),dtype=np.int64)
        assert core.accept(r)['accepted']
    finally:
        release.set();core.stop();worker.join(1);conn.close();server.shutdown();server.server_close();server_thread.join();core.close()


def test_feedback_invalid_after_alignment(live):
    obj,_=live;obj.acquire();obj.align_torso(lambda p:None)
    obj.monitor.torso=np.array([np.nan,0.,0.,0.])
    with pytest.raises(ExecutionError,match='invalid_component_state:astribot_torso'):obj.hold()
    obj.monitor.torso=obj.torso.target.copy()


def test_no_second_motion_when_already_aligned(live):
    obj,_=live;obj.monitor.torso=obj.torso.target.copy()
    obj.acquire();assert obj.torso.skip
    stages=[];obj.align_torso(stages.append)
    assert set(stages)=={'SETTLING_TORSO'}


def test_non_home_arms_do_not_block_torso_alignment(live):
    obj,_=live
    obj.monitor.left=HOME_LEFT+.2
    obj.monitor.q[0]+=.2
    obj.baseline=obj.monitor.snapshot()
    obj.acquire();obj.align_torso(lambda phase:None)
    assert obj.torso.ready
    for target in obj.backend.calls:
        np.testing.assert_allclose(target,obj.monitor.q)
