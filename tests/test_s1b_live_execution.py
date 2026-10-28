"""Model-to-SDK integration with real guards and a fake robot backend."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pressb.s1b_deployment.contract import VERSION, OFFSETS, read_config
from pressb.s1b_deployment.executor import PARTS, ExecutionError
from pressb.s1b_deployment.robot_state import ALL_PARTS
from pressb.s1b_deployment.live_execution import LiveExecution, ExecutionRunner, validate_execution_config
from pressb.s1b_deployment.scheduler import Scheduler, ScheduleError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('sign',[-1,1])
def test_clip_all_targets_accepts_large_overshoot_on_every_model_axis(sign):
    from pressb.s1b_deployment.executor import CommandGuard
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c['clip_all_targets']=True
    validate_execution_config(c)
    g=CommandGuard(c,np.zeros(10),0)
    raw=np.full(10,sign*1000.)
    decoded,bounded=g.prepare_prediction(raw)
    expected=np.array(c['position_lower' if sign<0 else 'position_upper'])
    np.testing.assert_array_equal(bounded,expected)
    np.testing.assert_array_equal(raw,np.full(10,sign*1000.))
    assert decoded[7]==(0 if sign<0 else 100)
    with pytest.raises(ExecutionError,match='target_position_limit'):g.validate_target(decoded)
    c['clip_all_targets']=False
    with pytest.raises(ExecutionError,match='target_position_limit'):g.prepare_prediction(raw)


@pytest.mark.parametrize('raw',[[float('nan')]*10,[float('inf')]*10,[-float('inf')]*10,[0.]*9])
def test_clip_all_targets_still_rejects_invalid_predictions(raw):
    from pressb.s1b_deployment.executor import CommandGuard
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text());c['clip_all_targets']=True
    g=CommandGuard(c,np.zeros(10),0)
    with pytest.raises(ExecutionError,match='invalid_target'):g.prepare_prediction(raw)


def test_clip_all_targets_records_and_sends_bounded_prediction():
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None);c['clip_all_targets']=True
    q=np.array(c['position_upper']);monitor=Monitor(q);backend=Backend(q);rows=[]
    obj=LiveExecution(c,rows.append,operator_present=True,estop_ready=True,monitor=monitor,backend=backend)
    obj.baseline=monitor.snapshot()
    raw=[1000.]*10
    row=dict(action_raw=raw,observation_id=1,step_index=0,deadline_ns=time.monotonic_ns()+1_000_000_000)
    try:
        obj.acquire();obj.send(row)
        np.testing.assert_array_equal(backend.calls[-1],q)
        assert row['action_raw']==raw
        clipped=next(r for r in rows if r['event']=='prediction_position_clipped')
        assert len(clipped['joint_names'])==9  # Gripper retains binary decoding.
        assert clipped['bounded_target']==q.tolist()
        assert clipped['clip_delta']==row['clip_delta']
        assert obj.executor.state=='EXECUTING'
    finally:obj.close()


@pytest.mark.parametrize('target,accepted',[
    (-.78,True), (-.7814302444458008,True), (-.7968653440475464,True), (-.80,True),
    (-.800001,False), (-.8009642362594604,False), (.76,True), (.760001,True), (.78,True), (.80,True), (.800001,False), (float('nan'),False),
])
def test_prediction_overshoot_is_bounded_before_sdk(target,accepted):
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)
    bound=.76 if target>0 else -.78
    q=np.zeros(10);q[5]=bound
    backend=Backend(q);rows=[];monitor=Monitor(q)
    obj=LiveExecution(c,rows.append,operator_present=True,estop_ready=True,
                      monitor=monitor,backend=backend)
    obj.baseline=monitor.snapshot()
    raw=q.copy();raw[5]=target
    row=dict(action_raw=raw.tolist(),observation_id=4949,step_index=1,
             deadline_ns=time.monotonic_ns()+1_000_000_000)
    try:
        obj.acquire()
        if not accepted:
            with pytest.raises(ExecutionError):obj.send(row)
            assert not backend.calls and obj.executor.state=='FAULT' and backend.stops==1
            assert not any(r['event']=='prediction_position_clipped' for r in rows)
            return
        obj.send(row)
        assert backend.calls[-1][5]==bound
        assert row['action_raw'][5]==target and row['decoded_target'][5]==target
        assert row['bounded_target'][5]==bound
        assert row['clip_delta'][5]==pytest.approx(bound-target)
        clipped=[r for r in rows if r['event']=='prediction_position_clipped']
        assert len(clipped)==int(target!=bound)
        if clipped:
            assert clipped[0]['joint_names']==['astribot_arm_right_joint_6']
            assert clipped[0]['observation_id']==4949 and clipped[0]['step_index']==1
            assert clipped[0]['action_raw'][5]==target
            assert clipped[0]['clip_delta'][5]==pytest.approx(bound-target)
            command=next(r for r in rows if r['event']=='command')
            assert command['source']['bounded_target'][5]==bound
            assert command['source']['clip_delta']==row['clip_delta']
    finally:obj.close()


def test_prediction_clipping_does_not_relax_other_axes_or_strict_command_guard():
    from pressb.s1b_deployment.executor import CommandGuard
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    q=np.zeros(10);g=CommandGuard(c,q,0);q[5]=-.78143
    with pytest.raises(ExecutionError,match='target_position_limit'):g.validate_target(q)
    q[0]=c['position_lower'][0]-.000001
    with pytest.raises(ExecutionError,match='target_position_limit'):g.prepare_prediction(q)
    q[0]=0;c.pop('prediction_lower_clip_tolerance')
    with pytest.raises(ExecutionError,match='target_position_limit'):g.prepare_prediction(q)
    q[5]=.79
    with pytest.raises(ExecutionError,match='target_position_limit'):g.validate_target(q)
    q[0]=c['position_upper'][0]+.000001
    with pytest.raises(ExecutionError,match='target_position_limit'):g.prepare_prediction(q)
    q[0]=0;c.pop('prediction_upper_clip_tolerance')
    with pytest.raises(ExecutionError,match='target_position_limit'):g.prepare_prediction(q)


@pytest.mark.parametrize('field',['prediction_lower_clip_tolerance','prediction_upper_clip_tolerance'])
@pytest.mark.parametrize('value',[[0.]*9,[-.001]*10,[float('nan')]*10,[.001]*10])
def test_invalid_prediction_clip_configuration_rejected(value,field):
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c[field]=value
    with pytest.raises(ExecutionError,match=field):
        validate_execution_config(c)


class Backend:
    def __init__(self, q):
        self.q=q.copy(); self.calls=[]; self.acquired=False; self.stops=0; self.released=False
    def acquire(self): self.acquired=True
    def rights(self): return self.acquired
    def position(self): return self.q.copy()
    def send(self, target): self.calls.append(target.copy())
    def stop(self): self.stops+=1
    def stopped(self): return self.stops>0
    def release(self): self.acquired=False; self.released=True


class Monitor:
    error=None
    def __init__(self,q): self.q=q; self.stale=False; self.drift=0.; self.closed=False
    def owners(self): return []
    def control_rights_available(self, timeout=.2): return False
    def snapshot(self):
        now=time.monotonic_ns()-(1_000_000_000 if self.stale else 0)
        positions={p:np.zeros(1) for p in ALL_PARTS}
        positions.update(dict(zip(PARTS,[self.q[:7],self.q[7:8],self.q[8:]])))
        positions['astribot_torso']=np.array([self.drift])
        return {p:dict(position=q.tolist(),velocity=np.zeros(len(q)).tolist(),receipt_ns=now) for p,q in positions.items()}
    def joint10(self): return self.q.copy(), self.snapshot()[PARTS[0]]['receipt_ns']
    def command10(self): return self.q.copy()
    def close(self): self.closed=True


@pytest.fixture
def live():
    c=json.loads((ROOT/'configs/s1b_execution_p4_v1.json').read_text())
    c['stop_stability_window_ns']=5_000_000
    q=(np.asarray(c['position_lower'])+c['position_upper'])/2
    q[7]=100
    monitor=Monitor(q); backend=Backend(q); rows=[]
    obj=LiveExecution(c,rows.append,operator_present=True,estop_ready=True,monitor=monitor,backend=backend)
    obj.baseline=monitor.snapshot()
    yield obj,rows
    obj.close()
    assert backend.released and monitor.closed


def response(live, oid=1):
    now=time.monotonic_ns();raw=live.monitor.q.astype(np.float32);raw[7]=1.
    return dict(protocol_version=np.asarray(VERSION),session_id=np.asarray('test'),
        model_sha256=np.asarray('model'),data_contract_sha256=np.asarray('contract'),
        observation_id=np.int64(oid),observation_monotonic_ns=np.int64(now),
        action_raw=np.tile(raw,(7,1)),target_offsets_ns=OFFSETS.copy())


def test_real_scheduler_decodes_model_and_records_actual_send(live):
    obj,rows=live;obj.acquire();time.sleep(.004)
    s=Scheduler('test','model','contract');r=response(obj)
    s.accept(r,time.monotonic_ns());s.enable_execution(obj,time.monotonic_ns())
    row=s.tick(time.monotonic_ns())
    assert row['sdk_target'][7]==100 and row['action_raw'][7]==1
    assert row['actual_send_ns'] and row['simulated_send_ns'] is None
    np.testing.assert_allclose(obj.backend.calls[-1],obj.monitor.q,atol=1e-7)
    s.fault('network_or_response_failure')
    assert not obj.active and obj.backend.stops==1
    with pytest.raises(ScheduleError):s.accept(response(obj,2),time.monotonic_ns())
    with pytest.raises(ExecutionError):obj.acquire()


@pytest.mark.parametrize('fault',['stale','drift','position','gripper_jump','log_failure'])
def test_failure_never_sends_model_target(live,fault):
    obj,rows=live;obj.acquire();time.sleep(.004)
    raw=response(obj)['action_raw'][0].tolist()
    row=dict(action_raw=raw,deadline_ns=time.monotonic_ns()+30_000_000)
    if fault=='stale':obj.monitor.stale=True
    if fault=='drift':obj.monitor.drift=.1
    if fault=='position':raw[0]+=1
    if fault=='gripper_jump':raw[7]=0
    if fault=='log_failure':
        def fail(row):
            if row['event']=='command':raise OSError('disk_full')
        obj.executor.emit=fail
    with pytest.raises((ExecutionError,OSError)):obj.send(row)
    assert obj.backend.calls==[] and obj.backend.stops==1 and not obj.active
    obj.monitor.stale=False;obj.monitor.drift=0.


def test_startup_holds_then_arms_only_fresh_prediction(live):
    obj,_=live;obj.acquire();time.sleep(.004)
    s=Scheduler('test','model','contract');runner=ExecutionRunner(s,obj,lambda row:None)
    runner.started_ns=time.monotonic_ns()
    assert runner.tick(time.monotonic_ns(),None)['event']=='startup_hold'
    np.testing.assert_allclose(obj.backend.calls[-1],obj.monitor.q)
    r=response(obj);s.accept(r,time.monotonic_ns())
    s.enable_execution(obj,time.monotonic_ns());time.sleep(.004)
    assert runner.tick(time.monotonic_ns(),None)['sdk_target'][7]==100
    s.tick(int(r['observation_monotonic_ns'])+300_000_000)
    assert s.state=='FAULT' and obj.backend.stops==1


def test_startup_timeout_and_expired_arm(live):
    obj,_=live;obj.acquire()
    s=Scheduler('test','model','contract');r=response(obj);s.accept(r,time.monotonic_ns())
    with pytest.raises(ScheduleError,match='fresh_target'):
        s.enable_execution(obj,int(r['observation_monotonic_ns'])+300_000_000)
    runner=ExecutionRunner(s,obj,lambda row:None);runner.started_ns=time.monotonic_ns()-2_000_000_000
    with pytest.raises(ExecutionError,match='prediction_timeout'):runner.tick(time.monotonic_ns(),None)


def test_explicit_config_and_operator_required(live):
    obj,_=live
    c=read_config(ROOT/'configs/s1b_deployment_v1.yaml','execute',obj.c)
    assert c['mode']=='execute' and c['execution']==obj.c
    with pytest.raises(ExecutionError,match='operator'):
        LiveExecution(obj.c,lambda row:None,operator_present=False,estop_ready=True)
    for field,value in [('max_velocity',[float('nan')]*10),('position_upper',[100]*10),('max_target_delta',[-1]*10)]:
        bad=copy.deepcopy(obj.c);bad[field]=value
        with pytest.raises(ExecutionError):validate_execution_config(bad)


def test_deadline_expiring_during_log_write_is_rejected(live):
    obj,_=live;obj.acquire();time.sleep(.004)
    def slow_log(row):
        if row['event']=='command':time.sleep(.015)
    obj.executor.emit=slow_log
    row=dict(action_raw=response(obj)['action_raw'][0].tolist(),deadline_ns=time.monotonic_ns()+5_000_000)
    with pytest.raises(ExecutionError,match='expired'):obj.send(row)
    assert not obj.backend.calls and obj.model_commands==0
    assert obj.active and obj.backend.stops==0


def test_failed_stop_observation_still_releases_and_closes_monitor(live):
    obj,_=live;obj.acquire()
    observer=obj.observer
    def fail():raise ExecutionError('unsettled')
    obj.observer=SimpleNamespace(wait=fail)
    with pytest.raises(ExecutionError,match='unsettled'):obj.close()
    assert obj.backend.released and obj.monitor.closed
    obj.observer=observer


@pytest.mark.parametrize('script,extra',[
    ('start_s1b_inference.py',[]),
    ('s1b_robot_bridge.py',['--output','unused','--operator-present','--estop-ready','--no-schedule']),
])
def test_cli_rejects_incomplete_execution_before_connections(script,extra,tmp_path):
    import subprocess,sys
    result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--mode','execute',*extra],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2
    assert 'execute requires' in result.stderr
    assert 'rclpy' not in result.stderr and 'ssh:' not in result.stderr
    assert not (tmp_path/'unused').exists()


def test_model_position_bounds_use_hardware_not_probe_window():
    from pressb.s1b_deployment.executor import CommandGuard
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)  # Legacy joint10-only fixture; torso tested separately.
    validate_execution_config(c)
    assert c['position_lower']==c['hardware_lower']
    assert c['position_upper']==c['hardware_upper']
    # Hold the demonstrated head pose; no temporary position window rejects it.
    q=np.zeros(10);q[8:]=[-.04289121356201172,.2740719752807617]
    g=CommandGuard(c,q,1_000_000_000)
    np.testing.assert_allclose(g.check(q,q,1_033_333_333,1_033_333_333),q)
    q[9]=c['hardware_upper'][9]+.001
    with pytest.raises(ExecutionError,match='position_limit'):
        g.check(q,q,1_033_333_333,1_033_333_333)


def test_sdk_environment_preserves_ros_and_quotes_sdk_path(tmp_path):
    import os,subprocess,sys,shlex
    from start_s1b_inference import sdk_environment
    root=str(tmp_path/'SDK with spaces')
    script='import os,json; print(json.dumps({k:os.environ.get(k) for k in ["PYTHONPATH","LD_LIBRARY_PATH","ROBOT_TYPE","ASTRIBOT_SDK_ROOT","ROS_DOMAIN_ID","FASTRTPS_DEFAULT_PROFILES_FILE"]}))'
    env=dict(os.environ,PYTHONPATH='/existing/python',LD_LIBRARY_PATH='/existing/lib',ROBOT_TYPE='S1',ROS_DOMAIN_ID='25',FASTRTPS_DEFAULT_PROFILES_FILE='/existing/dds.xml')
    result=subprocess.run(['bash','-c',sdk_environment({'sdk_root':root})+'; '+shlex.quote(sys.executable)+' -c '+shlex.quote(script)],env=env,capture_output=True,text=True,check=True)
    actual=json.loads(result.stdout)
    assert actual['PYTHONPATH'].split(':')==[root,root+'/third_party/astribot_ros_middleware_py','/existing/python']
    assert actual['ASTRIBOT_SDK_ROOT']==root
    assert actual['LD_LIBRARY_PATH'].split(':')[0]=='/existing/lib'
    assert root+'/third_party/drake/lib' in actual['LD_LIBRARY_PATH'].split(':')
    assert actual['ROBOT_TYPE']=='S1' and actual['ROS_DOMAIN_ID']=='25'
    assert actual['FASTRTPS_DEFAULT_PROFILES_FILE']=='/existing/dds.xml'


def test_launcher_surfaces_robot_root_cause(tmp_path):
    from start_s1b_inference import bridge_failure
    log=tmp_path/'bridge.log'
    log.write_text('Traceback:\nModuleNotFoundError: No module named astribot_sdk\n')
    assert 'ModuleNotFoundError' in bridge_failure(log)


def test_service_without_discoverable_owner_blocks_acquisition(live):
    obj,_=live
    obj.monitor.control_rights_available=lambda timeout=.2:True
    assert obj.monitor.owners()==[]
    with pytest.raises(ExecutionError,match='already_owned'):obj.acquire()
    assert not obj.backend.acquired and not obj.backend.calls


@pytest.mark.parametrize('handover',[False,True])
def test_prepare_uses_service_discovery_for_explicit_handover(live,monkeypatch,handover):
    import pressb.s1b_deployment.live_execution as module
    obj,_=live;obj.request_handover=handover
    available=[True];calls=[]
    obj.monitor.control_rights_available=lambda timeout=.2:available[0]
    def transfer():
        calls.append('handover');available[0]=False
    obj.monitor.request_handover=transfer
    monkeypatch.setattr(module.subprocess,'check_output',lambda *args,**kwargs:obj.c['sdk_commit'])
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    if handover:
        obj.prepare();assert calls==['handover'] and not available[0]
    else:
        with pytest.raises(ExecutionError,match='request-control-handover'):obj.prepare()
        assert not calls and available[0]
    assert not obj.backend.acquired and not obj.backend.calls


def test_model_startup_reproduces_real_head_offset_and_refreshes_command():
    """Regression for live_1793000653899948050: head command 0, feedback .05343."""
    from pressb.s1b_deployment.executor import GuardedExecutor
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)  # Legacy joint10-only fixture; torso tested separately.
    q=np.array([-.16112,-.02306,1.41910,1.65911,.34378,.11477,-.12457,-.09607,-.00599,.05343])
    accepted=np.array([-.159,-.022,1.420,1.660,.345,.115,-.125,0.,0.,0.])
    backend=Backend(q);rows=[];e=GuardedExecutor(c,backend,rows.append);calls=[]
    def fresh_command():
        assert backend.acquired
        calls.append('after_sdk_acquire')
        return accepted.copy()
    try:
        e.acquire(q,time.monotonic_ns(),operator_present=True,estop_ready=True,initial_command=fresh_command)
        assert calls==['after_sdk_acquire'] and e.state=='EXECUTING'
        np.testing.assert_allclose(e.guard.last,accepted)
        time.sleep(.004)
        e.send(accepted,q,time.monotonic_ns())
        np.testing.assert_allclose(backend.calls[-1],accepted)
        assert rows[0]['initial_tracking_error'][9]==pytest.approx(-.05343)
        assert not any(r.get('event')=='stop' for r in rows)
    finally:e.close()
    assert backend.released


def test_model_transitions_to_demonstrated_head_and_gripper_with_hardware_rate():
    from pressb.s1b_deployment.executor import CommandGuard
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)  # Legacy joint10-only fixture; torso tested separately.
    q=np.zeros(10);q[9]=.05343
    g=CommandGuard(c,q,1_000_000_000);g.last[9]=0
    raw=np.zeros(10);raw[8:]=[-.04289121356201172,.2740719752807617];raw[7]=1
    previous=g.last.copy()
    for step in range(1,5):
        now=1_000_000_000+step*33_333_333
        sent=g.check(raw,q,now,now,binary_gripper=True)
        assert np.all(np.abs(sent-previous)<=np.asarray(c['hardware_max_velocity'])*.033333333+1e-9)
        g.commit(sent,now);previous=sent.copy()
    assert sent[9]==pytest.approx(raw[9])
    assert sent[7]==100
    np.testing.assert_array_equal(raw[7:],[1,-.04289121356201172,.2740719752807617])


@pytest.mark.parametrize('fault',['position','nan','stale'])
def test_model_profile_still_rejects_invalid_targets_and_stale_feedback(fault):
    from pressb.s1b_deployment.executor import CommandGuard
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)  # Legacy joint10-only fixture; torso tested separately.
    q=np.zeros(10);g=CommandGuard(c,q,1_000_000_000);target=q.copy();stamp=1_033_333_333
    if fault=='position':target[9]=1.3
    if fault=='nan':target[0]=np.nan
    if fault=='stale':stamp=0
    with pytest.raises(ExecutionError):g.check(target,q,stamp,1_033_333_333)
    np.testing.assert_array_equal(g.last,q)


def test_live_model_profile_holds_then_sends_fresh_head_prediction():
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)  # Legacy joint10-only fixture; torso tested separately.
    c['stop_stability_window_ns']=5_000_000
    q=np.zeros(10);q[9]=.05343
    monitor=Monitor(q);monitor.command10=lambda:np.zeros(10)
    backend=Backend(q);rows=[]
    obj=LiveExecution(c,rows.append,operator_present=True,estop_ready=True,monitor=monitor,backend=backend)
    obj.baseline=monitor.snapshot()
    try:
        obj.acquire();time.sleep(.004);obj.hold()
        assert backend.calls[-1][9]==0
        time.sleep(.01)
        raw=np.zeros(10);raw[9]=.27407
        row=dict(action_raw=raw.tolist(),deadline_ns=time.monotonic_ns()+30_000_000)
        obj.send(row)
        assert row['sdk_target'][9]==0 and row['startup_alpha']==0 and obj.model_commands==1
        time.sleep(.01)
        row=dict(action_raw=raw.tolist(),deadline_ns=time.monotonic_ns()+30_000_000)
        obj.send(row)
        assert 0<row['sdk_target'][9]<.27407
        command=[r for r in rows if r['event']=='command'][-1]
        assert command['control_way']=='filter' and command['source']['action_raw'][9]==.27407
        assert command['source']['startup_alpha']>0
    finally:obj.close()
    assert backend.released


def test_driver_error_latches_before_late_prediction_and_normal_status_cannot_clear(live):
    import threading
    from pressb.s1b_deployment.robot_state import RobotState
    obj,rows=live;obj.acquire()
    monitor=RobotState.__new__(RobotState)
    monitor.lock=threading.Lock();monitor.driver_fault=None;monitor.on_fault=obj.executor.fault;monitor.emit=rows.append;monitor.error=None
    monitor.driver_callback(SimpleNamespace(data=[13000000,13143020]))
    assert obj.active and monitor.driver_fault is None
    monitor.driver_callback(SimpleNamespace(data=[13000000,13243026]))
    assert obj.backend.stops==1 and monitor.driver_fault=='driver_error:13243026'
    with pytest.raises(ExecutionError,match='driver_error:13243026'):obj.ensure_active()
    monitor.driver_callback(SimpleNamespace(data=[13000000]))
    assert not obj.active and monitor.driver_fault=='driver_error:13243026'
    assert not obj.backend.calls


def test_driver_fault_stops_before_logging(live):
    import threading
    from pressb.s1b_deployment.robot_state import RobotState
    obj,_=live;obj.acquire();monitor=RobotState.__new__(RobotState)
    monitor.lock=threading.Lock();monitor.driver_fault=None;monitor.on_fault=obj.executor.fault;monitor.error=None
    def log(row):
        assert not obj.active
        raise OSError('disk_full')
    monitor.emit=log
    monitor.driver_callback(SimpleNamespace(data=[13000000,13243026]))
    assert obj.backend.stops==1 and 'disk_full' in monitor.error


def test_monitor_closed_before_sdk_release(live):
    obj,_=live;events=[]
    close=obj.monitor.close;release=obj.backend.release
    def monitor_close():events.append('monitor');close()
    def sdk_release():events.append('sdk');release()
    obj.monitor.close=monitor_close;obj.backend.release=sdk_release
    obj.acquire();obj.close()
    assert events[:2]==['monitor','sdk']
