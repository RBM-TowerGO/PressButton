import copy
import json
from pathlib import Path
import threading
import time
import numpy as np
import pytest
from pressb.s1b_deployment.executor import CommandGuard,GuardedExecutor,ExecutionError

ROOT=Path(__file__).resolve().parents[1]
def config():return json.loads((ROOT/'configs/s1b_execution_p4_v1.json').read_text())
def initial(c):
    x=(np.asarray(c['position_lower'])+c['position_upper'])/2;x[7]=0;return x

class Backend:
    def __init__(self,state):self.q=state;self.acquired=False;self.calls=[];self.stop_calls=0;self.releases=0
    def acquire(self):self.acquired=True
    def rights(self):return self.acquired
    def position(self):return self.q.copy()
    def send(self,q):self.calls.append(q.copy())
    def stop(self):self.stop_calls+=1
    def stopped(self):return self.stop_calls>0
    def release(self):self.releases+=1;self.acquired=False

@pytest.mark.parametrize('fault',['position','rate','delta','state_nan','target_nan','stale','feedback'])
def test_whole_command_rejected_without_partial_send(fault):
    c=config();q=initial(c);g=CommandGuard(c,q,1_000_000_000);target=q.copy();state=q.copy();now=1_004_000_000;stamp=now
    if fault=='position':target[0]=c['hardware_upper'][0]+1
    if fault=='rate':target[0]+=.001
    if fault=='delta':target[0]+=.003
    if fault=='state_nan':state[0]=np.nan
    if fault=='target_nan':target[0]=np.inf
    if fault=='stale':stamp=now-c['state_max_age_ns']-1
    if fault=='feedback':state[7]=106
    with pytest.raises(ExecutionError):g.check(target,state,stamp,now)
    np.testing.assert_array_equal(g.last,q)


def test_binary_decode_and_long_idle_cannot_authorize_jump():
    c=config();q=initial(c);g=CommandGuard(c,q,1_000_000_000)
    x=q.copy();x[7]=.49
    assert g.check(x,q,1_004_000_000,1_004_000_000,binary_gripper=True)[7]==0
    x[7]=.5
    with pytest.raises(ExecutionError):g.check(x,q,1_004_000_000,1_004_000_000,binary_gripper=True)
    x=q.copy();x[0]+=.002
    with pytest.raises(ExecutionError,match='rate'):g.check(x,q,2_000_000_000,2_000_000_000)


def test_operator_required_before_sdk_factory():
    c=config();q=initial(c);b=Backend(q);e=GuardedExecutor(c,b,lambda x:None)
    with pytest.raises(ExecutionError):e.acquire(q,time.monotonic_ns(),operator_present=False,estop_ready=True)
    assert not b.acquired


def test_watchdog_stops_independently_of_blocked_caller_and_never_rearms():
    c=config();c['command_watchdog_ns']=30_000_000;q=initial(c);b=Backend(q);rows=[];e=GuardedExecutor(c,b,rows.append)
    e.acquire(q,time.monotonic_ns(),operator_present=True,estop_ready=True)
    until=time.monotonic()+.3
    while not b.stop_calls and time.monotonic()<until:time.sleep(.005)
    assert e.state=='FAULT' and b.stop_calls==1 and b.stopped()
    with pytest.raises(ExecutionError,match='not_active'):e.send(q,q,time.monotonic_ns())
    e.close();assert b.releases==1 and b.stop_calls==1


def test_invalid_command_or_log_error_latches_stop_before_backend_send():
    for bad_log in [False,True]:
        c=config();q=initial(c);b=Backend(q)
        def emit(row):
            if bad_log and row['event']=='command':raise OSError('disk_full')
        e=GuardedExecutor(c,b,emit);e.acquire(q,time.monotonic_ns(),operator_present=True,estop_ready=True)
        x=q.copy()
        if not bad_log:x[1]+=1
        with pytest.raises((ExecutionError,OSError)):e.send(x,q,time.monotonic_ns())
        assert e.state=='FAULT' and not b.calls and b.stop_calls==1
        e.close()


def test_rights_loss_and_manual_stop_are_idempotent():
    c=config();q=initial(c);b=Backend(q);e=GuardedExecutor(c,b,lambda x:None)
    e.acquire(q,time.monotonic_ns(),operator_present=True,estop_ready=True);b.acquired=False
    until=time.monotonic()+.2
    while e.state!='FAULT' and time.monotonic()<until:time.sleep(.005)
    assert e.state=='FAULT';e.stop();e.close();assert b.stop_calls==1

def test_command_monitor_uses_robot_joint_controller_command_field():
    from types import SimpleNamespace
    from pressb.s1b_deployment.robot_state import RobotState
    m=RobotState.__new__(RobotState);m.lock=threading.Lock();m.states={};m.commands={};rows=[];m.emit=rows.append
    header=SimpleNamespace(stamp=SimpleNamespace(sec=1,nanosec=2),frame_id='')
    packet=SimpleNamespace(header=header,command=[.1,.2],mode=1,name=[])
    m.callback('astribot_head',packet,True)
    assert m.commands['astribot_head']['position']==[.1,.2]
    assert rows[0]['event']=='command_topic'

def test_preserve_accepted_hold_target_instead_of_feedback_offset():
    c=config();q=initial(c);q[8]-=.003;q[9]+=.005
    command=q.copy();command[8]+=.003;command[9]-=.005
    b=Backend(q);e=GuardedExecutor(c,b,lambda x:None)
    e.acquire(q,time.monotonic_ns(),operator_present=True,estop_ready=True,initial_command=command)
    np.testing.assert_allclose(e.guard.last,command)
    time.sleep(.005);e.send(command,q,time.monotonic_ns());e.close()
    np.testing.assert_allclose(b.calls[0],command)


def test_stop_during_acquisition_cannot_later_enable_execution():
    c=config();q=initial(c);b=Backend(q);entered=threading.Event();release=threading.Event();failures=[]
    def acquire():entered.set();release.wait(1);b.acquired=True
    b.acquire=acquire;e=GuardedExecutor(c,b,lambda x:None)
    def start():
        try:e.acquire(q,time.monotonic_ns(),operator_present=True,estop_ready=True)
        except Exception as exc:failures.append(str(exc))
    t=threading.Thread(target=start);t.start();assert entered.wait(1);e.stop();release.set();t.join(1)
    assert e.state=='FAULT' and failures==['acquisition_cancelled'] and not b.acquired and not b.calls

def test_stop_feedback_observer_runs_while_caller_is_blocked():
    from types import SimpleNamespace
    from pressb.s1b_deployment.executor import StopObserver,PARTS
    c=config();c['stop_stability_window_ns']=20_000_000;c['max_stop_delay_ns']=150_000_000
    e=SimpleNamespace(c=c,stop_requested_ns=time.monotonic_ns())
    def snapshot():
        now=time.monotonic_ns()
        return {p:dict(receipt_ns=now,position=[0.]*n,velocity=[0.]*n) for p,n in zip(PARTS,[7,1,2])}
    observer=StopObserver(e,snapshot);observer.start();time.sleep(.08)
    settled=observer.wait()
    assert settled-e.stop_requested_ns<60_000_000


def test_stop_observer_rejects_motion_that_never_settles():
    from types import SimpleNamespace
    from pressb.s1b_deployment.executor import StopObserver,PARTS
    c=config();c['stop_stability_window_ns']=10_000_000;c['max_stop_delay_ns']=40_000_000
    e=SimpleNamespace(c=c,stop_requested_ns=time.monotonic_ns())
    def snapshot():
        now=time.monotonic_ns()
        return {p:dict(receipt_ns=now,position=[0.]*n,velocity=[100.]*n) for p,n in zip(PARTS,[7,1,2])}
    observer=StopObserver(e,snapshot);observer.start()
    with pytest.raises(ExecutionError,match='settle'):observer.wait()
    assert observer.diagnostics['velocity_violations']==list(range(10))
    assert observer.diagnostics['sample_count']>1 and not observer.diagnostics['settled']


def test_stop_observer_cannot_pass_using_feedback_returned_after_deadline():
    from types import SimpleNamespace
    from pressb.s1b_deployment.executor import StopObserver,PARTS
    c=config();c['stop_stability_window_ns']=5_000_000;c['max_stop_delay_ns']=30_000_000
    e=SimpleNamespace(c=c,stop_requested_ns=time.monotonic_ns());calls=[]
    def snapshot():
        if calls:time.sleep(.04)
        calls.append(True);now=time.monotonic_ns()
        return {p:dict(receipt_ns=now,position=[0.]*n,velocity=[0.]*n) for p,n in zip(PARTS,[7,1,2])}
    observer=StopObserver(e,snapshot);observer.start()
    with pytest.raises(ExecutionError,match='settle'):observer.wait()
    assert not observer.diagnostics['settled']
    assert observer.diagnostics['last_feedback_ns']>e.stop_requested_ns+c['max_stop_delay_ns']


def test_stop_logger_failure_preserves_original_fault_and_releases():
    c=config();q=initial(c);b=Backend(q);e=GuardedExecutor(c,b,lambda row:None)
    e.acquire(q,time.monotonic_ns(),operator_present=True,estop_ready=True)
    def failed_logger(row):raise OSError('log_queue_full')
    e.emit=failed_logger
    try:
        with pytest.raises(OSError,match='^log_queue_full$'):
            e.send(q,q,time.monotonic_ns())
        assert e.fault_reason=='command_rejected:log_queue_full'
        assert e.stop_log_error=="OSError('log_queue_full')"
        assert b.stop_calls==1 and not b.calls
    finally:e.close()
    assert b.releases==1 and e.state=='FAULT'
