"""Deadline and fault behavior with no ROS, GPU or robot SDK."""
import threading
import time
import numpy as np
import pytest
from pressb.s1b_deployment.contract import VERSION,OFFSETS
from pressb.s1b_deployment.scheduler import Scheduler,ScheduleError,FakeExecutor
from pressb.s1b_deployment.shadow import ShadowRunner

BASE=1_000_000_000

def response(oid=1,observed=BASE,session='test'):
    return dict(protocol_version=np.asarray(VERSION),session_id=np.asarray(session),model_sha256=np.asarray('model'),data_contract_sha256=np.asarray('contract'),observation_id=np.int64(oid),observation_monotonic_ns=np.int64(observed),action_raw=np.arange(70,dtype=np.float32).reshape(7,10)/100,target_offsets_ns=OFFSETS.copy())

def scheduler():return Scheduler('test','model','contract',FakeExecutor())

def test_deadlines_are_not_rebased_and_tick_at_deadline():
    s=scheduler();r=response();assert s.accept(r,BASE+100_000_000)
    row=s.tick(BASE+int(OFFSETS[3]),BASE+int(OFFSETS[3]))
    assert row['covered'] and row['step_index']==3 and row['deadline_ns']==BASE+int(OFFSETS[3])
    assert row['deadline_error_ns']==0 and row['sdk_target'] is None
    assert row['simulated_sdk_target10'][7] in (0,100)
    assert len(s.executor.calls)==1

def test_receive_at_deadline_is_expired_but_last_point_can_be_used():
    s=scheduler();assert s.accept(response(),BASE+int(OFFSETS[-2]))
    assert len(s.queue)==1 and s.queue[0]['step_index']==6
    row=s.tick(BASE+200_000_001)
    assert row['covered'] and row['step_index']==6
    assert not s.tick(BASE+233_333_334)['covered']
    assert len(s.executor.calls)==1

@pytest.mark.parametrize('delay',[233_333_333,233_333_334,1_000_000_000])
def test_all_expired_never_replay_first_or_extend_last(delay):
    s=scheduler();assert not s.accept(response(),BASE+delay)
    assert not s.tick(BASE+delay)['covered'];assert not s.executor.calls


def test_latest_observation_replaces_remaining_old_chunk():
    s=scheduler();s.accept(response(),BASE)
    assert s.tick(BASE)['step_index']==0
    s.accept(response(2,BASE+30_000_000),BASE+40_000_000)
    row=s.tick(BASE+50_000_000)
    assert row['observation_id']==2 and row['step_index']==0
    assert s.stats()['counts']['replaced_points']==6


def test_no_future_within_one_tick_is_not_interpolated():
    s=scheduler();s.accept(response(),BASE)
    assert not s.tick(BASE-2)['covered']
    assert not s.executor.calls

@pytest.mark.parametrize('field,value',[('session_id',np.asarray('wrong')),('model_sha256',np.asarray('wrong')),('data_contract_sha256',np.asarray('wrong')),('protocol_version',np.asarray('wrong')),('action_raw',np.zeros((7,8),np.float32)),('action_raw',np.zeros((7,10),np.float64)),('action_raw',np.full((7,10),np.nan,np.float32)),('action_raw',np.full((7,10),np.inf,np.float32)),('target_offsets_ns',OFFSETS.astype(np.float32)),('observation_monotonic_ns',np.asarray(BASE+3)),('observation_id',np.asarray(-1))])
def test_bad_response_faults_and_clears_old_queue(field,value):
    s=scheduler();s.accept(response(),BASE);r=response(2,BASE+1);r[field]=value
    with pytest.raises(ScheduleError):s.accept(r,BASE+2)
    assert s.state=='FAULT' and not s.queue
    assert not s.tick(BASE+1)['covered'] and not s.executor.calls

@pytest.mark.parametrize('oid,observed',[(1,BASE+1),(0,BASE+2),(2,BASE)])
def test_duplicate_or_late_response_rejected(oid,observed):
    s=scheduler();s.accept(response(),BASE)
    with pytest.raises(ScheduleError,match='out_of_order'):s.accept(response(oid,observed),BASE+10)
    assert not s.queue


def test_disconnect_never_restarts_fake_execution():
    s=scheduler();s.accept(response(),BASE)
    s.enable_fake_execution(session_id='test',model_sha256='model',explicit=True)
    assert s.state=='EXECUTING'
    s.fault('tunnel_disconnected');assert not s.tick(BASE+1)['covered']
    with pytest.raises(ScheduleError,match='new_session'):s.reconnect_shadow('test')
    s.reconnect_shadow('new');assert s.state=='SHADOW' and not s.queue
    assert not s.tick(BASE+2)['covered']
    s.accept(response(2,BASE+2,session='new'),BASE+2)
    assert s.state=='SHADOW'  # reconnect did not arm execution


def test_fake_execution_queue_exhaustion_and_stop_are_latched():
    s=scheduler();s.accept(response(),BASE)
    s.enable_fake_execution(session_id='test',model_sha256='model',explicit=True)
    assert not s.tick(BASE+300_000_000)['covered'];assert s.state=='FAULT'
    assert s.executor.stops==['queue_exhausted']
    s.stop();assert s.state=='STOPPED'
    with pytest.raises(ScheduleError):s.accept(response(2,BASE+1),BASE+2)


def test_slow_network_does_not_block_tick_thread():
    s=scheduler();rows=[];started=threading.Event();release=threading.Event()
    def slow_server():started.set();release.wait(2)
    network=threading.Thread(target=slow_server);network.start();started.wait()
    runner=ShadowRunner(s,rows.append);runner.start()
    try:
        deadline=time.monotonic()+1
        while len(rows)<6 and time.monotonic()<deadline:time.sleep(.005)
        assert len(rows)>=6 and network.is_alive()
        s.fault('server_crashed');before=len(rows);time.sleep(.08)
        assert len(rows)>before and not any(r['covered'] for r in rows)
    finally:release.set();network.join();runner.close()
    assert s.state=='STOPPED'


def test_tick_logging_failure_faults_without_sdk_calls():
    s=scheduler()
    def broken_log(row):raise OSError('disk_full')
    runner=ShadowRunner(s,broken_log);runner.start();runner.thread.join(1)
    assert 'disk_full' in runner.error and s.state=='FAULT' and not s.executor.calls
    runner.close()

def test_blocked_request_cannot_keep_fake_execution_alive_after_expiry():
    s=scheduler();now=time.monotonic_ns();s.accept(response(observed=now),now)
    s.enable_fake_execution(session_id='test',model_sha256='model',explicit=True)
    release=threading.Event();network=threading.Thread(target=lambda:release.wait(2));network.start()
    rows=[];runner=ShadowRunner(s,rows.append);runner.start()
    try:
        until=time.monotonic()+1
        while s.state!='FAULT' and time.monotonic()<until:time.sleep(.005)
        assert s.state=='FAULT' and network.is_alive() and not s.queue
        count=len(s.executor.calls);time.sleep(.1)
        assert len(s.executor.calls)==count
        assert s.executor.stops==['queue_exhausted']
    finally:release.set();network.join();runner.close()
