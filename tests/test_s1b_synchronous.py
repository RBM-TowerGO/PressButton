"""Serial chunks, robot-clock waiting budgets and real HTTP without hardware."""
import http.client
import json
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from pressb.s1b_deployment.synchronous import SynchronousScheduler
from pressb.s1b_deployment.scheduler import ScheduleError, FakeExecutor, PERIOD_NS
from pressb.s1b_deployment.control_service import ControlSession, make_control_server
from pressb.s1b_deployment.client import control_exchange
from pressb.s1b_deployment.protocol import encode
from pressb.s1b_deployment.live_execution import ExecutionRunner
from test_s1b_realtime import prediction

BASE=1_000_000_000


def scheduler():
    s=SynchronousScheduler('test','model','contract',FakeExecutor())
    s.begin_wait(BASE)
    return s


def finish(s, start):
    rows=[]
    now=start
    while True:
        row=s.tick(now);rows.append(row)
        if row['event']=='chunk_completed':return rows,now
        now+=4_000_000
        assert now-start<260_000_000


def response(oid, observed):
    r=prediction(oid,observed)
    r.update(robot_boot_id=np.asarray('boot'),execution_strategy=np.asarray('synchronous'),
             observation_receipt_ns=np.full(5,observed,dtype=np.int64))
    return r


def test_three_slow_predictions_execute_every_step_from_execution_time():
    s=scheduler();observed=BASE
    for oid in range(1,4):
        received=observed+269_900_000
        assert s.accept(prediction(oid,observed),received)
        assert s.last_observation_ns==observed
        rows,completed=finish(s,received)
        assert [r['step_index'] for r in rows if r['event']=='chunk_step_started']==list(range(7))
        assert 233_333_333<=completed-received<=237_333_333
        assert s.progress()['completed_chunk_id']==oid
        assert s.progress()['completed_ns']==completed
        observed=completed+1
        # Model wait still ticks the servo and keeps the last target.
        for tick in range(completed+4_000_000,observed+269_900_000,4_000_000):
            assert not s.tick(tick)['covered']
    assert s.completed_chunks==3
    assert not s.counts['expired_on_receive']


def test_pending_chunk_rejects_replacement_and_out_of_order_observation():
    s=scheduler();s.accept(prediction(1,BASE),BASE)
    with pytest.raises(ScheduleError,match='chunk_in_progress'):
        s.accept(prediction(2,BASE+1),BASE+1)
    assert len(s.queue)==7 and s.chunk_id==1
    _,end=finish(s,BASE)
    with pytest.raises(ScheduleError,match='observation_out_of_order'):
        s.accept(prediction(1,end+1),end+1)
    with pytest.raises(ScheduleError,match='predates_chunk_completion'):
        s.accept(prediction(2,end),end+1)


@pytest.mark.parametrize('stage',['startup','between_chunks'])
def test_wait_budget_stops_without_client_or_more_predictions(stage):
    s=scheduler();start=BASE
    if stage=='between_chunks':
        s.accept(prediction(1,BASE),BASE)
        _,start=finish(s,BASE)
    assert not s.tick(start+999_999_999)['covered']
    with pytest.raises(ScheduleError,match='synchronous_prediction_timeout'):
        s.tick(start+1_000_000_000)
    assert s.state=='FAULT' and not s.queue


def test_jitter_does_not_skip_steps_but_control_stall_faults():
    s=scheduler();s.accept(prediction(1,BASE),BASE)
    starts=[];now=BASE
    for i in range(100):
        row=s.tick(now)
        if row['event']=='chunk_step_started':starts.append(row['step_index'])
        if row['event']=='chunk_completed':break
        now += [3_000_000,5_000_000,4_000_000][i%3]
    assert starts==list(range(7))
    s=scheduler();s.accept(prediction(1,BASE),BASE);s.tick(BASE)
    with pytest.raises(ScheduleError,match='synchronous_control_stall'):
        s.tick(BASE+PERIOD_NS+1)
    assert len(s.executor.calls)==1 and s.state=='FAULT'


def test_dispatch_must_succeed_and_stop_cannot_complete_chunk():
    s=scheduler();s.accept(prediction(1,BASE),BASE)
    row=s.sample(BASE)
    assert s.chunk_started_ns is None
    with pytest.raises(ScheduleError,match='synchronous_control_stall'):
        s.sent(row,BASE+PERIOD_NS)
    assert s.completed_chunks==0
    s=scheduler();s.accept(prediction(1,BASE),BASE);s.tick(BASE);s.stop()
    with pytest.raises(ScheduleError,match='session_not_active'):s.tick(BASE+4_000_000)
    assert s.completed_chunks==0 and not s.queue


def test_live_runner_wait_holds_without_starvation_warning():
    holds=[];s=scheduler()
    live=SimpleNamespace(c={'sdk_frequency':250},ensure_active=lambda:None,
                         hold=lambda:(holds.append(True) or np.zeros(10)))
    runner=ExecutionRunner(s,live,lambda r:None)
    row=runner.tick(BASE,BASE)
    assert row['held'] and row['event']=='synchronous_wait'
    assert holds==[True] and runner.starved_ticks==0


@pytest.mark.parametrize('invalid',['strategy','shape','nan','identity'])
def test_bad_sync_preflight_does_not_start(invalid):
    c=ControlSession('test','boot',2000,'model','contract',lambda r:None,lambda r:None,
                     execution_strategy='synchronous')
    r=response(1,time.monotonic_ns()-269_900_000)
    if invalid=='strategy':r['execution_strategy']=np.asarray('continuous')
    elif invalid=='shape':r['action_raw']=np.zeros((7,8),np.float32)
    elif invalid=='nan':r['action_raw'][0,0]=np.nan
    else:r['model_sha256']=np.asarray('other')
    with pytest.raises(ValueError):c.request_start(r)
    assert not c.start_requested.is_set() and not c.scheduler.queue
    c.close()


def test_shadow_http_handshake_slow_prediction_and_post_completion_observation():
    ticks=[]
    c=ControlSession('test','boot',2000,'model','contract',lambda r:None,ticks.append,
                     execution_strategy='synchronous')
    server=make_control_server(c,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=1)
    try:
        status,_=control_exchange(conn,'POST','/start',encode(response(1,time.monotonic_ns()-269_900_000)))
        assert status==202 and c.start_requested.wait(1)
        assert not c.scheduler.queue and c.runner.started_ns is None
        c.start()
        observed=time.monotonic_ns();time.sleep(.2699)
        status,data=control_exchange(conn,'POST','/prediction',encode(response(2,observed)))
        assert status==200 and json.loads(data)['chunk_id']==2
        until=time.monotonic()+1
        while time.monotonic()<until:
            status,data=control_exchange(conn,'GET','/health');health=json.loads(data)
            assert status==200 and not health['fault']
            if health['completed_chunk_id']==2:break
            time.sleep(.01)
        assert health['completed_chunk_id']==2 and health['chunk_phase']=='WAITING'
        assert [r['step_index'] for r in ticks if r['event']=='chunk_step_started']==list(range(7))
        stale=response(3,time.monotonic_ns())
        stale['observation_receipt_ns'][1]=health['completed_ns']
        status,data=control_exchange(conn,'POST','/prediction',encode(stale))
        assert status==400 and 'chunk_boundary' in json.loads(data)['error']
    finally:
        c.stop();conn.close();server.shutdown();server.server_close();thread.join(2);c.close()


def test_worker_requires_matching_strategy_and_all_new_streams():
    from test_s1b_clock_monitor import worker
    c=ControlSession('test','boot',2000,'model','contract',lambda r:None,lambda r:None)
    health=c.health()
    assert health['execution_strategy']=='synchronous'
    model=dict(health,execution_strategy='continuous')
    with pytest.raises(ValueError,match='execution_strategy_mismatch'):
        worker.checked_health(health,'test',model)
    selected={key:SimpleNamespace(receipt_ns=BASE+1) for key in
              ('head','right','arm','gripper','head_state')}
    assert worker.observation_after(selected,BASE)
    for key in selected:
        selected[key].receipt_ns=BASE
        assert not worker.observation_after(selected,BASE)
        selected[key].receipt_ns=BASE+1
    c.close()


def test_subperiod_delay_extends_grid_instead_of_bursting_next_steps():
    s=scheduler();s.accept(prediction(1,BASE),BASE)
    s.tick(BASE);s.tick(BASE+30_000_000)
    row=s.tick(BASE+62_000_000)
    assert row['step_index']==1 and s.schedule_slip_ns>0
    assert s.tick(BASE+66_000_000)['step_index']==1
    assert s.tick(BASE+90_000_000)['step_index']==1
    assert s.tick(BASE+98_000_000)['step_index']==2
