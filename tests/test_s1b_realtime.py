"""Official-style SDK sends, independently timed from policy HTTP requests."""
import json
import http.client
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pressb.s1b_deployment.executor import SDKBackend, ExecutionError
from pressb.s1b_deployment.live_execution import LiveExecution, ExecutionRunner, validate_execution_config
from pressb.s1b_deployment.realtime import StartupBlend
from pressb.s1b_deployment.scheduler import Scheduler, ScheduleError, FakeExecutor
from pressb.s1b_deployment.contract import VERSION, OFFSETS
from test_s1b_live_execution import Backend, Monitor

ROOT=Path(__file__).resolve().parents[1]


def prediction(oid,observed):
    raw=np.zeros((7,10),np.float32);raw[:,9]=.27407
    return dict(protocol_version=np.asarray(VERSION),session_id=np.asarray('test'),
        model_sha256=np.asarray('model'),data_contract_sha256=np.asarray('contract'),
        observation_id=np.int64(oid),observation_monotonic_ns=np.int64(observed),
        action_raw=raw,target_offsets_ns=OFFSETS.copy())


def test_sdk_backend_uses_filter_and_preserves_controlled_parts():
    calls=[]
    b=SDKBackend('/unused',250,control_way='filter')
    b.client=SimpleNamespace(get_control_rights_status=lambda:True,
        set_joints_position=lambda *a,**kw:calls.append((a,kw)))
    b.send(np.arange(10,dtype=float))
    args,kw=calls[0]
    assert args[0]==['astribot_arm_right','astribot_gripper_right','astribot_head']
    assert args[1]==[list(range(7)),[7],[8,9]]
    assert kw==dict(control_way='filter',use_wbc=False,add_default_torso=False)


def test_transition_starts_at_accepted_command_and_uses_new_targets():
    b=StartupBlend(np.zeros(10),2.)
    first=np.ones(10);new=first*2
    q,alpha=b.target(first,1_000_000_000)
    assert alpha==0;np.testing.assert_array_equal(q,0)
    q,alpha=b.target(new,2_000_000_000)
    assert alpha==.5;np.testing.assert_array_equal(q,1)
    q,alpha=b.target(new,3_000_000_000)
    assert alpha==1;np.testing.assert_array_equal(q,new)


def test_sample_reuses_only_unexpired_30hz_targets():
    s=Scheduler('test','model','contract',FakeExecutor());base=1_000_000_000
    s.accept(prediction(1,base),base)
    s.enable_fake_execution(session_id='test',model_sha256='model',explicit=True)
    steps=[s.sample(base+i*4_000_000)['step_index'] for i in range(9)]
    assert steps==[0]*9
    assert s.sample(base+36_000_000)['step_index']==1
    assert s.sample(base+230_000_000)['step_index']==6
    with pytest.raises(ScheduleError,match='queue_exhausted'):s.sample(base+234_000_000)
    assert s.state=='FAULT' and s.executor.stops==['queue_exhausted']


def test_new_prediction_replaces_sampled_target_without_rebasing_time():
    s=Scheduler('test','model','contract',FakeExecutor());base=1_000_000_000
    s.accept(prediction(1,base),base)
    s.enable_fake_execution(session_id='test',model_sha256='model',explicit=True)
    s.sample(base+20_000_000)
    s.accept(prediction(2,base+30_000_000),base+50_000_000)
    row=s.sample(base+52_000_000)
    assert row['observation_id']==2
    assert row['deadline_ns']==base+30_000_000+int(OFFSETS[0])


def make_live(starvation_action='stop'):
    """Fake-hardware LiveExecution.

    Defaults to the historical 'stop' starvation policy so the safety tests keep
    pinning that behaviour; hold-mode tests pass 'hold' explicitly. The shipped
    config is checked separately in test_shipped_model_config_holds_on_starvation.
    """
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)  # Legacy joint10-only fixture; torso tested separately.
    c['stop_stability_window_ns']=5_000_000
    c['queue_exhausted_action']=starvation_action
    q=np.zeros(10);q[9]=.05343
    m=Monitor(q);m.command10=lambda:np.zeros(10)
    b=Backend(q);rows=[]
    obj=LiveExecution(c,rows.append,operator_present=True,estop_ready=True,monitor=m,backend=b)
    obj.baseline=m.snapshot();obj.acquire()
    return obj,rows


def test_invalid_model_target_not_hidden_by_zero_startup_blend():
    obj,_=make_live()
    try:
        row=dict(action_raw=[0.]*10,deadline_ns=time.monotonic_ns()+30_000_000)
        row['action_raw'][9]=2.
        with pytest.raises(ExecutionError,match='position_limit'):obj.send(row)
        assert not obj.backend.calls and obj.executor.state=='FAULT'
    finally:obj.close()


def test_250hz_thread_continues_during_network_wait_and_stops_at_horizon():
    # The main/inference thread sleeps between chunks; only runner may send SDK commands.
    obj,rows=make_live();s=Scheduler('test','model','contract');ticks=[]
    r=ExecutionRunner(s,obj,ticks.append)
    try:
        s.accept(prediction(1,time.monotonic_ns()),time.monotonic_ns())
        s.enable_execution(obj,time.monotonic_ns());r.start()
        for oid in range(2,6):
            time.sleep(.06)
            s.accept(prediction(oid,time.monotonic_ns()),time.monotonic_ns())
        assert obj.active and r.frequency==250
        # Broad lower bound distinguishes 250 Hz from the old 30 Hz loop.
        assert obj.model_commands>30
        assert any(t.get('step_index')==0 for t in ticks)
        assert all(t['control_way']=='filter' for t in ticks)
        before=obj.model_commands
        time.sleep(.30)  # Simulates inference/socket blocked beyond the last chunk.
        assert obj.model_commands>before and not obj.active
        assert s.state=='FAULT' and 'queue_exhausted' in obj.executor.fault_reason
        stopped_count=obj.model_commands
        time.sleep(.02);assert obj.model_commands==stopped_count
    finally:
        r.close();obj.close()


def test_shipped_model_config_holds_on_starvation():
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)  # Legacy joint10-only fixture; torso tested separately.
    assert c['queue_exhausted_action']=='hold'
    validate_execution_config(c)
    c['queue_exhausted_action']='ignore'
    with pytest.raises(ExecutionError,match='queue_exhausted_action'):validate_execution_config(c)


def test_starvation_holds_last_target_until_fresh_prediction():
    obj,_=make_live('hold');s=Scheduler('test','model','contract',starvation_action='hold')
    ticks=[];r=ExecutionRunner(s,obj,ticks.append)
    try:
        base=time.monotonic_ns();s.accept(prediction(1,base),base)
        s.enable_execution(obj,base)
        assert r.tick(time.monotonic_ns(),None)['covered']
        deadline=base+int(OFFSETS[-1])
        starved=None
        while starved is None:
            assert time.monotonic_ns()<deadline+50_000_000
            row=r.tick(time.monotonic_ns(),None)
            if row['covered']:assert row['observation_id']==1
            else:starved=row
            time.sleep(.005)
        # Past the last original deadline: hold instead of latching a stop.
        assert starved['held'] is True and starved['reason']=='queue_exhausted'
        assert s.state=='EXECUTING' and obj.active and obj.executor.fault_reason is None
        assert s.counts['starved_servo_ticks']>=1
        assert r.stats()['starvation_action']=='hold' and r.stats()['starved_ticks']>=1
        sent=obj.model_commands
        for _ in range(10):  # 100 ms of starved ticks keeps the SDK watchdog fed
            time.sleep(.01)
            held=r.tick(time.monotonic_ns(),None)
            assert held['held'] and not held['covered']
        assert obj.model_commands==sent and obj.active and obj.executor.fault_reason is None
        # A fresh prediction resumes model tracking on the same execution session.
        now=time.monotonic_ns();s.accept(prediction(2,now),now)
        fresh=r.tick(time.monotonic_ns(),None)
        assert fresh['covered'] and fresh['observation_id']==2
        assert obj.model_commands==sent+1 and r.stats()['max_starvation_ns']>0
    finally:r.close();obj.close()


def test_hold_mode_keeps_execution_alive_past_horizon_and_watchdog():
    obj,_=make_live('hold');s=Scheduler('test','model','contract',starvation_action='hold')
    ticks=[];r=ExecutionRunner(s,obj,ticks.append)
    try:
        base=time.monotonic_ns();s.accept(prediction(1,base),base)
        s.enable_execution(obj,base);r.start()
        time.sleep(.5)  # beyond the 233 ms horizon and the 100 ms command watchdog
        assert obj.active and s.state=='EXECUTING' and r.error is None
        assert obj.model_commands>0
        assert any(t.get('held') for t in ticks)
        assert r.stats()['starved_ticks']>10
        # The hold refreshed the SDK watchdog, so only an explicit stop ends it.
        assert obj.executor.fault_reason is None and not obj.backend.stops
    finally:r.close();obj.close()


def test_model_configuration_rejects_direct_or_bad_transition():
    c=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    c.pop('torso_alignment',None)  # Legacy joint10-only fixture; torso tested separately.
    c['control_way']='direct'
    with pytest.raises(ExecutionError,match='filter'):validate_execution_config(c)
    c['control_way']='filter';c['startup_blend_seconds']=float('nan')
    with pytest.raises(ExecutionError,match='startup_blend'):validate_execution_config(c)


def test_real_http_inference_and_servo_loop_with_delayed_response():
    from pressb.s1b_deployment.server import make_server
    from pressb.s1b_deployment.client import exchange
    from pressb.s1b_deployment.protocol import encode, decode
    from test_s1b_deployment import request
    delay=[.025]
    def predict(*args):
        time.sleep(delay[0])
        raw=np.zeros((7,10),np.float32);raw[:,9]=.27407
        return raw,dict(inference_seconds=delay[0])
    policy=SimpleNamespace(device='fake',identity=dict(step=2000,model_sha256='model',data_contract_sha256='contract'),predict=predict)
    server=make_server(policy,port=0)
    server_thread=threading.Thread(target=server.serve_forever,daemon=True);server_thread.start()
    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=1)
    obj,events=make_live();s=Scheduler('test','model','contract');ticks=[]
    runner=ExecutionRunner(s,obj,ticks.append);runner.start()
    try:
        fields=request();fields['data_contract_sha256']=np.asarray('contract')
        for oid in range(6):
            fields['observation_id']=np.int64(oid)
            fields['observation_monotonic_ns']=np.int64(time.monotonic_ns())
            status,body=exchange(connection,'POST','/predict',encode(fields))
            assert status==200
            assert s.accept(decode(body),time.monotonic_ns())
            if s.state=='SHADOW':s.enable_execution(obj,time.monotonic_ns())
        assert obj.active and obj.model_commands>15
        assert obj.model_commands>obj.policy_targets
        delay[0]=.30
        fields['observation_id']=np.int64(6)
        fields['observation_monotonic_ns']=np.int64(time.monotonic_ns())
        status,body=exchange(connection,'POST','/predict',encode(fields))
        assert status==200 and not obj.active
        with pytest.raises(ExecutionError,match='queue_exhausted'):obj.ensure_active()
        with pytest.raises(ScheduleError,match='session_not_active'):s.accept(decode(body),time.monotonic_ns())
        assert 'queue_exhausted' in obj.executor.fault_reason
    finally:
        runner.close();obj.close();connection.close()
        server.shutdown();server.server_close();server_thread.join(2)


def test_deadline_crossed_during_logging_reselects_without_empty_servo_tick():
    obj,events=make_live();s=Scheduler('test','model','contract')
    skipped_rows=[]
    runner=ExecutionRunner(s,obj,skipped_rows.append)
    original_emit=obj.executor.emit
    def cross_deadline(row):
        original_emit(row)
        if row['event']=='command':
            # Reproduce selection just before a 30 Hz boundary followed by a
            # delayed log write, with more valid points still queued.
            deadline=row['source']['deadline_ns']
            time.sleep(max(0,(deadline-time.monotonic_ns())/1e9)+.002)
            obj.executor.emit=original_emit
            assert not obj.backend.calls and obj.model_commands==0
            assert obj.executor.last_refresh_ns==refresh
            np.testing.assert_array_equal(obj.executor.guard.last,last)
    try:
        base=time.monotonic_ns()
        s.accept(prediction(1,base),base)
        s.enable_execution(obj,time.monotonic_ns())
        obj.executor.emit=cross_deadline
        last=obj.executor.guard.last.copy();refresh=obj.executor.last_refresh_ns
        sent=runner.tick(time.monotonic_ns(),None)
        assert len(skipped_rows)==1
        assert skipped_rows[0]['event']=='target_skipped'
        assert skipped_rows[0]['actual_send_ns'] is None
        assert obj.active and s.state=='EXECUTING'
        assert sent['step_index']==1 and sent['actual_send_ns'] is not None
        assert obj.model_commands==1 and len(obj.backend.calls)==1
        assert runner.stats()['expired_before_send']==1
        # Skipping never shifts the prediction horizon or allows stale replay.
        with pytest.raises(ScheduleError,match='queue_exhausted'):
            runner.tick(base+int(OFFSETS[-1])+1,None)
        assert not obj.active
    finally:
        runner.close();obj.close()


def test_expired_points_do_not_keep_watchdog_alive():
    from pressb.s1b_deployment.executor import TargetExpired
    obj,_=make_live()
    try:
        row=dict(action_raw=[0.]*10,deadline_ns=time.monotonic_ns()-1)
        with pytest.raises(TargetExpired):obj.send(row)
        assert obj.active and not obj.backend.calls
        until=time.monotonic()+.5
        while obj.active and time.monotonic()<until:
            time.sleep(.005)
        assert not obj.active and 'watchdog' in obj.executor.fault_reason
    finally:obj.close()


def test_blocked_log_disk_does_not_stall_sdk_loop():
    from pressb.s1b_deployment.async_log import AsyncLog
    from test_s1b_async_log import BlockedStream, Stream
    stream=BlockedStream();logger=AsyncLog({'events':stream,'ticks':Stream()},1_000_000)
    obj,_=make_live();s=Scheduler('test','model','contract')
    obj.executor.emit=lambda row:logger.log('events',row)
    runner=ExecutionRunner(s,obj,lambda row:logger.log('ticks',row))
    try:
        logger.log('events',{'event':'block_disk'})
        assert stream.entered.wait(1)
        base=time.monotonic_ns();s.accept(prediction(1,base),base)
        s.enable_execution(obj,time.monotonic_ns());runner.start()
        until=time.monotonic()+.15
        while obj.model_commands<20 and time.monotonic()<until:time.sleep(.002)
        assert obj.model_commands>=20 and obj.active and runner.error is None
        assert not stream.release.is_set()
    finally:
        stream.release.set()
        runner.close();obj.close();logger.close()
