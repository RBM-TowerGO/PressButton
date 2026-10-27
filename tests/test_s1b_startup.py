"""Execute startup across real sockets with slow fake SDK initialization."""
from collections import deque
import ctypes
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from pressb.s1b_deployment.control_service import ControlSession,make_control_server
from pressb.s1b_deployment.live_execution import LiveExecution
from pressb.s1b_deployment.client import control_exchange
from pressb.s1b_deployment.protocol import encode
from pressb.s1b_deployment.server import make_server
from pressb.s1b_deployment.observation import Sample,JpegFrame
from pressb.s1b_deployment.streams import LatestPublisher,camera_packet,state_packet
from test_s1b_live_execution import Backend,Monitor
from test_s1b_realtime import prediction
from test_s1b_split_transport import free_port

ROOT=Path(__file__).resolve().parents[1]


def live_backend(delay=0):
    config=json.loads((ROOT/'configs/s1b_execution_model_v1.json').read_text())
    config['stop_stability_window_ns']=5_000_000
    q=np.zeros(10);q[9]=.05343
    monitor=Monitor(q)
    class SlowBackend(Backend):
        def acquire(self):
            if delay:
                # Unlike time.sleep, PyDLL holds the GIL, reproducing native SDK
                # initialization that prevents this process serving HTTP briefly.
                ctypes.PyDLL(None).usleep(int(delay*1e6))
            super().acquire()
    backend=SlowBackend(q);events=[]
    live=LiveExecution(config,events.append,operator_present=True,estop_ready=True,monitor=monitor,backend=backend)
    live.baseline=monitor.snapshot()
    live.prepare=lambda:None
    return live,events


def response(oid=1,observed=None,model='model',contract='contract'):
    r=prediction(oid,time.monotonic_ns() if observed is None else observed)
    r.update(robot_boot_id=np.asarray('boot'),model_sha256=np.asarray(model),data_contract_sha256=np.asarray(contract))
    return r


def session(live,events):
    return ControlSession('test','boot',2000,'model','contract',events.append,lambda r:None,live)


def test_client_preflight_wait_does_not_acquire_or_start_one_second_timer():
    live,events=live_backend();core=session(live,events)
    server=make_control_server(core,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
    try:
        for _ in range(3):
            status,data=control_exchange(conn,'GET','/health')
            assert status==200 and json.loads(data)['phase']=='WAITING_FOR_CLIENT'
            time.sleep(.4)
        assert not live.backend.acquired and not live.backend.calls
        assert core.runner.started_ns is None
        status,_=control_exchange(conn,'POST','/start',encode(response()))
        assert status==202 and not live.backend.acquired
        assert core.start_requested.wait(1)
        core.start()
        assert core.runner.started_ns and not core.scheduler.queue
        assert live.model_commands==0
        assert core.accept(response(2))['accepted']
    finally:
        core.stop();conn.close();server.shutdown();server.server_close();thread.join(2);core.close()


@pytest.mark.parametrize('fault',['expired','identity','shape'])
def test_bad_preflight_never_acquires_sdk(fault):
    live,events=live_backend();core=session(live,events)
    r=response()
    if fault=='expired':r['observation_monotonic_ns']=np.int64(time.monotonic_ns()-1_000_000_000)
    elif fault=='identity':r['model_sha256']=np.asarray('wrong')
    else:r['action_raw']=np.zeros((7,8),np.float32)
    try:
        with pytest.raises(ValueError):core.request_start(r)
        assert not core.start_requested.is_set() and not live.backend.acquired
        assert not live.backend.calls
    finally:core.close()


def test_stop_during_prepare_prevents_later_acquisition():
    live,events=live_backend();core=session(live,events)
    entered=threading.Event();release=threading.Event()
    def prepare():entered.set();assert release.wait(2)
    live.prepare=prepare
    core.request_start(response());core.confirm_start();thread=threading.Thread(target=core.start);thread.start()
    try:
        assert entered.wait(1);core.stop();release.set();thread.join(1)
        assert not live.backend.acquired and core.runner.started_ns is None
        with pytest.raises(ValueError):core.request_start(response(2))
    finally:release.set();thread.join(2);core.close()


def test_start_cannot_acquire_until_http_acknowledgement_is_written():
    live,events=live_backend();core=session(live,events)
    try:
        core.request_start(response())
        with pytest.raises(ValueError,match='client_preflight_required'):core.start()
        assert not live.backend.acquired and not core.start_requested.is_set()
        core.stop()
        with pytest.raises(ValueError,match='start_not_allowed'):core.confirm_start()
        assert not core.start_requested.is_set()
    finally:core.close()


def test_failed_http_start_acknowledgement_never_triggers_acquisition():
    live,events=live_backend();core=session(live,events)
    server=make_control_server(core,0)
    reply=server.RequestHandlerClass.reply
    def fail_ack(handler,status,value):
        if status==202:raise ConnectionError('start_ack_write_failed')
        return reply(handler,status,value)
    server.RequestHandlerClass.reply=fail_ack
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
    try:
        status,_=control_exchange(conn,'POST','/start',encode(response()))
        assert status==400 and 'start_ack_write_failed' in core.error
        assert not core.start_requested.is_set() and not live.backend.acquired
        with pytest.raises(ValueError):core.start()
    finally:conn.close();server.shutdown();server.server_close();thread.join(2);core.close()


@pytest.mark.parametrize('fault',[False,True])
def test_stop_racing_inflight_tick_preserves_real_faults(fault):
    live,events=live_backend();core=session(live,events)
    entered=threading.Event();release=threading.Event()
    original_tick=core.runner.tick
    def tick(now,planned):
        entered.set();assert release.wait(1)
        if fault:raise RuntimeError('unrelated_driver_failure')
        return original_tick(now,planned)
    core.runner.tick=tick
    try:
        core.request_start(response());core.confirm_start();core.start()
        assert entered.wait(1);core.stop();release.set();core.runner.thread.join(1)
        assert bool(core.runner.error)==fault
        if fault:assert 'unrelated_driver_failure' in core.runner.error
        else:assert core.scheduler.state=='STOPPED'
        assert not live.backend.calls
    finally:release.set();core.close()


def test_preinitialization_chunk_cannot_be_replayed():
    live,events=live_backend();core=session(live,events)
    old=response();core.request_start(old);core.confirm_start();core.start()
    try:
        with pytest.raises(ValueError,match='predates_initialization'):core.accept(old)
        assert live.model_commands==0 and not live.active
    finally:core.close()


def test_first_prediction_timeout_still_stops_when_client_disappears_after_start():
    live,events=live_backend();core=session(live,events)
    core.request_start(response());core.confirm_start();core.start()
    try:
        until=time.monotonic()+2
        while live.active and time.monotonic()<until:time.sleep(.01)
        assert not live.active and 'first_fresh_prediction_timeout' in core.runner.error
        assert live.model_commands==0 and live.backend.stops>0
    finally:core.close()


@pytest.mark.parametrize('sdk_delay,slow_health',[(1.3,False),(10.2,False),(0,True)])
def test_actual_worker_with_native_blocking_sdk_start_and_real_streams(tmp_path,sdk_delay,slow_health):
    """Run the shipped workstation script, not a hand-written handshake client."""
    cv2=pytest.importorskip('cv2')
    identity=json.loads((ROOT/'configs/s1b_deployment_assets_v1.json').read_text())
    model=identity['models']['2000'];contract=identity['files']['data_contract.json']
    live,events=live_backend(delay=sdk_delay)
    core=ControlSession('test','boot',2000,model,contract,events.append,lambda r:None,live)
    slow_remaining=[3]
    health=core.health
    def delayed_health():
        if slow_health and live.model_commands>300 and slow_remaining[0]>0:
            slow_remaining[0]-=1;time.sleep(.065)
        return health()
    core.health=delayed_health
    control=make_control_server(core,0)
    http_thread=threading.Thread(target=control.serve_forever,daemon=True);http_thread.start()
    policy=SimpleNamespace(device='fake',identity=dict(step=2000,model_sha256=model,data_contract_sha256=contract),
        predict=lambda *args:(prediction(1,1)['action_raw'],dict(preprocess_seconds=0.,inference_seconds=.001)))
    policy_server=make_server(policy,0)
    model_thread=threading.Thread(target=policy_server.serve_forever,daemon=True);model_thread.start()
    ports={k:free_port() for k in ('head','right','states')}
    publisher=LatestPublisher({k:f'tcp://127.0.0.1:{v}' for k,v in ports.items()});publisher.start()
    stop=threading.Event();source_errors=[]
    jpeg={}
    for k,shape in [('head',(720,1280,3)),('right',(360,640,3))]:
        ok,jpg=cv2.imencode('.jpg',np.zeros(shape,np.uint8));assert ok
        jpeg[k]=JpegFrame(jpg.tobytes())
    def publish():
        seq=0;history={k:deque(maxlen=128) for k in ('arm','gripper','head_state')}
        try:
            while not stop.is_set():
                seq+=1;now=time.monotonic_ns()
                for k,dim in [('arm',7),('gripper',1),('head_state',2)]:
                    history[k].append(Sample(seq,now,now,now,{},np.zeros(dim,np.float32)))
                publisher.put('states',state_packet(history,'test','boot'))
                for k in jpeg:publisher.put(k,camera_packet(k,Sample(seq,now,now,now,{},jpeg[k]),'test','boot'))
                stop.wait(.02)
        except BaseException as exc:source_errors.append(exc)
    source_thread=threading.Thread(target=publish,daemon=True);source_thread.start()
    def initialize():
        if core.start_requested.wait(8):core.start()
    startup_thread=threading.Thread(target=initialize,daemon=True);startup_thread.start()
    output=tmp_path/'worker'
    try:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/s1b_split_inference.py'),
            '--session-id','test','--checkpoint','2000','--mode','execute','--duration','5' if slow_health else '1',
            '--head-port',str(ports['head']),'--right-port',str(ports['right']),
            '--state-port',str(ports['states']),'--control-port',str(control.server_port),
            '--policy-port',str(policy_server.server_port),'--output',str(output)],
            cwd=ROOT,capture_output=True,text=True,timeout=sdk_delay+15)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        summary=json.loads((output/'summary.json').read_text())
        assert summary['complete'] and summary['valid_requests']>5
        assert live.model_commands>0 and not core.runner.error
        preflight=next(r for r in events if r['event']=='client_preflight_ready')
        acquired=next(r for r in events if r['event']=='acquired')
        first=next(r for r in events if r['event']=='prediction_received')
        assert acquired['monotonic_ns']-preflight['monotonic_ns']>=int(sdk_delay*1e9)
        assert first['observation_monotonic_ns']>=core.accept_observations_after_ns
        assert first['received_ns']-core.runner.started_ns<1_000_000_000
        assert not source_errors
        if slow_health:
            probes=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines() if json.loads(line)['event']=='clock_probe']
            assert any(not row['accepted'] for row in probes)
            assert probes[-1]['accepted'] and slow_remaining[0]==0
    finally:
        stop.set();source_thread.join(2);startup_thread.join(2)
        core.stop();control.shutdown();control.server_close();http_thread.join(2);core.close()
        policy_server.shutdown();policy_server.server_close();model_thread.join(2);publisher.close()
