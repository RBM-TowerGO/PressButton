"""Split deployment protocols, real ZMQ/HTTP sockets, and guarded action expiry."""
from collections import deque
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import numpy as np
import pytest
from pressb.s1b_deployment.streams import (
    pack,unpack,camera_packet,state_packet,StreamBuffer,LatestPublisher,StreamReceiver,RobotClock)
from pressb.s1b_deployment.observation import JpegFrame,ObservationError
from pressb.s1b_deployment.control_service import ControlSession,make_control_server
from pressb.s1b_deployment.client import exchange,control_exchange
from pressb.s1b_deployment.protocol import encode
from test_s1b_observation import sample
from test_s1b_realtime import prediction,make_live

ROOT=Path(__file__).resolve().parents[1]
MAPPING=json.loads((ROOT/'configs/s1b_observation_v1.json').read_text())


def packet(key,stamp=1_000_000_000,seq=1):
    s=sample(key,stamp,seq);s.value=JpegFrame(b'jpeg')
    return camera_packet(key,s,'test','boot')


def histories(stamps):
    return {k:[sample(k,stamp,seq) for seq,stamp in enumerate(stamps,1)]
            for k in ('arm','gripper','head_state')}


def test_delayed_images_match_state_history_despite_conflated_state_packets():
    b=StreamBuffer(MAPPING,'test','boot')
    stamps=[1_000_000_000+i*4_000_000 for i in range(50)]
    # Simulate receiving only the latest state packet; old samples needed by
    # the cameras still arrive in this packet, not as separate lossy messages.
    b.ingest('states',state_packet(histories(stamps),'test','boot'))
    for key in ('head','right'):
        s=sample(key,stamps[20],1,receipt=stamps[-1]);s.value=JpegFrame(b'jpeg')
        b.ingest(key,camera_packet(key,s,'test','boot'))
    selected,state=b.buffer.snapshot(stamps[-1]+1_000_000)
    assert selected['arm'].sequence==21
    assert selected['head'].source_ns==stamps[20]
    assert selected['head'].receipt_ns==stamps[-1]
    assert state.shape==(10,)
    b.ingest('states',state_packet(histories(stamps+[stamps[-1]+4_000_000]),'test','boot'))
    assert len(b.buffer.buffers['arm'])==51
    with pytest.raises(ObservationError,match='stale_head'):b.buffer.snapshot(stamps[-1]+81_000_000)


@pytest.mark.parametrize('service,boot',[('wrong','boot'),('test','wrong')])
def test_stream_restart_or_wrong_service_rejected(service,boot):
    with pytest.raises(ValueError,match='identity'):unpack(packet('head'),service,boot)


def test_misroute_duplicate_and_bad_packet_rejected():
    b=StreamBuffer(MAPPING,'test','boot')
    with pytest.raises(ValueError,match='route'):b.ingest('right',packet('head'))
    b.ingest('head',packet('head'))
    with pytest.raises(ValueError,match='sequence'):b.ingest('head',packet('head'))
    for payload in (b'',b'\xff'*5):
        with pytest.raises(ValueError):unpack(payload,'test','boot')


def test_nonfinite_or_out_of_order_state_history_rejected():
    for kind in ('nan','reversed'):
        h=histories([1_000_000_000,1_004_000_000])
        if kind=='nan':h['arm'][0].value[0]=np.nan
        else:h['arm'].reverse()
        with pytest.raises(ValueError,match='invalid_state_history'):
            StreamBuffer(MAPPING,'test','boot').ingest('states',state_packet(h,'test','boot'))


def test_clock_domains_conservative_age_and_drift_checks():
    clock=RobotClock();clock.update(1_000_000_000,1_004_000_000,10_002_000_000)
    # True offset is 9 s in this symmetric example; estimate is deliberately older.
    assert clock.now(1_005_000_000)>=10_005_000_000
    assert clock.rtt_ns==4_000_000
    clock.update(2_000_000_000,2_002_000_000,11_001_000_000)
    with pytest.raises(ValueError,match='changed'):clock.update(3_000_000_000,3_002_000_000,50_000_000_000)
    with pytest.raises(ValueError,match='stale'):clock.now(20_000_000_000)
    with pytest.raises(ValueError,match='slow'):RobotClock().update(0,60_000_000,70_000_000)


def free_port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def test_real_zmq_latest_packets_and_state_alignment():
    endpoints={k:f'tcp://127.0.0.1:{free_port()}' for k in ('head','right','states')}
    publisher=LatestPublisher(endpoints)
    receiver=StreamReceiver(endpoints,MAPPING,'test','boot')
    publisher.start();receiver.start()
    try:
        for seq in range(1,80):
            stamp=1_000_000_000+seq*4_000_000
            publisher.put('head',packet('head',stamp,seq))
            publisher.put('right',packet('right',stamp,seq))
            publisher.put('states',state_packet(histories([1_000_000_000+i*4_000_000 for i in range(1,seq+1)]),'test','boot'))
            time.sleep(.01);receiver.check()
            try:
                selected,state=receiver.store.buffer.snapshot(stamp+10_000_000)
                if selected['head'].sequence>=5:break
            except ObservationError:pass
        else:pytest.fail('no synchronized ZMQ observation')
        assert state.shape==(10,)
        assert receiver.store.wire_bytes['head']>0
        assert publisher.counts['sent_head']>0
    finally:receiver.close();publisher.close()


def core(live=None):
    c=ControlSession('test','boot',2000,'model','contract',lambda row:None,lambda row:None,live)
    c.request_start(response(0));c.confirm_start()
    if live:
        # make_live already acquires this fake backend; slow real acquisition
        # ordering is covered separately in test_s1b_startup.py.
        live.prepare=lambda:None
        live.acquire=lambda:None
    c.start()
    return c


def response(oid=1,observed=None):
    r=prediction(oid,time.monotonic_ns() if observed is None else observed)
    r['robot_boot_id']=np.asarray('boot');return r


def test_control_http_accepts_only_small_predictions_and_robot_deadlines():
    c=core();server=make_control_server(c,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=1)
    try:
        status,data=exchange(conn,'GET','/health');health=json.loads(data)
        assert status==200 and health['boot_id']=='boot' and health['mode']=='shadow'
        time.sleep(.31)
        old=response(observed=c.accept_observations_after_ns+1)
        status,data=exchange(conn,'POST','/prediction',encode(old))
        assert status==200 and not json.loads(data)['accepted']
        r=response(2);status,data=exchange(conn,'POST','/prediction',encode(r))
        assert status==200 and json.loads(data)['accepted']
        assert 0<json.loads(data)['horizon_remaining_ns']<234_000_000
        status,data=exchange(conn,'POST','/stop',b'{"session_id":"test"}')
        assert status==200 and c.stopping.is_set()
        status,_=exchange(conn,'POST','/prediction',encode(response(3)))
        assert status==400 and not c.health()['ready']
    finally:conn.close();server.shutdown();server.server_close();thread.join(2);c.close()


@pytest.mark.parametrize('field,value',[('robot_boot_id','other'),('session_id','other'),('model_sha256','other')])
def test_control_identity_failure_latches(field,value):
    c=core();r=response();r[field]=np.asarray(value)
    with pytest.raises(ValueError):c.accept(r)
    assert c.error and not c.health()['ready']
    with pytest.raises(ValueError,match='inactive'):c.accept(response(2))
    c.close()


def test_execute_controller_stops_at_original_horizon_without_network_updates():
    live,_=make_live();c=core(live)
    # The fake hardware has already been acquired by make_live.
    try:
        r=response();c.accept(r)
        until=time.monotonic()+1
        while live.active and time.monotonic()<until:time.sleep(.005)
        assert live.model_commands>0 and not live.active
        assert live.executor.fault_reason=='queue_exhausted'
        sent=len(live.backend.calls)
        with pytest.raises(ValueError):c.accept(response(2))
        assert len(live.backend.calls)==sent
        assert live.executor.stop_requested_ns>=int(r['observation_monotonic_ns'])+int(r['target_offsets_ns'][-1])
    finally:c.close()


def test_execute_controller_holds_and_accepts_prediction_after_horizon():
    live,_=make_live('hold');c=core(live)
    try:
        first=c.accept(response(1))
        assert first['accepted'] and live.active
        # No prediction for longer than the 233 ms horizon: the runner keeps the
        # last guarded target instead of latching queue_exhausted.
        time.sleep(.35)
        assert live.active and c.health()['fault'] is None and c.runner.error is None
        assert c.stats()['control_loop']['starved_ticks']>0
        assert c.stats()['control_loop']['starvation_action']=='hold'
        later=c.accept(response(2))
        assert later['accepted'] and live.active and live.model_commands>0
    finally:c.close()


def test_control_invalid_http_body_latches_fault():
    c=core();server=make_control_server(c,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=1)
    try:
        status,_=exchange(conn,'POST','/prediction',b'x'*65537)
        assert status==400 and c.error and not c.health()['ready']
    finally:conn.close();server.shutdown();server.server_close();thread.join(2);c.close()


def test_slow_health_probe_does_not_block_action_admission():
    c=core();entered=threading.Event();release=threading.Event();errors=[]
    original=c.health
    def slow_health():
        entered.set();assert release.wait(2);return original()
    c.health=slow_health
    server=make_control_server(c,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def probe():
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
        try:control_exchange(conn,'GET','/health')
        except BaseException as exc:errors.append(exc)
        finally:conn.close()
    health_thread=threading.Thread(target=probe);health_thread.start()
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=.5)
    try:
        assert entered.wait(1)
        status,data=control_exchange(conn,'POST','/prediction',encode(response()))
        assert status==200 and json.loads(data)['accepted']
        assert health_thread.is_alive()
    finally:
        release.set();health_thread.join(2);conn.close()
        server.shutdown();server.server_close();thread.join(2);c.close()
    assert not errors


def test_shadow_coverage_reports_original_deadline_gap():
    c=core();time.sleep(.23)
    c.accept(response(observed=c.accept_observations_after_ns+10_000_000))
    time.sleep(.02)
    c.accept(response(2))
    assert c.stats()['coverage_gaps']==1
    assert c.stats()['max_coverage_gap_ms']>0
    c.close()


def test_launcher_cannot_auto_start_execute_or_share_ssh_connection():
    from s1b_split_launch import tunnel_command
    cmd=tunnel_command('S1B',[6200])
    assert 'ControlPath=none' in cmd and 'ControlMaster=no' in cmd
    assert '127.0.0.1:6200:127.0.0.1:6200' in cmd
    result=subprocess.run([sys.executable,str(ROOT/'scripts/start_s1b_inference.py'),
        '--mode','execute','--operator-present','--estop-ready','--auto-start'],capture_output=True,text=True)
    assert result.returncode!=0 and 'restricted to shadow' in result.stderr
