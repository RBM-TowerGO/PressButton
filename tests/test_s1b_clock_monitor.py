"""Real HTTP timing failures must not replace a valid clock calibration."""
import http.client
import importlib.util
from pathlib import Path
import threading
import time
import pytest
from pressb.s1b_deployment.control_service import ControlSession,make_control_server
from pressb.s1b_deployment.streams import RobotClock

spec=importlib.util.spec_from_file_location('split_worker',Path(__file__).resolve().parents[1]/'scripts/s1b_split_inference.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


@pytest.fixture
def monitor():
    core=ControlSession('test','boot',2000,'model','contract',lambda r:None,lambda r:None)
    health=core.health;model=health();settings=dict(delay=.06,boot='boot',offset=0)
    def delayed_health():
        time.sleep(settings['delay'])
        row=health();row['boot_id']=settings['boot'];row['robot_monotonic_ns']+=settings['offset'];return row
    core.health=delayed_health
    server=make_control_server(core,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    clock=RobotClock();now=time.monotonic_ns();clock.update(now,now,now)
    events=[];watch=worker.ClockMonitor(server.server_port,clock,'test',model,'boot',events.append)
    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=1)
    yield watch,connection,settings,events
    connection.close();server.shutdown();server.server_close();thread.join(2);core.close()


def test_slow_round_discarded_without_refreshing_expiry_then_recovers(monitor):
    watch,conn,settings,events=monitor;at=watch.clock.at
    assert watch.refresh(conn)==.25
    assert watch.clock.at==at and watch.clock.rtt_ns==0
    watch.check()
    assert not events[-1]['accepted'] and min(events[-1]['rtts_ns'])>50_000_000
    settings['delay']=0
    assert watch.refresh(conn)==3
    assert watch.clock.at>at and events[-1]['accepted']


def test_repeated_slow_probes_still_fail_when_original_calibration_expires(monitor):
    watch,conn,_,_=monitor
    watch.clock.at=time.monotonic_ns()-10_000_000_000
    with pytest.raises(ValueError,match='calibration_stale'):watch.refresh(conn)
    with pytest.raises(ValueError,match='calibration_stale'):watch.check()


def test_bad_identity_is_not_hidden_by_slow_probe(monitor):
    watch,conn,settings,_=monitor;settings['boot']='rebooted'
    with pytest.raises(ValueError,match='robot_rebooted'):watch.refresh(conn)


def test_real_clock_jump_still_fails(monitor):
    watch,conn,settings,_=monitor;settings.update(delay=0,offset=1_000_000_000)
    with pytest.raises(ValueError,match='robot_clock_changed'):watch.refresh(conn)
