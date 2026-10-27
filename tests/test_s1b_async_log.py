import io
import json
import threading
import time
import pytest
from pressb.s1b_deployment.async_log import AsyncLog


class Stream(io.StringIO):
    def close(self):
        self.saved=self.getvalue()
        super().close()


class BlockedStream(Stream):
    def __init__(self):
        super().__init__()
        self.entered=threading.Event()
        self.release=threading.Event()
    def write(self,data):
        self.entered.set()
        assert self.release.wait(2)
        return super().write(data)


def test_blocked_disk_does_not_block_producer_and_close_drains(tmp_path):
    stream=BlockedStream();log=AsyncLog({'events':stream},10000)
    try:
        log.log('events',{'event':'first'})
        assert stream.entered.wait(1)
        completed=threading.Event()
        def produce():
            log.log('events',{'event':'second'})
            log.archive(tmp_path/'image.npz',b'npz-data')
            completed.set()
        thread=threading.Thread(target=produce);thread.start()
        assert completed.wait(.2)
        thread.join(1)
    finally:
        stream.release.set();log.close()
    assert [json.loads(x)['event'] for x in stream.saved.splitlines()]==['first','second']
    assert (tmp_path/'image.npz').read_bytes()==b'npz-data'


def test_full_queue_fails_instead_of_blocking_or_silently_dropping():
    stream=BlockedStream();log=AsyncLog({'events':stream},10000,capacity=1)
    try:
        log.log('events',{'i':0});assert stream.entered.wait(1)
        log.log('events',{'i':1})
        with pytest.raises(OSError,match='log_queue_full'):log.log('events',{'i':2})
        with pytest.raises(OSError,match='log_queue_full'):log.check()
    finally:
        stream.release.set()
        with pytest.raises(OSError,match='log_queue_full'):log.close()


@pytest.mark.parametrize('failure',['write','limit','json'])
def test_background_errors_are_reported_to_producer_and_close(failure):
    class FailedStream(Stream):
        def write(self,data):raise OSError('disk_failed')
    stream=FailedStream() if failure=='write' else Stream()
    log=AsyncLog({'events':stream},1 if failure=='limit' else 10000)
    log.log('events',{'value':float('nan') if failure=='json' else 1})
    until=time.monotonic()+1
    while log.error is None and time.monotonic()<until:time.sleep(.001)
    with pytest.raises(OSError,match='log_writer_failed'):log.log('events',{'next':1})
    with pytest.raises(OSError,match='log_writer_failed'):log.close()
    assert stream.closed


def test_telemetry_flood_keeps_latest_per_part_without_filling_critical_queue():
    stream=BlockedStream();log=AsyncLog({'events':stream},2_000_000,capacity=2)
    original={'event':'state','part':'arm','position':[0.]}
    try:
        log.log('events',{'event':'acquire'});assert stream.entered.wait(1)
        for i in range(10000):
            for event in ('state','accepted_command','command_topic'):
                log.log('events',dict(original,event=event,position=[float(i)]))
        log.log('events',{'event':'stop','reason':'operator'})
        stats=log.stats()
        assert stats['telemetry_received']==30000
        assert stats['telemetry_coalesced']==29997
        assert stats['latest_slots']==3 and stats['pending_critical']==1
        assert log.error is None
    finally:
        stream.release.set();log.close()
    rows=[json.loads(x) for x in stream.saved.splitlines()]
    assert [r['event'] for r in rows[:2]]==['acquire','stop']
    sampled=[r for r in rows if r['event'] in ('state','accepted_command','command_topic')]
    assert len(sampled)==3
    assert all(r['position']==[9999.] and r['telemetry_received_count']==10000 for r in sampled)
    assert 'telemetry_received_count' not in original


def test_driver_errors_and_invalid_sources_are_not_coalesced():
    stream=Stream();log=AsyncLog({'events':stream,'sources':Stream()},10000)
    try:
        for i in range(3):
            log.log('events',dict(event='driver_status',error_codes=[13243026],sequence=i))
            log.log('sources',dict(stream='head',valid=False,error='decode',sequence=i))
    finally:log.close()
    assert len(stream.saved.splitlines())==3
    assert len(log.streams['sources'].saved.splitlines())==3
    assert log.stats()['telemetry_received']==0
