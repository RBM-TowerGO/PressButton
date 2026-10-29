#!/usr/bin/env python3
"""Workstation-only observation alignment, RGB decoding and policy requests."""
import argparse
from collections import Counter
import http.client
import json
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.contract import VERSION, OFFSETS
from pressb.s1b_deployment.protocol import encode,decode
from pressb.s1b_deployment.client import exchange,control_exchange
from pressb.s1b_deployment.streams import StreamReceiver,RobotClock
from pressb.s1b_deployment.observation import ObservationError,selected_rgb
from pressb.s1b_deployment.async_log import AsyncLog


def checked_health(value, session, model):
    if model.get('execution_strategy') is not None and value.get('execution_strategy')!=model['execution_strategy']:
        raise ValueError('control_execution_strategy_mismatch')
    if value.get('fault'):raise ValueError('control_fault:'+str(value['fault']))
    if (not value['ready'] or value['session_id']!=session
            or value.get('startup_protocol')!=1
            or any(value[k]!=model[k] for k in ('protocol_version','step','model_sha256','data_contract_sha256','action_dim','horizon'))):
        raise ValueError('control_health_identity_or_fault:'+str(value))


class DurationElapsed(Exception):
    """Normal run deadline, including time spent in blocking inference/HTTP."""


def observation_after(selected, threshold):
    return threshold is None or all(sample.receipt_ns > threshold for sample in selected.values())


def wait_chunk(connection, session, model, boot_id, chunk_id, check=lambda:None):
    """No action retries: poll idempotent health until this exact chunk finishes."""
    until=time.monotonic()+2
    while time.monotonic()<until:
        check()
        status,data=control_exchange(connection,'GET','/health')
        if status!=200:raise ValueError('control_health_http')
        health=json.loads(data);checked_health(health,session,model)
        if health['boot_id']!=boot_id:raise ValueError('robot_rebooted')
        if health['completed_chunk_id']==chunk_id:return health
        if health['chunk_id']!=chunk_id:raise ValueError('control_chunk_identity_mismatch')
        if health['phase'] in ('STOPPED','FAULT'):raise ValueError('control_stopped_during_chunk')
        time.sleep(.01)
    raise TimeoutError('chunk_completion_timeout')


class SlowClockProbe(ValueError):
    """An unusable timing sample, not evidence that the robot clock changed."""


def calibrate(connection, clock, session, model, boot_id=None, emit=None):
    probes=[]
    for _ in range(3):
        sent=time.monotonic_ns();status,data=control_exchange(connection,'GET','/health');received=time.monotonic_ns()
        if status!=200:raise ValueError('control_health_http')
        health=json.loads(data);checked_health(health,session,model)
        if boot_id is not None and health['boot_id']!=boot_id:raise ValueError('robot_rebooted')
        probes.append((received-sent,sent,received,health))
    rtt,sent,received,health=min(probes,key=lambda x:x[0])
    if emit:emit(dict(event='clock_probe',rtts_ns=[p[0] for p in probes],accepted=rtt<=50_000_000))
    if rtt>50_000_000:raise SlowClockProbe('clock_probe_too_slow:'+str(rtt))
    clock.update(sent,received,health['robot_monotonic_ns'])
    return health


def wait_initialized(connection,clock,session,model,boot_id,timeout=90):
    """Poll only idempotent health reads while the SDK initializes.

    Streams are already receiving. A slow initialization response is not used
    for clock calibration; the next fast health response updates the clock.
    No stale preflight chunk or ambiguous action request is retried.
    """
    until=time.monotonic()+timeout
    previous_timeout=connection.timeout
    try:
        while time.monotonic()<until:
            connection.timeout=max(.001,until-time.monotonic())
            if connection.sock:connection.sock.settimeout(connection.timeout)
            sent=time.monotonic_ns()
            status,data=control_exchange(connection,'GET','/health')
            received=time.monotonic_ns()
            if status!=200:raise ValueError('control_health_http')
            health=json.loads(data);checked_health(health,session,model)
            if health['boot_id']!=boot_id:raise ValueError('robot_rebooted')
            if received-sent<=50_000_000:
                clock.update(sent,received,health['robot_monotonic_ns'])
                if health['phase']=='AWAITING_PREDICTION':return health
            if health['phase'] not in ('INITIALIZING','ALIGNING_TORSO','SETTLING_TORSO','AWAITING_PREDICTION'):
                raise ValueError('unexpected_startup_phase:'+health['phase'])
            time.sleep(.01)
        raise TimeoutError('control_initialization_timeout')
    finally:
        connection.timeout=previous_timeout
        if connection.sock:connection.sock.settimeout(previous_timeout)


class ClockMonitor:
    """Health/clock probes have their own connection and never pause inference."""
    def __init__(self,port,clock,session,model,boot_id,emit=None):
        self.port,self.clock,self.session,self.model,self.boot_id=port,clock,session,model,boot_id
        self.stop=threading.Event();self.error=None
        self.emit=emit
        self.thread=threading.Thread(target=self.run,daemon=True,name='s1b-clock')
    def start(self):self.thread.start()
    def run(self):
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=1)
        try:
            delay=3
            while not self.stop.wait(delay):delay=self.refresh(connection)
        except BaseException as exc:self.error=repr(exc)
        finally:connection.close()
    def refresh(self,connection):
        try:
            calibrate(connection,self.clock,self.session,self.model,self.boot_id,self.emit)
            return 3
        except SlowClockProbe:
            # Keep the last valid calibration, including its original expiry
            # and drift allowance. Retry health only, never an action request.
            self.clock.now()
            return .25
    def check(self):
        if self.error:raise RuntimeError('clock_monitor_failed:'+self.error)
        self.clock.now()  # A run of slow probes still stops at the 10 s limit.
    def close(self):
        self.stop.set();self.thread.join(4)
        if self.thread.is_alive():raise RuntimeError('clock_monitor_did_not_stop')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-id',required=True)
    p.add_argument('--checkpoint',type=int,required=True)
    p.add_argument('--mode',choices=['shadow','execute'],default='shadow')
    p.add_argument('--mapping',default='configs/s1b_observation_v1.json')
    p.add_argument('--identity',default='configs/s1b_deployment_assets_v1.json')
    p.add_argument('--head-port',type=int,default=55555)
    p.add_argument('--right-port',type=int,default=55557)
    p.add_argument('--state-port',type=int,default=55558)
    p.add_argument('--control-port',type=int,default=6200)
    p.add_argument('--policy-port',type=int,default=8796)
    p.add_argument('--duration',type=float,default=600)
    p.add_argument('--output',required=True)
    p.add_argument('--execution-strategy',choices=['synchronous','continuous'],default='synchronous')
    a=p.parse_args()
    if not 0<a.duration<=86400:p.error('invalid_duration')
    import cv2
    cv2.setNumThreads(1)
    mapping=json.loads(Path(a.mapping).read_text());identity=json.loads(Path(a.identity).read_text())
    from pressb.s1b_deployment.contract import validate_checkpoint
    validate_checkpoint(identity, a.checkpoint)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    writer=AsyncLog({k:(out/(k+'.jsonl')).open('w',buffering=65536) for k in ('requests','events')},mapping['log_disk_limit_bytes'])
    policy=http.client.HTTPConnection('127.0.0.1',a.policy_port,timeout=2)
    control=http.client.HTTPConnection('127.0.0.1',a.control_port,timeout=1)
    receiver=None;clock=RobotClock();clock_monitor=None;fatal=[];requests=[];rejects=Counter();last_sequence=-1
    started=None;stop_requested=False;health=None;boot_id=None
    last_archive=0.;image_bytes=0;image_drops=0
    def stop(*_):raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        status,data=exchange(policy,'GET','/health');model=json.loads(data)
        if (status!=200 or not model['ready'] or model['step']!=a.checkpoint
                or model['protocol_version']!=VERSION or model['action_dim']!=10 or model['horizon']!=7
                or model['model_sha256']!=identity['models'][str(a.checkpoint)]
                or model['data_contract_sha256']!=identity['files']['data_contract.json']):
            raise ValueError('policy_identity_mismatch')
        model['execution_strategy']=a.execution_strategy
        until=time.monotonic()+90
        while True:
            try:
                status,data=control_exchange(control,'GET','/health')
                if status!=200:raise ValueError('control_health_http')
                health=json.loads(data);checked_health(health,a.session_id,model)
                break
            except OSError:
                if time.monotonic()>=until:raise
                time.sleep(.05)
        boot_id=health['boot_id']
        if health['mode']!=a.mode:raise ValueError('control_mode_mismatch')
        health=calibrate(control,clock,a.session_id,model)
        receiver=StreamReceiver({k:f'tcp://127.0.0.1:{port}' for k,port in
            [('head',a.head_port),('right',a.right_port),('states',a.state_port)]},mapping,a.session_id,boot_id)
        receiver.start();preflight_started=time.monotonic();accept_after=None
        if health['phase']!='WAITING_FOR_CLIENT':raise ValueError('control_session_already_started')
        last_prediction_local_ns=None
        writer.log('events',dict(event='ready',control=health,clock_rtt_ns=clock.rtt_ns))
        print(json.dumps(dict(event='ready',session_id=a.session_id,mode=health['mode'])),flush=True)
        while started is None or time.monotonic()-started<a.duration:
            writer.check();receiver.check()
            if clock_monitor:clock_monitor.check()
            if started is not None and health.get('torso_only',False):
                time.sleep(.05)
                continue
            if started is None and time.monotonic()-preflight_started>5:
                raise RuntimeError('client_preflight_timeout:'+str(rejects))
            if last_prediction_local_ns is not None and time.monotonic_ns()-last_prediction_local_ns>mapping['max_request_gap_seconds']*1e9:
                raise RuntimeError('observation_or_inference_stream_stalled')
            try:selected,state=receiver.store.buffer.snapshot(clock.now(),last_sequence)
            except ObservationError as exc:
                rejects[str(exc)]+=1
                time.sleep(.002);continue
            head=selected['head'];last_sequence=head.sequence
            if a.execution_strategy=='synchronous':
                if not observation_after(selected,accept_after):
                    rejects['observation_predates_chunk_boundary']+=1;continue
            elif accept_after is not None and not observation_after(selected,accept_after):
                rejects['observation_predates_initialization']+=1;continue
            decode_started=time.monotonic_ns()
            try:images=selected_rgb(selected)
            except ObservationError as exc:
                rejects[str(exc)]+=1;continue
            decoded=time.monotonic_ns()
            f=dict(protocol_version=VERSION,session_id=a.session_id,observation_id=np.int64(head.sequence),
                observation_monotonic_ns=np.int64(head.receipt_ns),seed=np.int64(mapping['seed']),
                head=images['head'],right=images['right'],state10=state,
                source_time_ns=np.asarray([selected[k].source_ns for k in ('head','right','arm')],np.int64),
                mapped_time_ns=np.asarray([selected[k].mapped_ns for k in ('head','right','arm')],np.int64),
                model_sha256=model['model_sha256'],data_contract_sha256=model['data_contract_sha256'])
            body=encode(f);send=time.monotonic_ns();age=clock.now(send)-head.receipt_ns
            if not 0<=age<=mapping['head_age_ns']:
                rejects['stale_after_encoding']+=1;continue
            status,data=exchange(policy,'POST','/predict',body,{'Content-Type':'application/octet-stream'})
            returned=time.monotonic_ns()
            if status!=200:raise ValueError('policy_http_'+str(status)+':'+data.decode(errors='replace'))
            result=decode(data)
            for key in ('protocol_version','session_id','observation_id','observation_monotonic_ns','model_sha256','data_contract_sha256'):
                if result[key].item()!=np.asarray(f[key]).item():raise ValueError('prediction_identity_'+key)
            if (result['action_raw'].shape!=(7,10) or result['action_raw'].dtype!=np.float32
                    or not np.isfinite(result['action_raw']).all() or not np.array_equal(result['target_offsets_ns'],OFFSETS)):
                raise ValueError('invalid_prediction')
            result['robot_boot_id']=np.asarray(boot_id)
            result['execution_strategy']=np.asarray(a.execution_strategy)
            result['observation_receipt_ns']=np.asarray([selected[k].receipt_ns for k in
                ('head','right','arm','gripper','head_state')],dtype=np.int64)
            action=encode(result)
            if started is None:
                status,data=control_exchange(control,'POST','/start',action,{'Content-Type':'application/octet-stream'})
                if status!=202:raise ValueError('control_start_rejected:'+data.decode(errors='replace'))
                writer.log('events',dict(event='client_preflight_ready',observation_id=head.sequence))
                print(json.dumps(dict(event='initializing_control',mode=a.mode)),flush=True)
                health=wait_initialized(control,clock,a.session_id,model,boot_id)
                accept_after=health['accept_observations_after_ns']
                started=time.monotonic();next_status=started+5
                def duration_elapsed(*_):raise DurationElapsed()
                signal.signal(signal.SIGALRM,duration_elapsed)
                signal.setitimer(signal.ITIMER_REAL,a.duration)
                clock_monitor=ClockMonitor(a.control_port,clock,a.session_id,model,boot_id,
                    lambda row:writer.log('events',row));clock_monitor.start()
                writer.log('events',dict(event='control_initialized',control=health,clock_rtt_ns=clock.rtt_ns))
                # This preflight prediction predates acquisition and is never sent
                # to /prediction. Select a new post-initialization observation.
                continue
            row=dict(observation_id=head.sequence,observation_monotonic_ns=head.receipt_ns,
                source_mapped_ns={k:s.mapped_ns for k,s in selected.items()},
                source_sequence={k:s.sequence for k,s in selected.items()},state10=state.tolist(),
                head_age_at_policy_send_ns=age,decode_ns=decoded-decode_started,
                policy_rtt_ns=returned-send,inference_seconds=float(result['inference_seconds']),
                preprocess_seconds=float(result['preprocess_seconds']),action_raw=result['action_raw'].tolist(),
                action_bytes=len(action),action_send_local_ns=time.monotonic_ns())
            try:
                status,data=control_exchange(control,'POST','/prediction',action,{'Content-Type':'application/octet-stream'})
                row.update(action_response_local_ns=time.monotonic_ns(),control_reply=json.loads(data))
                if status!=200:raise ValueError('control_rejected:'+str(row['control_reply']))
            except DurationElapsed:raise
            except BaseException as exc:
                row['error']=repr(exc);writer.log('requests',row);raise
            last_prediction_local_ns=time.monotonic_ns()
            requests.append(row)
            if a.execution_strategy=='synchronous':
                def check_wait():
                    writer.check();receiver.check()
                    if clock_monitor:clock_monitor.check()
                try:
                    health=wait_chunk(control,a.session_id,model,boot_id,row['control_reply']['chunk_id'],check_wait)
                    accept_after=health['completed_ns']
                    row['chunk_completion']=health
                finally:
                    # Preserve admission evidence even if the run ends mid-chunk.
                    writer.log('requests',row)
            else:writer.log('requests',row)
            if time.monotonic()-last_archive>=mapping['image_save_interval_seconds']:
                if image_bytes+len(body)<=mapping['image_disk_limit_bytes']:
                    writer.archive(out/f'observation_{head.sequence:08d}.npz',body);image_bytes+=len(body)
                else:image_drops+=1
                last_archive=time.monotonic()
            if time.monotonic()>=next_status:
                print(json.dumps(dict(event='progress',requests=len(requests),
                    rtt_ms=row['policy_rtt_ns']/1e6,execution_strategy=a.execution_strategy,
                    **(dict(completed_chunks=health['completed_chunks']) if a.execution_strategy=='synchronous' else
                       dict(horizon_remaining_ms=row['control_reply']['horizon_remaining_ns']/1e6)))),flush=True)
                next_status=time.monotonic()+5
        if not requests and not health.get('torso_only',False):raise RuntimeError('no_predictions')
    except DurationElapsed:
        pass
    except KeyboardInterrupt:
        stop_requested=True
    except BaseException as exc:
        fatal.append(repr(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            status,data=control_exchange(control,'POST','/stop',json.dumps(dict(session_id=a.session_id)).encode())
            if status!=200:raise ValueError('stop_http_'+str(status))
        except Exception as exc:fatal.append('stop_request_failed:'+repr(exc))
        if clock_monitor:
            try:clock_monitor.close()
            except Exception as exc:fatal.append('clock_close_failed:'+repr(exc))
        if receiver:
            try:receiver.close()
            except Exception as exc:fatal.append('receiver_close_failed:'+repr(exc))
        policy.close();control.close()
        try:writer.close()
        except Exception as exc:fatal.append('log_close_failed:'+repr(exc))
        def pct(values):return np.percentile(values,[0,50,95,99,100]).tolist() if values else []
        elapsed=time.monotonic()-started if started else 0
        summary=dict(session_id=a.session_id,mode=health['mode'] if health else None,elapsed_seconds=elapsed,
            valid_requests=len(requests),requests_per_second=len(requests)/max(elapsed,.001),fatal=fatal,
            interrupted=stop_requested,selection_rejections=dict(rejects),
            image_bytes=image_bytes,image_drops=image_drops,
            wire_bytes=dict(receiver.store.wire_bytes) if receiver else {},
            packets=dict(receiver.store.packets) if receiver else {},
            buffer=receiver.store.buffer.stats() if receiver else {},
            decode_ms_p0_p50_p95_p99_max=pct([r['decode_ns']/1e6 for r in requests]),
            policy_rtt_ms_p0_p50_p95_p99_max=pct([r['policy_rtt_ns']/1e6 for r in requests]),
            execution_strategy=a.execution_strategy,
            completed_chunks=sum('chunk_completion' in r for r in requests),
            horizon_remaining_ms_p0_p50_p95_p99_max=pct([r['control_reply']['horizon_remaining_ns']/1e6 for r in requests
                if 'horizon_remaining_ns' in r['control_reply']]),
            sdk_imported=any(n.startswith('astribot_sdk') for n in sys.modules),
            torso_only=bool(health and health.get('torso_only',False)),
            complete=(bool(requests) or bool(health and health.get('torso_only',False))) and not fatal and not stop_requested and elapsed>=a.duration)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(event='finished',**summary)),flush=True)
        if fatal and sys.exc_info()[0] is None:raise RuntimeError(';'.join(fatal))


if __name__=='__main__':main()
