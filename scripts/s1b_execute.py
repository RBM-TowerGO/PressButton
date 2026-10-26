#!/usr/bin/env python3
"""Explicit, finite P4 motion probes and stop-path validation on S1B."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.executor import GuardedExecutor,SDKBackend,ExecutionError,PARTS,StopObserver
from pressb.s1b_deployment.contract import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/s1b_execution_p4_v1.json')
    p.add_argument('--profile',choices=['acquire-hold','joint-probe','gripper-endpoints','stop-test'],default='acquire-hold')
    p.add_argument('--joint',type=int,default=6,help='0-based joint10 index, excludes gripper index 7')
    p.add_argument('--amplitude',type=float,default=.004)
    p.add_argument('--duration',type=float,default=4.)
    p.add_argument('--stop-reason',choices=['manual','network_disconnect','queue_exhausted','service_failure','watchdog','rights_loss'],default='manual')
    p.add_argument('--enable-real',action='store_true')
    p.add_argument('--request-control-handover',action='store_true')
    p.add_argument('--operator-present',action='store_true')
    p.add_argument('--estop-ready',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if a.joint not in [0,1,2,3,4,5,6,8,9] or not 0<abs(a.amplitude)<=.006 or not 2<=a.duration<=30:raise ValueError('Invalid bounded probe parameters')
    if c['parts']!=PARTS or c['sdk_frequency']!=250:raise ValueError('Unsupported SDK contract')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'resolved_config.json').write_text(json.dumps(dict(config=c,sha256=sha(a.config),arguments=vars(a)),indent=2)+'\n')
    if not a.enable_real:
        (out/'summary.json').write_text(json.dumps(dict(mode='dry-run',sdk_constructed=False,profile=a.profile))+'\n')
        print('Dry-run: no SDK client constructed. Use explicit local enable flags for real probes.');return
    if not a.operator_present or not a.estop_ready:raise ExecutionError('local_operator_and_estop_required')
    commit=subprocess.check_output(['git','-C',c['sdk_root'],'rev-parse','HEAD'],text=True).strip()
    if commit!=c['sdk_commit']:raise ExecutionError('SDK version changed')
    streams={n:(out/(n+'.jsonl')).open('w',buffering=1) for n in ['events','monitor']};write_lock=threading.Lock()
    def emit(row):
        text=json.dumps(row,allow_nan=False)+'\n'
        with write_lock:streams['monitor' if row['event'] in ('state','command_topic','accepted_command') else 'events'].write(text)
    from pressb.s1b_deployment.robot_state import RobotState,ALL_PARTS
    state=RobotState(emit);backend=SDKBackend(c['sdk_root'],c['sdk_frequency']);executor=GuardedExecutor(c,backend,emit)
    started=time.monotonic_ns();failure=None;stop_ns=None;stop_velocity_ns=None;baseline=None;initial=None;measured_initial=None;hold_command=None;max_drift={};max_deviation=np.zeros(10);loop_intervals=[];last_tick=None;observer=None;source_process=None;source_socket=None;source_stream=None
    def handle_stop(signum,frame):executor.stop('operator_signal')
    signal.signal(signal.SIGINT,handle_stop);signal.signal(signal.SIGTERM,handle_stop)
    try:
        ready_until=time.monotonic()+6
        while True:
            try:initial,stamp=state.joint10();break
            except ExecutionError:
                if time.monotonic()>ready_until:raise
                time.sleep(.05)
        # Let the live graph converge; never rely on the ros2 daemon cache.
        time.sleep(.5)
        owners=state.owners()
        if owners:
            if not a.request_control_handover:raise ExecutionError('control_rights_already_owned:'+','.join(owners))
            emit(dict(event='requesting_control_handover',owners=owners,monotonic_ns=time.monotonic_ns()))
            state.request_handover();time.sleep(.1)
            until=time.monotonic()+3
            while state.owners() and time.monotonic()<until:time.sleep(.05)
            if state.owners():raise ExecutionError('control_handover_owner_did_not_release')
        initial,stamp=state.joint10();baseline=state.snapshot();hold_command=state.command10()
        if a.profile=='stop-test' and a.stop_reason in ('network_disconnect','queue_exhausted','service_failure'):
            base_path=out/'test_source_base.json';base_path.write_text(json.dumps(hold_command.tolist()))
            source_process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('s1b_p4_test_source.py')),'--base',str(base_path),'--joint',str(a.joint),'--amplitude',str(a.amplitude),'--duration',str(a.duration),'--fault',a.stop_reason],stdout=subprocess.PIPE,text=True)
            source_info=json.loads(source_process.stdout.readline())
            source_socket=socket.create_connection(('127.0.0.1',source_info['port']),timeout=1.);source_stream=source_socket.makefile('r')
            emit(dict(event='test_source_ready',**source_info))
        initial,stamp=state.joint10()
        executor.acquire(initial,stamp,operator_present=a.operator_present,estop_ready=a.estop_ready,initial_command=hold_command)
        time.sleep(.02)
        own=state.owners()
        if len(own)!=1 or not own[0].startswith('pressb_p4_execution_'):raise ExecutionError('control_rights_owner_not_exclusive:'+str(own))
        measured_initial=backend.position()
        initial=hold_command.copy();initial[7]=max(0,min(100,initial[7]))
        observer=StopObserver(executor,state.snapshot);observer.start()
        if source_socket:source_socket.sendall(b'S')
        emit(dict(event='ready',monotonic_ns=time.monotonic_ns(),state10=initial.tolist(),control_owners=own))
        start=time.monotonic();ordinal=0;phase='moving';profile_duration=a.duration if a.profile!='gripper-endpoints' else 12.
        while time.monotonic()-start<profile_duration:
            if executor.state!='EXECUTING':break
            now=time.monotonic_ns()
            if last_tick is not None:loop_intervals.append(now-last_tick)
            last_tick=now
            measured,stamp=state.joint10();snapshot=state.snapshot()
            if state.error:raise ExecutionError('state_monitor_failure:'+state.error)
            for part in ALL_PARTS:
                drift=float(np.max(np.abs(np.asarray(snapshot[part]['position'])-baseline[part]['position'])))
                max_drift[part]=max(max_drift.get(part,0),drift)
                if part not in PARTS and drift>c['uncontrolled_max_position_drift'].get(part,.003):raise ExecutionError('uncontrolled_component_drift:'+part)
            target=initial.copy();elapsed=time.monotonic()-start
            if a.profile in ('joint-probe','stop-test'):
                target[a.joint]+=a.amplitude*.5*(1-np.cos(2*np.pi*min(elapsed/a.duration,1.)))
            elif a.profile=='gripper-endpoints':
                # Explicit calibration trajectory, not policy-output smoothing.
                if elapsed<1.:target[7]=initial[7]
                elif elapsed<5.:target[7]=initial[7]+(100-initial[7])*(1-np.cos(np.pi*(elapsed-1)/4))*.5
                elif elapsed<6.:target[7]=100.
                elif elapsed<10.:target[7]=100*(1+np.cos(np.pi*(elapsed-6)/4))*.5
                else:target[7]=0.
            if source_stream:
                try:
                    packet=source_stream.readline()
                    if not packet:
                        executor.fault(a.stop_reason);break
                    packet=json.loads(packet)
                    if not 0<=time.monotonic_ns()-packet['source_ns']<=c['state_max_age_ns']:raise ExecutionError('stale_test_source')
                    target=np.asarray(packet['target'],np.float64)
                except (OSError,ValueError) as exc:
                    executor.fault(a.stop_reason+':'+str(exc));break
            max_deviation=np.maximum(max_deviation,np.abs(measured-measured_initial))
            executor.send(target,measured,stamp)
            if a.profile=='stop-test' and not source_stream and elapsed>=a.duration/4:
                if a.stop_reason=='watchdog':time.sleep(c['command_watchdog_ns']/1e9+.1)
                elif a.stop_reason=='rights_loss':state.request_handover();time.sleep(.12)
                elif a.stop_reason=='manual':executor.stop('operator_stop')
                else:executor.fault(a.stop_reason)
                break
            ordinal+=1
            delay=start+ordinal/c['sdk_frequency']-time.monotonic()
            if delay>0:time.sleep(delay)
            elif delay<-.03:raise ExecutionError('control_loop_stall')
        executor.stop('profile_complete');stop_ns=executor.stop_requested_ns
        stop_velocity_ns=observer.wait()
        time.sleep(.15)
        emit(dict(event='stopped_observation',monotonic_ns=time.monotonic_ns(),states=state.snapshot(),logical_stopped=backend.stopped(),rights=backend.rights()))
    except BaseException as exc:
        failure=repr(exc)
        if backend.client is not None and backend.rights():executor.fault('probe_error:'+str(exc))
        raise
    finally:
        if observer and executor.stop_requested_ns is None:executor.stop('cleanup')
        if source_stream:source_stream.close()
        if source_socket:source_socket.close()
        if source_process:
            if source_process.poll() is None:source_process.terminate()
            source_process.wait(timeout=3)
        try:executor.close()
        except BaseException as exc:failure=(failure or '')+';cleanup:'+repr(exc)
        try:state.close()
        except BaseException as exc:failure=(failure or '')+';monitor_close:'+repr(exc)
        summary=dict(sdk_constructed=backend.client is not None,control_acquired=executor.acquired_once,sent_commands=executor.sent_commands,profile=a.profile,joint=a.joint,stop_reason=a.stop_reason,elapsed_seconds=(time.monotonic_ns()-started)/1e9,executor_state=executor.state,fault_reason=executor.fault_reason,failure=failure,initial_state10=measured_initial.tolist() if measured_initial is not None else initial.tolist() if initial is not None else None,initial_command10=hold_command.tolist() if hold_command is not None else None,max_measured_delta=max_deviation.tolist(),max_component_drift=max_drift,stop_requested_ns=stop_ns,velocity_settled_ns=stop_velocity_ns,stop_delay_ns=stop_velocity_ns-stop_ns if stop_velocity_ns and stop_ns else None,control_loop_gap_p50_p95_p99_max_ns=np.percentile(loop_intervals,[50,95,99,100]).tolist() if loop_intervals else [],rights_released=not backend.rights(),sdk_commit=commit,physical_motion_enabled=True)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        for f in streams.values():f.close()
        print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
