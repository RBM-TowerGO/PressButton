#!/usr/bin/env python3
"""Robot action server, independent of camera callbacks and model inference."""
import argparse
import json
import os
import queue
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.async_log import AsyncLog
from pressb.s1b_deployment.control_service import ControlSession, make_control_server
from pressb.s1b_deployment.executor import ExecutionError


def finish_service(out, summary, *, exit_sdk_process=False):
    """Persist results after explicit cleanup, before SDK object destruction.

    The standalone SDK worker must not run a second implicit teardown during
    Python finalization: Astribot.__del__ calls the interface destructor even
    after shutdown(), and native SDK objects also own background threads.
    Only this dedicated process bypasses finalizers; imported callers do not.
    """
    active_error = sys.exc_info()
    code = 1 if summary['fatal'] or active_error[0] is not None else 0
    direct_exit = exit_sdk_process and summary['sdk_imported']
    summary.update(process_exit_code=code,
                   process_exit_method='after_explicit_cleanup' if direct_exit else 'python')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(event='finished',**summary)),flush=True)
    if direct_exit:
        if active_error[0] is not None:
            traceback.print_exception(*active_error)
        elif summary['fatal']:
            print('Control service failed: '+'; '.join(summary['fatal']),file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


def main(*, exit_sdk_process=False):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torso-only',action='store_true')
    p.add_argument('--session-id',required=True)
    p.add_argument('--mode',choices=['shadow','execute'],default='shadow')
    p.add_argument('--checkpoint',type=int,required=True)
    p.add_argument('--identity',default='configs/s1b_deployment_assets_v1.json')
    p.add_argument('--execution-config',default='configs/s1b_execution_model_v1.json')
    p.add_argument('--operator-present',action='store_true')
    p.add_argument('--estop-ready',action='store_true')
    p.add_argument('--request-control-handover',action='store_true')
    p.add_argument('--port',type=int,default=6200)
    p.add_argument('--duration',type=float,default=720)
    p.add_argument('--output',required=True)
    p.add_argument('--execution-strategy',choices=['synchronous','continuous'],default='synchronous')
    a=p.parse_args()
    if not 0<a.duration<=86520:p.error('invalid_duration')
    if a.torso_only and a.mode!='execute':p.error('--torso-only requires execute')
    if a.mode=='execute' and (not a.operator_present or not a.estop_ready):
        p.error('execute requires --operator-present and --estop-ready')
    identity=json.loads(Path(a.identity).read_text())
    from pressb.s1b_deployment.contract import validate_checkpoint
    validate_checkpoint(identity, a.checkpoint)
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    writer=AsyncLog({k:(out/(k+'.jsonl')).open('w',buffering=65536) for k in ('events','ticks')},512*1024**2)
    warnings=queue.SimpleQueue()
    def emit(row):
        writer.log('events',row)
        if row.get('event')=='warning':warnings.put(row)
        elif row.get('event') in ('torso_progress','torso_aligned'):print(json.dumps(row),flush=True)
    def print_warnings():
        while True:
            try:row=warnings.get_nowait()
            except queue.Empty:return
            print(json.dumps(row),flush=True)
    live=None
    if a.mode=='execute':
        from pressb.s1b_deployment.live_execution import LiveExecution
        execution=json.loads(Path(a.execution_config).read_text())
        execution['torso_only']=a.torso_only
        if a.torso_only and not execution.get('torso_alignment',{}).get('enabled'):p.error('torso alignment required')
        live=LiveExecution(execution,emit,
            operator_present=a.operator_present,estop_ready=a.estop_ready,request_handover=a.request_control_handover)
    core=ControlSession(a.session_id,boot_id,a.checkpoint,identity['models'][str(a.checkpoint)],
        identity['files']['data_contract.json'],lambda row:writer.log('events',row),lambda row:writer.log('ticks',row),live,execution_strategy=a.execution_strategy)
    def stop(*_):core.stop()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    server=None;http_thread=None;fatal=[];started=time.monotonic()
    try:
        # Bind before any SDK acquisition; a port conflict must never take rights.
        server=make_control_server(core,a.port)
        if live:
            import rclpy
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        # Identity/clock requests work before SDK acquisition. The workstation
        # requests /start only after streams, clock and model preflight succeed.
        http_thread=threading.Thread(target=server.serve_forever,kwargs=dict(poll_interval=.02),daemon=True)
        http_thread.start()
        print(json.dumps(dict(event='ready',**core.health())),flush=True)
        started=time.monotonic()
        while not core.stopping.is_set() and time.monotonic()-started<a.duration:
            writer.check()
            print_warnings()
            if core.runner.error or core.error:raise RuntimeError(core.runner.error or core.error)
            if core.start_requested.is_set() and core.phase=='INITIALIZING':
                core.start()
                for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
                print(json.dumps(dict(event='control_initialized',**core.health())),flush=True)
            if live and live.executor.acquired_once:
                try:live.ensure_active()
                except ExecutionError as exc:
                    # /stop is served concurrently with this health check.
                    if not (core.stopping.is_set() and live.executor.state=='STOPPED'
                            and str(exc)=='operator_stop'):raise
            core.stopping.wait(.01)
    except BaseException as exc:
        fatal.append(repr(exc));core.fail('control_service_failure:'+str(exc));raise
    finally:
        shutdown_started=time.monotonic()
        def shutdown_stage(stage):
            print(json.dumps(dict(event='shutdown_stage',stage=stage,
                elapsed_seconds=time.monotonic()-shutdown_started)),flush=True)
        shutdown_stage('closing_control')
        try:core.close()
        except BaseException as exc:fatal.append(repr(exc))
        shutdown_stage('control_closed')
        print_warnings()
        if server:
            if http_thread:
                server.shutdown();http_thread.join(2)
            server.server_close()
        shutdown_stage('http_closed')
        if live:
            try:
                import rclpy
                if rclpy.ok():rclpy.shutdown()
            except BaseException as exc:fatal.append('ros_shutdown:'+repr(exc))
        shutdown_stage('ros_closed')
        try:writer.close()
        except BaseException as exc:fatal.append(repr(exc))
        shutdown_stage('writer_closed')
        summary=dict(core.stats(),fatal=fatal,elapsed_seconds=time.monotonic()-started,
            sdk_imported=any(n.startswith('astribot_sdk') for n in sys.modules))
        finish_service(out,summary,exit_sdk_process=exit_sdk_process)
        if fatal and sys.exc_info()[0] is None:raise RuntimeError(';'.join(fatal))


if __name__=='__main__':main(exit_sdk_process=True)
