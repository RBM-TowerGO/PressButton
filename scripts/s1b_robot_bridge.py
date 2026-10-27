#!/usr/bin/env python3
"""Live S1B observations and explicit guarded model execution."""
import argparse
from collections import Counter
import http.client
import json
from pathlib import Path
import sys
import signal
import threading
import time
import uuid
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.contract import VERSION,OFFSETS,canonical,read_config,sha
from pressb.s1b_deployment.observation import Sample,TimeMapper,ObservationBuffer,ObservationError,JpegFrame,selected_rgb
from pressb.s1b_deployment.protocol import encode,decode,MAX_BYTES
from pressb.s1b_deployment.scheduler import Scheduler,ScheduleError
from pressb.s1b_deployment.shadow import ShadowRunner
from pressb.s1b_deployment.client import exchange
from pressb.s1b_deployment.async_log import AsyncLog

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/s1b_deployment_v1.yaml')
    p.add_argument('--mapping',default='configs/s1b_observation_v1.json')
    p.add_argument('--mode',choices=['shadow','execute'],default='shadow')
    p.add_argument('--duration',type=float,default=600)
    p.add_argument('--output',required=True)
    p.add_argument('--schedule',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--execution-config',default='configs/s1b_execution_model_v1.json')
    p.add_argument('--operator-present',action='store_true')
    p.add_argument('--estop-ready',action='store_true')
    p.add_argument('--request-control-handover',action='store_true')
    p.add_argument('--checkpoint',type=int,choices=[2000,8000])
    a=p.parse_args()
    if not 0<a.duration<=86400:p.error('duration must be in (0,86400]')
    if a.mode=='execute' and (not a.schedule or not a.operator_present or not a.estop_ready):
        p.error('execute requires scheduling, --operator-present and --estop-ready')
    execution_config=json.loads(Path(a.execution_config).read_text()) if a.mode=='execute' else None
    c=read_config(a.config,a.mode,execution_config)
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    mapping=json.loads(Path(a.mapping).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('resolved_config',c),('mapping',mapping)]:
        (out/(name+'.json')).write_text(json.dumps(dict(value=value,sha256=canonical(value)),indent=2)+'\n')
    logs={name:(out/(name+'.jsonl')).open('w',buffering=65536) for name in ('sources','requests','events','ticks')}
    writer=AsyncLog(logs,mapping['log_disk_limit_bytes'])
    log=writer.log
    buffer=ObservationBuffer(mapping['buffer_capacity'],mapping['head_age_ns'],mapping['state_skew_ns'],mapping['right_skew_ns'])
    mapper=TimeMapper(time.time_ns(),time.monotonic_ns(),mapping['clock']['max_offset_jump_ns'])
    callback_counts=Counter();errors=Counter();fatal=[];intervals={k:[] for k in buffer.buffers};last_stamp={};max_age={};min_age={}
    import rclpy
    from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy,DurabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from astribot_msgs.msg import RobotJointState
    import cv2
    # Small image operations must not fan out across every robot CPU core.
    cv2.setNumThreads(1)
    log('events',dict(event='camera_decode_config',opencv_threads=cv2.getNumThreads(),
        decode_location='selected_observation',callback_payload='jpeg'))
    rclpy.init();node=rclpy.create_node('pressb_s1b_readonly_bridge')
    qos=QoSProfile(depth=5,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,durability=DurabilityPolicy.VOLATILE)
    def receive(key,msg):
        receipt=time.monotonic_ns();wall=time.time_ns();callback_counts[key]+=1
        header=dict(sec=msg.header.stamp.sec,nanosec=msg.header.stamp.nanosec,frame_id=msg.header.frame_id)
        source=header['sec']*10**9+header['nanosec']
        row=dict(stream=key,sequence=callback_counts[key],header=header,receipt_ns=receipt,wall_ns=wall)
        try:
            mapped=mapper.map(source,wall,receipt)
            if key in mapping['cameras']:
                if 'jpeg' not in msg.format.lower():raise ObservationError('unsupported_camera_format')
                value=JpegFrame(bytes(msg.data))
                row.update(format=msg.format,encoded_bytes=len(value.data),decode_deferred=True)
            else:
                expected=mapping['states'][key]['names'];names=list(msg.name)
                if names!=expected and not (not names and mapping['allow_empty_names_with_versioned_mapping']):raise ObservationError('state_names_'+key)
                value=np.asarray(msg.position,dtype=np.float32)
                if value.shape!=(len(expected),):raise ObservationError('state_length_'+key)
                row.update(names=names,position=value.tolist(),mapping_version=mapping['version'])
            buffer.add(key,Sample(callback_counts[key],source,mapped,receipt,header,value))
            if key in last_stamp:intervals[key].append(source-last_stamp[key])
            last_stamp[key]=source
            age=receipt-mapped;max_age[key]=max(age,max_age.get(key,age));min_age[key]=min(age,min_age.get(key,age))
            row.update(mapped_ns=mapped,source_to_receipt_ns=age,valid=True)
        except ObservationError as exc:
            errors[str(exc)]+=1;row.update(valid=False,error=str(exc))
        except Exception as exc:
            fatal.append(repr(exc));row.update(valid=False,error=repr(exc))
        try:log('sources',row)
        except Exception as exc:
            error=repr(exc)
            if error not in fatal:fatal.append(error)
    for key,topic in mapping['cameras'].items():node.create_subscription(CompressedImage,topic,lambda m,k=key:receive(k,m),qos)
    for key,spec in mapping['states'].items():node.create_subscription(RobotJointState,spec['topic'],lambda m,k=key:receive(k,m),qos)
    def spin():
        try:rclpy.spin(node)
        except Exception as exc:
            if rclpy.ok():fatal.append(repr(exc))
    thread=threading.Thread(target=spin,daemon=True);thread.start()
    expected=json.loads(Path(c['model']['identity']).read_text());conn=None;health=None;reconnections=0
    def connect():
        nonlocal conn,health,reconnections
        if conn:conn.close();reconnections+=1
        conn=http.client.HTTPConnection('127.0.0.1',c['network']['port'],timeout=c['network']['timeout_seconds'])
        status,data=exchange(conn,'GET','/health')
        if status!=200:raise ValueError('health_failed')
        candidate=json.loads(data)
        if health is not None and any(candidate.get(k)!=health.get(k) for k in ('model_sha256','data_contract_sha256','step')):raise RuntimeError('health_changed_requires_new_session')
        health=candidate
        if a.checkpoint is not None and health['step']!=a.checkpoint:raise RuntimeError('checkpoint_mismatch')
        contract_hash=expected['files']['data_contract.json']
        if not health['ready'] or health['protocol_version']!=VERSION or health['action_dim']!=10 or health['horizon']!=7 or health['model_sha256']!=expected['models'][str(health['step'])] or health['data_contract_sha256']!=contract_hash:raise ValueError('health_identity_mismatch')
        log('events',dict(event='health',health=health,robot_monotonic_ns=time.monotonic_ns()))
    session=str(uuid.uuid4());start=time.monotonic();started_ns=time.monotonic_ns();last_sequence=-1;last_saved=0.;image_bytes=0;image_drops=0;requests=[];rejects=Counter();last_rejection_log_ns=0;timeouts=0;response_errors=0;wire_bytes=0;max_inflight=0
    scheduler=None;runner=None;live=None
    try:
        connect()
        if a.mode=='execute':
            from pressb.s1b_deployment.live_execution import LiveExecution,ExecutionRunner
            live=LiveExecution(execution_config,lambda row:log('events',row),
                operator_present=a.operator_present,estop_ready=a.estop_ready,
                request_handover=a.request_control_handover)
            live.prepare()
        if a.schedule:
            scheduler=Scheduler(session,health['model_sha256'],health['data_contract_sha256'])
            runner=(ExecutionRunner(scheduler,live,lambda row:log('ticks',row)) if live else ShadowRunner(scheduler,lambda row:log('ticks',row)))
            if live is None:runner.start()
        print(json.dumps(dict(event='ready',session=session,step=health['step'],control_hz=execution_config['sdk_frequency'] if live else None,control_way=execution_config.get('control_way') if live else None)),flush=True)
        while time.monotonic()-start<a.duration:
            writer.check()
            if fatal:raise RuntimeError(';'.join(fatal))
            if runner and runner.error:raise RuntimeError(runner.error)
            if live and live.executor.acquired_once and (not live.active or scheduler.state in ('FAULT','STOPPED')):
                raise RuntimeError('execution_stopped:'+str(live.executor.fault_reason))
            if mapper.failed:raise RuntimeError('clock_jump')
            selection_started_ns=time.monotonic_ns()
            try:selected,state=buffer.snapshot(selection_started_ns,last_sequence)
            except ObservationError as exc:
                rejects[str(exc)]+=1
                rejected_ns=time.monotonic_ns()
                if rejected_ns-last_rejection_log_ns>=100_000_000:
                    log('events',dict(event='observation_rejected',reason=str(exc),
                        monotonic_ns=rejected_ns,after_sequence=last_sequence,buffer=buffer.stats()))
                    last_rejection_log_ns=rejected_ns
                time.sleep(.005);continue
            selected_ns=time.monotonic_ns()
            head=selected['head'];last_sequence=head.sequence
            decode_started_ns=time.monotonic_ns()
            try:images=selected_rgb(selected)
            except ObservationError as exc:
                rejects[str(exc)]+=1
                log('events',dict(event='rejected',reason=str(exc),observation_id=head.sequence))
                continue
            decoded_ns=time.monotonic_ns()
            row=dict(session_id=session,observation_id=head.sequence,observation_monotonic_ns=head.receipt_ns,source_headers={k:v.header for k,v in selected.items()},source_receipt_ns={k:v.receipt_ns for k,v in selected.items()},source_mapped_ns={k:v.mapped_ns for k,v in selected.items()},source_sequence={k:v.sequence for k,v in selected.items()},state10=state.tolist(),buffer=buffer.stats(),seed=mapping['seed'],sdk_target=None)
            f=dict(protocol_version=VERSION,session_id=session,observation_id=np.int64(head.sequence),observation_monotonic_ns=np.int64(head.receipt_ns),seed=np.int64(mapping['seed']),head=images['head'],right=images['right'],state10=state,source_time_ns=np.array([selected[k].source_ns for k in ('head','right','arm')],np.int64),mapped_time_ns=np.array([selected[k].mapped_ns for k in ('head','right','arm')],np.int64),state_component_source_ns=np.array([selected[k].source_ns for k in ('arm','gripper','head_state')],np.int64),model_sha256=health['model_sha256'],data_contract_sha256=health['data_contract_sha256'])
            encode_started_ns=time.monotonic_ns()
            body=encode(f);send_ns=time.monotonic_ns();row['send_ns']=send_ns;row['head_receipt_age_at_send_ns']=send_ns-head.receipt_ns
            row.update(selection_started_ns=selection_started_ns,selected_ns=selected_ns,decode_started_ns=decode_started_ns,decoded_ns=decoded_ns,encode_started_ns=encode_started_ns)
            if send_ns-head.receipt_ns>mapping['head_age_ns']:
                rejects['stale_after_encoding']+=1;log('events',dict(event='rejected',reason='stale_after_encoding',observation_id=head.sequence));continue
            log('events',dict(event='prediction_request',observation_id=head.sequence,
                observation_monotonic_ns=head.receipt_ns,send_ns=send_ns,request_bytes=len(body),
                selection_started_ns=selection_started_ns,selected_ns=selected_ns,decode_started_ns=decode_started_ns,decoded_ns=decoded_ns,encode_started_ns=encode_started_ns))
            try:
                max_inflight=1
                status,data=exchange(conn,'POST','/predict',body,{'Content-Type':'application/octet-stream'})
                received_ns=time.monotonic_ns();wire_bytes+=len(body)+len(data)
                if status!=200:raise ValueError('prediction_http_'+str(status))
                result=decode(data)
                for k in ('protocol_version','session_id','observation_id','observation_monotonic_ns','model_sha256','data_contract_sha256'):
                    if result[k].item()!=np.asarray(f[k]).item():raise ValueError('response_identity_'+k)
                action=result['action_raw']
                if action.shape!=(7,10) or action.dtype!=np.float32 or not np.isfinite(action).all() or not np.array_equal(result['target_offsets_ns'],OFFSETS):raise ValueError('invalid_action_response')
                row.update(received_ns=received_ns,rtt_ns=received_ns-send_ns,observation_to_response_ns=received_ns-head.receipt_ns,source_to_response_ns=received_ns-head.mapped_ns,action_raw=action.tolist(),gripper_binary=(action[:,7]>=.5).astype(int).tolist(),preprocess_seconds=float(result['preprocess_seconds']),inference_seconds=float(result['inference_seconds']),request_bytes=len(body),response_bytes=len(data),valid=True)
            except (OSError,ValueError,KeyError,http.client.HTTPException) as exc:
                if live:live.executor.fault('network_or_response_failure')
                if scheduler:scheduler.fault('network_or_response_failure')
                if isinstance(exc,TimeoutError):timeouts+=1
                response_errors+=1;row.update(valid=False,error=str(exc));log('requests',row)
                if live:raise RuntimeError('execution_network_or_response_failure') from exc
                try:
                    connect()
                    session=str(uuid.uuid4())
                    if scheduler:scheduler.reconnect_shadow(session)
                    log('events',dict(event='reconnected_shadow',session_id=session))
                except RuntimeError:raise
                except Exception as connect_exc:log('events',dict(event='reconnect_failed',error=str(connect_exc)));time.sleep(.1)
                continue
            if live and live.executor.acquired_once:
                try:live.ensure_active()
                except Exception as exc:
                    row.update(scheduler_accepted=False,execution_error=str(exc))
                    requests.append(row);log('requests',row)
                    raise
            if scheduler:
                row['scheduler_accept_ns']=time.monotonic_ns()
                try:row['scheduler_accepted']=scheduler.accept(result,received_ns)
                except ScheduleError as exc:
                    row['scheduler_accepted']=False;row['scheduler_error']=str(exc)
                    log('events',dict(event='schedule_rejected',error=str(exc),observation_id=head.sequence))
                    if live:
                        requests.append(row);log('requests',row)
                        raise
                if live and row['scheduler_accepted'] and not live.executor.acquired_once:
                    live.acquire()
                    # Acquisition can outlive a chunk. Hold the accepted target,
                    # then arm only on the next freshly observed response.
                    scheduler.discard_pending()
                    runner.start()
                elif live and row['scheduler_accepted'] and scheduler.state=='SHADOW':
                    scheduler.enable_execution(live,time.monotonic_ns())
            requests.append(row);log('requests',row)
            if time.monotonic()-last_saved>=mapping['image_save_interval_seconds']:
                if image_bytes+len(body)>mapping['image_disk_limit_bytes']:image_drops+=1
                else:
                    path=out/f'observation_{head.sequence:08d}.npz';writer.archive(path,body);image_bytes+=len(body)
                last_saved=time.monotonic()
            if len(requests)%100==0:print(json.dumps(dict(event='progress',elapsed=time.monotonic()-start,requests=len(requests),errors=response_errors,rtt_ms=row['rtt_ns']/1e6)),flush=True)
    except BaseException as exc:
        fatal.append(type(exc).__name__+': '+str(exc))
        for stop in ([lambda:live.executor.fault('bridge_failure:'+str(exc))] if live else []) + ([lambda:scheduler.fault('bridge_failure')] if scheduler else []):
            try:stop()
            except Exception as stop_exc:fatal.append('stop_error:'+repr(stop_exc))
        raise
    finally:
        original_error=sys.exc_info()[1]
        elapsed=time.monotonic()-start
        # Every cleanup stage runs even if stopping/feedback verification fails.
        cleanup_errors=[]
        for cleanup in ([runner.close] if runner else []) + ([live.close] if live else []):
            try:cleanup()
            except BaseException as exc:cleanup_errors.append(repr(exc))
        fatal.extend(cleanup_errors)
        if live and original_error is None and (live.model_commands==0 or live.executor.state=='FAULT' or (runner and runner.error)):
            cleanup_errors.append('model_execution_incomplete:'+str(live.executor.fault_reason))
            fatal.append(cleanup_errors[-1])
        if conn:conn.close()
        if rclpy.ok():rclpy.shutdown()
        thread.join(timeout=5);node.destroy_node()
        try:writer.close()
        except Exception as exc:
            cleanup_errors.append(repr(exc));fatal.append(repr(exc))
        def percentiles(values):return np.percentile(values,[50,95,99,100]).tolist() if len(values) else []
        valid_span=(requests[-1]['received_ns']-requests[0]['received_ns'])/1e9 if len(requests)>1 else 0.
        max_request_gap=max((b['received_ns']-a['received_ns'])/1e9 for a,b in zip(requests,requests[1:])) if len(requests)>1 else 0.
        summary=dict(scheduler=scheduler.stats() if scheduler else None,shadow_thread_error=runner.error if runner else None,skipped_control_ticks=runner.skipped_ticks if runner else 0,valid_request_span_seconds=valid_span,max_valid_request_gap_seconds=max_request_gap,session_id=session,step=health['step'] if health else None,elapsed_seconds=elapsed,requested_duration_seconds=a.duration,valid_requests=len(requests),response_errors=response_errors,timeouts=timeouts,reconnections=reconnections,max_inflight=max_inflight,wire_payload_bytes=wire_bytes,mean_payload_mbps=wire_bytes*8/max(elapsed,1)/1e6,buffer=buffer.stats(),selection_rejections=dict(rejects),callback_errors=dict(errors),callbacks=dict(callback_counts),fatal=fatal,mapper_offset_ns=mapper.offset,clock_jump=mapper.failed,source_interval_p50_p95_p99_max_ns={k:percentiles(v) for k,v in intervals.items()},source_hz={k:(len(v)*1e9/sum(v) if v else 0) for k,v in intervals.items()},source_age_min_ns=min_age,source_age_max_ns=max_age,rtt_p50_p95_p99_max_ns=percentiles([r['rtt_ns'] for r in requests]),observation_to_response_p50_p95_p99_max_ns=percentiles([r['observation_to_response_ns'] for r in requests]),head_age_at_send_p50_p95_p99_max_ns=percentiles([r['head_receipt_age_at_send_ns'] for r in requests]),cross_stream_skew_p50_p95_p99_max_ns={k:percentiles([abs(r['source_mapped_ns'][k]-r['source_mapped_ns']['head']) for r in requests]) for k in ('right','arm','gripper','head_state')},estimated_missing_camera_frames={k:int(sum(max(0,round(dt/(1e9/30))-1) for dt in intervals[k])) for k in ('head','right')},unsent_replaced_head_frames=max(0,callback_counts['head']-len(requests)),image_bytes=image_bytes,image_saved_count=len(list(out.glob('observation_*.npz'))),image_drops=image_drops,log_bytes=writer.bytes_written,log_writer=writer.stats(),stop_log_error=live.executor.stop_log_error if live else None,sdk_imported=any(n=='astribot' or n.startswith('astribot_sdk') for n in sys.modules),application_publishers_created=None if live else 0,ros_service_clients_created=None if live else 0,execution_enabled=bool(live and live.executor.acquired_once),sent_commands=live.executor.sent_commands if live else 0,model_commands=live.model_commands if live else 0,policy_targets=live.policy_targets if live else 0,control_loop=runner.stats() if live and runner else None,execution_fault=live.executor.fault_reason if live else None,stop_settled_ns=live.stop_result if live else None,complete=elapsed>=a.duration and bool(requests) and not fatal and not mapper.failed and (a.duration<600 or valid_span>=600) and max_request_gap<=mapping['max_request_gap_seconds'])
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(event='finished',**summary)),flush=True)
        if cleanup_errors and original_error is None:raise RuntimeError('cleanup_failed:'+ ';'.join(cleanup_errors))
if __name__=='__main__':main()
