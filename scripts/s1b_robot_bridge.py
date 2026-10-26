#!/usr/bin/env python3
"""Read-only ROS subscriptions and single-flight inference. No SDK control client."""
import argparse
from collections import Counter
import http.client
import json
from pathlib import Path
import sys
import threading
import time
import uuid
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.contract import VERSION,OFFSETS,canonical,read_config,sha
from pressb.s1b_deployment.observation import Sample,TimeMapper,ObservationBuffer,ObservationError
from pressb.s1b_deployment.protocol import encode,decode,MAX_BYTES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/s1b_deployment_v1.yaml')
    p.add_argument('--mapping',default='configs/s1b_observation_v1.json')
    p.add_argument('--mode',choices=['shadow','execute'],default='shadow')
    p.add_argument('--duration',type=float,default=600)
    p.add_argument('--output',required=True)
    a=p.parse_args();c=read_config(a.config,a.mode)
    if a.mode!='shadow':raise ValueError('P2 supports only read-only shadow mode')
    mapping=json.loads(Path(a.mapping).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('resolved_config',c),('mapping',mapping)]:
        (out/(name+'.json')).write_text(json.dumps(dict(value=value,sha256=canonical(value)),indent=2)+'\n')
    logs={name:(out/(name+'.jsonl')).open('w',buffering=1) for name in ('sources','requests','events')}
    log_lock=threading.Lock();log_bytes=0
    def log(name, row):
        nonlocal log_bytes
        data=json.dumps(row,allow_nan=False)+'\n'
        with log_lock:
            if log_bytes+len(data)>mapping['log_disk_limit_bytes']:raise OSError('log_disk_limit')
            logs[name].write(data);log_bytes+=len(data)
    buffer=ObservationBuffer(mapping['buffer_capacity'],mapping['head_age_ns'],mapping['state_skew_ns'],mapping['right_skew_ns'])
    mapper=TimeMapper(time.time_ns(),time.monotonic_ns(),mapping['clock']['max_offset_jump_ns'])
    callback_counts=Counter();errors=Counter();fatal=[];intervals={k:[] for k in buffer.buffers};last_stamp={};max_age={};min_age={}
    import rclpy
    from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy,DurabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from astribot_msgs.msg import RobotJointState
    import cv2
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
                bgr=cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR)
                if bgr is None:raise ObservationError('jpeg_decode_failed')
                value=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                row.update(format=msg.format,encoded_bytes=len(msg.data),shape=list(value.shape),rgb_conversion='cv2 BGR2RGB exactly once')
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
        except Exception as exc:fatal.append(repr(exc))
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
        conn.request('GET','/health');response=conn.getresponse();data=response.read(MAX_BYTES+1)
        if response.status!=200 or len(data)>MAX_BYTES:raise ValueError('health_failed')
        candidate=json.loads(data)
        if health is not None and any(candidate.get(k)!=health.get(k) for k in ('model_sha256','data_contract_sha256','step')):raise RuntimeError('health_changed_requires_new_session')
        health=candidate
        contract_hash=expected['files']['data_contract.json']
        if not health['ready'] or health['protocol_version']!=VERSION or health['action_dim']!=10 or health['horizon']!=7 or health['model_sha256']!=expected['models'][str(health['step'])] or health['data_contract_sha256']!=contract_hash:raise ValueError('health_identity_mismatch')
        log('events',dict(event='health',health=health,robot_monotonic_ns=time.monotonic_ns()))
    session=str(uuid.uuid4());start=time.monotonic();started_ns=time.monotonic_ns();last_sequence=-1;last_saved=0.;image_bytes=0;image_drops=0;requests=[];rejects=Counter();timeouts=0;response_errors=0;wire_bytes=0;max_inflight=0
    try:
        connect();print(json.dumps(dict(event='ready',session=session,step=health['step'])),flush=True)
        while time.monotonic()-start<a.duration:
            if fatal:raise RuntimeError(';'.join(fatal))
            if mapper.failed:raise RuntimeError('clock_jump')
            try:selected,state=buffer.snapshot(time.monotonic_ns(),last_sequence)
            except ObservationError as exc:
                rejects[str(exc)]+=1;time.sleep(.005);continue
            head=selected['head'];last_sequence=head.sequence
            row=dict(session_id=session,observation_id=head.sequence,observation_monotonic_ns=head.receipt_ns,source_headers={k:v.header for k,v in selected.items()},source_receipt_ns={k:v.receipt_ns for k,v in selected.items()},source_mapped_ns={k:v.mapped_ns for k,v in selected.items()},source_sequence={k:v.sequence for k,v in selected.items()},state10=state.tolist(),buffer=buffer.stats(),seed=mapping['seed'],sdk_target=None)
            f=dict(protocol_version=VERSION,session_id=session,observation_id=np.int64(head.sequence),observation_monotonic_ns=np.int64(head.receipt_ns),seed=np.int64(mapping['seed']),head=head.value,right=selected['right'].value,state10=state,source_time_ns=np.array([selected[k].source_ns for k in ('head','right','arm')],np.int64),mapped_time_ns=np.array([selected[k].mapped_ns for k in ('head','right','arm')],np.int64),state_component_source_ns=np.array([selected[k].source_ns for k in ('arm','gripper','head_state')],np.int64),model_sha256=health['model_sha256'],data_contract_sha256=health['data_contract_sha256'])
            body=encode(f);send_ns=time.monotonic_ns();row['send_ns']=send_ns;row['head_receipt_age_at_send_ns']=send_ns-head.receipt_ns
            if send_ns-head.receipt_ns>mapping['head_age_ns']:
                rejects['stale_after_encoding']+=1;log('events',dict(event='rejected',reason='stale_after_encoding',observation_id=head.sequence));continue
            try:
                max_inflight=1
                conn.request('POST','/predict',body,{'Content-Type':'application/octet-stream'})
                response=conn.getresponse();data=response.read(MAX_BYTES+1);received_ns=time.monotonic_ns();wire_bytes+=len(body)+len(data)
                if response.status!=200 or len(data)>MAX_BYTES:raise ValueError('prediction_http_'+str(response.status))
                result=decode(data)
                for k in ('protocol_version','session_id','observation_id','observation_monotonic_ns','model_sha256','data_contract_sha256'):
                    if result[k].item()!=np.asarray(f[k]).item():raise ValueError('response_identity_'+k)
                action=result['action_raw']
                if action.shape!=(7,10) or action.dtype!=np.float32 or not np.isfinite(action).all() or not np.array_equal(result['target_offsets_ns'],OFFSETS):raise ValueError('invalid_action_response')
                row.update(received_ns=received_ns,rtt_ns=received_ns-send_ns,observation_to_response_ns=received_ns-head.receipt_ns,source_to_response_ns=received_ns-head.mapped_ns,action_raw=action.tolist(),gripper_binary=(action[:,7]>=.5).astype(int).tolist(),preprocess_seconds=float(result['preprocess_seconds']),inference_seconds=float(result['inference_seconds']),request_bytes=len(body),response_bytes=len(data),valid=True)
            except (OSError,ValueError,KeyError,http.client.HTTPException) as exc:
                if isinstance(exc,TimeoutError):timeouts+=1
                response_errors+=1;row.update(valid=False,error=str(exc));log('requests',row)
                try:connect()
                except Exception as connect_exc:log('events',dict(event='reconnect_failed',error=str(connect_exc)));time.sleep(.1)
                continue
            requests.append(row);log('requests',row)
            if time.monotonic()-last_saved>=mapping['image_save_interval_seconds']:
                if image_bytes+len(body)>mapping['image_disk_limit_bytes']:image_drops+=1
                else:
                    path=out/f'observation_{head.sequence:08d}.npz';path.write_bytes(body);image_bytes+=len(body)
                last_saved=time.monotonic()
            if len(requests)%100==0:print(json.dumps(dict(event='progress',elapsed=time.monotonic()-start,requests=len(requests),errors=response_errors,rtt_ms=row['rtt_ns']/1e6)),flush=True)
    except BaseException as exc:
        fatal.append(type(exc).__name__+': '+str(exc))
        raise
    finally:
        elapsed=time.monotonic()-start
        if conn:conn.close()
        if rclpy.ok():rclpy.shutdown()
        thread.join(timeout=5);node.destroy_node()
        def percentiles(values):return np.percentile(values,[50,95,99,100]).tolist() if len(values) else []
        valid_span=(requests[-1]['received_ns']-requests[0]['received_ns'])/1e9 if len(requests)>1 else 0.
        max_request_gap=max((b['received_ns']-a['received_ns'])/1e9 for a,b in zip(requests,requests[1:])) if len(requests)>1 else 0.
        summary=dict(valid_request_span_seconds=valid_span,max_valid_request_gap_seconds=max_request_gap,session_id=session,step=health['step'] if health else None,elapsed_seconds=elapsed,requested_duration_seconds=a.duration,valid_requests=len(requests),response_errors=response_errors,timeouts=timeouts,reconnections=reconnections,max_inflight=max_inflight,wire_payload_bytes=wire_bytes,mean_payload_mbps=wire_bytes*8/max(elapsed,1)/1e6,buffer=buffer.stats(),selection_rejections=dict(rejects),callback_errors=dict(errors),callbacks=dict(callback_counts),fatal=fatal,mapper_offset_ns=mapper.offset,clock_jump=mapper.failed,source_interval_p50_p95_p99_max_ns={k:percentiles(v) for k,v in intervals.items()},source_hz={k:(len(v)*1e9/sum(v) if v else 0) for k,v in intervals.items()},source_age_min_ns=min_age,source_age_max_ns=max_age,rtt_p50_p95_p99_max_ns=percentiles([r['rtt_ns'] for r in requests]),observation_to_response_p50_p95_p99_max_ns=percentiles([r['observation_to_response_ns'] for r in requests]),head_age_at_send_p50_p95_p99_max_ns=percentiles([r['head_receipt_age_at_send_ns'] for r in requests]),cross_stream_skew_p50_p95_p99_max_ns={k:percentiles([abs(r['source_mapped_ns'][k]-r['source_mapped_ns']['head']) for r in requests]) for k in ('right','arm','gripper','head_state')},estimated_missing_camera_frames={k:int(sum(max(0,round(dt/(1e9/30))-1) for dt in intervals[k])) for k in ('head','right')},unsent_replaced_head_frames=max(0,callback_counts['head']-len(requests)),image_bytes=image_bytes,image_saved_count=len(list(out.glob('observation_*.npz'))),image_drops=image_drops,log_bytes=log_bytes,sdk_imported=any(n=='astribot' or n.startswith('astribot_sdk') for n in sys.modules),application_publishers_created=0,ros_service_clients_created=0,execution_enabled=False,complete=elapsed>=a.duration and bool(requests) and not fatal and not mapper.failed and (a.duration<600 or valid_span>=600) and max_request_gap<=mapping['max_request_gap_seconds'])
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        for stream in logs.values():stream.close()
        print(json.dumps(dict(event='finished',**summary)),flush=True)
if __name__=='__main__':main()
