#!/usr/bin/env python3
"""Robot-local camera OR state publisher; never creates a control SDK client."""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.observation import Sample, JpegFrame, TimeMapper, ObservationBuffer, ObservationError
from pressb.s1b_deployment.streams import LatestPublisher, camera_packet, state_packet, HISTORY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['images','states'],required=True)
    p.add_argument('--service-id',required=True)
    p.add_argument('--mapping',default='configs/s1b_observation_v1.json')
    p.add_argument('--bind',default='127.0.0.1')
    p.add_argument('--head-port',type=int,default=55555)
    p.add_argument('--right-port',type=int,default=55557)
    p.add_argument('--state-port',type=int,default=55558)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    mapping=json.loads(Path(a.mapping).read_text())
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    routes=({'head':a.head_port,'right':a.right_port} if a.kind=='images' else {'states':a.state_port})
    publisher=LatestPublisher({k:f'tcp://{a.bind}:{port}' for k,port in routes.items()})
    mapper=TimeMapper(time.time_ns(),time.monotonic_ns(),mapping['clock']['max_offset_jump_ns'])
    buffer=ObservationBuffer(capacity=HISTORY)
    counts=Counter();errors=Counter();fatal=[];stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from astribot_msgs.msg import RobotJointState
    # Our handler first stops callbacks, then the explicit cleanup shuts ROS
    # down once. ROS's asynchronous signal shutdown would race that cleanup.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=rclpy.create_node('pressb_split_'+a.kind)
    def receive(key,msg):
        receipt=time.monotonic_ns();wall=time.time_ns();counts[key]+=1
        source=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        header=dict(sec=msg.header.stamp.sec,nanosec=msg.header.stamp.nanosec,frame_id=msg.header.frame_id)
        try:
            mapped=mapper.map(source,wall,receipt)
            if a.kind=='images':
                if 'jpeg' not in msg.format.lower():raise ObservationError('unsupported_camera_format')
                value=JpegFrame(bytes(msg.data))
            else:
                names=list(msg.name);expected=mapping['states'][key]['names']
                if names!=expected and not (not names and mapping['allow_empty_names_with_versioned_mapping']):
                    raise ObservationError('state_names_'+key)
                value=np.asarray(msg.position,dtype=np.float32)
            sample=Sample(counts[key],source,mapped,receipt,header,value)
            buffer.add(key,sample)
            if a.kind=='images':publisher.put(key,camera_packet(key,sample,a.service_id,boot_id))
        except ObservationError as exc:
            errors[str(exc)]+=1
            if mapper.failed:fatal.append(str(exc))
        except BaseException as exc:fatal.append(repr(exc))
    streams=mapping['cameras'] if a.kind=='images' else mapping['states']
    for key,spec in streams.items():
        node.create_subscription(CompressedImage if a.kind=='images' else RobotJointState,
            spec if a.kind=='images' else spec['topic'],lambda msg,k=key:receive(k,msg),qos_profile_sensor_data)
    def spin():
        try:rclpy.spin(node)
        except BaseException as exc:
            if not stop.is_set():fatal.append(repr(exc))
    thread=threading.Thread(target=spin,daemon=True)
    started=time.monotonic();next_status=started
    try:
        publisher.start();thread.start()
        print(json.dumps(dict(event='ready',kind=a.kind,service_id=a.service_id,boot_id=boot_id,routes=routes)),flush=True)
        while not stop.wait(.02):
            publisher.check()
            if fatal:raise RuntimeError(';'.join(fatal))
            if a.kind=='states':
                with buffer.lock:history={key:list(buffer.buffers[key]) for key in streams}
                packet=state_packet(history,a.service_id,boot_id)
                if packet:publisher.put('states',packet)
            if time.monotonic()>=next_status:
                print(json.dumps(dict(event='progress',counts=dict(counts),errors=dict(errors),publisher=dict(publisher.counts))),flush=True)
                next_status=time.monotonic()+5
    except BaseException as exc:
        fatal.append(repr(exc));raise
    finally:
        stop.set()
        if rclpy.ok():rclpy.shutdown()
        if thread.ident:thread.join(2)
        node.destroy_node()
        try:publisher.close()
        finally:
            summary=dict(kind=a.kind,service_id=a.service_id,boot_id=boot_id,elapsed_seconds=time.monotonic()-started,
                callbacks=dict(counts),errors=dict(errors),fatal=fatal,publisher=dict(publisher.counts),
                sdk_imported=any(n.startswith('astribot_sdk') for n in sys.modules),buffer=buffer.stats())
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
