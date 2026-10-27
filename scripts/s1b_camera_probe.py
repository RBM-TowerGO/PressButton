#!/usr/bin/env python3
"""Subscribe and validate existing camera streams without starting drivers."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.s1b_deployment.cameras import CameraReadiness


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mapping',default='configs/s1b_observation_v1.json')
    p.add_argument('--timeout',type=float,default=5)
    a=p.parse_args()
    if not 0<a.timeout<=30:p.error('timeout must be in (0,30]')
    mapping=json.loads(Path(a.mapping).read_text())
    import cv2
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    cv2.setNumThreads(1)
    rclpy.init();node=rclpy.create_node('pressb_camera_readiness')
    probe=CameraReadiness(mapping,time.time_ns(),time.monotonic_ns())
    def receive(key,msg):
        receipt=time.monotonic_ns();wall=time.time_ns()
        probe.receive(key,msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                      receipt,wall,msg.format,bytes(msg.data))
    try:
        for key,topic in mapping['cameras'].items():
            node.create_subscription(CompressedImage,topic,lambda msg,k=key:receive(k,msg),qos_profile_sensor_data)
        until=time.monotonic()+a.timeout
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=.05)
            report=probe.report(time.monotonic_ns())
            if report['ready']:break
        report=probe.report(time.monotonic_ns())
        report['sdk_imported']=any(n.startswith('astribot_sdk') for n in sys.modules)
        print(json.dumps(report),flush=True)
        return 0 if report['ready'] else 2
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
