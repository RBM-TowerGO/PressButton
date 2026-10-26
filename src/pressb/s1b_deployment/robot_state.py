"""Read-only live state/command monitor for P4; no SDK imports."""
import threading
import time
import numpy as np
from .executor import PARTS,ExecutionError

ALL_PARTS=['astribot_chassis','astribot_torso','astribot_arm_left','astribot_gripper_left',*PARTS]

class RobotState:
    def __init__(self, emit):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from astribot_msgs.msg import RobotJointState,RobotJointController
        if not rclpy.ok():rclpy.init()
        self.rclpy=rclpy;self.node=rclpy.create_node('pressb_p4_state_monitor')
        self.lock=threading.Lock();self.states={};self.commands={};self.accepted={};self.emit=emit;self.error=None
        for part in ALL_PARTS:
            self.node.create_subscription(RobotJointState,'/'+part+'/joint_space_states',lambda m,p=part:self.callback(p,m,False),qos_profile_sensor_data)
            self.node.create_subscription(RobotJointController,'/'+part+'/joint_space_command',lambda m,p=part:self.callback(p,m,True),qos_profile_sensor_data)
            self.node.create_subscription(RobotJointState,'/'+part+'/joint_space_command_recv',lambda m,p=part:self.accepted_callback(p,m),qos_profile_sensor_data)
        self.thread=threading.Thread(target=self.spin,daemon=True);self.thread.start()

    def callback(self, part, message, command):
        row=dict(event='command_topic' if command else 'state',part=part,receipt_ns=time.monotonic_ns(),header=dict(sec=message.header.stamp.sec,nanosec=message.header.stamp.nanosec,frame_id=message.header.frame_id),position=list(message.command if command else message.position),mode=int(message.mode),names=list(message.name))
        if not command:row['velocity']=list(message.velocity);row['torque']=list(message.torque)
        with self.lock:(self.commands if command else self.states)[part]=row
        try:self.emit(row)
        except BaseException as exc:self.error=repr(exc)

    def accepted_callback(self, part, message):
        row=dict(event='accepted_command',part=part,receipt_ns=time.monotonic_ns(),position=list(message.position),mode=int(message.mode),header=dict(sec=message.header.stamp.sec,nanosec=message.header.stamp.nanosec))
        with self.lock:self.accepted[part]=row
        try:self.emit(row)
        except BaseException as exc:self.error=repr(exc)

    def command10(self):
        with self.lock:
            if any(p not in self.accepted for p in PARTS):raise ExecutionError('missing_accepted_command')
            rows=[self.accepted[p] for p in PARTS]
            if any(len(r['position'])!=n for r,n in zip(rows,[7,1,2])):raise ExecutionError('accepted_command_length')
            if time.monotonic_ns()-min(r['receipt_ns'] for r in rows)>80_000_000:raise ExecutionError('accepted_command_stale')
            if any(r['mode']!=1 for r in rows):raise ExecutionError('accepted_command_mode_unverified')
            return np.concatenate([r['position'] for r in rows]).astype(np.float64)

    def spin(self):
        try:self.rclpy.spin(self.node)
        except Exception as exc:
            if self.rclpy.ok():self.error=repr(exc)

    def snapshot(self):
        with self.lock:
            if any(part not in self.states for part in ALL_PARTS):raise ExecutionError('missing_component_state')
            return {k:dict(v) for k,v in self.states.items()}

    def joint10(self):
        states=self.snapshot()
        if any(len(states[p]['position'])!=n for p,n in zip(PARTS,[7,1,2])):raise ExecutionError('component_state_length')
        return np.concatenate([states[p]['position'] for p in PARTS]),min(states[p]['receipt_ns'] for p in PARTS)

    def owners(self):
        found=[]
        for name,namespace in self.node.get_node_names_and_namespaces():
            try:
                if any(s=='/astribot/control_rights' for s,t in self.node.get_service_names_and_types_by_node(name,namespace)):found.append(name)
            except Exception:pass
        return found

    def request_handover(self, timeout=3.):
        from astribot_msgs.srv import RawRequest
        client=self.node.create_client(RawRequest,'/astribot/control_rights')
        try:
            if not client.wait_for_service(timeout_sec=timeout):raise ExecutionError('control_handover_service_unavailable')
            future=client.call_async(RawRequest.Request());until=time.monotonic()+timeout
            while not future.done() and time.monotonic()<until:time.sleep(.005)
            if not future.done() or future.result() is None:raise ExecutionError('control_handover_timeout')
            result=future.result().response
            if result!='success':raise ExecutionError('control_handover_rejected:'+str(result))
            self.emit(dict(event='control_handover_response',monotonic_ns=time.monotonic_ns(),response=result))
            return result
        finally:self.node.destroy_client(client)

    def close(self):
        if self.rclpy.ok():self.rclpy.shutdown()
        self.thread.join(1)
        self.node.destroy_node()
