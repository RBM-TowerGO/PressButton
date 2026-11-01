"""Native SDK Cartesian executor. No model, HTTP or disk work on servo thread."""
import copy
import math
import threading
import time
import traceback
from collections import deque

from .common import number


class SDKBackend:
    def __init__(self, robot, monitor, frequency=250):
        self.robot,self.monitor,self.frequency=robot,monitor,number(frequency)
        if not 1<=self.frequency<=500:raise ValueError('invalid_frequency')
        self.lock=threading.RLock()
        self.done=threading.Event();self.error=None
        self.first_command=threading.Event()
        self.names=['astribot_arm_right','astribot_torso']
        poses=robot.get_current_cartesian_pose(self.names,frame='chassis')
        self.target,self.torso=copy.deepcopy(poses)
        self.plane_pose=list(self.target)
        self.command_pose=list(self.target)
        self.sampling_error=None
        self.error_traceback=None
        self.trajectory=None;self.measured=None;self.history=deque(maxlen=6000)
        self.measured=self._read()
        self.history.append(copy.deepcopy(self.measured))
        self.reader=threading.Thread(target=self._sample,daemon=True)
        self.reader.start()
        self.thread=threading.Thread(target=self._servo,name='agent-cartesian-servo',daemon=True)
        self.thread.start()
        until=time.monotonic()+3
        while not self.first_command.is_set() and not self.error and time.monotonic()<until:time.sleep(.01)
        try:
            self.snapshot()
            if not self.first_command.is_set():raise RuntimeError('initial_command_timeout')
        except Exception as exc:
            self.close()
            raise RuntimeError('SDK backend startup failed: '+str(exc)+'\n'+(self.error_traceback or '')) from exc

    def _read(self):
        rows=self.monitor.snapshot()
        needed=('astribot_arm_right','astribot_torso','astribot_gripper_right')
        poses=self.robot.get_current_cartesian_pose(self.names,frame='chassis')
        pose=[number(x) for x in poses[0]]
        with self.lock:target=list(self.command_pose)
        return dict(pose_m_xyzw=pose,command_pose_m_xyzw=target,frame_id='chassis',reference_point='sdk_end_effector',
                    receipt_ns=time.monotonic_ns(),joints={k:rows[k] for k in needed})

    def _servo(self):
        from s1b_keyboard_direct import run_cartesian_loop, interpolate_pose
        complete=False
        def targets():
            nonlocal complete
            complete=False
            if self.trajectory:
                start,goal,began,duration=self.trajectory
                self.target,complete=interpolate_pose(start,goal,time.monotonic()-began,duration)
            self.command_pose=list(self.target)
            return self.names,self.target,self.torso,None
        def sent():
            if complete:self.trajectory=None
            self.first_command.set()
        def failed(exc):
            self.error_traceback=traceback.format_exc()
            self.error=str(exc);self.stop()
        run_cartesian_loop(self.robot,self.frequency,self.done,self.lock,targets,failed,sent)

    def _sample(self):
        """Observation recording only; feedback does not gate keyboard motion."""
        while not self.done.is_set():
            try:
                measured=self._read()
                with self.lock:
                    self.measured=measured;self.history.append(measured)
                    self.sampling_error=None
            except Exception as exc:
                self.sampling_error=str(exc)
            self.done.wait(.02)

    def snapshot(self):
        with self.lock:
            if self.error:raise RuntimeError(self.error)
            if self.measured is None or time.monotonic_ns()-self.measured['receipt_ns']>300_000_000:
                raise RuntimeError('state_unavailable')
            return copy.deepcopy(self.measured)

    def state_for_capture(self, bundle):
        source=bundle['frames']['right']['source_ns']
        def stamp(row):
            h=row['joints']['astribot_arm_right']['header']
            return h['sec']*10**9+h['nanosec']
        with self.lock:
            if not self.history:raise RuntimeError('no_state_for_capture')
            state=copy.deepcopy(min(self.history,key=lambda row:abs(stamp(row)-source)))
        state['rgb_state_skew_ms']=abs(stamp(state)-source)/1e6
        if state['rgb_state_skew_ms']>80:raise RuntimeError('image_state_skew')
        return state

    def _segment(self, goal, cancel):
        with self.lock:
            start=list(self.target)
            duration=max(.2,1.875*math.dist(start[:3],goal[:3])/.020)
            self.trajectory=(start,list(goal),time.monotonic(),duration)
        while True:
            if cancel.is_set() or self.done.is_set():raise RuntimeError(self.error or 'stopped')
            with self.lock:
                if self.trajectory is None:return
            cancel.wait(.01)

    def move(self, kind, data, cancel, phase):
        from s1b_keyboard_direct import target_for_key
        with self.lock:start=list(self.target)
        goal=list(start)
        if kind=='move':
            goal=target_for_key('d',goal,self.plane_pose[0],data['right_mm']/1000)
            goal=target_for_key('w',goal,self.plane_pose[0],data['up_mm']/1000)
            goal[0]=self.plane_pose[0]
        else:goal[0]+=data['forward_mm']/1000
        goal[3:]=self.plane_pose[3:]
        began=time.monotonic_ns();apex=None
        if kind=='move':
            phase('MOVING');self._segment(goal,cancel)
        else:
            phase('EXTENDING');self._segment(goal,cancel)
            apex=time.monotonic_ns()
            retract=list(start);retract[3:]=self.plane_pose[3:]
            phase('RETRACTING');self._segment(retract,cancel)
        phase('SETTLING')
        with self.lock:
            final=copy.deepcopy(self.measured)
            history=[copy.deepcopy(row) for row in self.history if row['receipt_ns']>=began]
        return dict(start_pose_m_xyzw=start,target_pose_m_xyzw=goal,final_state=final,
            returned_to_start=None,command_returned_to_start=True if kind=='press' else None,
            completion_basis='command_interpolation',started_monotonic_ns=began,
            apex_monotonic_ns=apex,completed_monotonic_ns=time.monotonic_ns(),state_trace=history,
            execution_mode='keyboard_public_cartesian')

    def diagnostic(self):
        with self.lock:
            return dict(error=self.error,traceback=self.error_traceback,
                last_state=copy.deepcopy(self.measured),command_pose=list(self.command_pose),
                state_trace=copy.deepcopy(list(self.history)),sampling_error=self.sampling_error)

    def stop(self):
        self.done.set()
        with self.lock:
            if self.robot.get_control_rights_status():self.robot.stop_robot()

    def close(self):
        self.stop();self.thread.join(3);self.reader.join(3)
        if self.thread.is_alive() or self.reader.is_alive():raise RuntimeError('SDK thread did not stop')
