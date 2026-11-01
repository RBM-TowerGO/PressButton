"""State sampling and service adapter around the shared keyboard executor."""
import copy
import math
import threading
import time
from collections import deque

from .common import number


class SDKBackend:
    def __init__(self, robot, monitor, frequency=250, speed_mm_s=20., head_speed_deg_s=20.):
        from s1b_keyboard_control import CommandExecutor
        self.robot, self.monitor = robot, monitor
        self.controller = CommandExecutor(robot, frequency, speed_mm_s, head_speed_deg_s)
        self.names = self.controller.names
        self.lock = threading.RLock()
        self.done = self.controller.stop_event
        self.measured = None
        self.history = deque(maxlen=6000)
        self.sampling_error = None
        try:
            self.measured = self._read()
            self.history.append(copy.deepcopy(self.measured))
            self.reader = threading.Thread(target=self._sample, daemon=True)
            self.reader.start()
        except BaseException:
            self.controller.close()
            raise

    @property
    def error(self):
        return self.controller.error

    @property
    def error_traceback(self):
        return self.controller.error_traceback

    def _read(self):
        rows=self.monitor.snapshot()
        needed=('astribot_arm_right','astribot_torso','astribot_gripper_right')
        poses=self.robot.get_current_cartesian_pose(self.names,frame='chassis')
        pose=[number(x) for x in poses[0]]
        target=self.controller.command_pose()
        return dict(pose_m_xyzw=pose,command_pose_m_xyzw=target,frame_id='chassis',reference_point='sdk_end_effector',
                    receipt_ns=time.monotonic_ns(),joints={k:rows[k] for k in needed},
                    head_deg=[math.degrees(v) for v in self.robot.get_current_joints_position(['astribot_head'])[0]],
                    gripper_command=self.controller.gripper_target)

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

    def move(self, kind, data, cancel, phase):
        field = {'move': 'xyz_mm', 'head': 'head_deg', 'gripper': 'key'}[kind]
        prepared = self.controller.prepare(dict(type=kind, **{field: data[field]}))
        if cancel.is_set():
            raise RuntimeError('stopped')
        phase('MOVING')
        task = self.controller.start(prepared)
        self.controller.wait(task, cancel)
        began = task['started']['monotonic_ns']
        with self.lock:
            final = copy.deepcopy(self.measured)
            history = [copy.deepcopy(row) for row in self.history if row['receipt_ns'] >= began]
        return dict(start_pose_m_xyzw=task['start'], target_pose_m_xyzw=task['goal'], final_state=final,
            returned_to_start=None, completion_basis='first_successful_send' if kind == 'gripper' else 'command_interpolation',
            started_monotonic_ns=began, completed_monotonic_ns=task['ended']['monotonic_ns'],
            state_trace=history, execution_mode='shared_keyboard_executor')

    def diagnostic(self):
        with self.lock:
            return dict(error=self.error, traceback=self.error_traceback,
                last_state=copy.deepcopy(self.measured), command_pose=self.controller.command_pose(),
                state_trace=copy.deepcopy(list(self.history)), sampling_error=self.sampling_error)

    def stop(self):
        self.controller.stop()

    def close(self):
        try:
            self.controller.close()
        finally:
            self.reader.join(3)
            if self.reader.is_alive():
                raise RuntimeError('SDK state reader did not stop')
