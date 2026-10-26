"""Bounded S1B joint10 execution; SDK import occurs only during explicit acquire."""
from collections import deque
import builtins
import json
from pathlib import Path
import threading
import time
import numpy as np

PARTS = ['astribot_arm_right', 'astribot_gripper_right', 'astribot_head']

class ExecutionError(ValueError):
    pass

class CommandGuard:
    def __init__(self, config, initial, now_ns):
        self.c = config
        self.last = np.asarray(initial, dtype=np.float64).copy()
        if self.last.shape != (10,) or not np.isfinite(self.last).all():
            raise ExecutionError('invalid_initial_state')
        self.last[7] = min(100., max(0., self.last[7]))
        self.last_ns = now_ns
        self.validate_state(initial)

    def validate_state(self, state):
        x = np.asarray(state, dtype=np.float64)
        if x.shape != (10,) or not np.isfinite(x).all():
            raise ExecutionError('invalid_measured_state')
        if (x < np.asarray(self.c['feedback_lower'])).any() or (x > np.asarray(self.c['feedback_upper'])).any():
            raise ExecutionError('feedback_out_of_range')
        return x

    def check(self, target, state, state_ns, now_ns, *, binary_gripper=False):
        x = np.asarray(target, dtype=np.float64).copy()
        if x.shape != (10,) or not np.isfinite(x).all():
            raise ExecutionError('invalid_target')
        measured = self.validate_state(state)
        if not 0 <= now_ns - state_ns <= self.c['state_max_age_ns']:
            raise ExecutionError('stale_state')
        if binary_gripper:
            x[7] = 100. if x[7] >= .5 else 0.
        lo = np.maximum(self.c['hardware_lower'], self.c['position_lower'])
        hi = np.minimum(self.c['hardware_upper'], self.c['position_upper'])
        if (x < lo).any() or (x > hi).any():
            raise ExecutionError('target_position_limit')
        if (np.abs(x - measured) > np.asarray(self.c['max_tracking_error'])).any():
            raise ExecutionError('target_measured_deviation')
        delta = np.abs(x - self.last)
        if (delta > np.asarray(self.c['max_target_delta']) + 1e-12).any():
            raise ExecutionError('target_delta_limit')
        dt = (now_ns - self.last_ns) / 1e9
        if dt <= 0:
            raise ExecutionError('nonmonotonic_command_time')
        # Long idle gaps cannot authorize a large target jump.
        dt = min(dt, self.c['max_rate_interval_seconds'])
        if (delta / dt > np.asarray(self.c['max_velocity']) + 1e-8).any():
            raise ExecutionError('target_rate_limit')
        return x

    def commit(self, target, now_ns):
        self.last = np.asarray(target, dtype=np.float64).copy()
        self.last_ns = now_ns

class SDKBackend:
    """Low-priority control only. Never seize another user's rights."""
    def __init__(self, sdk_root, frequency):
        self.sdk_root = str(sdk_root)
        self.frequency = frequency
        self.client = None
        self.released = False

    def acquire(self):
        from astribot_sdk.core.astribot_api.astribot_client import Astribot
        previous_input = builtins.input
        # SDK's EOF exception fallback may create rights; decline its takeover
        # prompt explicitly rather than letting a noninteractive EOF reach it.
        builtins.input = lambda *a, **kw: ''
        try:
            self.client = Astribot(freq=self.frequency, high_control_rights=False,
                                   node_name='pressb_p4_execution')
        finally:
            builtins.input = previous_input
        if not self.rights():
            raise ExecutionError('control_rights_unavailable')
        self.client.set_head_follow_effector(False)

    def rights(self):
        return self.client is not None and self.client.get_control_rights_status() is True

    def position(self):
        rows = self.client.get_current_joints_position(names=PARTS)
        return np.concatenate([np.asarray(x, np.float64) for x in rows])

    def send(self, target):
        if not self.rights():
            raise ExecutionError('control_rights_lost')
        self.client.set_joints_position(PARTS,
            [target[:7].tolist(), target[7:8].tolist(), target[8:].tolist()],
            control_way='direct', use_wbc=False, add_default_torso=False)

    def stop(self):
        if self.rights():
            self.client.stop_robot()

    def stopped(self):
        if self.client is None:return True
        core = self.client.astribot_interface._AstribotInterface__astribot_class
        return bool(core.is_stopped())

    def release(self):
        if self.client is None or self.released:
            return
        self.released = True
        interface = self.client.astribot_interface
        # Synchronous SDK release sets have_control_rights=False and stops;
        # avoids destroying the node ahead of its 50 ms transfer timer.
        if self.rights():
            interface.shutdown_control_rights_srv()
        interface.shutdown()

class GuardedExecutor:
    """Latched stop, bounded command admission and independent watchdog."""
    def __init__(self, config, backend, emit):
        self.c = config
        self.backend = backend
        self.emit = emit
        self.state = 'DISCONNECTED'
        self.guard = None
        self.last_refresh_ns = 0
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.stop_lock = threading.Lock()
        self.fault_reason = None
        self.watchdog = None
        self.stop_requested_ns = None
        self.acquired_once = False
        self.sent_commands = 0

    def acquire(self, measured, state_ns, *, operator_present, estop_ready, initial_command=None):
        if self.state != 'DISCONNECTED':raise ExecutionError('new_executor_instance_required')
        if not operator_present or not estop_ready:
            raise ExecutionError('local_operator_and_estop_required')
        now = time.monotonic_ns()
        guard = CommandGuard(self.c, measured, now)
        if not 0 <= now - state_ns <= self.c['state_max_age_ns']:
            raise ExecutionError('stale_initial_state')
        self.state = 'READY'
        try:
            self.backend.acquire()
            if self.stop_event.is_set() or self.state != 'READY':raise ExecutionError('acquisition_cancelled')
            if not self.backend.rights():
                raise ExecutionError('control_rights_unavailable')
            actual = self.backend.position()
            guard = CommandGuard(self.c, actual, time.monotonic_ns())
            if initial_command is not None:
                command=np.asarray(initial_command,np.float64)
                if command.shape!=(10,) or not np.isfinite(command).all():raise ExecutionError('invalid_initial_command')
                if (command<np.maximum(self.c['hardware_lower'],self.c['position_lower'])).any() or (command>np.minimum(self.c['hardware_upper'],self.c['position_upper'])).any():raise ExecutionError('initial_command_position_limit')
                if (np.abs(command-actual)>np.asarray(self.c['max_tracking_error'])).any():raise ExecutionError('initial_command_tracking_error')
                guard.last=command.copy()
                guard.last_ns=time.monotonic_ns()
            self.guard = guard
            self.last_refresh_ns = time.monotonic_ns()
            self.state = 'EXECUTING'
            self.acquired_once = True
            self.emit(dict(event='acquired',monotonic_ns=self.last_refresh_ns,
                           initial_state=actual.tolist(),control_way='direct',use_wbc=False,
                           add_default_torso=False,head_follow_effector=False))
            self.watchdog = threading.Thread(target=self._watch,daemon=True,name='s1b-stop-watchdog')
            self.watchdog.start()
        except BaseException:
            self.backend.stop()
            self.backend.release()
            self.state = 'FAULT'
            raise

    def send(self, target, measured, state_ns, *, binary_gripper=False):
        with self.lock:
            if self.state != 'EXECUTING':
                raise ExecutionError('executor_not_active')
            now = time.monotonic_ns()
            try:
                if not self.backend.rights():
                    raise ExecutionError('control_rights_lost')
                x = self.guard.check(target,measured,state_ns,now,binary_gripper=binary_gripper)
                # Evidence write failure must stop before accepting the command.
                self.emit(dict(event='command',monotonic_ns=now,raw_target=np.asarray(target).tolist(),
                    sdk_target=x.tolist(),measured=np.asarray(measured).tolist()))
                if self.state != 'EXECUTING':raise ExecutionError('stopped_before_send')
                self.backend.send(x)
                if self.state != 'EXECUTING':raise ExecutionError('stopped_during_send')
                self.sent_commands += 1
                self.guard.commit(x,now)
                self.last_refresh_ns = now
                return x
            except BaseException as exc:
                self._stop('command_rejected:'+str(exc),'FAULT')
                raise

    def _stop(self, reason, state):
        with self.stop_lock:
            if self.state in ('FAULT','STOPPED'):
                return
            requested = time.monotonic_ns()
            self.stop_requested_ns = requested
            self.state = state
            self.fault_reason = reason
            self.stop_event.set()
            self.backend.stop()
            returned = time.monotonic_ns()
            self.emit(dict(event='stop',reason=reason,requested_ns=requested,
                           sdk_returned_ns=returned,logical_stopped=self.backend.stopped()))

    def stop(self, reason='operator_stop'):
        self._stop(reason,'STOPPED')

    def fault(self, reason):
        self._stop(reason,'FAULT')

    def _watch(self):
        while not self.stop_event.wait(.005):
            if not self.backend.rights():
                self.fault('control_rights_lost')
                return
            if time.monotonic_ns()-self.last_refresh_ns > self.c['command_watchdog_ns']:
                self.fault('command_watchdog')
                return

    def close(self):
        self.stop()
        if self.watchdog and self.watchdog is not threading.current_thread():
            self.watchdog.join(timeout=1)
        self.backend.release()

class StopObserver:
    """Measure feedback settling on a separate thread while callers may block."""
    def __init__(self, executor, snapshot):
        self.executor=executor;self.snapshot=snapshot;self.done=threading.Event()
        self.settled_ns=None;self.error=None;self.thread=threading.Thread(target=self.run,daemon=True,name='s1b-stop-feedback')
    def start(self):self.thread.start()
    def run(self):
        try:
            while self.executor.stop_requested_ns is None:
                if self.done.wait(.002):return
            start=self.executor.stop_requested_ns;c=self.executor.c;samples=[];last=-1
            while time.monotonic_ns()-start<=c['max_stop_delay_ns']:
                rows=self.snapshot();stamp=min(rows[p]['receipt_ns'] for p in PARTS)
                if stamp>last and stamp>=start:
                    q=np.concatenate([rows[p]['position'] for p in PARTS]);v=np.concatenate([rows[p]['velocity'] for p in PARTS]);last=stamp;samples.append((stamp,q,v))
                    while len(samples)>1 and samples[1][0]<=stamp-c['stop_stability_window_ns']:samples.pop(0)
                    if stamp-samples[0][0]>=c['stop_stability_window_ns']:
                        spread=np.ptp(np.stack([s[1] for s in samples]),axis=0)
                        speeds=np.max(np.abs(np.stack([s[2] for s in samples])),axis=0)
                        if np.all(spread<=np.asarray(c['stop_position_spread'])) and np.all(speeds<=np.asarray(c['stop_velocity_threshold'])):
                            self.settled_ns=time.monotonic_ns();return
                time.sleep(.002)
            self.error='stop_did_not_settle_within_budget'
        except BaseException as exc:self.error=repr(exc)
        finally:self.done.set()
    def wait(self):
        if not self.done.wait(self.executor.c['max_stop_delay_ns']/1e9+.5):raise ExecutionError('stop_observer_timeout')
        if self.error:raise ExecutionError(self.error)
        return self.settled_ns
