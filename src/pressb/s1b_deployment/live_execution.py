"""Bridge deadline-selected model actions to the guarded SDK executor."""
import subprocess
import threading
import time
import numpy as np
from .contract import JOINTS
from .realtime import StartupBlend
from .executor import GuardedExecutor, SDKBackend, StopObserver, ExecutionError, TargetExpired, PARTS


def validate_execution_config(c):
    if c['schema'] != 's1b-execution-p4-v1' or c['parts'] != PARTS or c['joint_names'] != JOINTS:
        raise ExecutionError('execution_contract_mismatch')
    if c['sdk_frequency'] != 250 or c['head_follow_effector'] is not False:
        raise ExecutionError('execution_sdk_contract_mismatch')
    profile=c.get('command_guard_profile','probe')
    for name in ('uncontrolled_drift_action','stop_feedback_action'):
        if c.get(name,'stop') not in ('warn','stop'):raise ExecutionError('invalid_execution_config:'+name)
    if c.get('queue_exhausted_action','stop') not in ('stop','hold'):
        raise ExecutionError('invalid_execution_config:queue_exhausted_action')
    if profile not in ('probe','model_hardware'):raise ExecutionError('unknown_command_guard_profile')
    if profile=='model_hardware':
        if c.get('control_way')!='filter':raise ExecutionError('model_requires_sdk_filter')
        for name in ('startup_blend_seconds','first_prediction_timeout_seconds'):
            if not np.isfinite(c[name]) or c[name]<=0:raise ExecutionError('invalid_execution_config:'+name)
    vectors = ('hardware_lower', 'hardware_upper', 'position_lower', 'position_upper',
               'feedback_lower', 'feedback_upper', 'max_velocity', 'hardware_max_velocity',
               'stop_velocity_threshold',
               'stop_position_spread')
    if profile=='probe':vectors+=('max_target_delta','max_tracking_error')
    for name in vectors:
        x = np.asarray(c[name], dtype=float)
        if x.shape != (10,) or not np.isfinite(x).all():
            raise ExecutionError('invalid_execution_config:' + name)
        if name.startswith(('max_', 'stop_')) and (x <= 0).any():
            raise ExecutionError('invalid_execution_config:' + name)
    for prefix in ('hardware', 'position', 'feedback'):
        if (np.asarray(c[prefix+'_lower']) >= c[prefix+'_upper']).any():
            raise ExecutionError('invalid_execution_bounds:' + prefix)
    if (np.asarray(c['position_lower']) < c['hardware_lower']).any() or (np.asarray(c['position_upper']) > c['hardware_upper']).any():
        raise ExecutionError('execution_bounds_exceed_hardware')
    if (np.asarray(c['max_velocity']) > c['hardware_max_velocity']).any():
        raise ExecutionError('execution_velocity_exceeds_hardware')
    for name in ('state_max_age_ns', 'command_watchdog_ns', 'max_stop_delay_ns',
                 'max_rate_interval_seconds', 'stop_stability_window_ns'):
        if not np.isfinite(c[name]) or c[name] <= 0:
            raise ExecutionError('invalid_execution_config:' + name)
    for part in ('astribot_chassis', 'astribot_torso', 'astribot_arm_left', 'astribot_gripper_left'):
        limit = c['uncontrolled_max_position_drift'][part]
        if not np.isfinite(limit) or limit <= 0:
            raise ExecutionError('invalid_uncontrolled_limit:' + part)


class LiveExecution:
    def __init__(self, config, emit, *, operator_present, estop_ready,
                 request_handover=False, monitor=None, backend=None):
        validate_execution_config(config)
        if not operator_present or not estop_ready:
            raise ExecutionError('local_operator_and_estop_required')
        self.c = config
        self.emit = emit
        self.request_handover = request_handover
        self.monitor = monitor
        self.backend = backend or SDKBackend(config['sdk_root'], config['sdk_frequency'],control_way=config.get('control_way','direct'))
        self.executor = GuardedExecutor(config, self.backend, emit)
        self.observer = None
        self.baseline = None
        self.stop_result = None
        self.model_commands = 0
        self.startup = None
        self.policy_targets = 0
        self.last_policy_key = None
        self.warning_lock=threading.Lock()
        self.warnings={};self.warning_emit_ns={}

    def warn(self,check,**details):
        now=time.monotonic_ns();key=(check,details.get('part'))
        with self.warning_lock:
            previous=self.warnings.get(key,{})
            row=dict(event='warning',check=check,monotonic_ns=now,
                     occurrences=previous.get('occurrences',0)+1,**details)
            if 'drift' in details:row['max_drift']=max(previous.get('max_drift',0),details['drift'])
            self.warnings[key]=row
            if now-self.warning_emit_ns.get(key,-10**18)<1_000_000_000:return
            self.warning_emit_ns[key]=now
        self.emit(row)

    def warning_snapshot(self):
        with self.warning_lock:return list(self.warnings.values())

    @property
    def active(self):
        return self.executor.state == 'EXECUTING'

    def prepare(self):
        commit = subprocess.check_output(['git', '-C', self.c['sdk_root'], 'rev-parse', 'HEAD'], text=True).strip()
        if commit != self.c['sdk_commit']:
            raise ExecutionError('SDK version changed')
        if self.monitor is None:
            from .robot_state import RobotState
            self.monitor = RobotState(self.emit, shared_context=True, on_fault=self.executor.fault,control_only=True)
        until = time.monotonic() + 6
        while True:
            try:
                self.monitor.joint10()
                self.monitor.command10()
                self.baseline = self.monitor.snapshot()
                break
            except ExecutionError:
                if time.monotonic() >= until:
                    raise
                time.sleep(.02)
        # Allow discovery to converge before checking existing control owners.
        time.sleep(.5)
        if self.monitor.control_rights_available(timeout=2.) or self.monitor.owners():
            owners=self.monitor.owners()
            self.emit(dict(event='existing_control_owner',owners=owners,service_available=True))
            if not self.request_handover:
                raise ExecutionError('control_rights_already_owned: use --request-control-handover for an explicit service handover')
            self.monitor.request_handover()
            until = time.monotonic() + 3
            while self.monitor.control_rights_available(timeout=.02) and time.monotonic() < until:
                time.sleep(.02)
            if self.monitor.control_rights_available(timeout=.02):
                raise ExecutionError('control_handover_owner_did_not_release')

    def ensure_active(self):
        if not self.active:
            raise ExecutionError(self.executor.fault_reason or 'executor_not_active')

    def check_state(self):
        if getattr(self.monitor,'driver_fault',None):
            raise ExecutionError(self.monitor.driver_fault)
        if self.monitor.error:
            raise ExecutionError('state_monitor_failure:' + self.monitor.error)
        snapshot = self.monitor.snapshot()
        now = time.monotonic_ns()
        for part, baseline in self.baseline.items():
            row = snapshot[part]
            q = np.asarray(row['position'], dtype=float)
            if q.shape != np.asarray(baseline['position']).shape or not np.isfinite(q).all():
                raise ExecutionError('invalid_component_state:' + part)
            if not 0 <= now - row['receipt_ns'] <= self.c['state_max_age_ns']:
                raise ExecutionError('stale_component_state:' + part)
            if part not in PARTS:
                drift=float(np.max(np.abs(q-baseline['position'])))
                limit=self.c['uncontrolled_max_position_drift'][part]
                if drift>limit:
                    if self.c.get('uncontrolled_drift_action','stop')=='warn':
                        self.warn('uncontrolled_component_drift',part=part,drift=drift,limit=limit,
                                  position=q.tolist(),baseline=baseline['position'])
                    else:raise ExecutionError('uncontrolled_component_drift:' + part)
        return self.monitor.joint10()

    def acquire(self):
        if self.monitor.control_rights_available() or self.monitor.owners():
            raise ExecutionError('control_rights_already_owned')
        measured, stamp = self.check_state()
        self.executor.acquire(measured, stamp, operator_present=True, estop_ready=True,
                              initial_command=self.monitor.command10)
        self.startup = StartupBlend(self.executor.guard.last, self.c.get('startup_blend_seconds',0.))
        self.observer = StopObserver(self.executor, self.monitor.snapshot)
        self.observer.start()

    def hold(self):
        """Re-send the already accepted guarded target with fresh state.

        Used for bounded startup and, when ``queue_exhausted_action='hold'`` is
        configured, for every servo tick without a fresh prediction. It never
        consumes or rebases a prediction point.
        """
        measured, stamp = self.check_state()
        return self.executor.send(self.executor.guard.last.copy(), measured, stamp)

    def send(self, row):
        try:
            if time.monotonic_ns() >= row['deadline_ns']:
                raise TargetExpired('target_expired_before_send')
            measured, stamp = self.check_state()
            # Reject invalid raw targets before blending; blending must not hide
            # a prediction outside hardware bounds.
            decoded = self.executor.guard.validate_target(row['action_raw'],binary_gripper=True)
            transition,alpha = self.startup.target(decoded,time.monotonic_ns())
            source=dict(action_raw=row['action_raw'],decoded_target=decoded.tolist(),
                observation_id=row.get('observation_id'),step_index=row.get('step_index'),
                deadline_ns=row['deadline_ns'],startup_alpha=alpha)
            target = self.executor.send(transition, measured, stamp, deadline_ns=row['deadline_ns'],source=source)
            self.model_commands += 1
            key=(row.get('observation_id'),row.get('step_index'))
            if key!=self.last_policy_key:
                self.policy_targets+=1;self.last_policy_key=key
            row.update(sdk_target=target.tolist(), actual_send_ns=time.monotonic_ns(), simulated_send_ns=None,
                decoded_target=decoded.tolist(),transition_target=transition.tolist(),startup_alpha=alpha,
                execution_phase='startup_transition' if alpha<1 else 'model_tracking',
                control_way=self.c.get('control_way','direct'))
        except TargetExpired:
            raise
        except BaseException as exc:
            self.executor.fault('live_execution:' + str(exc))
            raise

    def stop(self, reason):
        self.executor.stop(reason)

    def close(self):
        errors = []
        try:
            self.executor.stop('bridge_shutdown')
            if self.observer:
                try:
                    self.stop_result = self.observer.wait()
                except Exception as exc:
                    if (self.c.get('stop_feedback_action','stop')=='warn' and isinstance(exc,ExecutionError)
                            and str(exc) in ('stop_did_not_settle_within_budget','stop_observer_timeout')):
                        self.warn(str(exc),feedback=dict(self.observer.diagnostics))
                    else:errors.append(str(exc))
                finally:
                    try:self.emit(dict(event='stop_feedback',**self.observer.diagnostics))
                    except Exception as exc:errors.append('stop_feedback_log:'+str(exc))
        finally:
            try:
                self.executor.close()
            finally:
                if self.monitor:
                    self.monitor.close()
        if errors:
            raise ExecutionError(';'.join(errors))


class ExecutionRunner:
    """Independent 250 Hz SDK loop; the inference thread only updates the queue.

    Uses monotonic deadlines and interruptible waits rather than blocking the
    shutdown path in a middleware sleep. Does not replay missed control ticks.
    """
    def __init__(self, scheduler, live, emit):
        self.scheduler=scheduler
        self.live=live
        self.emit=emit
        self.frequency=live.c['sdk_frequency']
        self.starvation_action=live.c.get('queue_exhausted_action','stop')
        self.stop_event=threading.Event()
        self.thread=None
        self.error=None
        self.started_ns=None
        self.skipped_ticks=0
        self.ticks=0
        self.max_jitter_ns=0
        self.expired_before_send=0
        self.starved_ticks=0
        self.starvation_started_ns=None
        self.max_starvation_ns=0

    def start(self):
        if self.thread is not None:raise ExecutionError('control_runner_already_started')
        self.started_ns=time.monotonic_ns()
        self.thread=threading.Thread(target=self.run,name='s1b-sdk-250hz',daemon=True)
        self.thread.start()

    def starved(self,row):
        """No usable prediction point: hold the last guarded target.

        Only reached when ``queue_exhausted_action='hold'``. The scheduler never
        rebases or re-sends an expired prediction point; the SDK keeps receiving
        the last accepted target until a fresh prediction arrives or another
        independent fault (state staleness, driver error, rights loss, operator
        stop, logging failure) latches a stop.
        """
        now=time.monotonic_ns()
        if self.starvation_started_ns is None:self.starvation_started_ns=now
        self.starved_ticks+=1
        target=self.live.hold()
        row.update(held=True,starvation_ns=now-self.starvation_started_ns,
                   sdk_target=target.tolist(),actual_send_ns=time.monotonic_ns())
        self.live.warn('prediction_starvation',reason=row.get('reason'),
                       starved_ms=round((now-self.starvation_started_ns)/1e6,1))
        return row

    def tick(self, now, planned):
        self.live.ensure_active()
        if getattr(self.scheduler,'strategy',None)=='synchronous':
            row=self.scheduler.sample(now,planned)
            if row['covered']:
                # A failed or late dispatch faults the run; never skip a step.
                self.live.send(row)
                self.scheduler.sent(row,time.monotonic_ns())
            else:
                target=self.live.hold()
                row.update(held=True,sdk_target=target.tolist(),actual_send_ns=time.monotonic_ns())
            return row
        if self.scheduler.state=='SHADOW':
            timeout=self.live.c.get('first_prediction_timeout_seconds',1.)
            if now-self.started_ns>timeout*1e9:
                raise ExecutionError('first_fresh_prediction_timeout')
            self.live.hold()
            return dict(event='startup_hold',tick_ns=now,control_hz=self.frequency)
        # At most one reselection: an expired candidate was never dispatched,
        # so selecting its successor does not create a burst of SDK commands.
        for attempt in range(2):
            row=self.scheduler.sample(now,planned)
            if not row['covered']:
                return self.starved(row)
            try:
                self.live.send(row)
                if self.starvation_started_ns is not None:
                    self.max_starvation_ns=max(self.max_starvation_ns,time.monotonic_ns()-self.starvation_started_ns)
                    self.starvation_started_ns=None
                return row
            except TargetExpired:
                self.live.ensure_active()
                self.expired_before_send+=1
                row.update(event='target_skipped',reason='target_expired_before_send',
                           covered=False,actual_send_ns=None,simulated_send_ns=None,
                           control_way=self.live.c.get('control_way','direct'))
                if attempt==1:return row
                self.emit(row)
                now=time.monotonic_ns()
        return row

    def run(self):
        ordinal=0
        try:
            while not self.stop_event.is_set():
                planned=self.started_ns+round(ordinal*1e9/self.frequency)
                delay=(planned-time.monotonic_ns())/1e9
                if delay>0 and self.stop_event.wait(delay):break
                now=time.monotonic_ns()
                self.max_jitter_ns=max(self.max_jitter_ns,now-planned)
                row=self.tick(now,planned)
                row['tick_ordinal']=ordinal
                self.emit(row)
                self.ticks+=1
                ordinal+=1
                earliest=(time.monotonic_ns()-self.started_ns)*self.frequency//1_000_000_000
                if earliest>ordinal:
                    self.skipped_ticks+=earliest-ordinal
                    ordinal=earliest
        except BaseException as exc:
            # A tick already in flight may observe the explicitly stopped
            # executor. Preserve real faults, but do not relabel cancellation
            # as a control-thread failure.
            if (self.stop_event.is_set() and self.live.executor.state=='STOPPED'
                    and self.live.executor.fault_reason in ('operator_stop','control_loop_shutdown')
                    and str(exc) in ('operator_stop','control_loop_shutdown','executor_not_active',
                                     'stopped_before_send','stopped_during_send','session_not_active')):
                return
            self.error=repr(exc)
            # Stop the SDK even if the scheduler is not armed during warmup.
            try:self.live.executor.fault('control_thread_failure:'+str(exc))
            finally:self.scheduler.fault('control_thread_failure')

    def stats(self):
        return dict(requested_hz=self.frequency,ticks=self.ticks,skipped_ticks=self.skipped_ticks,
                    max_jitter_ns=self.max_jitter_ns,expired_before_send=self.expired_before_send,
                    starvation_action=self.starvation_action,starved_ticks=self.starved_ticks,
                    max_starvation_ns=self.max_starvation_ns,error=self.error)

    def close(self):
        self.stop_event.set()
        # Stop before joining a possibly stalled control thread.
        try:self.live.stop('control_loop_shutdown')
        finally:
            if self.thread:
                self.thread.join(timeout=5)
                if self.thread.is_alive():raise ExecutionError('control_thread_did_not_stop')
            self.scheduler.stop()
