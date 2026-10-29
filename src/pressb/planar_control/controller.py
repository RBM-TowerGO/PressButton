"""Serial tool admission and robot-local servo; Agent latency is off the servo path."""
from collections import OrderedDict, deque
import copy
import threading
import time
import uuid
import numpy as np
from .geometry import vector


class Controller:
    def __init__(self, planner, backend, config, emit=lambda row: None):
        self.planner, self.backend, self.c, self.emit = planner, backend, config, emit
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.phase = 'READY'
        self.error = None
        self.result = None
        self.task = None
        self.queue = deque()
        self.commands = OrderedDict()
        self.observations = OrderedDict()
        self.generation = 0
        self.completed_ns = time.monotonic_ns()
        self.completed_source_ns = time.time_ns()
        self.uv = vector(config['initial_uv_m'], 2)
        self.normal_m = 0.
        self.target = planner.plane.target(self.uv)
        initial = backend.snapshot()
        planner.check_pose(initial['tcp'], self.target)
        self.q = vector(initial['q'], 7)
        self.last_state = initial
        self.worker = None
        self.settle_since = None
        self.settle_deadline = None
        self.thread = threading.Thread(target=self.run, daemon=True, name='planar-servo')
        self.thread.start()

    def observe(self):
        with self.lock:
            if self.phase != 'READY':
                raise ValueError('observation_requires_ready')
            generation = self.generation
        observation = self.backend.observe()
        frames = observation['images']
        with self.lock:
            if self.phase != 'READY' or generation != self.generation:
                raise ValueError('observation_crossed_motion')
            if any(f['receipt_ns'] <= self.completed_ns or f['source_ns'] <= self.completed_source_ns
                   for f in frames.values()):
                raise ValueError('images_predate_completion')
            now = time.monotonic_ns()
            if any(not 0 <= now-f['receipt_ns'] <= self.c['observation_max_age_ns'] for f in frames.values()):
                raise ValueError('stale_images')
            oid = str(uuid.uuid4())
            self.observations[oid] = (min(f['receipt_ns'] for f in frames.values()), generation)
            while len(self.observations) > 32:
                self.observations.popitem(last=False)
            self.emit(dict(event='observation', observation_id=oid, **observation))
            return dict(observation_id=oid, **observation,
                        directions=dict(du_positive='robot right', dv_positive='up', units='mm'),
                        state=self.get_state())

    def submit(self, kind, observation_id, command_id, du_mm=0., dv_mm=0., normal_mm=0.):
        if kind not in ('move', 'press', 'air_probe', 'advance', 'retract') or not isinstance(command_id, str) or not 1 <= len(command_id) <= 128:
            raise ValueError('invalid_command')
        if kind == 'air_probe':
            if not np.isfinite(normal_mm) or not 0 < normal_mm/1000 <= self.c['air_probe_max_m']:
                raise ValueError('invalid_air_probe_extension')
        elif kind == 'advance':
            forward = self.c.get('keyboard_forward', {})
            if forward.get('verified') is not True or not forward.get('evidence'):
                raise ValueError('keyboard_forward_calibration_required')
            if not np.isfinite(normal_mm) or not 0 < normal_mm/1000 <= forward['max_distance_m']:
                raise ValueError('keyboard_forward_out_of_range')
        elif normal_mm != 0:
            raise ValueError('normal_displacement_is_local_only')
        delta = vector([du_mm, dv_mm], 2)/1000
        signature = (kind, observation_id, float(du_mm), float(dv_mm), float(normal_mm))
        with self.lock:
            if command_id in self.commands:
                previous, receipt = self.commands[command_id]
                if previous != signature:
                    raise ValueError('command_id_conflict')
                return dict(receipt)
            if len(self.commands) >= self.c['max_commands_per_session']:
                raise ValueError('session_command_limit')
            if self.phase != 'READY':
                raise ValueError('controller_not_ready')
            if self.normal_m != 0 and kind != 'retract':
                raise ValueError('retract_to_plane_first')
            if kind == 'retract' and self.normal_m == 0:
                raise ValueError('already_on_plane')
            observed = self.observations.get(observation_id)
            if (observed is None or observed[1] != self.generation
                    or not 0 <= time.monotonic_ns()-observed[0] <= self.c['observation_max_age_ns']):
                raise ValueError('stale_or_unknown_observation')
            if np.linalg.norm(delta) > self.c['max_move_m']:
                raise ValueError('move_too_large')
            goal = self.uv + delta if kind == 'move' else self.uv.copy()
            self.planner.plane.target(goal)
            if kind == 'press' and not self.c['press']['verified']:
                raise ValueError('press_calibration_required')
            receipt = dict(task_id=str(uuid.uuid4()), accepted=True, command_id=command_id)
            self.commands[command_id] = (signature, receipt)
            self.generation += 1
            self.phase = 'PLANNING'
            self.task = dict(**receipt, kind=kind, goal_uv_m=goal.tolist(), air_probe_m=normal_mm/1000,
                             goal_normal_m=normal_mm/1000 if kind == 'advance' else 0.)
            self.result = None
            self.emit(dict(event='task_accepted', **self.task))
            self.worker = threading.Thread(target=self.plan, args=(kind, goal), daemon=True, name='planar-planner')
            self.worker.start()
            return dict(receipt)

    def plan(self, kind, goal):
        try:
            with self.lock:
                start = np.r_[self.uv, self.normal_m]
                seed = self.q.copy()
            if kind == 'move':
                points = self.planner.path(start, np.r_[goal, 0.], seed, self.c['move_speed_m_s'], self.cancel.is_set)
            elif kind == 'air_probe':
                end = np.r_[self.uv, self.task['air_probe_m']]
                points = self.planner.path(start, end, seed, self.c['move_speed_m_s'], self.cancel.is_set)
                points += self.planner.path(end, start, points[-1].q, self.c['move_speed_m_s'], self.cancel.is_set)
            elif kind in ('advance', 'retract'):
                end = np.r_[self.uv, self.task['goal_normal_m']]
                points = self.planner.path(start, end, seed, self.c['keyboard_forward']['speed_m_s'], self.cancel.is_set)
            else:
                p = self.c['press']
                ranging = self.backend.range()
                if not ranging.get('trusted'):
                    raise ValueError('trusted_range_required')
                gap = float(ranging['gap_m'])
                if not np.isfinite(gap) or not p['minimum_gap_m'] <= gap <= p['maximum_gap_m']:
                    raise ValueError('gap_out_of_range')
                approach = np.r_[self.uv, max(0., gap-p['contact_margin_m'])]
                end = np.r_[self.uv, gap+p['overtravel_m']]
                if end[2] > p['maximum_extension_m']:
                    raise ValueError('maximum_extension')
                points = self.planner.path(start, approach, seed, p['approach_speed_m_s'], self.cancel.is_set)
                points += self.planner.path(approach, end, points[-1].q, p['contact_speed_m_s'], self.cancel.is_set)
                points += [points[-1]] * int(np.ceil(p['dwell_s']*self.c['frequency_hz']))
                points += self.planner.path(end, start, points[-1].q, p['retract_speed_m_s'], self.cancel.is_set)
                # Recheck depth after potentially slow IK; do not execute an expired estimate.
                fresh = self.backend.range()
                if (not fresh.get('trusted') or abs(fresh['gap_m']-gap) > p['range_change_tolerance_m']):
                    raise ValueError('range_changed_during_planning')
            with self.lock:
                if self.cancel.is_set():
                    return
                self.queue = deque(points)
                self.phase = 'PRESSING' if kind == 'press' else 'MOVING'
                self.emit(dict(event='path_validated', task_id=self.task['task_id'], points=len(points)))
        except Exception as exc:
            with self.lock:
                if not self.cancel.is_set():
                    self.phase = 'READY'
                    self.completed_ns = time.monotonic_ns()
                    self.completed_source_ns = time.time_ns()
                    self.result = dict(status='rejected', reason=str(exc), action_completed=False,
                                       contact_evidence=None, button_trigger_evidence=None)
                    self.emit(dict(event='task_rejected', result=self.result))

    def tick(self):
        state = self.backend.snapshot()
        with self.lock:
            if self.cancel.is_set():
                return
            self.last_state = state
            # Measured path must remain close to the previous commanded Cartesian point.
            self.planner.check_pose(state['tcp'], self.target)
            if self.queue:
                point = self.queue.popleft()
                self.q, self.target = point.q, point.target
            elif self.phase in ('MOVING', 'PRESSING'):
                self.phase = 'SETTLING'
                self.settle_deadline = time.monotonic()+self.c['settle_timeout_s']
                self.settle_since = None
            if self.phase == 'SETTLING':
                now = time.monotonic()
                if now > self.settle_deadline:
                    raise ValueError('settle_timeout')
                if np.max(np.abs(state['q']-self.q)) <= self.c['settle_joint_error_rad'] and np.max(np.abs(state['velocity'])) <= self.c['settle_velocity_rad_s']:
                    self.settle_since = self.settle_since or now
                    if now-self.settle_since >= self.c['settle_duration_s']:
                        self.uv = np.asarray(self.task['goal_uv_m'])
                        self.normal_m = self.task['goal_normal_m']
                        self.phase = 'READY'
                        self.completed_ns = time.monotonic_ns()
                        self.completed_source_ns = time.time_ns()
                        self.result = dict(status='completed', action_completed=True,
                                           contact_evidence=None, button_trigger_evidence=None)
                        self.emit(dict(event='task_completed', task_id=self.task['task_id'], result=self.result))
                else:
                    self.settle_since = None
            q = self.q.copy()
        # Backend stop latch serializes SDK sends against stop, independently of tool locks.
        self.backend.send(q, state)
        self.emit(dict(event='servo', receipt_ns=state['receipt_ns'], q=state['q'].tolist(),
                       target_q=q.tolist(), target_tcp=self.target.tolist(), phase=self.phase,
                       tcp=state['tcp'].tolist(), torque=state['torque'].tolist()))

    def run(self):
        period = 1/self.c['frequency_hz']
        due = time.monotonic()
        try:
            while not self.cancel.is_set():
                self.tick()
                due += period
                remaining = due-time.monotonic()
                if remaining < -self.c['max_servo_lateness_s']:
                    raise ValueError('servo_deadline_missed')
                self.cancel.wait(max(0, remaining))
        except Exception as exc:
            self.stop(str(exc), fault=True)

    def get_state(self):
        with self.lock:
            coordinates = self.planner.plane.coordinates(self.last_state['tcp'])
            return copy.deepcopy(dict(phase=self.phase, fault=self.error, task=self.task, result=self.result,
                completed_uv_m=self.uv.tolist(), measured_uvd_m=coordinates.tolist(),
                completed_normal_m=self.normal_m,
                tcp_error_m=float(np.linalg.norm(self.last_state['tcp'][:3, 3]-self.target[:3, 3])),
                state_receipt_ns=self.last_state['receipt_ns']))

    def stop(self, reason='operator_stop', fault=False):
        # Stop actuator first; neither planning nor an observation can delay this behind a lock.
        if self.cancel.is_set() and self.phase in ('FAULT', 'STOPPED'):
            return self.get_state()
        self.cancel.set()
        try:
            self.backend.stop(reason, fault)
        finally:
            with self.lock:
                if self.phase in ('PLANNING', 'MOVING', 'PRESSING', 'SETTLING'):
                    self.result = dict(status='interrupted', reason=reason, action_completed=False,
                                       contact_evidence=None, button_trigger_evidence=None)
                self.phase = 'FAULT' if fault else 'STOPPED'
                self.error = reason
                self.queue.clear()
        return self.get_state()

    def close(self):
        self.stop()
        self.thread.join(timeout=2)
        if self.worker:
            self.worker.join(timeout=2)
        if self.thread.is_alive() or (self.worker and self.worker.is_alive()):
            raise RuntimeError('stopped_but_sdk_call_has_not_returned')
        self.backend.close()
