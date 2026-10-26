"""Robot-local deadline consumption, independent of ROS, network and GPU."""
from collections import Counter, deque
import threading
import numpy as np
from .contract import VERSION, OFFSETS

PERIOD_NS = 33_333_334  # upper bound on one rational 30 Hz interval

class ScheduleError(ValueError):
    pass

class FakeExecutor:
    """Records simulated sends only; never imports or constructs an SDK client."""
    def __init__(self):
        self.calls = []
        self.stops = []
    def send(self, record):
        self.calls.append(record)
    def stop(self, reason):
        self.stops.append(reason)

class Scheduler:
    def __init__(self, session_id, model_sha256, contract_sha256, executor=None):
        self.session_id = session_id
        self.model_sha256 = model_sha256
        self.contract_sha256 = contract_sha256
        self.executor = executor
        self.state = 'SHADOW'
        self.queue = deque()
        self.last_observation_id = -1
        self.last_observation_ns = -1
        self.last_tick_ns = -1
        self.counts = Counter()
        self.lock = threading.Lock()

    def accept(self, response, received_ns):
        """Validate everything before replacing a queue; equality is expired here."""
        with self.lock:
            if self.state not in ('SHADOW', 'EXECUTING'):
                raise ScheduleError('session_not_active')
            identity = {'protocol_version': VERSION, 'session_id': self.session_id,
                        'model_sha256': self.model_sha256, 'data_contract_sha256': self.contract_sha256}
            try:
                for key, expected in identity.items():
                    x = np.asarray(response[key])
                    if x.shape != () or x.dtype.kind not in 'US' or x.item() != expected:
                        raise ScheduleError('identity_' + key)
                for key in ('observation_id', 'observation_monotonic_ns'):
                    x = np.asarray(response[key])
                    if x.shape != () or x.dtype.kind not in 'iu' or int(x) < 0:
                        raise ScheduleError('invalid_' + key)
                oid = int(response['observation_id'])
                observed = int(response['observation_monotonic_ns'])
                if oid <= self.last_observation_id or observed <= self.last_observation_ns:
                    raise ScheduleError('observation_out_of_order')
                if observed > received_ns:
                    raise ScheduleError('future_observation')
                action = response['action_raw']
                offsets = response['target_offsets_ns']
                if (not isinstance(action, np.ndarray) or action.dtype != np.float32
                        or action.shape != (7, 10) or not np.isfinite(action).all()):
                    raise ScheduleError('invalid_action')
                if (not isinstance(offsets, np.ndarray) or offsets.dtype != np.int64
                        or offsets.shape != (7,) or not np.array_equal(offsets, OFFSETS)):
                    raise ScheduleError('invalid_offsets')
            except (ScheduleError, KeyError, TypeError, OverflowError) as exc:
                self._fault('invalid_response')
                if isinstance(exc, ScheduleError):raise
                raise ScheduleError('missing_or_invalid_field') from exc
            # A newer observed prediction invalidates old pending targets even if
            # every point in the new response has expired.
            self.last_observation_id = oid
            self.last_observation_ns = observed
            self.counts['replaced_points'] += len(self.queue)
            self.queue.clear()
            for index, offset in enumerate(offsets):
                deadline = observed + int(offset)
                if deadline > received_ns:
                    self.queue.append(dict(session_id=self.session_id, observation_id=oid,
                        observation_monotonic_ns=observed, received_ns=received_ns,
                        step_index=index, deadline_ns=deadline, action_raw=action[index].copy()))
                else:
                    self.counts['expired_on_receive'] += 1
            if not self.queue:
                self.counts['expired_responses'] += 1
                if self.state == 'EXECUTING':
                    self._fault('all_points_expired')
                return False
            self.counts['accepted_responses'] += 1
            return True

    def tick(self, now_ns, planned_tick_ns=None):
        """Consume once; a queued target exactly at this tick is still allowed."""
        with self.lock:
            if now_ns <= self.last_tick_ns:
                raise ScheduleError('tick_out_of_order')
            self.last_tick_ns = now_ns
            self.counts['ticks'] += 1
            if self.state not in ('SHADOW', 'EXECUTING'):
                return dict(event='tick', state=self.state, tick_ns=now_ns, planned_tick_ns=planned_tick_ns, covered=False, reason='inactive')
            while self.queue and self.queue[0]['deadline_ns'] < now_ns:
                self.queue.popleft()
                self.counts['expired_on_tick'] += 1
            row = dict(event='tick', state=self.state, tick_ns=now_ns,
                       planned_tick_ns=planned_tick_ns, covered=False, queue_length=len(self.queue))
            if not self.queue or self.queue[0]['deadline_ns'] > now_ns + PERIOD_NS:
                self.counts['uncovered_ticks'] += 1
                row['reason'] = 'queue_exhausted' if not self.queue else 'no_point_in_tick_window'
                if self.state == 'EXECUTING':
                    self._fault(row['reason'])
                # SHADOW continues collecting evidence; it records where a real
                # execution would fault rather than enabling any actuator.
                return row
            item = self.queue.popleft()
            raw = item.pop('action_raw')
            row.update(item, covered=True, action_raw=raw.tolist(),
                       gripper_binary=int(raw[7] >= .5),
                       simulated_gripper_sdk_target=100 if raw[7] >= .5 else 0,
                       sdk_target=None, actual_send_ns=None,
                       simulated_send_ns=now_ns, deadline_error_ns=now_ns-item['deadline_ns'],
                       response_to_tick_ns=now_ns-item['received_ns'],
                       tick_jitter_ns=None if planned_tick_ns is None else now_ns-planned_tick_ns)
            simulated = raw.copy()
            simulated[7] = row['simulated_gripper_sdk_target']
            row['simulated_sdk_target10'] = simulated.tolist()
            self.counts['covered_ticks'] += 1
            if self.executor is not None:
                try:
                    self.executor.send(row)
                except Exception:
                    self._fault('executor_error')
                    raise
            return row

    def _fault(self, reason):
        self.queue.clear()
        self.state = 'FAULT'
        self.counts['faults'] += 1
        if self.executor is not None:
            self.executor.stop(reason)

    def fault(self, reason):
        with self.lock:
            self._fault(reason)

    def reconnect_shadow(self, session_id):
        """Reconnection always starts a fresh SHADOW session, never EXECUTING."""
        with self.lock:
            if session_id == self.session_id:
                raise ScheduleError('new_session_required')
            self.queue.clear()
            self.session_id = session_id
            self.last_observation_id = self.last_observation_ns = -1
            self.state = 'SHADOW'

    def enable_fake_execution(self, *, session_id, model_sha256, explicit=False):
        with self.lock:
            if (type(self.executor) is not FakeExecutor or not explicit or self.state != 'SHADOW'
                    or session_id != self.session_id or model_sha256 != self.model_sha256 or not self.queue):
                raise ScheduleError('fake_execution_not_armed')
            self.state = 'EXECUTING'

    def stop(self):
        with self.lock:
            self.queue.clear()
            self.state = 'STOPPED'
            if self.executor is not None:
                self.executor.stop('operator_stop')

    def stats(self):
        with self.lock:
            return dict(state=self.state, counts=dict(self.counts), queue_length=len(self.queue),
                        last_observation_id=self.last_observation_id)
