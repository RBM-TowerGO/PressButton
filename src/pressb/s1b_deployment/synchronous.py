"""Serial action chunks timed from execution, never from image observation."""
from .scheduler import Scheduler, ScheduleError, PERIOD_NS
from .contract import OFFSETS


class SynchronousScheduler(Scheduler):
    strategy = 'synchronous'

    def __init__(self, *args, wait_timeout_ns=1_000_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_timeout_ns = wait_timeout_ns
        self.wait_since_ns = None
        self.chunk_id = None
        self.completed_chunk_id = None
        self.completed_ns = None
        self.chunk_started_ns = None
        self.step_started_ns = None
        self.step_sent = False
        self.completed_chunks = 0
        self.schedule_slip_ns = 0

    def begin_wait(self, now_ns):
        with self.lock:
            self.wait_since_ns = now_ns

    def accept(self, response, received_ns):
        with self.lock:
            if self.state not in ('SHADOW', 'EXECUTING'):
                raise ScheduleError('session_not_active')
            if self.queue:
                raise ScheduleError('chunk_in_progress')
            if self.wait_since_ns is not None and received_ns-self.wait_since_ns >= self.wait_timeout_ns:
                raise ScheduleError('synchronous_prediction_timeout')
            # Reuse the complete wire validator without applying observation-age
            # deadlines. The original observation timestamp is never rewritten.
            check = Scheduler(self.session_id, self.model_sha256, self.contract_sha256)
            check.last_observation_id = self.last_observation_id
            check.last_observation_ns = self.last_observation_ns
            observed = int(response['observation_monotonic_ns'])
            if observed > received_ns:
                raise ScheduleError('future_observation')
            if self.completed_ns is not None and observed <= self.completed_ns:
                raise ScheduleError('observation_predates_chunk_completion')
            check.accept(response, observed)
            for item in check.queue:
                item['received_ns'] = received_ns
                item.pop('deadline_ns')
            self.queue = check.queue
            self.last_observation_id = check.last_observation_id
            self.last_observation_ns = observed
            self.chunk_id = self.last_observation_id
            self.chunk_started_ns = self.step_started_ns = None
            self.step_sent = False
            self.schedule_slip_ns = 0
            self.counts['accepted_responses'] += 1
            return True

    def sample(self, now_ns, planned_tick_ns=None):
        with self.lock:
            if self.state not in ('SHADOW', 'EXECUTING'):
                raise ScheduleError('session_not_active')
            if now_ns <= self.last_tick_ns:
                raise ScheduleError('tick_out_of_order')
            if self.queue and self.last_tick_ns >= 0 and now_ns-self.last_tick_ns > PERIOD_NS:
                self._fault('synchronous_control_stall')
                raise ScheduleError('synchronous_control_stall')
            self.last_tick_ns = now_ns
            self.counts['servo_ticks'] += 1
            completed = False
            if self.queue and self.step_started_ns is not None and now_ns >= self.chunk_started_ns+self.schedule_slip_ns+int(OFFSETS[self.queue[0]['step_index']]):
                if not self.step_sent:
                    self._fault('synchronous_step_not_sent')
                    raise ScheduleError('synchronous_step_not_sent')
                self.queue.popleft()
                self.step_started_ns = None
                self.step_sent = False
                if not self.queue:
                    self.completed_chunk_id = self.chunk_id
                    self.completed_ns = self.wait_since_ns = now_ns
                    self.completed_chunks += 1
                    completed = True
            row = dict(event='servo_tick', state=self.state, tick_ns=now_ns,
                       planned_tick_ns=planned_tick_ns, execution_strategy=self.strategy,
                       chunk_id=self.chunk_id, chunk_started_ns=self.chunk_started_ns,
                       covered=False, queue_length=len(self.queue))
            if not self.queue:
                if self.wait_since_ns is not None and now_ns-self.wait_since_ns >= self.wait_timeout_ns:
                    self._fault('synchronous_prediction_timeout')
                    raise ScheduleError('synchronous_prediction_timeout')
                row.update(event='chunk_completed' if completed else 'synchronous_wait',
                           completed_chunk_id=self.completed_chunk_id, completed_ns=self.completed_ns,
                           wait_ns=0 if self.wait_since_ns is None else now_ns-self.wait_since_ns)
                return row
            item = self.queue[0]
            # This dispatch deadline bounds a stalled SDK send, not model age.
            row.update({k:v for k,v in item.items() if k != 'action_raw'})
            row.update(covered=True, action_raw=item['action_raw'].tolist(),
                       deadline_ns=now_ns+PERIOD_NS, sdk_target=None, actual_send_ns=None)
            return row

    def sent(self, row, sent_ns):
        """Acknowledge actual dispatch before any transition to the next step."""
        with self.lock:
            if self.state not in ('SHADOW', 'EXECUTING'):
                raise ScheduleError('session_not_active')
            if not self.queue or (row['observation_id'],row['step_index']) != (self.chunk_id,self.queue[0]['step_index']):
                raise ScheduleError('chunk_dispatch_mismatch')
            if sent_ns >= row['deadline_ns']:
                self._fault('synchronous_control_stall')
                raise ScheduleError('synchronous_control_stall')
            if self.step_started_ns is None:
                self.step_started_ns = sent_ns
                if self.chunk_started_ns is None:
                    self.chunk_started_ns = sent_ns
                    row['wait_ns'] = 0 if self.wait_since_ns is None else sent_ns-self.wait_since_ns
                else:
                    nominal_start=self.chunk_started_ns+self.schedule_slip_ns+int(OFFSETS[row['step_index']-1])
                    # Ordinary 250 Hz quantization is <=4 ms. A later dispatch
                    # shifts the remaining grid instead of squeezing the next
                    # step into a short catch-up interval.
                    lateness=sent_ns-nominal_start
                    if lateness>4_000_000:self.schedule_slip_ns+=lateness
                row['event'] = 'chunk_step_started'
            self.step_sent = True
            row['chunk_started_ns'] = self.chunk_started_ns
            row['step_started_ns'] = self.step_started_ns
            row['schedule_slip_ns'] = self.schedule_slip_ns

    def tick(self, now_ns, planned_tick_ns=None):
        row = self.sample(now_ns, planned_tick_ns)
        if row['covered']:
            if self.executor is not None:
                self.executor.send(row)
            self.sent(row, now_ns)
            row['simulated_send_ns'] = now_ns
        return row

    def enable_execution(self, executor, now_ns):
        from .live_execution import LiveExecution
        with self.lock:
            if (not isinstance(executor, LiveExecution) or not executor.active
                    or self.executor is not None or self.state != 'SHADOW' or not self.queue):
                raise ScheduleError('execution_not_armed')
            self.executor = executor
            self.state = 'EXECUTING'

    def progress(self):
        with self.lock:
            return dict(execution_strategy=self.strategy, chunk_id=self.chunk_id,
                        completed_chunk_id=self.completed_chunk_id, completed_ns=self.completed_ns,
                        completed_chunks=self.completed_chunks, chunk_started_ns=self.chunk_started_ns,
                        schedule_slip_ns=self.schedule_slip_ns,
                        chunk_phase=('EXECUTING' if self.step_started_ns is not None else 'PENDING')
                            if self.queue else 'WAITING')

    def stats(self):
        return dict(super().stats(), **self.progress())
