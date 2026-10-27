"""Small action-only HTTP endpoint; deadlines and control stay robot-local."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import numpy as np
from .contract import VERSION
from .protocol import decode
from .scheduler import Scheduler
from .shadow import ShadowRunner
from .live_execution import ExecutionRunner


class ControlSession:
    def __init__(self, session_id, boot_id, step, model_sha, contract_sha, emit, tick_emit, live=None):
        self.session_id,self.boot_id,self.step=session_id,boot_id,step
        self.model_sha,self.contract_sha=model_sha,contract_sha
        self.live=live;self.emit=emit
        # Starvation policy belongs to the execution config: 'stop' keeps the
        # original latch, 'hold' keeps the last guarded target and waits.
        starvation=live.c.get('queue_exhausted_action','stop') if live else 'stop'
        self.scheduler=Scheduler(session_id,model_sha,contract_sha,starvation_action=starvation)
        self.runner=ExecutionRunner(self.scheduler,live,tick_emit) if live else ShadowRunner(self.scheduler,tick_emit)
        self.stopping=threading.Event();self.lock=threading.Lock()
        self.received=0;self.accepted=0;self.error=None
        self.last_received_ns=None;self.max_gap_ns=0;self.margins=[]
        self.last_deadline_ns=None;self.coverage_gaps=0;self.max_coverage_gap_ns=0
        self.phase='WAITING_FOR_CLIENT'
        self.start_requested=threading.Event()
        self.preflight_observation_id=None
        self.accept_observations_after_ns=None

    def request_start(self,response):
        """Validate and discard a fresh preflight before any SDK acquisition."""
        with self.lock:
            if self.phase!='WAITING_FOR_CLIENT' or self.stopping.is_set() or self.error:
                raise ValueError('start_not_allowed')
            self.validate_boot(response)
            check=Scheduler(self.session_id,self.model_sha,self.contract_sha)
            if not check.accept(response,time.monotonic_ns()):
                raise ValueError('preflight_prediction_expired')
            self.preflight_observation_id=int(response['observation_id'])
            self.emit(dict(event='client_preflight_ready',monotonic_ns=time.monotonic_ns(),
                observation_id=self.preflight_observation_id))
            self.phase='INITIALIZING'
            return dict(ok=True,phase=self.phase)

    def confirm_start(self):
        # The HTTP acknowledgement must be fully written before native SDK
        # initialization can hold the GIL and block the HTTP handler.
        with self.lock:
            if self.phase!='INITIALIZING' or self.stopping.is_set() or self.error:
                raise ValueError('start_not_allowed')
            self.start_requested.set()

    def validate_boot(self,response):
        boot=np.asarray(response.get('robot_boot_id'))
        if boot.shape!=() or boot.dtype.kind not in 'US' or boot.item()!=self.boot_id:
            raise ValueError('robot_boot_identity_mismatch')

    def start(self):
        if self.phase!='INITIALIZING' or not self.start_requested.is_set():
            raise ValueError('client_preflight_required')
        if self.stopping.is_set():return
        if self.live:
            self.live.prepare()
            if self.stopping.is_set():return
            self.live.acquire()
        if self.stopping.is_set():return
        self.accept_observations_after_ns=time.monotonic_ns()
        self.runner.start()
        self.phase='AWAITING_PREDICTION'
        self.emit(dict(event='control_initialized',monotonic_ns=time.monotonic_ns(),
            accept_observations_after_ns=self.accept_observations_after_ns,
            runner_started_ns=self.runner.started_ns))

    def health(self):
        fault=self.error or self.runner.error or (self.live.executor.fault_reason if self.live else None)
        return dict(ready=not self.stopping.is_set() and not fault,protocol_version=VERSION,
            startup_protocol=1,
            session_id=self.session_id,boot_id=self.boot_id,step=self.step,model_sha256=self.model_sha,
            data_contract_sha256=self.contract_sha,action_dim=10,horizon=7,
            phase=self.phase,accept_observations_after_ns=self.accept_observations_after_ns,
            mode='execute' if self.live else 'shadow',fault=fault,
            robot_monotonic_ns=time.monotonic_ns())

    def fail(self,reason):
        self.error=self.error or reason
        self.phase='FAULT'
        # Stop is latched even before the first fresh prediction arms scheduling.
        if self.live:self.live.executor.fault(reason)
        self.scheduler.fault(reason)

    def accept(self,response):
        with self.lock:
            try:
                if self.stopping.is_set() or self.error or self.runner.error:
                    raise ValueError('control_session_inactive')
                if self.phase not in ('AWAITING_PREDICTION','RUNNING'):
                    raise ValueError('control_not_initialized')
                if self.live:self.live.ensure_active()
                self.validate_boot(response)
                if (int(response['observation_id'])<=self.preflight_observation_id
                        or int(response['observation_monotonic_ns'])<self.accept_observations_after_ns):
                    raise ValueError('observation_predates_initialization')
                received=time.monotonic_ns()
                accepted=self.scheduler.accept(response,received)
                self.received+=1;self.accepted+=int(accepted)
                if self.last_received_ns is not None:
                    self.max_gap_ns=max(self.max_gap_ns,received-self.last_received_ns)
                self.last_received_ns=received
                margin=int(response['observation_monotonic_ns'])+int(response['target_offsets_ns'][-1])-received
                self.margins.append(margin)
                if self.last_deadline_ns is not None and received>self.last_deadline_ns:
                    self.coverage_gaps+=1
                    self.max_coverage_gap_ns=max(self.max_coverage_gap_ns,received-self.last_deadline_ns)
                self.last_deadline_ns=received+margin
                if accepted and self.live and self.scheduler.state=='SHADOW':
                    self.scheduler.enable_execution(self.live,time.monotonic_ns())
                if accepted:self.phase='RUNNING'
                row=dict(event='prediction_received',observation_id=int(response['observation_id']),
                    observation_monotonic_ns=int(response['observation_monotonic_ns']),
                    received_ns=received,horizon_remaining_ns=margin,accepted=accepted,
                    action_raw=response['action_raw'].tolist())
                self.emit(row)
                return dict(ok=True,accepted=accepted,received_ns=received,horizon_remaining_ns=margin)
            except BaseException as exc:
                self.fail('prediction_rejected:'+str(exc));raise

    def stop(self):
        self.stopping.set()
        self.runner.stop_event.set()
        if self.phase!='FAULT':self.phase='STOPPED'
        if self.live:self.live.stop('operator_stop')
        self.scheduler.stop()

    def close(self):
        self.stopping.set()
        try:self.runner.close()
        finally:
            if self.live:self.live.close()

    def stats(self):
        return dict(mode='execute' if self.live else 'shadow',session_id=self.session_id,
            phase=self.phase,preflight_observation_id=self.preflight_observation_id,
            accept_observations_after_ns=self.accept_observations_after_ns,
            received_predictions=self.received,accepted_predictions=self.accepted,
            max_prediction_gap_ms=self.max_gap_ns/1e6,
            coverage_gaps=self.coverage_gaps,max_coverage_gap_ms=self.max_coverage_gap_ns/1e6,
            horizon_remaining_ms_p0_p50_p95=(np.percentile(self.margins,[0,50,95])/1e6).tolist() if self.margins else [],
            scheduler=self.scheduler.stats(),runner_error=self.runner.error,error=self.error,
            execution_enabled=bool(self.live and self.live.executor.acquired_once),
            sent_commands=self.live.executor.sent_commands if self.live else 0,
            model_commands=self.live.model_commands if self.live else 0,
            control_loop=self.runner.stats() if self.live else None,
            execution_fault=self.live.executor.fault_reason if self.live else None,
            warnings=self.live.warning_snapshot() if self.live else [],
            stop_feedback=self.live.observer.diagnostics if self.live and self.live.observer else None,
            stop_settled_ns=self.live.stop_result if self.live else None)


def make_control_server(core, port=6200, host='127.0.0.1'):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        disable_nagle_algorithm=True
        def log_message(self,*args):pass
        def setup(self):
            super().setup();self.connection.settimeout(2)
        def reply(self,status,value):
            body=json.dumps(value,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Length',str(len(body)))
            self.send_header('Content-Type','application/json')
            if status!=200 or core.stopping.is_set():
                self.send_header('Connection','close');self.close_connection=True
            self.end_headers();self.wfile.write(body);self.wfile.flush()
        def do_GET(self):
            if self.path=='/health':self.reply(200,core.health())
            else:self.reply(404,dict(error='unknown_endpoint'))
        def do_POST(self):
            try:
                if self.headers.get('Transfer-Encoding'):raise ValueError('unsupported_transfer_encoding')
                n=int(self.headers.get('Content-Length','-1'))
                if not 0<n<=65536:raise ValueError('invalid_content_length')
                data=self.rfile.read(n)
                if len(data)!=n:raise ValueError('incomplete_body')
                if self.path=='/prediction':result=core.accept(decode(data))
                elif self.path=='/start':result=core.request_start(decode(data))
                elif self.path=='/stop':
                    if json.loads(data).get('session_id')!=core.session_id:raise ValueError('session_identity_mismatch')
                    core.stop();result=dict(ok=True)
                else:raise ValueError('unknown_endpoint')
                self.reply(202 if self.path=='/start' else 200,result)
                if self.path=='/start':core.confirm_start()
            except Exception as exc:
                core.fail('control_request_failure:'+str(exc))
                self.reply(400,dict(error=str(exc)))
    server=ThreadingHTTPServer((host,port),Handler)
    server.timeout=.1
    return server
