"""Robot-local serial tasks, immutable observations and loopback HTTP API."""
import base64
import copy
import fcntl
import io
import json
import os
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .common import atomic_json, json_bytes, request, sha256, SESSION_SCHEMA
from s1b_keyboard_control import validate_command



class Camera:
    def __init__(self, base='http://127.0.0.1:8765'):
        self.base = base

    def selection(self):
        return request(self.base, '/api/snapshot')['selection']

    def capture_now(self):
        return request(self.base, '/api/capture', timeout=2)

    def capture(self, after_ns, cancel):
        deadline = time.monotonic()+10
        reason='frames_predate_request'
        while not cancel.is_set() and time.monotonic() < deadline:
            try:
                value = self.capture_now()
                if all(f['receipt_ns'] > after_ns for f in value['frames'].values()):
                    return value
            except (OSError, ValueError) as exc:
                reason=str(exc)
            cancel.wait(.1)
        raise ValueError('fresh_capture_unavailable:'+reason)


class Service:
    def __init__(self, backend, camera, root):
        self.backend, self.camera = backend, camera
        self.cancel = threading.Event()
        # Motion cancellation is also used for faults. Only an explicit stop
        # closes HTTP, so clients can retrieve the failed task and its evidence.
        self.shutdown_requested = threading.Event()
        self.lock = threading.RLock()
        self.phase, self.fault = 'READY', None
        self.tasks, self.commands, self.observations, self.artifacts = {}, {}, {}, {}
        self.latest = None
        self.worker = None
        Path(root).mkdir(parents=True,exist_ok=True)
        self.root_lock=(Path(root)/'.service.lock').open('a')
        try:fcntl.flock(self.root_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.root_lock.close();raise RuntimeError('another_agent_service_owns_output_root')
        self.session_id = uuid.uuid4().hex
        self.root = Path(root)/self.session_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.initial_pose = backend.snapshot()['pose_m_xyzw']
        controller = getattr(backend, 'controller', None)
        self.session = dict(schema=SESSION_SCHEMA, session_id=self.session_id,
            speed_mm_s=getattr(controller, 'speed_mm_s', 20.),
            head_speed_deg_s=getattr(controller, 'head_speed_deg_s', 20.),
            control_hz=getattr(controller, 'frequency', 250.),
            reference_point='sdk_end_effector', frame_id='chassis',
            initial_pose_m_xyzw=self.initial_pose, completion_basis='command_interpolation_or_first_gripper_send',
            commands=['move', 'head', 'gripper'], execution_mode='shared_keyboard_executor',
            coordinates=dict(x='chassis +X forward', y='chassis +Y left', z='chassis +Z up'),
            motion_units='mm', head_units='degrees_absolute')
        atomic_json(self.root/'session.json', self.session)
        # Previous processes are never resumed. Mark incomplete journals explicitly.
        for p in Path(root).glob('*/tasks/*.json'):
            row = json.loads(p.read_text())
            if row['status'] not in ('completed', 'failed', 'interrupted', 'rejected'):
                row.update(status='interrupted', error='service_restarted_no_replay')
                atomic_json(p, row)

    def state(self):
        try:measured=self.backend.snapshot()
        except Exception as exc:measured=dict(error=str(exc))
        with self.lock:
            return dict(session_id=self.session_id, phase=self.phase, fault=self.fault,
                        latest_observation_id=self.latest, measured=measured)

    def stop(self):
        self.cancel.set()
        try:self.backend.stop()
        finally:
            with self.lock:self.phase='STOPPED'
            self.shutdown_requested.set()
        return dict(phase='STOPPED', session_id=self.session_id)

    def backend_failed(self):
        self.cancel.set()
        with self.lock:
            if not self.shutdown_requested.is_set():self.phase='FAULT'
            self.fault=self.backend.error

    def close(self):
        try:
            self.stop()
            if self.worker:self.worker.join(15)
            self.backend.close()
        finally:self.root_lock.close()

    def task(self, tid):
        with self.lock:return copy.deepcopy(self.tasks[tid])

    def submit(self, data, observe=False):
        if not isinstance(data, dict):raise ValueError('invalid_arguments')
        common = {'session_id','command_id'}
        kind = 'observe' if observe else data.get('type')
        if observe:
            if set(data) != common:
                raise ValueError('invalid_arguments')
        else:
            if not (common | {'type', 'observation_id'}) <= set(data):
                raise ValueError('invalid_arguments')
            validate_command({k: v for k, v in data.items() if k not in common | {'observation_id'}})
        if data['session_id']!=self.session_id:raise ValueError('session_mismatch')
        cid = data['command_id']
        if not isinstance(cid,str) or not 1<=len(cid)<=128:raise ValueError('invalid_command_id')
        signature = json.dumps(data, sort_keys=True, allow_nan=False)
        with self.lock:
            if cid in self.commands:
                old, tid = self.commands[cid]
                if old!=signature:raise ValueError('command_id_conflict')
                return dict(task_id=tid, command_id=cid, session_id=self.session_id, accepted=True)
            if self.cancel.is_set() or self.backend.error:raise ValueError('controller_stopped')
            if self.phase not in (('READY','NEEDS_OBSERVATION') if observe else ('READY',)):
                raise ValueError('controller_not_ready')
            if not observe:
                if self.latest is None or data['observation_id']!=self.latest:raise ValueError('old_observation')
            tid=uuid.uuid4().hex
            task=dict(task_id=tid,session_id=self.session_id,command_id=cid,request=data,
                status='running',phase='ACCEPTED',motion=dict(action_completed=False,
                returned_to_start=None,contact_evidence=None,button_trigger_evidence=None,speed_mm_s=self.session['speed_mm_s']),
                capture=dict(status='pending'),storage=dict(status='pending'),error=None)
            atomic_json(self.root/'tasks'/f'{tid}.json',task)  # Durable admission before motion.
            self.tasks[tid]=task;self.commands[cid]=(signature,tid)
            self.phase='CAPTURING' if observe else 'MOVING'
            self.worker=threading.Thread(target=self._run,args=(tid,kind,data),daemon=True)
            self.worker.start()
            return dict(task_id=tid,command_id=cid,session_id=self.session_id,accepted=True)

    def _phase(self, task, phase):
        with self.lock:
            if self.cancel.is_set():raise RuntimeError('stopped')
            self.phase=phase;task['phase']=phase

    def _run(self, tid, kind, data):
        task=self.tasks[tid]
        try:
            if kind!='observe':
                task['motion'].update(self.backend.move(kind, data, self.cancel,
                    lambda value: self._phase(task, value)))
                task['motion']['action_completed'] = True
            self._phase(task,'CAPTURING')
            after=time.monotonic_ns()
            bundle=self.camera.capture(after,self.cancel)
            state=(self.backend.state_for_capture(bundle) if hasattr(self.backend,'state_for_capture')
                   else self.backend.snapshot())
            if self.cancel.is_set():raise RuntimeError('stopped')
            self._phase(task,'SAVING')
            obs=self._save_observation(bundle,state,tid)
            task['capture']=dict(status='complete',observation_id=obs['observation_id'])
            task['storage']=dict(status='committed')
            task['observation_url']='/v2/observations/'+obs['observation_id']
            with self.lock:
                if self.cancel.is_set():raise RuntimeError('stopped')
                completed=dict(task,status='completed',phase='READY',next_action_allowed=True)
                atomic_json(self.root/'tasks'/f'{tid}.json',completed)
                task.update(completed)
                self.latest=obs['observation_id'];self.phase='READY';self.fault=None
        except Exception as exc:
            task['error']=self.backend.error or str(exc);task['next_action_allowed']=False
            if hasattr(self.backend,'diagnostic'):task['execution_diagnostics']=self.backend.diagnostic()
            try:task['failure_state']=self.backend.snapshot()
            except Exception as state_error:task['failure_state']=dict(error=str(state_error))
            failed_motion=kind!='observe' and not task['motion']['action_completed']
            if failed_motion or self.backend.error:
                self.cancel.set()
                try:self.backend.stop()
                except Exception:pass
            task['status']='interrupted' if self.cancel.is_set() else 'failed'
            with self.lock:
                self.phase='FAULT' if self.cancel.is_set() else 'NEEDS_OBSERVATION'
                self.fault=str(exc)
            try:atomic_json(self.root/'tasks'/f'{tid}.json',task)
            except OSError:self.cancel.set();self.backend.stop()

    def _save_observation(self, bundle, state, tid):
        import numpy as np
        oid=uuid.uuid4().hex
        parent=self.root/'observations';parent.mkdir(exist_ok=True)
        tmp=parent/('.'+oid);tmp.mkdir()
        artifacts={}
        try:
            files={k+'.jpg':base64.b64decode(bundle['frames'][k]['data_base64'],validate=True) for k in ('head','right')}
            depth=np.frombuffer(base64.b64decode(bundle['frames']['depth']['data_base64'],validate=True),dtype='<u2')
            meta=bundle['metadata'];shape=(meta['depth']['height'],meta['depth']['width'])
            buf=io.BytesIO();np.save(buf,depth.reshape(shape),allow_pickle=False)
            files['right_depth_raw.npy']=buf.getvalue()
            files['depth_metadata.json']=json_bytes(meta)
            files['state.json']=json_bytes(state)
            for name,content in files.items():
                path=tmp/name
                with path.open('wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
                aid=uuid.uuid4().hex
                artifacts[name]=dict(artifact_id=aid,url='/v2/artifacts/'+aid,bytes=len(content),sha256=sha256(content))
            obs=dict(schema='pressb-observation-v2',observation_id=oid,task_id=tid,
                session_id=self.session_id,captured_monotonic_ns=time.monotonic_ns(),utc_ns=time.time_ns(),
                state=state,depth=bundle['depth'],selection=bundle['selection'],
                frames={k:{a:v for a,v in row.items() if a!='data_base64'} for k,row in bundle['frames'].items()},
                artifacts=artifacts,tcp_pose=None,tcp_surface_gap_mm=None)
            atomic_json(tmp/'manifest.json',obs)
            os.replace(tmp,parent/oid)
            with self.lock:
                self.observations[oid]=obs
                for name,row in artifacts.items():self.artifacts[row['artifact_id']]=parent/oid/name
            return obs
        finally:
            if tmp.exists():shutil.rmtree(tmp)


def make_server(service, port=6210):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def setup(self):
            super().setup();self.connection.settimeout(3)
        def reply(self,code,value,binary=False):
            body=value if binary else json_bytes(value)
            self.send_response(code);self.send_header('Content-Type','application/octet-stream' if binary else 'application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                if self.path=='/v2/session':value=service.session
                elif self.path=='/v2/state':value=service.state()
                elif self.path.startswith('/v2/tasks/'):value=service.task(self.path.rsplit('/',1)[1])
                elif self.path.startswith('/v2/observations/'):value=service.observations[self.path.rsplit('/',1)[1]]
                elif self.path.startswith('/v2/artifacts/'):
                    self.reply(200,service.artifacts[self.path.rsplit('/',1)[1]].read_bytes(),True);return
                else:raise KeyError('unknown_endpoint')
                self.reply(200,value)
            except KeyError:self.reply(404,dict(error='not_found'))
            except Exception as exc:self.reply(503,dict(error=str(exc)))
        def do_POST(self):
            try:
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('json_required')
                origin=self.headers.get('Origin')
                if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                    self.reply(403,dict(error='cross_origin_request'));return
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=8192 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_body')
                data=json.loads(self.rfile.read(n))
                if not isinstance(data,dict):raise ValueError('object_required')
                if self.path=='/v2/stop':
                    if data:raise ValueError('unexpected_arguments')
                    self.reply(200,service.stop());return
                if self.path not in ('/v2/actions','/v2/observations'):
                    self.reply(404,dict(error='not_found'));return
                self.reply(202,service.submit(data,self.path.endswith('observations')))
            except (ValueError,TypeError,KeyError) as exc:self.reply(400,dict(error=str(exc)))
            except Exception as exc:self.reply(503,dict(error=str(exc)))
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    # Service shutdown must wait for the /stop handler to finish its response.
    # Otherwise the standalone process exits while the acknowledgement is pending.
    server.daemon_threads=False
    return server
