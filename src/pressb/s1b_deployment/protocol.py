"""Bounded NPZ wire format: no pickle, fixed scalar strings and numeric arrays."""
import io
import zipfile
import numpy as np
from .contract import VERSION, OFFSETS
MAX_BYTES = 8 * 1024**2

class ProtocolError(ValueError): pass

def encode(fields):
    if any(np.asarray(value).dtype.hasobject for value in fields.values()): raise ProtocolError('object_arrays_forbidden')
    out = io.BytesIO()
    np.savez(out, **fields)
    if out.tell() > MAX_BYTES: raise ProtocolError('payload_too_large')
    return out.getvalue()

def decode(data):
    if len(data) > MAX_BYTES: raise ProtocolError('payload_too_large')
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(x.file_size for x in z.infolist()) > MAX_BYTES or len(z.infolist()) > 40: raise ProtocolError('expanded_payload_too_large')
            if len(z.namelist()) != len(set(z.namelist())): raise ProtocolError('duplicate_fields')
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (ValueError, OSError, zipfile.BadZipFile) as exc: raise ProtocolError('invalid_npz') from exc

def scalar(fields, key, kind):
    if key not in fields or fields[key].shape != () or fields[key].dtype.kind not in kind: raise ProtocolError('invalid_' + key)
    return fields[key].item()

def validate_request(f):
    if scalar(f, 'protocol_version', 'US') != VERSION: raise ProtocolError('protocol_mismatch')
    if not scalar(f, 'session_id', 'US'): raise ProtocolError('empty_session')
    for k in ('observation_id', 'observation_monotonic_ns', 'seed'):
        if scalar(f, k, 'iu') < 0: raise ProtocolError('negative_' + k)
    if int(f['seed']) >= 2**32: raise ProtocolError('invalid_seed')
    for k, shape, dtype in [('head', (720,1280,3), np.uint8), ('right',(360,640,3),np.uint8), ('state10',(10,),np.float32)]:
        if k not in f or f[k].shape != shape or f[k].dtype != dtype or not np.isfinite(f[k]).all(): raise ProtocolError('invalid_' + k)
    for k in ('source_time_ns', 'mapped_time_ns'):
        if k not in f or f[k].shape != (3,) or f[k].dtype != np.int64: raise ProtocolError('invalid_' + k)
    for k in ('model_sha256', 'data_contract_sha256'): scalar(f,k,'US')

class Application:
    def __init__(self, policy, recorder=None):
        import threading
        self.policy = policy
        self.recorder = recorder
        self.lock = threading.Lock()
        self.last = {}
    def health(self):
        return dict(ready=True, protocol_version=VERSION, action_dim=10, horizon=7, device=str(self.policy.device), **{k:v for k,v in self.policy.identity.items() if k != 'normalization'})
    def predict(self, f):
        validate_request(f)
        for k in ('model_sha256','data_contract_sha256'):
            if str(f[k]) != self.policy.identity[k]: raise ProtocolError('identity_mismatch')
        if not self.lock.acquire(blocking=False): raise ProtocolError('busy')
        try:
            session, oid = str(f['session_id']), int(f['observation_id'])
            if oid <= self.last.get(session,-1): raise ProtocolError('observation_out_of_order')
            if session not in self.last and len(self.last) >= 1024: raise ProtocolError('session_limit')
            self.last[session] = oid
            action, timing = self.policy.predict(f['head'],f['right'],f['state10'],int(f['seed']))
            if self.recorder is not None:
                self.recorder.append('prediction', session_id=session, observation_id=oid, seed=int(f['seed']), model_sha256=self.policy.identity['model_sha256'], data_contract_sha256=self.policy.identity['data_contract_sha256'], source_time_ns=f['source_time_ns'].tolist(), mapped_time_ns=f['mapped_time_ns'].tolist(), observation_monotonic_ns=int(f['observation_monotonic_ns']), state10=f['state10'].tolist(), action_raw=action.tolist(), gripper_binary=(action[:,7]>=.5).astype(int).tolist(), sdk_target=None, fault=None, **timing)
            response = {k:f[k] for k in ('protocol_version','session_id','observation_id','observation_monotonic_ns','model_sha256','data_contract_sha256')}
            return dict(response, action_raw=action, target_offsets_ns=OFFSETS, **timing)
        finally: self.lock.release()
