#!/usr/bin/env python3
"""Synthetic HTTP/NPZ to shadow-control smoke check. No models, ROS, SSH or SDK."""
import http.client
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pressb.openpi_s1b.transforms import Inputs, Outputs, fit_stats
from pressb.s1b_deployment.contract import VERSION, MIXED_TASK
from pressb.s1b_deployment.protocol import encode, decode
from pressb.s1b_deployment.server import make_server
from pressb.s1b_deployment.client import control_exchange
from pressb.s1b_deployment.control_service import ControlSession, make_control_server


def run():
    stats = fit_stats(np.zeros((8,10)), np.zeros((8,10)))
    def predict(head, right, state, seed):
        Inputs(stats)(dict(head=head, right=right, state=state, prompt=MIXED_TASK))
        actions = np.zeros((7,32), np.float32)
        actions[:,7] = np.arange(7) % 2
        return Outputs(stats)({'actions': actions})['actions'], dict(inference_seconds=0., preprocess_seconds=0.)
    # Test-only identity is intentionally unsupported by production backend selection.
    policy = SimpleNamespace(device='test-only', identity=dict(backend='test_only', step=123,
        model_sha256='synthetic-model', data_contract_sha256='synthetic-contract'), predict=predict)
    ticks = []
    core = ControlSession('offline', 'offline-boot', 123, 'synthetic-model', 'synthetic-contract',
                          lambda row: None, ticks.append, execution_strategy='synchronous')
    servers = [make_server(policy,0), make_control_server(core,0)]
    threads = [threading.Thread(target=s.serve_forever, kwargs={'poll_interval':.01}, daemon=True) for s in servers]
    for t in threads: t.start()
    connections = [http.client.HTTPConnection('127.0.0.1', s.server_port, timeout=3) for s in servers]
    def infer(oid):
        observed = time.monotonic_ns()
        f = {k:np.asarray(v) for k,v in dict(protocol_version=VERSION, session_id='offline',
            observation_id=np.int64(oid), observation_monotonic_ns=np.int64(observed), seed=np.int64(42),
            head=np.zeros((720,1280,3), np.uint8), right=np.zeros((360,640,3), np.uint8),
            state10=np.zeros(10,np.float32), source_time_ns=np.full(3,observed,np.int64),
            mapped_time_ns=np.full(3,observed,np.int64), model_sha256='synthetic-model',
            data_contract_sha256='synthetic-contract').items()}
        status, body = control_exchange(connections[0], 'POST', '/predict', encode(f))
        if status != 200: raise RuntimeError(body.decode())
        r = decode(body)
        r.update(robot_boot_id=np.asarray('offline-boot'), execution_strategy=np.asarray('synchronous'),
                 observation_receipt_ns=np.full(5,observed,np.int64))
        return r
    try:
        status, body = control_exchange(connections[1], 'POST', '/start', encode(infer(1)))
        if status != 202: raise RuntimeError(body.decode())
        core.start()
        status, body = control_exchange(connections[1], 'POST', '/prediction', encode(infer(2)))
        if status != 200: raise RuntimeError(body.decode())
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            status, body = control_exchange(connections[1], 'GET', '/health')
            health = json.loads(body)
            if status != 200 or health['fault']: raise RuntimeError(str(health))
            if health['completed_chunk_id'] == 2: break
            time.sleep(.01)
        started = [row for row in ticks if row['event']=='chunk_step_started']
        if [row['step_index'] for row in started] != list(range(7)) or health['completed_chunk_id'] != 2:
            raise RuntimeError('Seven-step shadow execution incomplete')
        return dict(backend='test_only', http_npz_roundtrip=True, shadow_steps=7,
                    sdk_imported=any(n.startswith('astribot_sdk') for n in sys.modules),
                    model_inference_verified=False, robot_connected=False)
    finally:
        core.stop()
        for conn in connections: conn.close()
        for server in servers: server.shutdown(); server.server_close()
        for thread in threads: thread.join(2)
        core.close()


if __name__ == '__main__': print(json.dumps(run(), indent=2))
