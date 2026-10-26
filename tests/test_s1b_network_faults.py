"""Actual local HTTP failure injection against the production protocol/server."""
import http.client
import io
import json
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from pressb.s1b_deployment.server import make_server
from pressb.s1b_deployment.protocol import encode,decode
from pressb.s1b_deployment.contract import VERSION
from pressb.s1b_deployment.scheduler import Scheduler,FakeExecutor
from pressb.s1b_deployment.shadow import ShadowRunner


def fields(oid=1):
    return dict(protocol_version=VERSION,session_id='fault-test',observation_id=np.int64(oid),observation_monotonic_ns=np.int64(1),seed=np.int64(42),head=np.zeros((720,1280,3),np.uint8),right=np.zeros((360,640,3),np.uint8),state10=np.zeros(10,np.float32),source_time_ns=np.zeros(3,np.int64),mapped_time_ns=np.zeros(3,np.int64),model_sha256='model',data_contract_sha256='contract')

@pytest.fixture
def http_policy():
    policy=SimpleNamespace(device='fake',identity=dict(step=2000,model_sha256='model',data_contract_sha256='contract'),predict=lambda *a:(np.zeros((7,10),np.float32),{}))
    server=make_server(policy,0);thread=threading.Thread(target=server.serve_forever);thread.start()
    yield policy,server
    server.shutdown();server.server_close();thread.join()


def post(server,body,headers=None):
    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
    try:
        connection.request('POST','/predict',body,headers or {})
        response=connection.getresponse();return response.status,response.read()
    finally:connection.close()

@pytest.mark.parametrize('kind',['oversize','pickle','shape','dtype','nan','inf','identity','missing'])
def test_malformed_http_never_returns_action(http_policy,kind):
    policy,server=http_policy
    called=[];policy.predict=lambda *a:called.append(1)
    f=fields()
    if kind=='oversize':body=b'';headers={'Content-Length':str(8*1024**2+1)}
    elif kind=='pickle':
        out=io.BytesIO();np.savez(out,x=np.array([{}],object));body=out.getvalue();headers={}
    else:
        if kind=='shape':f['state10']=np.zeros(8,np.float32)
        if kind=='dtype':f['state10']=np.zeros(10,np.float64)
        if kind in ('nan','inf'):f['state10']=np.full(10,float(kind),np.float32)
        if kind=='identity':f['model_sha256']='other'
        if kind=='missing':del f['session_id']
        body=encode(f);headers={}
    status,body=post(server,body,headers)
    assert status==400 and 'error' in json.loads(body) and not called
    assert b'action_raw' not in body


def test_busy_slow_server_and_independent_30hz_consumer(http_policy):
    policy,server=http_policy;entered=threading.Event();release=threading.Event();result=[]
    def predict(*args):
        entered.set();release.wait(2);return np.zeros((7,10),np.float32),{}
    policy.predict=predict
    network=threading.Thread(target=lambda:result.append(post(server,encode(fields()))));network.start();assert entered.wait(1)
    ticks=[];s=Scheduler('fault-test','model','contract',FakeExecutor());runner=ShadowRunner(s,ticks.append);runner.start()
    try:
        status,body=post(server,encode(fields(2)))
        assert status==400 and json.loads(body)['error']=='busy'
        deadline=time.monotonic()+1
        while len(ticks)<6 and time.monotonic()<deadline:time.sleep(.005)
        assert len(ticks)>=6 and network.is_alive()
        assert not s.executor.calls
    finally:release.set();network.join();runner.close()
    assert result[0][0]==200


def test_server_failure_returns_error_not_zero_or_previous_action(http_policy):
    policy,server=http_policy
    def failed(*args):raise RuntimeError('injected_gpu_failure')
    policy.predict=failed
    status,body=post(server,encode(fields()))
    assert status==500 and json.loads(body)=={'error':'inference_failed'}
    assert b'action_raw' not in body

@pytest.mark.parametrize('enabled',[True,False])
def test_tcp_nodelay_configuration_is_applied_to_accepted_socket(enabled):
    import socket
    sockets=[]
    def predict(*args):
        assert bool(sockets[-1].getsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY))==enabled
        return np.zeros((7,10),np.float32),{}
    policy=SimpleNamespace(device='fake',identity=dict(step=2000,model_sha256='model',data_contract_sha256='contract'),predict=predict)
    server=make_server(policy,0,tcp_nodelay=enabled)
    original=server.get_request
    def capture():
        connection,address=original();sockets.append(connection);return connection,address
    server.get_request=capture
    thread=threading.Thread(target=server.serve_forever);thread.start()
    try:assert post(server,encode(fields()))[0]==200
    finally:server.shutdown();server.server_close();thread.join()
