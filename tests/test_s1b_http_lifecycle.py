"""Real sockets reproduce SDK startup exceeding server keep-alive lifetime."""
import http.client
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from pressb.s1b_deployment.client import exchange,control_exchange


@pytest.fixture
def server():
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup();self.connection.settimeout(.05)
        def log_message(self,*args):pass
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200);self.send_header('Content-Length','2');self.end_headers();self.wfile.write(b'{}')
        def do_POST(self):
            requests.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            if self.path=='/disconnect':
                self.close_connection=True;return
            self.send_response(200);self.send_header('Content-Length','2');self.end_headers();self.wfile.write(b'{}')
    s=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
    yield s.server_port,requests
    s.shutdown();s.server_close();t.join(2)


def test_old_keepalive_connection_reproduces_reported_disconnect(server):
    port,requests=server;c=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
    try:
        c.request('GET','/health');assert c.getresponse().read()==b'{}'
        time.sleep(.15)  # Represents SDK initialization longer than idle timeout.
        with pytest.raises((http.client.RemoteDisconnected,ConnectionError)):
            c.request('POST','/predict',b'payload');c.getresponse()
        assert requests==['/health']
    finally:c.close()


def test_health_warmup_sdk_delay_and_predictions_use_fresh_connections(server):
    port,requests=server;c=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
    assert exchange(c,'GET','/health')==(200,b'{}')
    assert c.sock is None
    assert exchange(c,'POST','/predict',b'warmup')==(200,b'{}')
    time.sleep(.15)
    for _ in range(3):
        assert exchange(c,'POST','/predict',b'fresh')==(200,b'{}')
        assert c.sock is None
    assert requests==['/health']+['/predict']*4


def test_real_disconnect_closes_socket_without_retry(server):
    port,requests=server;c=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
    with pytest.raises(http.client.RemoteDisconnected):exchange(c,'POST','/disconnect',b'payload')
    assert requests==['/disconnect'] and c.sock is None


def test_control_connection_reused_and_expired_idle_connection_reopened(server):
    port,requests=server;c=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
    try:
        assert control_exchange(c,'GET','/health')==(200,b'{}')
        sock=c.sock
        assert control_exchange(c,'POST','/predict',b'fresh')==(200,b'{}')
        assert c.sock is sock
        time.sleep(.6)  # Beyond client idle lifetime and server idle timeout.
        assert control_exchange(c,'POST','/predict',b'new')==(200,b'{}')
        assert c.sock is not sock
        assert requests==['/health','/predict','/predict']
    finally:c.close()


def test_control_disconnect_never_retries_action(server):
    port,requests=server;c=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
    with pytest.raises(http.client.RemoteDisconnected):control_exchange(c,'POST','/disconnect',b'payload')
    assert requests==['/disconnect'] and c.sock is None
