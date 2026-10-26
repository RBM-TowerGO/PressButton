"""Loopback-only HTTP inference, with explicit errors and no robot dependencies."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from .protocol import Application, ProtocolError, MAX_BYTES, encode, decode

def make_server(policy, port=8796, recorder=None, tcp_nodelay=True):
    app = Application(policy, recorder)
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        disable_nagle_algorithm = tcp_nodelay
        def log_message(self, *args): pass
        def setup(self):
            super().setup()
            self.connection.settimeout(5)
        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path == '/health': self.send(200,json.dumps(app.health()).encode(),'application/json')
            else: self.send(404,b'{}','application/json')
        def do_POST(self):
            try:
                if self.path != '/predict': raise ProtocolError('unknown_endpoint')
                if self.headers.get('Transfer-Encoding'): raise ProtocolError('unsupported_transfer_encoding')
                n = int(self.headers.get('Content-Length','-1'))
                if not 0 < n <= MAX_BYTES: raise ProtocolError('invalid_content_length')
                body = self.rfile.read(n)
                if len(body) != n: raise ProtocolError('incomplete_body')
                self.send(200,encode(app.predict(decode(body))),'application/octet-stream')
            except (ValueError, OSError) as exc:
                self.close_connection = True
                self.send(400,json.dumps({'error':str(exc)}).encode(),'application/json')
            except Exception:
                self.close_connection = True
                self.send(500,b'{"error":"inference_failed"}','application/json')
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)
