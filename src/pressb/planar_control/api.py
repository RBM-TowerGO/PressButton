"""Loopback-only HTTP tools and a standard-library Python client."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def make_server(controller, port=6210):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def reply(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/state':
                self.reply(200, controller.get_state())
            else:
                self.reply(404, dict(error='unknown_endpoint'))

        def do_POST(self):
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError('unsupported_transfer_encoding')
                n = int(self.headers.get('Content-Length', '-1'))
                if not 0 < n <= 8192:
                    raise ValueError('invalid_body_size')
                body = self.rfile.read(n)
                if len(body) != n:
                    raise ValueError('incomplete_body')
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError('object_required')
                if self.path in ('/observe', '/stop'):
                    if data:
                        raise ValueError('unexpected_arguments')
                    result = controller.observe() if self.path == '/observe' else controller.stop()
                elif self.path in ('/move', '/press'):
                    expected = {'command_id', 'observation_id'}
                    if self.path == '/move':
                        expected |= {'du_mm', 'dv_mm'}
                    if set(data) != expected or not isinstance(data['observation_id'], str):
                        raise ValueError('invalid_tool_arguments')
                    if self.path == '/move' and any(type(data[k]) not in (int, float) for k in ('du_mm','dv_mm')):
                        raise ValueError('numeric_displacement_required')
                    result = controller.submit(self.path[1:], **data)
                else:
                    self.reply(404, dict(error='unknown_endpoint'))
                    return
                self.reply(202 if self.path in ('/move','/press') else 200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, dict(error=str(exc)))
            except Exception as exc:
                controller.stop('tool_server_error:'+str(exc), fault=True)
                self.reply(500, dict(error='tool_server_error'))
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


class PlanarTools:
    """Use an SSH tunnel for remote callers; all motion units are millimetres."""
    def __init__(self, base_url='http://127.0.0.1:6210', timeout=5):
        self.base_url, self.timeout = base_url.rstrip('/'), timeout

    def _call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        req = Request(self.base_url+path, data=data, headers={'Content-Type':'application/json'})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise ValueError(json.load(exc).get('error', 'http_error')) from exc

    def observe(self):
        return self._call('/observe', {})

    def move_in_plane(self, du_mm, dv_mm, observation_id, command_id):
        return self._call('/move', dict(du_mm=du_mm, dv_mm=dv_mm,
                         observation_id=observation_id, command_id=command_id))

    def press(self, observation_id, command_id):
        return self._call('/press', dict(observation_id=observation_id, command_id=command_id))

    def get_state(self):
        return self._call('/state')

    def stop(self):
        return self._call('/stop', {})
