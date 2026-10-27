"""Single-flight HTTP exchanges with no idle socket across SDK initialization."""
from .protocol import MAX_BYTES
import time


def control_exchange(connection, method, path, body=None, headers=None):
    """Reuse the small control connection only within its bounded idle lifetime.

    The control server has a 2 s socket timeout, including during startup.
    Close locally after 0.5 s idle; never retry an ambiguous action submission.
    This avoids creating a new SSH forwarding channel on every prediction.
    """
    now=time.monotonic_ns()
    last=getattr(connection,'_pressb_last_response_ns',None)
    if last is None or now-last>500_000_000:connection.close()
    try:
        connection.request(method,path,body,dict(headers or {}))
        response=connection.getresponse()
        data=response.read(MAX_BYTES+1)
        if len(data)>MAX_BYTES:raise ValueError('response_too_large')
        if response.will_close:connection.close()
        connection._pressb_last_response_ns=time.monotonic_ns()
        return response.status,data
    except BaseException:
        connection.close();connection._pressb_last_response_ns=None
        raise


def exchange(connection, method, path, body=None, headers=None):
    """One request per connection; never retry an ambiguous model request.

    The HTTPConnection object can be reused: close() discards its socket and the
    next request connects afresh. SDK initialization can therefore take longer
    than the server's HTTP keep-alive timeout without retaining a dead socket.
    """
    headers=dict(headers or {})
    headers['Connection']='close'
    try:
        connection.request(method,path,body,headers)
        response=connection.getresponse()
        data=response.read(MAX_BYTES+1)
        if len(data)>MAX_BYTES:
            raise ValueError('response_too_large')
        return response.status,data
    finally:
        connection.close()
