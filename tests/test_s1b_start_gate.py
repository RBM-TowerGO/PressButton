"""Exercise the start gate with a real pseudo-terminal, without SDK or SSH."""
import importlib.util
import io
import os
from pathlib import Path
import pty
import termios
import threading
import pytest

spec=importlib.util.spec_from_file_location('s1b_start_gate',Path(__file__).resolve().parents[1]/'scripts/start_s1b_inference.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


@pytest.mark.parametrize('key',[b'S',b's'])
def test_waits_for_fresh_s_without_enter_and_restores_terminal(key):
    master,slave=pty.openpty()
    stream=os.fdopen(slave,'r');previous=termios.tcgetattr(slave)
    ready=threading.Event();finished=threading.Event();errors=[]
    def run():
        try:launcher.wait_for_start(ready.set,stream)
        except BaseException as exc:errors.append(exc)
        finally:finished.set()
    # A key typed while the model is loading must not start it automatically.
    os.write(master,b'S\n')
    thread=threading.Thread(target=run);thread.start()
    try:
        assert ready.wait(1)
        assert not finished.wait(.05)
        os.write(master,b'x\n')
        assert not finished.wait(.05)
        os.write(master,key)
        assert finished.wait(1)
        assert not errors
        assert termios.tcgetattr(slave)==previous
    finally:
        os.write(master,b'S')
        thread.join(1);stream.close();os.close(master)


@pytest.mark.parametrize('error',[KeyboardInterrupt(),RuntimeError('server exited')])
def test_cancellation_or_service_exit_restores_terminal(error):
    master,slave=pty.openpty();stream=os.fdopen(slave,'r')
    previous=termios.tcgetattr(slave)
    def check():raise error
    try:
        with pytest.raises(type(error)):launcher.wait_for_start(check,stream)
        assert termios.tcgetattr(slave)==previous
    finally:stream.close();os.close(master)


def test_piped_s_is_not_an_interactive_start():
    with pytest.raises(RuntimeError,match='交互终端'):
        launcher.wait_for_start(lambda:None,io.StringIO('S\n'))


def test_failure_reports_fault_before_large_summary_and_warnings(tmp_path):
    import json
    log=tmp_path/'bridge.log'
    log.write_text('WARNING: frequency mismatch\n'*30+json.dumps(dict(
        event='finished',execution_fault='queue_exhausted',
        fatal=['ExecutionError: stop_did_not_settle_within_budget'],
        telemetry='x'*10000))+'\nTraceback: queue_exhausted\n')
    error=launcher.bridge_failure(log)
    assert error.splitlines()[1]=='execution_fault=queue_exhausted'
    assert 'stop_did_not_settle_within_budget' in error[:200]
    assert len(error)<1500


def test_port_probe_allows_recently_closed_connection_but_rejects_listener():
    import socket
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0));listener.listen(1)
        port=listener.getsockname()[1]
        with pytest.raises(OSError):launcher.check_port_available(port)
        with socket.create_connection(('127.0.0.1',port)) as client:
            connection,_=listener.accept()
            connection.close()
            assert client.recv(1)==b''
    launcher.check_port_available(port)
