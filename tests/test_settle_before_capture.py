import threading
import time
from collections import deque

from pressb.agent_loop import sdk_backend
from pressb.agent_loop.sdk_backend import SDKBackend
from pressb.agent_loop.service import Service

from test_agent_loop import FakeBackend, FakeCamera, observe, action, wait


class Sampler:
    """SDKBackend.wait_settled on a synthetic state stream; no SDK."""
    def __init__(self, moving_until_s=0.0, speed_m_s=0.02):
        self.lock = threading.RLock()
        self.history = deque(maxlen=6000)
        self.start = time.monotonic_ns()
        self.done = threading.Event()
        def run():
            while not self.done.is_set():
                now = time.monotonic_ns()
                t = (now-self.start)/1e9
                x = 0.4 + speed_m_s*min(t, moving_until_s)
                with self.lock:
                    self.history.append(dict(receipt_ns=now, pose_m_xyzw=[x, -.2, 1., 0, 0, 0, 1], head_deg=[0., 30.]))
                self.done.wait(.01)
        self.thread = threading.Thread(target=run, daemon=True); self.thread.start()
        time.sleep(.05)
    wait_settled = SDKBackend.wait_settled
    def close(self):
        self.done.set(); self.thread.join(1)


def test_waits_for_rest_then_reports_settled():
    s = Sampler(moving_until_s=0.4)
    try:
        result = s.wait_settled(threading.Event(), timeout_s=2.0)
    finally:
        s.close()
    assert result['settled'] and result['reason'] == 'settled'
    assert 300 <= result['waited_ms'] < 1500


def test_still_moving_times_out_without_raising():
    s = Sampler(moving_until_s=10.0)
    try:
        result = s.wait_settled(threading.Event(), timeout_s=0.3)
    finally:
        s.close()
    assert not result['settled'] and result['reason'] == 'timeout' and result['waited_ms'] < 600


def test_stop_ends_the_wait_immediately():
    s = Sampler(moving_until_s=10.0); cancel = threading.Event(); cancel.set()
    try:
        result = s.wait_settled(cancel, timeout_s=5.0)
    finally:
        s.close()
    assert result['reason'] == 'stopped' and result['waited_ms'] < 200


def test_service_photographs_even_if_settling_fails_or_times_out(tmp_path):
    class Unsettled(FakeBackend):
        def __init__(self, mode):
            super().__init__(); self.mode = mode
        def wait_settled(self, cancel):
            if self.mode == 'raise':
                raise RuntimeError('sdk glitch')
            return dict(settled=False, reason='timeout', waited_ms=1500.0, timeout_ms=1500.0)
    for mode in ('timeout', 'raise'):
        service = Service(Unsettled(mode), FakeCamera(), tmp_path/mode)
        try:
            obs = observe(service)
            task = wait(service, service.submit(action(service, obs, 'm', type='move', xyz_mm=[-10, 0, 0])))
            assert task['status'] == 'completed' and task['capture']['status'] == 'complete'
            assert task['settle']['settled'] is False
            assert ('settle_check_failed' in task['settle']['reason']) == (mode == 'raise')
        finally:
            service.close()


def test_defaults_are_soft_and_short():
    assert sdk_backend.SETTLE_TIMEOUT_S <= 2.0
