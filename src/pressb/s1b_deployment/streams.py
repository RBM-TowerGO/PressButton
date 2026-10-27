"""Bounded timestamped JPEG/state transport; no ROS or robot SDK imports."""
from collections import Counter
import json
import struct
import threading
import time
import numpy as np
from .observation import JpegFrame, Sample, ObservationBuffer, ObservationError
from .protocol import encode, decode

STREAM_VERSION = 's1b-stream-v1'
STATE_KEYS = ('arm', 'gripper', 'head_state')
STATE_SIZES = (7, 1, 2)
HISTORY = 128
MAX_PACKET = 8 * 1024**2


def pack(meta, payload=b''):
    header = json.dumps(dict(version=STREAM_VERSION, **meta), allow_nan=False,
                        separators=(',', ':')).encode()
    if len(header) > 16384 or len(payload) + len(header) + 4 > MAX_PACKET:
        raise ValueError('stream_packet_too_large')
    return struct.pack('!I', len(header)) + header + payload


def unpack(packet, service_id, boot_id):
    if not 4 <= len(packet) <= MAX_PACKET:
        raise ValueError('invalid_stream_packet_size')
    n = struct.unpack('!I', packet[:4])[0]
    if not 0 < n <= min(16384, len(packet)-4):
        raise ValueError('invalid_stream_header_size')
    meta = json.loads(packet[4:4+n])
    if (not isinstance(meta, dict) or meta.get('version') != STREAM_VERSION
            or meta.get('service_id') != service_id or meta.get('boot_id') != boot_id):
        raise ValueError('stream_identity_mismatch')
    return meta, packet[4+n:]


def camera_packet(key, sample, service_id, boot_id):
    return pack(dict(stream=key, service_id=service_id, boot_id=boot_id,
        sequence=sample.sequence, source_ns=sample.source_ns, mapped_ns=sample.mapped_ns,
        receipt_ns=sample.receipt_ns, header=sample.header), sample.value.data)


def state_packet(histories, service_id, boot_id):
    fields = {}
    for key in STATE_KEYS:
        samples = list(histories[key])[-HISTORY:]
        if not samples:
            return None
        fields[key+'_times'] = np.asarray([[s.sequence, s.source_ns, s.mapped_ns, s.receipt_ns]
                                          for s in samples], np.int64)
        fields[key+'_position'] = np.stack([s.value for s in samples]).astype(np.float32)
    return pack(dict(stream='states', service_id=service_id, boot_id=boot_id), encode(fields))


class StreamBuffer:
    def __init__(self, mapping, service_id, boot_id):
        self.service_id, self.boot_id = service_id, boot_id
        self.buffer = ObservationBuffer(mapping['buffer_capacity'], mapping['head_age_ns'],
                                        mapping['state_skew_ns'], mapping['right_skew_ns'])
        self.last = Counter()
        self.packets = Counter()
        self.wire_bytes = Counter()

    def ingest(self, channel, packet):
        meta, payload = unpack(packet, self.service_id, self.boot_id)
        if meta.get('stream') != channel:
            raise ValueError('stream_route_mismatch')
        pending = []
        if channel in ('head', 'right'):
            times = [meta.get(k) for k in ('sequence', 'source_ns', 'mapped_ns', 'receipt_ns')]
            if any(type(x) is not int or x < 0 for x in times) or times[0] < 1:
                raise ValueError('invalid_camera_time')
            pending.append((channel, Sample(*times, meta.get('header', {}), JpegFrame(payload))))
        elif channel == 'states':
            f = decode(payload)
            for key, size in zip(STATE_KEYS, STATE_SIZES):
                times, positions = f[key+'_times'], f[key+'_position']
                if (times.dtype != np.int64 or times.ndim != 2 or times.shape[1] != 4
                        or not 1 <= len(times) <= HISTORY or positions.shape != (len(times), size)
                        or positions.dtype != np.float32 or not np.isfinite(positions).all()
                        or (times < 0).any() or (times[:, 0] < 1).any()
                        or (np.diff(times, axis=0) <= 0).any()):
                    raise ValueError('invalid_state_history:'+key)
                for t, q in zip(times, positions):
                    if int(t[0]) > self.last[key]:
                        pending.append((key, Sample(*(int(x) for x in t), {}, q.copy())))
        else:
            raise ValueError('unknown_stream')
        for key, sample in pending:
            if sample.sequence <= self.last[key]:
                raise ValueError('stream_sequence_regressed:'+key)
            self.buffer.add(key, sample)
            self.last[key] = sample.sequence
        self.packets[channel] += 1
        self.wire_bytes[channel] += len(packet)


class LatestPublisher:
    """One socket owner, one replaceable packet per route; callbacks never send."""
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.latest = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.counts = Counter()
        self.thread = threading.Thread(target=self.run, daemon=True, name='s1b-publish')

    def start(self):
        self.thread.start()
        if not self.ready.wait(5):raise RuntimeError('publisher_start_timeout')
        self.check()

    def check(self):
        if self.error:raise RuntimeError('publisher_failed:'+self.error)

    def put(self, route, packet):
        self.check()
        if route not in self.endpoints:raise ValueError('unknown_publish_route')
        with self.lock:
            if route in self.latest:self.counts['replaced_'+route] += 1
            self.latest[route] = packet
        self.wake.set()

    def run(self):
        import zmq
        context = zmq.Context()
        sockets = {}
        try:
            for route, endpoint in self.endpoints.items():
                sock = context.socket(zmq.PUB);sockets[route] = sock
                sock.setsockopt(zmq.SNDHWM, 1)
                sock.setsockopt(zmq.CONFLATE, 1)
                sock.setsockopt(zmq.LINGER, 0)
                sock.bind(endpoint)
            self.ready.set()
            while not self.stop_event.is_set():
                self.wake.wait(.1);self.wake.clear()
                with self.lock:
                    batch, self.latest = self.latest, {}
                for route, packet in batch.items():
                    try:sockets[route].send(packet, zmq.NOBLOCK)
                    except zmq.Again:self.counts['dropped_'+route] += 1
                    else:self.counts['sent_'+route] += 1
        except BaseException as exc:
            self.error = repr(exc)
        finally:
            self.ready.set()
            for sock in sockets.values():sock.close()
            context.term()

    def close(self):
        self.stop_event.set();self.wake.set();self.thread.join(2)
        if self.thread.is_alive():raise RuntimeError('publisher_did_not_stop')
        self.check()


class StreamReceiver:
    def __init__(self, endpoints, mapping, service_id, boot_id):
        self.endpoints = endpoints
        self.store = StreamBuffer(mapping, service_id, boot_id)
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True, name='s1b-receive')

    def start(self):self.thread.start()

    def check(self):
        if self.error:raise RuntimeError('receiver_failed:'+self.error)

    def run(self):
        import zmq
        context = zmq.Context();sockets = {}
        try:
            poller = zmq.Poller()
            for channel, endpoint in self.endpoints.items():
                sock = context.socket(zmq.SUB);sockets[sock] = channel
                sock.setsockopt(zmq.RCVHWM, 1)
                sock.setsockopt(zmq.CONFLATE, 1)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.MAXMSGSIZE, MAX_PACKET)
                sock.setsockopt(zmq.SUBSCRIBE, b'')
                sock.connect(endpoint);poller.register(sock, zmq.POLLIN)
            while not self.stop_event.is_set():
                for sock, _ in poller.poll(50):
                    self.store.ingest(sockets[sock], sock.recv())
        except BaseException as exc:self.error = repr(exc)
        finally:
            for sock in sockets:sock.close()
            context.term()

    def close(self):
        self.stop_event.set();self.thread.join(2)
        if self.thread.is_alive():raise RuntimeError('receiver_did_not_stop')


class RobotClock:
    """Conservative robot-time estimate, calibrated over the control connection.

    Offset upper bound uses local request-send time; network delay can only
    make observations appear older. Actual command deadlines remain robot-local.
    """
    def __init__(self):
        self.upper = self.lower = self.at = self.rtt_ns = None
        self.lock = threading.Lock()

    def update(self, sent, received, robot_ns):
        with self.lock:self._update(sent,received,robot_ns)

    def _update(self, sent, received, robot_ns):
        rtt = received-sent
        if not 0 <= rtt <= 50_000_000:raise ValueError('clock_probe_too_slow')
        lower, upper = robot_ns-received, robot_ns-sent
        if self.at is not None:
            tolerance = 5_000_000 + max(0, received-self.at)//10000
            if lower > self.upper+tolerance or upper < self.lower-tolerance:
                raise ValueError('robot_clock_changed')
        self.lower, self.upper, self.at, self.rtt_ns = lower, upper, received, rtt

    def now(self, local_ns=None):
        with self.lock:return self._now(local_ns)

    def _now(self, local_ns=None):
        local_ns = time.monotonic_ns() if local_ns is None else local_ns
        if self.at is None or not 0 <= local_ns-self.at <= 10_000_000_000:
            raise ValueError('clock_calibration_stale')
        return local_ns+self.upper+(local_ns-self.at)//10000
