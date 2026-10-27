"""Bounded evidence writer: servo and ROS threads never wait for disk writes."""
import json
import queue
import threading
import time
from collections import Counter


class AsyncLog:
    def __init__(self, streams, byte_limit, capacity=4096):
        self.streams = streams
        self.byte_limit = byte_limit
        self.bytes_written = 0
        self.pending = queue.Queue(maxsize=capacity)
        self.error = None
        self.latest = {}
        self.telemetry_counts = Counter()
        self.coalesced = 0
        self.closing = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name='s1b-log-writer', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise OSError('log_writer_failed: ' + str(self.error)) from self.error

    def _put(self, item):
        with self.lock:
            self.check()
            if self.closing.is_set():
                raise OSError('log_writer_closed')
            try:
                self.pending.put_nowait(item)
            except queue.Full as exc:
                self.error = OSError('log_queue_full')
                raise self.error from exc

    def log(self, name, row):
        # Caller transfers ownership of nested values; it must not mutate them.
        event = row.get('event')
        telemetry = ((name == 'sources' and row.get('valid') is True)
                     or (name == 'events' and event in ('state', 'accepted_command', 'command_topic'))
                     or (name == 'events' and event == 'driver_status' and not row.get('error_codes')))
        if not telemetry:
            self._put(('json', name, dict(row)))
            return
        key = (name, event or row.get('stream'), row.get('part'))
        with self.lock:
            self.check()
            if self.closing.is_set():
                raise OSError('log_writer_closed')
            self.telemetry_counts[key] += 1
            if key in self.latest:
                self.coalesced += 1
            self.latest[key] = (name, dict(row))

    def archive(self, path, data):
        self._put(('archive', path, data))

    def _write(self, name, value):
        data = json.dumps(value, allow_nan=False) + '\n'
        size = len(data.encode('utf-8'))
        if self.bytes_written + size > self.byte_limit:
            raise OSError('log_disk_limit')
        self.streams[name].write(data)
        self.bytes_written += size

    def stats(self):
        with self.lock:
            return dict(telemetry_received=sum(self.telemetry_counts.values()),
                        telemetry_coalesced=self.coalesced, latest_slots=len(self.latest),
                        pending_critical=self.pending.qsize(), error=str(self.error) if self.error else None)

    def _flush_latest(self):
        with self.lock:
            batch = self.latest
            self.latest = {}
            counts = dict(self.telemetry_counts)
        for key, (name, row) in batch.items():
            row['telemetry_received_count'] = counts[key]
            self._write(name, row)

    def _run(self):
        next_flush = time.monotonic() + .1
        try:
            while not self.closing.is_set() or not self.pending.empty():
                try:
                    kind, destination, value = self.pending.get(timeout=.01)
                except queue.Empty:
                    kind = None
                if kind == 'archive':
                    destination.write_bytes(value)
                elif kind == 'json':
                    self._write(destination, value)
                if time.monotonic() >= next_flush:
                    self._flush_latest()
                    for stream in self.streams.values():
                        stream.flush()
                    next_flush = time.monotonic() + .1
            self._flush_latest()
            for stream in self.streams.values():
                stream.flush()
        except BaseException as exc:
            self.error = exc

    def close(self):
        with self.lock:
            self.closing.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise OSError('log_writer_did_not_stop')
        try:
            self.check()
        finally:
            for stream in self.streams.values():
                stream.close()
