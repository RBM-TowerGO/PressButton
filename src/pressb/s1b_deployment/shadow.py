"""Independent 30 Hz shadow consumer; network calls never run on its thread."""
import threading
import time

class ShadowRunner:
    def __init__(self, scheduler, emit):
        self.scheduler = scheduler
        self.emit = emit
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.started_ns = None
        self.frequency = 250 if getattr(scheduler, "strategy", None)=="synchronous" else 30
        self.skipped_ticks = 0

    def start(self):
        self.started_ns = time.monotonic_ns()
        self.thread = threading.Thread(target=self.run, name='s1b-shadow-30hz', daemon=True)
        self.thread.start()

    def run(self):
        ordinal = 0
        try:
            while not self.stop_event.is_set():
                planned = self.started_ns + round(ordinal * 1e9 / self.frequency)
                remaining = (planned - time.monotonic_ns()) / 1e9
                if remaining > 0 and self.stop_event.wait(remaining):
                    break
                now = time.monotonic_ns()
                row = self.tick(now, planned)
                row['tick_ordinal'] = ordinal
                self.emit(row)
                ordinal += 1
                # Do not replay a burst of historical ticks after a stall.
                earliest = (time.monotonic_ns() - self.started_ns) * self.frequency // 1_000_000_000
                if earliest > ordinal:
                    self.skipped_ticks += earliest - ordinal
                    ordinal = earliest
        except BaseException as exc:
            self.error = repr(exc)
            self.scheduler.fault('shadow_thread_failure')

    def tick(self, now, planned):
        return self.scheduler.tick(now, planned)

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError('shadow_thread_did_not_stop')
        self.scheduler.stop()
