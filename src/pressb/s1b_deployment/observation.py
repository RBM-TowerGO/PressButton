"""Bounded read-only stream synchronization; no ROS, SDK or CUDA imports."""
from collections import Counter, deque
from dataclasses import dataclass
import threading
import numpy as np

class ObservationError(ValueError):
    pass

CAMERA_SHAPES = {'head': (720, 1280, 3), 'right': (360, 640, 3)}


@dataclass(frozen=True)
class JpegFrame:
    """Own compressed bytes in callbacks; decode only selected observations.

    Decoding runs outside the buffer lock and ROS spin thread. In particular,
    slow image work must not prevent joint feedback from reaching the buffer.
    """
    data: bytes

    def __post_init__(self):
        if not isinstance(self.data, bytes) or not 0 < len(self.data) <= 8 * 1024**2:
            raise ObservationError('invalid_jpeg_payload')

    def rgb(self, key):
        import cv2
        try:
            bgr = cv2.imdecode(np.frombuffer(self.data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ObservationError('jpeg_decode_failed')
            if bgr.shape != CAMERA_SHAPES[key] or bgr.dtype != np.uint8:
                raise ObservationError('bad_' + key)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except cv2.error as exc:
            raise ObservationError('jpeg_decode_failed') from exc


def selected_rgb(selected):
    """Materialize the two selected images without changing their timestamps."""
    return {key: sample.value.rgb(key) if isinstance(sample.value, JpegFrame) else sample.value
            for key, sample in selected.items() if key in CAMERA_SHAPES}


@dataclass
class Sample:
    sequence: int
    source_ns: int
    mapped_ns: int
    receipt_ns: int
    header: dict
    value: np.ndarray | JpegFrame

class TimeMapper:
    """Map epoch header stamps into local monotonic, retaining proxy limitations."""
    def __init__(self, wall_ns, monotonic_ns, max_jump_ns=5_000_000):
        self.offset = monotonic_ns - wall_ns
        self.max_jump_ns = max_jump_ns
        self.failed = False
    def map(self, source_ns, wall_ns, receipt_ns):
        if self.failed or abs((receipt_ns-wall_ns)-self.offset)>self.max_jump_ns:
            self.failed = True
            raise ObservationError('clock_jump')
        if not -5_000_000 <= wall_ns-source_ns <= 2_000_000_000:
            raise ObservationError('unmapped_source_time')
        return source_ns+self.offset

class ObservationBuffer:
    def __init__(self, capacity=256, head_age_ns=80_000_000, state_skew_ns=10_000_000, right_skew_ns=40_000_000):
        if capacity<2: raise ValueError('capacity must be >=2')
        self.buffers={k:deque(maxlen=capacity) for k in ('head','right','arm','gripper','head_state')}
        self.capacity=capacity
        self.head_age_ns=head_age_ns;self.state_skew_ns=state_skew_ns;self.right_skew_ns=right_skew_ns
        self.counts=Counter();self.rejections=Counter();self.overwrites=Counter();self.high_water=Counter()
        self.lock=threading.Lock()
        self.selection_counts=Counter()
    def add(self, key, sample):
        shape={'head':(720,1280,3),'right':(360,640,3),'arm':(7,),'gripper':(1,),'head_state':(2,)}[key]
        dtype=np.uint8 if key in ('head','right') else np.float32
        if isinstance(sample.value, JpegFrame):
            if key not in CAMERA_SHAPES:raise ObservationError('bad_'+key)
        elif (sample.value.shape!=shape or sample.value.dtype!=dtype
              or (dtype!=np.uint8 and not np.isfinite(sample.value).all())):
            raise ObservationError('bad_'+key)
        with self.lock:
            b=self.buffers[key]
            if b and (sample.source_ns<=b[-1].source_ns or sample.receipt_ns<=b[-1].receipt_ns):
                self.rejections['nonmonotonic_'+key]+=1
                raise ObservationError('nonmonotonic_'+key)
            if len(b)==self.capacity:self.overwrites[key]+=1
            b.append(sample);self.counts[key]+=1;self.high_water[key]=max(self.high_water[key],len(b))
    def snapshot(self, now_ns, after_sequence=-1):
        with self.lock:
            if any(not b for b in self.buffers.values()):raise ObservationError('missing_stream')
            head=self.buffers['head'][-1]
            if head.sequence<=after_sequence:raise ObservationError('no_new_head')
            if not 0<=now_ns-head.receipt_ns<=self.head_age_ns:raise ObservationError('stale_head')
            # Check stream liveness independently of alignment. A fresh latest
            # head may fall inside a state gap; try earlier unused fresh heads
            # instead of starving inference until another camera frame arrives.
            for key in ('right','arm','gripper','head_state'):
                if now_ns-self.buffers[key][-1].receipt_ns>self.head_age_ns:
                    raise ObservationError('stale_'+key)
            newest_reason=None
            for head in reversed(self.buffers['head']):
                if head.sequence<=after_sequence:break
                if not 0<=now_ns-head.receipt_ns<=self.head_age_ns:continue
                selected={'head':head}
                for key in ('right','arm','gripper','head_state'):
                    candidate=min(self.buffers[key],key=lambda s:abs(s.mapped_ns-head.mapped_ns))
                    limit=self.right_skew_ns if key=='right' else self.state_skew_ns
                    if abs(candidate.mapped_ns-head.mapped_ns)>limit:
                        reason='skew_'+key
                        if newest_reason is None:newest_reason=reason
                        self.selection_counts['candidate_'+reason]+=1
                        break
                    selected[key]=candidate
                else:
                    if head.sequence!=self.buffers['head'][-1].sequence:
                        self.selection_counts['aligned_older_head']+=1
                    state=np.concatenate([selected[k].value for k in ('arm','gripper','head_state')]).astype(np.float32)
                    return selected,state
            raise ObservationError(newest_reason or 'no_fresh_aligned_head')
    def stats(self):
        with self.lock:
            return dict(selection_counts=dict(self.selection_counts),counts=dict(self.counts),invalid=dict(self.rejections),overwrites=dict(self.overwrites),high_water=dict(self.high_water),occupancy={k:len(b) for k,b in self.buffers.items()},capacity=self.capacity,latest={k:dict(sequence=b[-1].sequence,mapped_ns=b[-1].mapped_ns,receipt_ns=b[-1].receipt_ns) for k,b in self.buffers.items() if b})
