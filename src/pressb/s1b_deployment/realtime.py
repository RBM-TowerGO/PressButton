"""Startup transition for fresh online targets; no SDK or ROS dependencies."""
import numpy as np


class StartupBlend:
    """Blend the accepted initial command into freshly selected model targets.

    This is an explicit startup policy, not a replay of the first prediction.
    The caller must validate each target and its original deadline on every tick.
    """
    def __init__(self, initial, duration_seconds):
        self.initial=np.asarray(initial,dtype=np.float64).copy()
        if self.initial.shape!=(10,) or not np.isfinite(self.initial).all():
            raise ValueError('invalid_startup_command')
        if not np.isfinite(duration_seconds) or duration_seconds<0:
            raise ValueError('invalid_startup_duration')
        self.duration_ns=round(duration_seconds*1e9)
        self.started_ns=None
        self.last_ns=None

    def target(self, decoded, now_ns):
        if self.last_ns is not None and now_ns<self.last_ns:
            raise ValueError('nonmonotonic_transition_time')
        self.last_ns=now_ns
        if self.started_ns is None:self.started_ns=now_ns
        u=min(1.,(now_ns-self.started_ns)/self.duration_ns) if self.duration_ns else 1.
        # Quintic blend: zero first/second derivatives at the endpoints for a
        # fixed target. Changing model targets still require the SDK filter.
        alpha=u*u*u*(10+u*(-15+6*u))
        return self.initial+alpha*(np.asarray(decoded)-self.initial),alpha
