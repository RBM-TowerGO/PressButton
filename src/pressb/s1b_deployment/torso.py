"""Explicit torso alignment, separate from the immutable joint10 model contract."""
import time
import numpy as np
from .executor import ExecutionError

TORSO = 'astribot_torso'



def validate_config(c):
    if not isinstance(c, dict) or type(c.get('enabled')) is not bool:
        raise ExecutionError('invalid_torso_config')
    if not c['enabled']:return
    q=np.asarray(c['target'],float)
    if q.shape!=(4,) or not np.isfinite(q).all():raise ExecutionError('invalid_torso_target')
    for key in ('initial_tracking_tolerance','minimum_duration_seconds',
                'max_velocity','max_acceleration','timeout_seconds','position_tolerance',
                'velocity_tolerance','stable_seconds','tracking_tolerance','tracking_seconds'):
        if not np.isfinite(c[key]) or c[key]<=0:raise ExecutionError('invalid_torso_config:'+key)
    if c['minimum_duration_seconds']+c['stable_seconds']>=c['timeout_seconds']:
        raise ExecutionError('invalid_torso_timeout')
    source=c['source']
    if (source['frame']!=0 or source['field']!='joints_dict/joints_position_command[0,3:7]'
            or len(source['sha256'])!=64 or any(x not in '0123456789abcdef' for x in source['sha256'])):
        raise ExecutionError('invalid_torso_source')


def duration(start, target, velocity, acceleration, minimum):
    distance=np.abs(np.asarray(target)-start)
    return max(minimum,float(np.max(1.875*distance/velocity)),
               float(np.sqrt(np.max((10/np.sqrt(3))*distance/acceleration))))


def interpolate(start,target,u):
    u=float(np.clip(u,0,1));alpha=10*u**3-15*u**4+6*u**5
    return np.asarray(start)+(np.asarray(target)-start)*alpha


class TorsoAlignment:
    def __init__(self, config, live):
        self.c=config;self.live=live;self.target=np.asarray(config['target'],float)
        self.ready=False;self.bad_since=None;self.progress=None
        self.lower=self.upper=self.velocity=None

    def state(self, snapshot, now):
        row=snapshot[TORSO];q=np.asarray(row['position'],float);v=np.asarray(row['velocity'],float)
        if q.shape!=(4,) or v.shape!=(4,) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ExecutionError('invalid_torso_feedback')
        if not 0<=now-row['receipt_ns']<=self.live.c['state_max_age_ns']:
            raise ExecutionError('stale_torso_feedback')
        if self.lower is not None and ((q<self.lower).any() or (q>self.upper).any()):
            raise ExecutionError('torso_feedback_limit')
        return q,v

    def prepare(self):
        live=self.live
        lower,upper,velocity=live.backend.torso_limits()
        values=[np.asarray(x,float) for x in (lower,upper,velocity)]
        if any(x.shape!=(4,) or not np.isfinite(x).all() for x in values):
            raise ExecutionError('invalid_torso_sdk_limits')
        self.lower,self.upper,self.velocity=values
        if (self.lower>=self.upper).any() or (self.velocity<=0).any():raise ExecutionError('invalid_torso_sdk_limits')
        if ((self.target<self.lower).any() or (self.target>self.upper).any()):raise ExecutionError('torso_target_limit')
        self.start=np.asarray(live.monitor.torso_command(),float)
        q,_=self.state(live.monitor.snapshot(),time.monotonic_ns())
        if (self.start.shape!=(4,) or not np.isfinite(self.start).all()
                or (self.start<self.lower).any() or (self.start>self.upper).any()
                or np.max(abs(self.start-q))>self.c['initial_tracking_tolerance']):
            raise ExecutionError('invalid_initial_torso_command')
        self.seconds=duration(self.start,self.target,np.minimum(self.velocity,self.c['max_velocity']),
                              self.c['max_acceleration'],self.c['minimum_duration_seconds'])
        if self.seconds+self.c['stable_seconds']>=self.c['timeout_seconds']:raise ExecutionError('torso_transition_exceeds_budget')
        # Skip only if accepted command is already the reference; otherwise even
        # a small difference needs a smooth transition, not a target jump.
        self.skip=bool(np.max(abs(q-self.target))<=self.c['position_tolerance'] and
                       np.max(abs(self.start-self.target))<1e-6)
        live.backend.torso_target=self.start.copy()
        live.emit(dict(event='torso_reference',reference=self.target.tolist(),source=self.c['source'],
                       initial_command=self.start.tolist(),initial_measured=q.tolist(),duration_seconds=self.seconds,
                       sdk_lower=self.lower.tolist(),sdk_upper=self.upper.tolist(),sdk_velocity=self.velocity.tolist()))

    def check(self,snapshot,now):
        q,v=self.state(snapshot,now)
        if self.ready:
            if np.max(abs(q-self.target))>self.c['tracking_tolerance']:
                if self.bad_since is None:self.bad_since=now
                if now-self.bad_since>=self.c['tracking_seconds']*1e9:raise ExecutionError('torso_tracking_error')
            else:self.bad_since=None
        return q,v

    def run(self,phase):
        live=self.live;started=time.monotonic();stable=None;last_log=-1.;next_tick=started
        while True:
            if live.executor.stop_event.is_set():raise ExecutionError('torso_alignment_cancelled')
            now=time.monotonic();elapsed=now-started
            if elapsed>=self.c['timeout_seconds']:raise ExecutionError('torso_alignment_timeout')
            u=1. if self.skip else min(1.,elapsed/self.seconds)
            stage='SETTLING_TORSO' if u>=1 else 'ALIGNING_TORSO'
            phase(stage)
            live.backend.torso_target=interpolate(self.start,self.target,u)
            live.hold()
            q,v=self.state(live.monitor.snapshot(),time.monotonic_ns())
            error=float(np.max(abs(q-self.target)))
            if u>=1 and error<=self.c['position_tolerance'] and np.max(abs(v))<=self.c['velocity_tolerance']:
                if stable is None:stable=now
                if now-stable>=self.c['stable_seconds']:break
            else:stable=None
            if elapsed-last_log>=.5:
                self.progress=dict(event='torso_progress',phase=stage,progress=u,elapsed_seconds=elapsed,
                                   max_error_rad=error,measured=q.tolist(),target=self.target.tolist())
                live.emit(self.progress.copy());last_log=elapsed
            next_tick=max(next_tick+.004,time.monotonic())
            live.executor.stop_event.wait(max(0,next_tick-time.monotonic()))
        self.ready=True;live.backend.torso_target=self.target.copy()
        live.emit(dict(event='torso_aligned',elapsed_seconds=time.monotonic()-started,
                       target=self.target.tolist(),measured=q.tolist(),max_error_rad=error))
