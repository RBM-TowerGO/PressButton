"""CPU behavioral tests; no robot, ROS or SDK required."""
import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from pressb.planar_control.geometry import Plane, Planner, SDKKinematics, pose_list
from pressb.planar_control.sensing import depth_distance
from pressb.planar_control.controller import Controller
from pressb.planar_control.runtime import validate_config
from pressb.planar_control.api import PlanarTools, make_server


@pytest.fixture
def config():
    c = json.loads(Path('configs/s1b_planar_v1.json').read_text())
    c['plane']['origin_m'] = [0,0,0]
    c['control']['joint_lower'] = [-2]*7
    c['control']['joint_upper'] = [2]*7
    c['control']['joint_velocity_rad_s'] = [1]*7
    c['control']['frequency_hz'] = 100
    c['control']['move_speed_m_s'] = .05
    c['control']['settle_duration_s'] = .01
    c['control']['max_servo_lateness_s'] = .5
    return c


class CartesianKin:
    def __init__(self, plane):
        self.plane = plane
        self.jump = 0
        self.bad_fk = False
        self.distance = 1
        self.gate = None

    def solve(self, target, seed):
        if self.gate:
            self.gate.wait(2)
        return np.r_[target[:3,3], self.jump, 0,0,0]

    def fk(self, q):
        t = np.eye(4)
        t[:3,3] = q[:3]
        if self.bad_fk:
            t[0,3] += .1
        return t

    def clearance(self, q):
        return self.distance


def planner_for(c):
    plane = Plane(c['plane'])
    return Planner(plane, CartesianKin(plane), c['control'])


def test_plane_formula_and_inverse(config):
    p = Plane(config['plane'])
    t = p.target([.01,.015], .03)
    np.testing.assert_allclose(t[:3,3], [.03,-.01,.015])
    np.testing.assert_allclose(p.coordinates(t), [.01,.015,.03])
    with pytest.raises(ValueError, match='workspace'):
        p.target([.1,0])
    config['plane']['forward'] = [0,0,1]
    with pytest.raises(ValueError, match='orthonormal'):
        Plane(config['plane'])


def test_full_path_stays_in_plane_and_press_normal(config):
    p = planner_for(config)
    points = p.path([0,0,0], [.01,.01,0], np.zeros(7), .02)
    assert len(points)>2
    for pt in points:
        assert p.plane.coordinates(p.kin.fk(pt.q))[2] == pytest.approx(0)
    press = p.path([.01,.01,0], [.01,.01,.02], points[-1].q, .02)
    for pt in press:
        np.testing.assert_allclose(p.plane.coordinates(pt.target)[:2],[.01,.01])
    assert press[-1].target[0,3] == pytest.approx(.02)


@pytest.mark.parametrize('case,error', [('jump','branch_jump'),('fk','tracking_error'),('collision','clearance'),('velocity','velocity_limit'),('bounds','joint_limit')])
def test_path_rejects_invalid_ik(config, case, error):
    p = planner_for(config)
    if case=='jump':p.kin.jump=.1
    if case=='fk':p.kin.bad_fk=True
    if case=='collision':p.kin.distance=0
    if case=='velocity':p.c['joint_velocity_rad_s']=[1e-8]*7
    if case=='bounds':p.c['joint_lower']=[.1]*7
    with pytest.raises(ValueError, match=error):
        p.path([0,0,0],[.01,0,0],np.zeros(7),.01)


def test_sdk_tcp_conversion_and_torso_rejection():
    offset=np.eye(4);offset[2,3]=.1
    captured={}
    def ik(names, poses):
        captured['poses']=poses
        return True,dict(astribot_torso=[0]*4,astribot_arm_right=[0]*7)
    client=SimpleNamespace(get_forward_kinematics=lambda *a:dict(astribot_torso=[0,0,0,0,0,0,1],astribot_arm_right=[0,0,0,0,0,0,1]),get_inverse_kinematics=ik)
    k=SDKKinematics(client,[0]*4,[0]*7,offset)
    np.testing.assert_allclose(k.fk([0]*7)[:3,3],[0,0,.1])
    k.solve(offset,[0]*7)
    np.testing.assert_allclose(captured['poses'][1][:3],[0,0,0])
    client.get_inverse_kinematics=lambda *a:(True,dict(astribot_torso=[.1]*4,astribot_arm_right=[0]*7))
    with pytest.raises(ValueError,match='torso'):
        k.solve(offset,[0]*7)
    client.get_inverse_kinematics=lambda *a:(False,None)
    with pytest.raises(ValueError,match='ik_failed'):
        k.solve(offset,[0]*7)


def test_depth_geometry(config):
    c=config['depth'];c['ray_radius_m']=.04
    depth=np.full((60,80),.2)
    k=np.array([[200,0,40],[0,200,30],[0,0,1]])
    result=depth_distance(depth,k,np.eye(4),[0,0,.15],[0,0,1],c)
    assert result['gap_m']==pytest.approx(.05)
    assert result['tilt_rad']<1e-5
    depth[:10]=np.nan
    result=depth_distance(depth,k,np.eye(4),[0,0,.15],[0,0,1],c)
    assert result['trusted']
    with pytest.raises(ValueError,match='support'):
        depth_distance(depth,k,np.eye(4),[0,0,.15],[0,0,1],c,np.ones(depth.shape,bool))
    tilted=.2/(1-.2*(np.indices(depth.shape)[1]-40)/200)
    with pytest.raises(ValueError,match='tilt|ambiguous'):
        depth_distance(tilted,k,np.eye(4),[0,0,.15],[0,0,1],c)


class FakeBackend:
    def __init__(self, planner):
        self.p=planner;self.q=np.zeros(7);self.sent=[];self.stopped=False
        self.failure=None;self.range_trusted=True;self.gap=.003
        self.lock=threading.Lock();self.image_stamp=None
    def snapshot(self):
        if self.failure:raise ValueError(self.failure)
        return dict(q=self.q.copy(),tcp=self.p.kin.fk(self.q),velocity=np.zeros(7),torque=np.zeros(7),receipt_ns=time.monotonic_ns())
    def send(self,q,state):
        with self.lock:
            if self.stopped:raise ValueError('stopped')
            self.q=q.copy();self.sent.append(q.copy())
    def observe(self):
        now=self.image_stamp or time.monotonic_ns()
        return dict(images={k:dict(receipt_ns=now,source_ns=time.time_ns(),jpeg_base64='') for k in ('head','right')},range=dict(trusted=self.range_trusted))
    def range(self):return dict(gap_m=self.gap,trusted=self.range_trusted)
    def stop(self,reason,fault):
        with self.lock:self.stopped=True
    def close(self):pass


@pytest.fixture
def live(config):
    p=planner_for(config);b=FakeBackend(p)
    controller=Controller(p,b,config['control'])
    yield controller,b
    controller.close()


def wait_for(predicate):
    until=time.monotonic()+3
    while not predicate():
        assert time.monotonic()<until,'timeout'
        time.sleep(.005)


def test_move_idempotency_and_new_observation(live):
    c,b=live
    oid=c.observe()['observation_id']
    receipt=c.submit('move',oid,'one',du_mm=2,dv_mm=1)
    assert c.submit('move',oid,'one',du_mm=2,dv_mm=1)==receipt
    with pytest.raises(ValueError,match='conflict'):
        c.submit('move',oid,'one',du_mm=3)
    wait_for(lambda:c.get_state()['phase']=='READY')
    np.testing.assert_allclose(c.get_state()['completed_uv_m'],[.002,.001])
    with pytest.raises(ValueError,match='observation'):
        c.submit('move',oid,'two',du_mm=1)
    assert c.observe()['observation_id']!=oid


def test_press_retracts_and_never_claims_trigger(live):
    c,b=live;c.c['press']['verified']=True
    c.c['press'].update(approach_speed_m_s=.05,contact_speed_m_s=.05,retract_speed_m_s=.05,dwell_s=.01)
    oid=c.observe()['observation_id']
    receipt=c.submit('press',oid,'press-once')
    assert c.submit('press',oid,'press-once')==receipt
    wait_for(lambda:c.get_state()['phase']=='READY')
    assert max(q[0] for q in b.sent) == pytest.approx(.004)
    assert b.q[0] == pytest.approx(0)
    result=c.get_state()['result']
    assert result['action_completed'] is True
    assert result['button_trigger_evidence'] is None
    assert result['contact_evidence'] is None


def test_missing_range_rejects_without_motion(live):
    c,b=live;c.c['press']['verified']=True;b.range_trusted=False
    c.submit('press',c.observe()['observation_id'],'no-depth')
    wait_for(lambda:c.get_state()['phase']=='READY')
    assert c.get_state()['result']['status']=='rejected'
    assert all(np.allclose(q,0) for q in b.sent)


def test_stop_cancels_blocked_planning(live):
    c,b=live;c.planner.kin.gate=threading.Event()
    oid=c.observe()['observation_id']
    c.submit('move',oid,'blocked',du_mm=2)
    with pytest.raises(ValueError,match='ready'):
        c.submit('move',oid,'second',du_mm=1)
    c.stop();count=len(b.sent)
    c.planner.kin.gate.set()
    c.worker.join(1)
    time.sleep(.03)
    assert len(b.sent)==count
    assert c.get_state()['phase']=='STOPPED'
    with pytest.raises(ValueError,match='ready'):
        c.submit('move',oid,'later',du_mm=1)


def test_feedback_failure_latches_fault(live):
    c,b=live;b.failure='stale_state'
    wait_for(lambda:c.get_state()['phase']=='FAULT')
    count=len(b.sent)
    b.failure=None;time.sleep(.03)
    assert len(b.sent)==count
    assert b.stopped


def test_stale_images_and_unverified_press(live):
    c,b=live
    oid=c.observe()['observation_id']
    with pytest.raises(ValueError,match='calibration'):
        c.submit('press',oid,'uncalibrated')
    b.image_stamp=c.completed_ns-1
    with pytest.raises(ValueError,match='predate'):
        c.observe()
    c.observations[oid]=(0,c.generation)
    with pytest.raises(ValueError,match='observation'):
        c.submit('move',oid,'old',du_mm=1)


def test_http_tools(live):
    c,b=live
    server=make_server(c,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        api=PlanarTools('http://127.0.0.1:'+str(server.server_port))
        oid=api.observe()['observation_id']
        r=api.move_in_plane(1,0,oid,'http-one')
        assert api.move_in_plane(1,0,oid,'http-one')==r
        with pytest.raises(ValueError):api._call('/move',dict(du_mm=True,dv_mm=0,observation_id=oid,command_id='bad'))
        assert api.stop()['phase']=='STOPPED'
    finally:
        server.shutdown();server.server_close();thread.join(1)


def test_template_cannot_execute(config):
    with pytest.raises(ValueError,match='calibration_required'):
        validate_config(config)
    for k in config['calibration']:
        config['calibration'][k]=True if k!='evidence' else 'test-only'
    validate_config(config)
    config['control']['press']['verified']=True
    with pytest.raises(ValueError,match='torque_and_evidence'):
        validate_config(config)


def test_local_air_probe_without_press_calibration(live):
    c,b=live
    c.submit('air_probe',c.observe()['observation_id'],'air',normal_mm=3)
    wait_for(lambda:c.get_state()['phase']=='READY')
    assert max(q[0] for q in b.sent)==pytest.approx(.003)
    np.testing.assert_allclose(b.q[:3],[0,0,0],atol=1e-10)
    assert not c.c['press']['verified']


def test_nonfinite_fk_rejected(config):
    p=planner_for(config)
    bad=np.eye(4);bad[0,3]=np.nan
    with pytest.raises(ValueError,match='transform'):
        p.check_pose(bad,np.eye(4))


def test_depth_change_during_planning_rejects_press(live):
    c,b=live;c.c['press']['verified']=True
    calls=[]
    def ranging():
        calls.append(1)
        return dict(trusted=True,gap_m=.003 if len(calls)==1 else .02)
    b.range=ranging
    c.submit('press',c.observe()['observation_id'],'changing-depth')
    wait_for(lambda:c.get_state()['phase']=='READY')
    assert 'range_changed' in c.get_state()['result']['reason']
    assert all(np.allclose(q,0) for q in b.sent)


def test_close_preserves_original_fault(live):
    c,b=live
    c.stop('torque_anomaly',fault=True)
    c.close()
    assert c.get_state()['phase']=='FAULT'
    assert c.get_state()['fault']=='torque_anomaly'


def test_live_backend_feedback_guards(config):
    from pressb.planar_control.runtime import LiveBackend
    names=['astribot_arm_right','astribot_gripper_right','astribot_head','astribot_torso','astribot_chassis','astribot_arm_left','astribot_gripper_left']
    sizes=[7,1,2,4,3,7,1]
    now=time.time_ns()
    rows={p:dict(position=[0]*n,velocity=[0]*n,torque=[0]*n,names=[],receipt_ns=time.monotonic_ns(),header=dict(sec=now//10**9,nanosec=now%10**9)) for p,n in zip(names,sizes)}
    monitor=SimpleNamespace(snapshot=lambda:rows,error=None,driver_fault=None)
    b=LiveBackend.__new__(LiveBackend)
    b.c=config;b.sensors=SimpleNamespace(monitor=monitor)
    b.executor=SimpleNamespace(state='EXECUTING',fault_reason=None,c=dict(state_max_age_ns=10**9,joint_names=['j'+str(i) for i in range(7)]))
    b.baseline={p:np.zeros(n) for p,n in zip(names,sizes)}
    b.kin=CartesianKin(Plane(config['plane']))
    assert b.snapshot()['q'].shape==(7,)
    rows['astribot_head']['position'][0]=.1
    with pytest.raises(ValueError,match='fixed_part_drift'):b.snapshot()
    rows['astribot_head']['position'][0]=0
    config['torque'].update(verified=True,max_deviation_nm=[.1]*7)
    rows['astribot_arm_right']['torque'][0]=1
    with pytest.raises(ValueError,match='torque_anomaly'):b.snapshot()
    rows['astribot_arm_right']['torque'][0]=0
    rows['astribot_torso']['receipt_ns']=0
    with pytest.raises(ValueError,match='stale_part'):b.snapshot()


def test_sensor_timestamp_and_depth_decoding(config):
    from pressb.planar_control.sensing import Sensors
    s=Sensors.__new__(Sensors)
    s.c=config['sensors'];s.c['depth_scale_m']['16UC1']=.001
    s.lock=threading.Lock();s.depth=None;s.error=None
    now=time.time_ns()
    header=SimpleNamespace(stamp=SimpleNamespace(sec=now//10**9,nanosec=now%10**9),frame_id='depth')
    values=np.array([[200,201,999],[202,203,999]],dtype='>u2')
    message=SimpleNamespace(header=header,encoding='16UC1',is_bigendian=True,height=2,width=2,step=6,data=values.tobytes())
    s.on_depth(message)
    np.testing.assert_allclose(s.depth['array'],[[.2,.201],[.202,.203]])
    s.on_depth(message)
    assert s.error=='nonmonotonic_depth'
    header.stamp.sec=0;header.stamp.nanosec=0
    with pytest.raises(ValueError,match='stale_sensor'):s.stamp(message)


def test_delayed_pre_motion_exposure_is_rejected(live):
    c,b=live
    original=b.observe
    def delayed():
        observation=original()
        for frame in observation['images'].values():
            frame['source_ns']=c.completed_source_ns-1
        return observation
    b.observe=delayed
    with pytest.raises(ValueError,match='predate'):
        c.observe()


def test_video_export_keeps_both_timelines(tmp_path):
    import base64
    import io
    import shutil
    from PIL import Image
    from export_planar_video import export
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    buffer=io.BytesIO();Image.new('RGB',(32,24),'green').save(buffer,format='JPEG')
    encoded=base64.b64encode(buffer.getvalue()).decode()
    log=tmp_path/'events.jsonl'
    rows=[]
    for i in range(3):
        for key,offset in [('head',0),('right',50_000_000)]:
            rows.append(dict(event='camera_frame',camera=key,source_ns=1_000_000_000+i*100_000_000+offset,jpeg_base64=encoded))
    log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=export(log,tmp_path/'video')
    assert result['right']['leading_hold_s']==pytest.approx(.05)
    for key in ('head','right'):
        assert (tmp_path/'video'/f'{key}.mp4').stat().st_size>0


def test_press_fault_does_not_auto_retract(live):
    c,b=live;c.c['press']['verified']=True
    c.c['press'].update(contact_speed_m_s=.05,approach_speed_m_s=.05,retract_speed_m_s=.05)
    original=b.send
    def contact_failure(q,state):
        original(q,state)
        if q[0]>.0002:
            b.failure='torque_anomaly'
    b.send=contact_failure
    c.submit('press',c.observe()['observation_id'],'faulting-press')
    wait_for(lambda:c.get_state()['phase']=='FAULT')
    count=len(b.sent);time.sleep(.03)
    assert len(b.sent)==count
    assert b.q[0]>.0002
    assert c.get_state()['result']['status']=='interrupted'
    assert c.get_state()['result']['action_completed'] is False


def test_keyboard_forward_100mm_holds_then_retracts(live):
    from pressb.planar_control.keyboard import submit_key
    c,b=live
    c.c['keyboard_forward'].update(verified=True,evidence='test-only',speed_m_s=.2)
    submit_key(c,'f')
    wait_for(lambda:c.get_state()['phase']=='READY')
    assert c.get_state()['completed_normal_m']==pytest.approx(.1)
    assert b.q[0]==pytest.approx(.1)
    for key in ('w','f'):
        with pytest.raises(ValueError,match='retract_to_plane_first'):
            submit_key(c,key)
    submit_key(c,'r')
    wait_for(lambda:c.get_state()['phase']=='READY')
    assert c.get_state()['completed_normal_m']==0
    assert b.q[0]==pytest.approx(0)
    submit_key(c,'d',2)
    wait_for(lambda:c.get_state()['phase']=='READY')
    assert c.get_state()['completed_uv_m']==pytest.approx([.002,0])


def test_keyboard_forward_requires_calibrated_clearance(live):
    from pressb.planar_control.keyboard import submit_key
    c,b=live
    with pytest.raises(ValueError,match='keyboard_forward_calibration'):
        submit_key(c,'f')
    assert all(np.allclose(q,0) for q in b.sent)
    with pytest.raises(ValueError,match='already_on_plane'):
        submit_key(c,'r')
