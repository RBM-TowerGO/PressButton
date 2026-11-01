"""Delayed SDK feedback, path safety and fault retrieval regressions (no robot)."""
import math
import threading
from types import SimpleNamespace

import pytest

from pressb.agent_loop.sdk_backend import SDKBackend
from pressb.agent_loop.client import RobotClient
from pressb.agent_loop.common import request
from pressb.agent_loop.service import make_server
from test_agent_loop import setup, observe, action, wait


def test_keyboard_interpolation_reaches_full_target_without_feedback_clamp():
    from s1b_keyboard_direct import interpolate_pose
    start=[.4,-.2,1.,0,0,0,1];goal=list(start);goal[2]+=.015
    middle,done=interpolate_pose(start,goal,.703125,1.40625)
    assert middle[2]==pytest.approx(1.0075) and not done
    final,done=interpolate_pose(start,goal,1.40625,1.40625)
    assert final==pytest.approx(goal) and done


def test_stationary_feedback_does_not_block_keyboard_commands():
    import time
    class Robot:
        pose=[.4,-.2,1.,0,0,0,1]
        stopped=False
        commands=[]
        def get_control_rights_status(self):return True
        def get_current_cartesian_pose(self,*a,**kw):return [list(self.pose),list(self.pose)]
        def set_cartesian_pose(self,names,poses,**kw):self.commands.append(list(poses[0]))
        def get_current_joints_position(self,*a):return [[0,0]]
        def stop_robot(self):self.stopped=True
    robot=Robot()
    def rows():
        now=time.time_ns()
        return {k:dict(header=dict(sec=now//10**9,nanosec=now%10**9))
                for k in ('astribot_arm_right','astribot_torso','astribot_gripper_right')}
    backend=SDKBackend(robot,SimpleNamespace(snapshot=rows),100)
    try:
        for z in (1.020,1.040):
            result=backend.move('move',dict(xyz_mm=[0,0,20]),threading.Event(),lambda p:None)
            assert robot.commands[-1][2]==pytest.approx(z)
            assert result['completion_basis']=='command_interpolation'
            assert result['final_state']['pose_m_xyzw']==robot.pose
            assert not robot.stopped and backend.error is None
    finally:backend.close()
    assert robot.stopped


def test_backend_fault_remains_queryable_and_rejects_new_motion(setup,tmp_path):
    service,backend,_=setup;obs=observe(service)
    def fail(*args):
        backend.error='SDK send failed'
        raise RuntimeError(backend.error)
    backend.move=fail
    receipt=service.submit(action(service,obs,type='move',xyz_mm=[0,0,20]))
    result=wait(service,receipt)
    service.backend_failed()
    assert service.cancel.is_set() and backend.stopped
    assert not service.shutdown_requested.is_set()
    assert result['status']=='interrupted'
    server=make_server(service,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=RobotClient('http://127.0.0.1:'+str(server.server_port),tmp_path/'client')
        with pytest.raises(ValueError,match='task_failed:SDK send failed'):
            client.wait(receipt)
        assert request(client.base,'/v2/state')['phase']=='FAULT'
        with pytest.raises(ValueError,match='controller_stopped'):
            service.submit(action(service,obs,cid='new',type='move',xyz_mm=[0,-1,0]))
        assert client.stop()['phase']=='STOPPED'
        assert service.shutdown_requested.is_set()
    finally:
        server.shutdown();server.server_close();thread.join()


@pytest.mark.parametrize('delta,angle', [([.0021,0,0],0),([0,.0051,0],0),([0,0,0],.51)])
def test_feedback_error_is_recorded_without_motion_gate(delta,angle):
    import time
    backend=SDKBackend.__new__(SDKBackend)
    start=[.4,-.2,1.,0,0,0,1];pose=list(start)
    pose[:3]=[x+d for x,d in zip(start[:3],delta)]
    pose[5:]=[math.sin(math.radians(angle)/2),math.cos(math.radians(angle)/2)]
    torso=[0,0,0,0,0,0,1]
    def rows():
        stamp=time.time_ns()
        return {name:dict(receipt_ns=time.monotonic_ns(),position=[0]*4,
            header=dict(sec=stamp//10**9,nanosec=stamp%10**9))
            for name in ('astribot_arm_right','astribot_torso','astribot_gripper_right')}
    backend.monitor=SimpleNamespace(snapshot=rows,error=None,driver_fault=None)
    backend.robot=SimpleNamespace(get_current_cartesian_pose=lambda *a,**kw:[pose,torso],
                                  get_current_joints_position=lambda *a:[[0,0]])
    backend.controller=SimpleNamespace(command_pose=lambda:start,gripper_target=None)
    backend.names=['astribot_arm_right','astribot_torso'];backend.torso=torso;backend.torso_joints=[0]*4
    backend.lock=threading.RLock();backend.cartesian_active=True
    backend.constraint=('move',start);backend.command_pose=start
    assert backend._read()['pose_m_xyzw']==pose
