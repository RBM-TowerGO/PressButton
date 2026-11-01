from types import SimpleNamespace

import pytest

from pressb.agent_loop.sdk_compat import install_local_route, install_native_local_route


class Native:
    def __init__(self):self.calls=[]
    def cartesian_pose_control(self,names,commands,use_wbc=False,remote_wbc_option=True,**kwargs):
        self.calls.append((names,commands,use_wbc,remote_wbc_option,kwargs))
        if remote_wbc_option and len(commands.get('astribot_torso',[]))==4:
            raise IndexError('index 1 is out of bounds for axis 0 with size 1')
        return 'local'


def test_mixed_command_is_filtered_and_explicitly_local():
    native=Native()
    poses={'astribot_arm_right':[.4,-.2,1,0,0,0,1],'astribot_torso':[.29,-.58,.29,0],'astribot_gripper_right':[100]}
    with pytest.raises(IndexError):native.cartesian_pose_control(['astribot_arm_right'],poses,use_wbc=False)
    original=native.cartesian_pose_control
    restore=install_native_local_route(native)
    assert native.cartesian_pose_control(['astribot_arm_right'],poses,remote_wbc_option=True)=='local'
    names,filtered,use_wbc,remote,kwargs=native.calls[-1]
    assert names==list(filtered)==['astribot_arm_right']
    assert use_wbc is False and remote is False
    assert poses['astribot_torso']==[.29,-.58,.29,0] and poses['astribot_gripper_right']==[100]
    restore();assert native.cartesian_pose_control==original


def test_route_is_instance_local_and_validates_contract():
    one,two=Native(),Native()
    robot=SimpleNamespace(astribot_interface=SimpleNamespace(_AstribotInterface__astribot_class=one))
    restore=install_local_route(robot)
    with pytest.raises(ValueError,match='7_values'):one.cartesian_pose_control(['torso'],{'torso':[0]*4})
    with pytest.raises(ValueError,match='local_cartesian'):one.cartesian_pose_control([],{},use_wbc=True)
    assert not two.calls and not one.calls
    restore()
    with pytest.raises(RuntimeError,match='unsupported_SDK_interface'):
        install_local_route(SimpleNamespace(astribot_interface=SimpleNamespace()))
    with pytest.raises(RuntimeError,match='remote_wbc_option'):
        install_native_local_route(SimpleNamespace(cartesian_pose_control=lambda names,commands:None))


def test_backend_constructor_preserves_servo_traceback_and_stops(monkeypatch):
    from pressb.agent_loop.sdk_backend import SDKBackend
    pose=[.4,-.2,1.,0.,0.,0.,1.]
    class Robot:
        stopped=0
        def get_control_rights_status(self):return True
        def get_current_cartesian_pose(self,*args,**kwargs):return [pose,pose]
        def set_different_type_command(self,*args,**kwargs):
            raise IndexError('native-startup-index-error')
        set_cartesian_pose=set_different_type_command
        def stop_robot(self):self.stopped+=1
    robot=Robot()
    monitor=SimpleNamespace(snapshot=lambda:{'astribot_torso':dict(position=[0]*4),
                                           'astribot_arm_right':dict(position=[0]*7)})
    import time
    monkeypatch.setattr(SDKBackend,'_read',lambda self:dict(pose_m_xyzw=pose,receipt_ns=time.monotonic_ns()))
    with pytest.raises(RuntimeError) as exc:SDKBackend(robot,monitor,100)
    message=str(exc.value)
    assert 'Traceback' in message and 'IndexError: native-startup-index-error' in message
    assert robot.stopped>=1
