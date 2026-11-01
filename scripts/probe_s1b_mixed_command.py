#!/usr/bin/env python3
"""Exercise native SDK routing with inert objects: no robot/client/node is created.

Run in S1B SDK Python environment. All robot interfaces, IK and publishers are
stubs; only native routing and quaternion arithmetic execute. Output JSON proves
both the failure and the instance-local workaround, not physical tracking.
"""
import json
from pathlib import Path
import sys
import time
import traceback
from collections import defaultdict
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def probe():
    import numpy as np
    from astribot_sdk.core.common.util.astribot_function import AstribotFunction as Native
    from pressb.agent_loop.sdk_compat import install_native_local_route
    pose=[.4,-.2,1.,0.,0.,0.,1.]
    joints=[.29,-.58,.29,0.]

    class Part:
        def __init__(self,count):
            self.joint_interface_=self.kinodynamic_=self;self.count=count;self.writes=[]
        def GetCurrentPoseWeldToTool(self):return np.array([0.,0.,0.,0.,0.,0.,1.])
        GetDesiredPoseWeldToTool=GetCurrentPoseWeldToTool
        def GetDesiredJointPosition(self):return np.zeros(self.count)
        def SetDesiredJointPosition(self,*args):self.writes.append(args)

    class Mapping:
        def get(self,key):
            assert key=='filter'
            return lambda name,command,kind:command

    class Stub:
        def __init__(self):
            self.robot_dict={'astribot_arm_right':Part(7),'astribot_torso':Part(4),'astribot_gripper_right':Part(1)}
            self.hight_level_stop_robot_flag=False
            self.wbc_service_exist_flag=True;self.wbc_first_loop=False
            self.received_time=defaultdict(time.time);self.control_frequency=250;self.delta_t=.004
            self.control_mode=None
            self.logger=SimpleNamespace(warning=lambda *a:None)
            self.torso_name='astribot_torso';self.head_name='astribot_head';self.chassis_name='astribot_chassis';self.arm_names=['astribot_arm_left','astribot_arm_right']
            self.ik_names=['astribot_arm_right']
            self.last_desired_cartesian_pose_dict={};self.last_desired_joints_position_dict={}
            self.last_pose_recorder=SimpleNamespace(pose_world_to_torso_desired=np.array(joints),
                pose_world_to_left_arm_desired=np.array(pose),pose_world_to_right_arm_desired=np.array(pose),update=lambda *a:None)
            self._AstribotFunction__control_function_mapping=Mapping()
            self.remote_calls=0;self.local_calls=0;self.joint_calls=[];self.cartesian_keys=[]
        def reset_wbc_first_loop_flag(self):pass
        def get_pose_world_to_robot_desired_dict(self,**kwargs):
            return defaultdict(lambda:np.array([0.,0.,0.,0.,0.,0.,1.]))
        def set_different_type_command_init(self,*args):return Native.set_different_type_command_init(self,*args)
        def joints_position_control(self,names,commands,**kwargs):
            self.joint_calls.append({n:commands[n] for n in names})
        def cartesian_pose_control(self,names,commands,use_wbc=False,remote_wbc_option=True,**kwargs):
            self.cartesian_keys.append(list(commands))
            return Native.cartesian_pose_control(self,names,commands,use_wbc,remote_wbc_option=remote_wbc_option,**kwargs)
        def execute_ik_and_set_position(self,*args,**kwargs):
            if kwargs.get('remote_wbc_option',True):self.remote_calls+=1
            return Native.execute_ik_and_set_position(self,*args,**kwargs)
        def calculate_twist_from_poses(self,*args,**kwargs):return Native.calculate_twist_from_poses(self,*args,**kwargs)
        def IK(self,names,commands,**kwargs):
            self.local_calls+=1
            assert names==['astribot_arm_right'] and list(commands)==names
            assert kwargs['torso_open_loop'] is True
            return True,{'astribot_arm_right':[0.]*7}
        def list_to_numpy(self,value):return np.asarray(value,dtype=float)
        def gripper_control(self,*args):pass
        def publish_wbc_command(self,*args,**kwargs):raise AssertionError('No real publication permitted')

    command=dict(name_list=['astribot_arm_right','astribot_torso','astribot_gripper_right'],
        type_list=['cartesian','joints','joints'],command_list=[pose,joints,[100.]],control_way='filter',use_wbc=False)
    # Verify assignment on the installed native class without its constructor.
    native_binding_verified=False
    if Native.__new__ is object.__new__ and not any('__del__' in cls.__dict__ for cls in Native.__mro__):
        native_shell=object.__new__(Native)
        undo=install_native_local_route(native_shell);undo()
        native_binding_verified=True
    broken=Stub()
    try:Native.set_different_type_command(broken,command)
    except IndexError as exc:
        error=str(exc);stack=traceback.format_exc()
    else:raise AssertionError('Expected the installed SDK mixed/WBC bug to reproduce')
    fixed=Stub();restore=install_native_local_route(fixed)
    for _ in range(3):Native.set_different_type_command(fixed,command)
    assert fixed.remote_calls==0 and fixed.local_calls==3
    assert fixed.joint_calls[-1]=={'astribot_torso':joints,'astribot_gripper_right':[100.]}
    restore()
    return dict(native_instance_binding_verified=native_binding_verified,broken_error=error,broken_traceback=stack,broken_remote_calls=broken.remote_calls,
        fixed_remote_calls=fixed.remote_calls,fixed_local_ik_calls=fixed.local_calls,
        fixed_cartesian_keys=fixed.cartesian_keys,fixed_joint_command=fixed.joint_calls[-1],
        hardware_commands_sent=0,scope='Native routing/math; IK, robot interfaces and publishers are stubs')


if __name__=='__main__':print(json.dumps(probe(),ensure_ascii=False,indent=2))
