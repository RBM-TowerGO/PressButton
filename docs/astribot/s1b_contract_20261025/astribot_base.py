
import os

class AstribotBase:
    def __init__(self):
        self.whole_body_names = ['astribot_chassis', 'astribot_torso',
                                 'astribot_arm_left','astribot_gripper_left',
                                 'astribot_arm_right', 'astribot_gripper_right',
                                 'astribot_head']
        self.head_name = 'astribot_head'
        self.torso_name = 'astribot_torso'
        self.chassis_name = 'astribot_chassis'
        self.arm_left_name = 'astribot_arm_left'
        self.arm_right_name = 'astribot_arm_right'
        self.effector_left_name = 'astribot_gripper_left'
        self.effector_right_name = 'astribot_gripper_right'
        self.elbow_left_name = 'astribot_left_elbow'
        self.elbow_right_name = 'astribot_right_elbow'

        self.arm_names = ['astribot_arm_left', 'astribot_arm_right']
        self.elbow_names = ['astribot_left_elbow', 'astribot_right_elbow']
        self.effector_names = ['astribot_gripper_left', 'astribot_gripper_right']

        self.world_frame_name = 'world'
        self.chassis_frame_name = 'chassis'
        self.freme_names = [self.world_frame_name, self.chassis_frame_name]

        self.whole_body_dofs = [2, 4, 7, 1, 7, 1, 2]
        self.whole_body_cartesian_dofs = [7, 7, 7, 1, 7, 1, 7]
        self.head_dof = 2
        self.arm_dof = 7
        self.effector_dof = 1
        self.torso_dof = 4
        self.chassis_dof = 2

        self.robot_type = os.getenv('ROBOT_TYPE')
        if self.robot_type == 'S1':
            self.whole_body_dofs = [3, 4, 7, 1, 7, 1, 2]
            self.chassis_dof = 3