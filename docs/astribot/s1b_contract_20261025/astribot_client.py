import os
import time
import copy
from typing import List, Dict, Optional, Any
from astribot_sdk.core.common.astribot_base import AstribotBase
from astribot_sdk.core.astribot_api.astribot_interface import AstribotInterface
from astribot_sdk.core.common.logger import default_logger

def check_ros_version():
    """
    Brief:
        Check the system ros version to determine the communication method used.

    Args:
        None

    Returns:
        None

    Example:
        output=check_ros_version()

        ---output---
        output: "ROS1" or "ROS2" or "Unknown Version" or "ROS not installed"
    """
    if "ROS_DISTRO" in os.environ:
        ros_distro = os.environ["ROS_DISTRO"]
        if ros_distro.lower().startswith("noetic") or ros_distro.lower().startswith("melodic"):
            return "ROS1"
        elif ros_distro.lower().startswith("foxy") or ros_distro.lower().startswith("galactic") or ros_distro.lower().startswith("humble"):
            return "ROS2"
        else:
            return "Unknown Version"
    else:
        return "ROS not installed"

class Astribot(AstribotBase):
    def __init__(self, freq=250.0, ip=None, interface:str=None, high_control_rights:bool=False, node_name:str="use_astribot",logger=default_logger) -> None:
        super().__init__()
        if self.robot_type not in ['S0', 'S1']:
            raise ValueError("Invalid robot type")
        self._logger = logger
        self.freq = freq
        self._logger.info("astribot_interface is initializing...")
        self.astribot_interface = AstribotInterface(freq, high_control_rights, node_name, logger)
        self._logger.info("astribot_interface initialized.")
        self._logger.info("Waiting for interface to be alive...")
        self.is_alive = self.wait_for_interface_alive()
        self._logger.info("Interface is alive.")

        if self.is_alive:
            robot_mode = self.astribot_interface.get_robot_mode()
            self.__in_simulation = False
            if robot_mode == "safe":
                self._logger.info(f"Astribot S1 is connected! Robot mode is safe")
            elif robot_mode == "professional":
                self._logger.warning(f"Astribot S1 is connected! Robot mode is standard")
            elif robot_mode == "extremity":
                self._logger.error(f"Astribot S1 is connected! Robot mode is extreme")
            else:
                self.__in_simulation = True
                self._logger.info("Astribot S1 is connected! Currently using robots in simulation")
        else:
            self._logger.error("No simulation or real robot is started.")
            raise RuntimeError("No simulation or real robot is started.")

    def get_control_rights_status(self):
        """
        Brief:
            Get the control rights status of the robot.

        Args:
            None

        Returns:
            A bool value to show if the control rights is acquired.

        Example:
            output=Astribot.get_control_rights_status()

            ---output---
            output: True or False
        """
        return self.astribot_interface.have_control_rights

    ### 1. Robot Information ###
    def get_info(self):
        """
        Brief:
            Prints the information about the robot.

        Args:
            None

        Returns:
            None

        Example:
            Astribot.get_info()

            ---output---
            Robot is alive: True
            Whole body names: ['astribot_torso', 'astribot_arm_left', 'astribot_gripper_left', 'astribot_arm_right', 'astribot_gripper_right', 'astribot_head']
            Whole body dofs: [4, 7, 1, 7, 1, 2]

        """
        self._logger.info(f"Robot is alive: {self.is_alive}")
        self._logger.info(f"Whole body names: {self.whole_body_names}")
        self._logger.info(f"Whole body dofs: {self.whole_body_dofs}")

    def get_dof(self, names: Optional[List] = None):
        """
        Brief:
            Get the degree of freedom of the robot.

        Args:
            names: A list of joints' name. If None, the whole body names will be used.

        Returns:
            A list of degree of freedom of the robot.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_dof(input)

            ---output---
            output: [2, 7, 7]
        """
        return self.astribot_interface.get_general_info(
            "get_dof", names if names is not None else self.whole_body_names)

    def get_joints_position_limit(self, names: Optional[List] = None):
        """
        Brief:
            Get the limit of joints' position of the robot with the given names.

        Args:
            names: A list of joints' name.If None, the whole body names will be used.

        Returns:
            Two lists of joints' position limit, the first list is the lower limit and the second list is the upper limit.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_joints_position_limit(input)

            ---output---
            output:[[-1.57, -1.22], [-3.01, -1.5, -3.14, 0.001, -2.4, -0.75, -1.57], [0.0]],
                    [[1.57, 1.22], [3.01, 0.25, 3.14, 2.618, 2.4, 0.75, 1.57], [100.0]]
        """
        return self.astribot_interface.get_general_info(
            "get_joints_position_limit", names if names is not None else self.whole_body_names)

    def get_joints_velocity_limit(self, names: Optional[List] = None):
        """
        Brief:
            Get the limit of joints' velocity of the robot with the given names.

        Args:
            names: A list of joints' name. If None, the whole body names will be used.

        Returns:
            A lists of joints' velocity limit.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_joints_velocity_limit(input)

            ---output---
            output:[[4.0, 4.0], [7.4, 7.4, 7.4, 13.1, 18.1, 18.1, 18.1], [1000.0]]
        """
        return self.astribot_interface.get_general_info(
            "get_joints_velocity_limit", names if names is not None else self.whole_body_names)

    def get_joints_torque_limit(self, names: Optional[List] = None):
        """
        Brief:
            Get the limit of joints' torque of the robot with the given names.

        Args:
            names: A list of joints' name. If None, the whole body names will be used.

        Returns:
            A lists of joints' torque limit.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_joints_torque_limit(input)

            ---output---
            output:[[100.0, 100.0], [265.0, 265.0, 132.5, 142.0, 71.0, 20.0, 20.0], [1.0]]
        """
        return self.astribot_interface.get_general_info(
            "get_joints_torque_limit", names if names is not None else self.whole_body_names)

    def get_current_joints_position(self, names: Optional[List] = None):
        """
        Brief:
            Get the joints' current position of the robot with the given names.

        Args:
            names: A list of joints' name. If None, the whole body names will be used.

        Returns:
            A lists of joints' current position.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_current_joints_position(input)

            ---output---
            output: [[head_q0, head_q1], [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6], [effector_q0]]
        """
        return self.astribot_interface.get_joints_info(
            "get_current_joints_position", names if names is not None else self.whole_body_names)

    def get_current_joints_velocity(self, names: Optional[List] = None):
        """
        Brief:
            Get the joints' current velocity of the robot with the given names.

        Args:
            names: A list of joints' name. If None, the whole body names will be used.

        Returns:
            A lists of joints' current velocity.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_current_joints_velocity(input)

            ---output---
            output: [[head_qdot0, head_qdot1], [arm_qdot0, arm_qdot1, arm_qdot2, arm_qdot3, arm_qdot4, arm_qdot5, arm_qdot6], [effector_qdot0]]
        """
        return self.astribot_interface.get_joints_info(
            "get_current_joints_velocity", names if names is not None else self.whole_body_names)

    def get_current_joints_acceleration(self, names: Optional[List] = None):
        """
        Brief:
            Get the joints' current acceleration of the robot with the given names.

        Args:
            names: A list of joints' name.

        Returns:
            A lists of joints' current acceleration.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_current_joints_acceleration(input)

            ---output---
            output: [[head_qddot0, head_qddot1], [arm_qddot0, arm_qddot1, arm_qddot2, arm_qddot3, arm_qddot4, arm_qddot5, arm_qddot6], [effector_qddot0]]
        """
        return self.astribot_interface.get_joints_info(
            "get_current_joints_acceleration", names if names is not None else self.whole_body_names)

    def get_current_joints_torque(self, names: Optional[List] = None):
        """
        Brief:
            Get the joints' current torque of the robot with the given names.

        Args:
            names: A list of joints' name.

        Returns:
            A lists of joints' current torque.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_current_joints_torque(input)

            ---output---
            output: [[head_qtor0, head_qtor1], [arm_qtor0, arm_qtor1, arm_qtor2, arm_qtor3, arm_qtor4, arm_qtor5, arm_qtor6], [effector_qtor0]]
        """
        return self.astribot_interface.get_joints_info(
            "get_current_joints_torque", names if names is not None else self.whole_body_names)

    def get_desired_joints_position(self, names: Optional[List] = None):
        """
        Brief:
            Get the joints' desired position of the robot with the given names, which means the last position command.

        Args:
            names: A list of joints' name.

        Returns:
            A lists of joints' desired position.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_desired_joints_position(input)

            ---output---
            output: [[head_q0, head_q1], [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6], [effector_q0]]
        """
        return self.astribot_interface.get_joints_info(
            "get_desired_joints_position", names if names is not None else self.whole_body_names)

    def get_desired_joints_velocity(self, names: Optional[List] = None):
        """
        Brief:
            Get the joints' desired velocity of the robot with the given names, which means the last velocity command.

        Args:
            names: A list of joints' name.

        Returns:
            A lists of joints' desired velocity.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_desired_joints_velocity(input)

            ---output---
            output: [[head_qdot0, head_qdot1], [arm_qdot0, arm_qdot1, arm_qdot2, arm_qdot3, arm_qdot4, arm_qdot5, arm_qdot6], [effector_qdot0]]
        """
        return self.astribot_interface.get_joints_info(
            "get_desired_joints_velocity", names if names is not None else self.whole_body_names)

    def get_desired_joints_torque(self, names: Optional[List] = None):
        """
        Brief:
            Get the joints' desired torque of the robot with the given names, which means the last torque command.

        Args:
            names: A list of joints' name.

        Returns:
            A lists of joints' desired torque.

        Example:
            input=["astribot_head", "astribot_arm_left", "astribot_arm_right"]
            output=Astribot.get_desired_joints_torque(input)

            ---output---
            output: [[head_qtor0, head_qtor1], [arm_qtor0, arm_qtor1, arm_qtor2, arm_qtor3, arm_qtor4, arm_qtor5, arm_qtor6], [effector_qtor0]]
        """
        return self.astribot_interface.get_joints_info(
            "get_desired_joints_torque", names if names is not None else self.whole_body_names)

    def get_current_cartesian_pose(self, names: Optional[List] = None, frame:str="chassis"):
        """
        Brief:
            Get the cartesian pose of current joints' state in a given frame ("chassis" or "world").
            This method supports only global frames such as the robot's chassis or the world frame.

        Args:
            names: A list of parts' name, and specified coordinate frame with string format.
            frame: A reference frame to get the cartesian pose, can only be "chassis", "world"

        Returns:
            A lists of current cartesian pose. [Note]: If input name is effector, the output will be the effector's joint angle.

        Example:
            input=["astribot_arm_left", "astribot_arm_right"], "world"
            output=Astribot.get_current_cartesian_pose(input)

            ---output---
            output: [[x, y, z, qx, qy, qz, qw], [x, y, z, qx, qy, qz, qw]]
        """
        if frame not in self.freme_names:
            raise ValueError("Invalid frame.")
        return self.astribot_interface.get_cartesian_info(
            "current_cartesian_pose", names if names is not None else self.whole_body_names, frame)

    def get_desired_cartesian_pose(self, names: Optional[List] = None, frame:str="chassis"):
        """
        Brief:
            Get the cartesian pose of desired joints' state in a given frame ("chassis" or "world").
            This method supports only global frames such as the robot's chassis or the world frame.

        Args:
            names: A list of parts' name, and specified coordinate frame with string format.
            frame: A reference frame to get the cartesian pose, can be "chassis", "world"

        Returns:
            A lists of desired cartesian pose. [Note]: If input name is effector, the output will be the effector's joint angle.

        Example:
            input=["astribot_arm_left", "astribot_arm_right"], "world"
            output=Astribot.get_desired_cartesian_pose(input)

            ---output---
            output: [[x, y, z, qx, qy, qz, qw], [x, y, z, qx, qy, qz, qw]]
        """
        if frame not in self.freme_names:
            raise ValueError("Invalid frame.")
        return self.astribot_interface.get_cartesian_info(
            "desired_cartesian_pose", names if names is not None else self.whole_body_names, frame)

    def get_desired_wbc_pose(self, names: Optional[List] = None, frame:str="chassis"):
        """
        Brief:
            Get the cartesian pose of desired joints' state in a given frame ("chassis" or "world").
            Specially, the arms' cartesian pose is calculated by current torso joints and desired arms' joints.
            This method supports only global frames such as the robot's chassis or the world frame.

        Args:
            names: A list of parts' name, and specified coordinate frame with string format.
            frame: A reference frame to get the cartesian pose, can be "chassis", "world"

        Returns:
            A lists of desired cartesian pose (arms' cartesian pose is calculated by current torso joints and desired arms' joints).
            [Note]: If input name is effector, the output will be the effector's joint angle.

        Example:
            input=["astribot_arm_left", "astribot_arm_right"], "world"
            output=Astribot.get_desired_cartesian_pose(input)

            ---output---
            output: [[x, y, z, qx, qy, qz, qw], [x, y, z, qx, qy, qz, qw]]
        """
        return self.astribot_interface.get_cartesian_info(
            "desired_wbc_pose", names if names is not None else self.whole_body_names, "chassis")

    def get_forward_kinematics(self, names: List[str] , joints_position: List[List[float]] ) :
        """
        Brief:
            Get the forward kinematics result of the robot with the given names and joints' position.
            Please notice that do not include effector in the names, and at least torso joint info must be included.

        Args:
            names: A list of joints' name, a list of joints' position.
            joints_position: A list of joints' position.

        Returns:
            A lists of cartesian pose, which is the forward kinematics result.

        Example:
            input=["astribot_arm_left", "astribot_torso"], [[arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6], [torso_q0, torso_q1, torso_q2, torso_q3]]
            output=Astribot.get_forward_kinematics(input)

            ---output---
            output: {
                'astribot_arm_left': [x, y, z, qx, qy, qz, qw],
                'astribot_torso': [x, y, z, qx, qy, qz, qw]
            }
        """
        if "astribot_gripper_left" in names or "astribot_gripper_right" in names:
            raise ValueError("Forward kinematic does not support for effector.")
        if "astribot_torso" not in names:
            raise ValueError("At least torso joint info must be included.")

        return self.astribot_interface.get_forward_kinematics(names, joints_position)

    def get_inverse_kinematics(self, names: List[str] , cartesian_pose_list: List[List[float]] ) :
        """
        Brief:
            Get the inverse kinematics result of the robot with the given names and cartesian pose.
            Please notice that now we just support for elbow, arm and torso inverse kinematic.
            Do not include effector, head, and chassis in the names.

        Args:
            names: A list of names, a list of cartesian pose. [Note]: Now can only be input elbow, arm and torso name.
            cartesian_pose_list: A list of cartesian pose.

        Returns:
            A bool value which means if the result of inverse kinematics is valid,
            And a list of joints' position if the result is valid.

        Example:
            input=['astriobt_torso', 'astribot_arm_left'], [[x, y, z, qx, qy, qz, qw], [x, y, z, qx, qy, qz, qw]]
            output=Astribot.get_inverse_kinematics(input)

            ---output---
            output: {'astribot_arm_left': [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6],
                     'astribot_torso': [torso_q0, torso_q1, torso_q2, torso_q3]}
        """
        if "astribot_head" in names or "astribot_chassis" in names or "astribot_gripper_left" in names or "astribot_gripper_right" in names:
            raise ValueError("inverse kinematic now just support for elbow, arm and torso.")
        return self.astribot_interface.get_inverse_kinematics(names, cartesian_pose_list)

    def get_self_closest_point(self, torso_joints_position=None, arm_left_joints_position=None, arm_right_joints_position=None):
        """
        Brief:
            Get the closest part and distance value of the robot itself.

        Args:
            torso_joints_position: a list of astribot-torso's joints position, if None, use current joints position
            arm_left_joints_position: a list of astribot-arm-left's joints position, if None, use current joints position
            arm_right_joints_position: a list of astribot-arm-right's joints position, if None, use current joints position

        Returns:
            Closest distance value, name of the closest part A and B, and the closest point pose on A and B.
        """
        min_distance, link_A_name, link_B_name, closest_point_on_A_of_torso_frame, closest_point_on_B_of_torso_frame = \
            self.astribot_interface.get_self_closest_point(torso_joints_position, arm_left_joints_position, arm_right_joints_position)
        return min_distance, link_A_name, link_B_name, closest_point_on_A_of_torso_frame, closest_point_on_B_of_torso_frame

    ### 2. Robot Control ###
    def move_joints_position(self, names: List[str] , commands: List[List[float]] , duration=5.0, use_wbc:bool=False, add_default_torso:bool=True):
        """
        Brief:
            Make the joints' position of the robot with given names move to the given position offline.
            When using this function to control chassis, the posture is in the world system.

        Args:
            names: A list of names
            commands: A list of joints' position
            duration: duration to reach the goal
            use_wbc: A bool value to determine if the whole body control mode is used.
            add_default_torso: A bool value to determine if the default torso joint is added.

        Returns:
            A string to show the result of the move.

        Example:
            Reference 103-move_to_joint_position.py

        """

        if self.torso_name not in names and add_default_torso:
            new_names = copy.deepcopy(names)
            new_cmds = copy.deepcopy(commands)
            new_names.append(self.torso_name)
            torso_desired_position = self.get_desired_joints_position([self.torso_name])[0]
            new_cmds.append(torso_desired_position)
            return self.astribot_interface.move_to_joint_position(new_names, new_cmds, duration, use_wbc)
        else:
            return self.astribot_interface.move_to_joint_position(names, commands, duration, use_wbc)

    def move_cartesian_pose(self, names: List[str] , commands: List[List[float]] , duration=5.0, use_wbc:bool=True, add_default_torso:bool=True):
        """
        Brief:
            Make the robot with given names move to the given cartesian pose offline.
            When using this function to control chassis, the posture is in the world system.

        Args:
            names: A list of names
            commands: A list of cartesian pose
            duration: duration to reach the goal
            use_wbc: A bool value to determine if the whole body control mode is used.
            add_default_torso: A bool value to determine if the default torso joint is added.

        Returns:
            A string to show the result of the move.

        Example:
            Reference 107-move_to_cartesian_pose.py
        """

        if self.torso_name not in names and add_default_torso:
            new_names = copy.deepcopy(names)
            new_cmds = copy.deepcopy(commands)
            new_names.append(self.torso_name)
            torso_desired_pose = self.get_desired_cartesian_pose([self.torso_name])[0]
            new_cmds.append(torso_desired_pose)
            return self.astribot_interface.move_to_cartesian_pose(new_names, new_cmds, duration, use_wbc)
        else:
            return self.astribot_interface.move_to_cartesian_pose(names, commands, duration, use_wbc)

    def move_to_home(self, duration=-1.0, use_wbc:bool=False):
        """
        Brief:
            Make the robot move to the home pose offline.

        Args:
            duration: duration to reach home pose.
            use_wbc: A bool value to determine if the whole body control mode is used.

        Returns:
            A string to show the result of the move.

        Example:
            Astribot.move_to_home()
        """

        self._logger.info("Moving to home")
        joints_name_list = ['astribot_torso',
                            'astribot_arm_left', 'astribot_gripper_left',
                            'astribot_arm_right', 'astribot_gripper_right',
                            'astribot_head']
        joints_command_list = [[ 0.29, -0.58, 0.29, 0.0],
                               [ 0.159, -0.022, -1.42, 1.66, -0.345, 0.115, 0.125],
                               [ 0.0],
                               [-0.159, -0.022, 1.42, 1.66, 0.345, 0.115, -0.125],
                               [ 0.0],
                               [ 0.0, 0.0]]
        max_duration = 5.0
        current_joints = self.astribot_interface.get_joints_info("get_current_joints_position", [self.torso_name, self.arm_left_name, self.effector_left_name, self.arm_right_name, self.effector_right_name])

        if not self.__in_simulation:
            check_start_time = time.time()
            collision_flag = self.astribot_interface._AstribotInterface__astribot_class.check_to_joint_position(
                {   "name_list": joints_name_list,
                    "command_list": joints_command_list,
                    "duration": max_duration,
                    "safe_distance": 0.03
                })
            check_use_time = time.time() - check_start_time
            max_duration = max_duration - check_use_time
        
            if not collision_flag:
                left_arm_may_collision, right_arm_may_collision = False, False
                if abs(current_joints[1][0]) < 0.3 and current_joints[1][1] > 0.2:
                    left_arm_may_collision = True
                if abs(current_joints[3][0]) < 0.3 and current_joints[3][1] > 0.2:
                    right_arm_may_collision = True
                else:
                    self._logger.warning("Returning from the current position to the home pose will cause a collision.\nPlease drag the robot to a reasonable position and run the program again.")
                    self.astribot_interface._pub_heartbeat([16000000, 16200000])
                    return "move failed"
            
                if left_arm_may_collision and right_arm_may_collision:
                    mid_point_names = [self.arm_left_name, self.arm_right_name]
                    mid_point = [current_joints[1], current_joints[3]]
                    mid_point[0][1], mid_point[1][1] = -0.1, -0.1
                elif left_arm_may_collision:
                    mid_point_names = [self.arm_left_name]
                    mid_point = [current_joints[1]]
                    mid_point[0][1] = -0.1
                elif right_arm_may_collision:
                    mid_point_names = [self.arm_right_name]
                    mid_point = [current_joints[3]]
                    mid_point[0][1] = -0.1
                
                if left_arm_may_collision or right_arm_may_collision:
                    self.astribot_interface.move_to_joint_position(mid_point_names, mid_point, duration=1.0, use_wbc=use_wbc)
                    max_duration = max_duration - 1.0

        if duration == -1:
            duration = 0.0
            joints_vel_limit = self.astribot_interface.get_general_info("get_joints_velocity_limit", [self.torso_name, self.arm_left_name, self.effector_left_name, self.arm_right_name, self.effector_right_name])
            if joints_vel_limit is None:
                duration = max_duration
            else:
                for i in range(len(current_joints)):
                    for j in range(len(current_joints[i])):
                        if i == 0:
                            cur_duration = abs(joints_command_list[i][j] - current_joints[i][j]) / joints_vel_limit[i][j] * 12.0
                        else:
                            cur_duration = abs(joints_command_list[i][j] - current_joints[i][j]) / joints_vel_limit[i][j] * 26.0
                        duration = max(duration, cur_duration)
        duration = min(duration, max_duration)
        return self.astribot_interface.move_to_joint_position(joints_name_list, joints_command_list, duration, use_wbc)

    def move_joints_waypoints(self, names: List[str] , waypoints: List[List[List[float]]] , time_list: List[float] , use_wbc:bool=False, joy_controller=False, add_default_torso:bool=True):
        """
        Brief:
            Make the joints' position of the robot with given names move to the target joints' position in sequence and in time.

        Args:
            names: A list of names
            waypoints: A list of waypoints, every waypoint in waypoints is a list of joints' position,
            time_list: A list of time to reach the target point in sequence, the length of time_list should be the same as the length of waypoints,
            use_wbc: A bool value to determine if the whole body control mode is used.
            joy_controller: A bool value to determine if the joy controller is used.
            add_default_torso: A bool value to determine if the default torso joint is added.

        Returns:
            A string to show the result of the move.

        Example:
            Reference 210-traj_replay.py
        """

        if self.torso_name not in names and add_default_torso:
            new_names = names.copy()
            new_waypoints = waypoints.copy()
            new_names.append(self.torso_name)
            torso_desired_position = self.get_desired_joints_position([self.torso_name])[0]
            for waypoint in new_waypoints:
                waypoint.append(torso_desired_position)

            return self.astribot_interface.move_joints_waypoints(new_names, new_waypoints, time_list, use_wbc, joy_controller)
        else:
            return self.astribot_interface.move_joints_waypoints(names, waypoints, time_list, use_wbc, joy_controller)

    def move_cartesian_waypoints(self, names: List[str] , waypoints: List[List[List[float]]] , time_list: List[float] , use_wbc:bool=True, joy_controller=False, add_default_torso:bool=True):
        """
        Brief:
            Make the robot with given names move to the target cartesian pose in sequence and in time.

        Args:
            names: A list of names
            waypoints: A list of waypoints, every waypoint in waypoints is a list of cartesian pose,
            time_list: A list of time to reach the target point in sequence, the length of time_list should be the same as the length of waypoints,
            use_wbc: A bool value to determine if the whole body control mode is used.
            joy_controller: A bool value to determine if the joy controller is used.
            frame: A reference frame to get the cartesian pose, can be "chassis", "world"
            add_default_torso: A bool value to determine if the default torso joint is added.

        Returns:
            A string to show the result of the move.

        Example:
            Reference 206-follow_cartesian_waypoints.py

        """

        if self.torso_name not in names and add_default_torso:
            new_names = names.copy()
            new_waypoints = waypoints.copy()
            new_names.append(self.torso_name)
            torso_desired_pose = self.get_desired_cartesian_pose([self.torso_name])[0]
            for waypoint in new_waypoints:
                waypoint.append(torso_desired_pose)
            return self.astribot_interface.move_cartesian_waypoints(new_names, new_waypoints, time_list, use_wbc, joy_controller)
        else:
            return self.astribot_interface.move_cartesian_waypoints(names, waypoints, time_list, use_wbc, joy_controller)

    def set_joints_position(self, names: List[str] , position: Optional[List[List[float]]] , control_way:str="filter" , use_wbc:bool=False, add_default_torso:bool=True):
        """
        Brief:
            Set the joints' position of the robot with the given names online.

        Args:
            names: A list of names
            position: A list of joints' position,
            control_way: A string to determine the control way, default is "filter", and the other option is "direct".
            use_wbc: A bool value to determine if the whole body control mode is used. default is False.
            add_default_torso: A bool value to determine if the default torso joint is added. default is True.

        Returns:
            None

        Example:
            Reference examples/104_set_joints_position.py
        """
        self.astribot_interface.set_joints_position(names, position, control_way, use_wbc, add_default_torso)

    def set_joints_velocity(self, names: List[str] , velocity: List[List[float]] ) :
        """
        Brief:
            Set the joints' velocity of the robot with the given names online.

        Args:
            names: A list of names
            velocity: A list of joints' velocity.

        Returns:
            None

        Example:
            Reference examples/106_set_joints_velocity.py
        """
        self.astribot_interface.set_joints_velocity(names, velocity)

    def set_joints_torque(self, names: List[str] , torque: List[List[float]] ) :
        """
        Brief:
            Set the joints' torque of the robot with the given names online.

        Args:
            names: A list of names
            torque: A list of joints' torque.

        Returns:
            None

        Example:
            Reference examples/209-arm_gravity_compensation.py

        """
        self.astribot_interface.set_joints_torque(names, torque)

    def set_cartesian_pose(self, names: List[str] , cartesian_pose: List[List[float]] , control_way:str="filter", use_wbc:bool=False, add_default_torso:bool=True, remote_wbc_option:bool=True):
        """
        Brief:
            Set the cartesian pose of the robot with the given names online.

        Args:
            A list of names, and a list of cartesian pose,
            control_way: A string to determine the control way, default is "filter", and the other option is "direct".
            use_wbc: A bool value to determine if the whole body control mode is used.
            remote_wbc_option: A bool value to determine if remote wbc service is used when wbc_service exists.

        Returns:
            None

        Example:
            Reference examples/108_set_cartesian_pose.py
        """
        allowed = {self.effector_names[0], self.effector_names[1]}
        if set(names).issubset(allowed) and bool(names):
            self.astribot_interface.set_joints_position(names, cartesian_pose, control_way, use_wbc=False, add_default_torso=add_default_torso)
        else:
            self.astribot_interface.set_cartesian_pose(names, cartesian_pose, control_way, use_wbc, add_default_torso, remote_wbc_option)
    
    def set_different_type_command(self, names: List[str], types: List[str], command_list: List[List[float]], control_way="filter", use_wbc:bool=False):
        """
        Brief:
            Set the command of the robot with different type("joints or cartesian").

        Args:
            A list of names, a list of different types, a list of command,
            control_way: A string to determine the control way, default is "filter", and the other option is "direct".
            use_wbc: A bool value to determine if the whole body control mode is used.

        Returns:
            None
        """
        self.astribot_interface.set_different_type_command(names, types, command_list, control_way, use_wbc)

    def open_effector(self, names: Optional[List] = None, duration=1.0):
        """
        Brief:
            Open the specified gripper.

        Args:
            names: A list of names, only the gripper name can be entered.

        Returns:
            A string to show the result of the move.

        Example:
            Reference 109-effector_open_close.py
        """
        if names is None or names == self.effector_names:
            names = self.effector_names
            joints_list = [[0.0], [0.0]]
        elif names == [self.effector_left_name] or names == [self.effector_right_name]:
            joints_list = [[0.0]]
        else:
            raise ValueError("Invalid effector name.")
        return self.astribot_interface.move_to_joint_position(names, joints_list, duration, False)

    def close_effector(self, names: Optional[List] = None, duration=1.0):
        """
        Brief:
            Close the specified gripper.

        Args:
            names: A list of names, only the gripper name can be entered.

        Returns:
            A string to show the result of the move.

        Example:
            Reference 109-effector_open_close.py
        """
        if names is None or names == self.effector_names:
            names = self.effector_names
            joints_list = [[100.0], [100.0]]
        elif names == [self.effector_left_name] or names == [self.effector_right_name]:
            joints_list = [[100.0]]
        else:
            raise ValueError("Invalid effector name.")
        return self.astribot_interface.move_to_joint_position(names, joints_list, duration, False)

    def stop_robot(self):
        """
        Brief:
            Stop the robot.

        Args:
            None

        Returns:
            None
        """
        return self.astribot_interface.stop_robot()

    def restart_robot(self):
        """
        Brief:
            Restart the robot.

        Args:
            None

        Returns:
            None
        """
        return self.astribot_interface.restart_robot()

    ### 3. Sensor Control ###
    # Camera
    def activate_camera(self, cameras_setting: Optional[Dict[str, Any]] = None):
        """
        Brief:
            Activate the camera of the robot.

        Args:
            cameras_setting: A dict of camera setting

        Returns:
            A bool value to show the result of the activation.

        Example:
            Reference 300-get_images.py
        """
        if cameras_setting is None:
            cameras_setting = {
                'left_D405': {'flag_getdepth': True, 'flag_getIR': True},
                'right_D405': {'flag_getdepth': True, 'flag_getIR': True},
                'Gemini335': {'flag_getdepth': True, 'flag_getIR': True},
                'Bolt': {'flag_getdepth': True},
                'Stereo': {}
            }
        return self.astribot_interface.activate_camera_abs(cameras_setting)

    def deactivate_camera(self):
        """
        Brief:
            Deactivate the camera of the robot.

        Args:
            None

        Returns:
            A bool value to show the result of the deactivation.

        Example:
            Reference 300-get_images.py
        """
        return self.astribot_interface.deactivate_camera_abs()

    def register_image_callback(self, camera_name: str, image_type: str, callback, need_decode: bool):
        """
        Brief:
            Register callback function to get latest image.

        Args:
            camera_name: camera name
            image_type: depth, color or gray
            callback: callback function which used for register as ROS subscriber callback
            need_decode: flag which to decide if SDK decode JPEG to BGR

        Returns:
            A bool value to show the result of the deactivation.

        Example:
            Reference 300-get_images.py
        """
        return self.astribot_interface.register_image_callback(camera_name, image_type,
                            lambda topic_name, msg, width, height, array: callback(topic_name, msg, width, height, array), need_decode)

    def get_camera_name_from_topic_name(self, topic_name: str) -> str:
        """
        Brief:
            Get the camera name from the topic name.

        Args:
            topic_name: The topic name of the camera image.

        Returns:
            A string of the camera name.
        """
        parts = topic_name.split('/')
        return parts[2]
    
    def get_image_type_from_topic_name(self, topic_name: str) -> str:
        """
        Brief:
            Get the image type from the topic name.

        Args:
            topic_name: The topic name of the camera image.

        Returns:
            A string of the image type.
        """
        parts = topic_name.split('/')
        image_type = parts[3].split('_')
        return image_type[0]

    def get_cameras_info(self):
        """
        Brief:
            Get the cameras information of the robot. (Connected camera, serial number, USB type)
            Only activate status is support at current SDK version.

        Args:
            None

        Returns:
            A dict of cameras information.

        Example:
            output=Astribot.get_cameras_info()

            ---output---
            output: {
                'head_rgbd': {
                    'activate': True,
                    'serial_number': '123456789',
                    'USB_type': 'USB3.0',
                    },
                'left_wrist_rgbd': {
                    'activate': True,
                    'serial_number': '987654321',
                    'USB_type': 'USB3.0',
                    },
                }
                'right_wrist_rgbd': {
                    'activate': True,
                    'serial_number': '123456789',
                    'USB_type': 'USB3.0',
                    },
            }
        """
        camera_name_list = [
            "head_rgbd",
            "torso_rgbd",
            "left_wrist_rgbd",
            "right_wrist_rgbd",
            "head_stereo"  #head_stereo camera did not support depth image type
        ]
        output = {}
        for camera in camera_name_list:
            output[camera] = {}
            # we can only check color topic to decide whether camera is activated or not
            output[camera]['activate'] = self.astribot_interface.is_camera_activated(camera, "color")
        return output

    # Lidar
    def activate_lidar(self):
        """
        Brief:
            Activate the lidar of the robot.

        Args:
            None

        Returns:
            A bool value to show the result of the activation.

        Example:
            Reference 111-get_lidar.py
        """
        return self.astribot_interface.activate_lidar()

    def deactivate_lidar(self):
        """
        Brief:
            Deactivate the lidar of the robot.

        Args:
            None

        Returns:
            A bool value to show the result of the deactivation.

        Example:
            Reference 111-get_lidar.py
        """
        return self.astribot_interface.deactivate_lidar()

    # Microphone and Speaker
    def activate_audio(self,audio_setting: Optional[Dict[str, Any]] = None):
        """
        Brief:
            Activate the audio of the robot.

        Args:
            audio_setting: A dict of audio setting

        Returns:
            A bool value to show the result of the activation.

        Example:
            Reference 302-get_and_play_audio.py
        """
        return self.astribot_interface.activate_audio(audio_setting)
    
    def deactivate_audio(self):
        """
        Brief:
            Deactivate the audio of the robot.

        Args:
            audio_setting: A dict of audio setting

        Returns:
            A bool value to show the result of the deactivation.

        Example:
            Reference 302-get_and_play_audio.py
        """
        return self.astribot_interface.deactivate_audio()

    ### 4. High Level API ###
    def set_head_follow_effector(self, enable:bool=True, arm_name:str="dual"):
        """
        Brief:
            Set the head motion mode to follow the effector or not.

        Args:
            enable: Bool value to determine if the head motion mode is set to follow the effector or not.
            arm_name: A string to determine which arm's effector is followed, can be "dual", "left" or "right". Default is "dual".

        Returns:
            A string to show the result of the setting.

        Example:
            Astribot.set_head_follow_effector(True)
        """
        return self.astribot_interface.set_mode("set_head_follow_effector", enable, arm_name)

    def set_wbc_collision_avoidance(self, enable:bool=True):
        """
        Brief:
            Set the robot motion mode to avoid self-collision or not.

        Args:
            Bool value to determine if the head motion mode is set to avoid self-collision or not.

        Returns:
            A string to show the result of the setting.

        Example:
            Astribot.set_wbc_collision_avoidance(True)
        """
        return self.astribot_interface.set_mode("set_wbc_collision_avoidance", enable)

    def set_filter_parameters(self, filter_scale, gripper_filter_scale):
        """
        Brief:
            Set the filter parameters of the robot. lower is higher smooth.

        Args:
            filter_scale: A value to determine the filter scale
            gripper_filter_scale: A value to determine the gripper filter scale.

        Returns:
            None

        Example:
            Astribot.set_filter_parameters(0.1, 0.5)
        """
        self.astribot_interface.set_filter_parameters(filter_scale, gripper_filter_scale)

    def set_effector_max_force(self, names, max_force):
        """
        Brief:
            Set gripper max force, max_force is changed by each grasp.
            The minimum clamping force is 10N
            The maximum clamping force varies with the robot mode, 40N in safe mode, 60N in standard mode, and 80N in extreme mode.

        Returns:
            None

        Example:
            Astribot.set_effector_max_force(Astribot.effector_names, max_force=[40, 40])
        """
        if self.__in_simulation:
            return
        default_max_force = [None, None]
        for i, name in enumerate(names):
            if name == self.effector_left_name:
                default_max_force[0] = max_force[i]
            if name == self.effector_right_name:
                default_max_force[1] = max_force[i]
        self.astribot_interface.set_effector_max_force(default_max_force)

    def wait_for_robot_ready(self, timeout=10.0):
        """
        Brief:
            Wait for the robot to be ready.

        Args:
            timeout: A float value to determine the timeout time in seconds.

        Returns:
            A bool value to show if the robot is ready or not.
        """
        while not self.astribot_interface.ready:
            time.sleep(0.1)
            timeout -= 0.1
            if timeout <= 0:
                self._logger.error("Robot is not ready, timeout.")
                return False
        self._logger.info("Robot is ready.")
        return True

    def wait_for_interface_alive(self, timeout_s=2):
        while not self.astribot_interface.is_alive():
            time.sleep(0.1)
            timeout_s -= 0.1
            if timeout_s <= 0:
                self._logger.error("Interface is not alive, timeout.")
                return False
        return True

    def __del__(self):
        self.astribot_interface.__del__()
