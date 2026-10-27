#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024, Astribot Co., Ltd.
# All rights reserved.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Astribot Team
# -----------------------------------------------------------------------------

"""
File: 104-set_joint_position.py
Brief: Control the robot to reach the desired joint position in real-time.

Overview
========
1. start a control loop
2. set desired control in each control loop
3. sleep according to the set frequency and then enter the next loop
"""



import copy
import numpy as np
from astribot_sdk.core.astribot_api.astribot_client import Astribot
import astribot_ros_middleware as ast_ros_middleware

if __name__ == '__main__':
    # Connect astribot
    freq = 250.0    # Control frequency of robot (in Hz)
    dt = 1.0 / freq
    astribot = Astribot(freq=freq)

    # Create rate object
    rate = ast_ros_middleware.Rate(freq)

    astribot.move_to_home()

    names = [astribot.arm_left_name, astribot.arm_right_name]
    init_pose = astribot.get_desired_joints_position(names=names)

    t = 0.0
    while ast_ros_middleware.ok():
        command_list = copy.deepcopy(init_pose)
        command_list[0][3] = init_pose[0][3] + 0.3 * np.sin(t)
        command_list[1][3] = init_pose[1][3] + 0.3 * np.sin(t)
        astribot.set_joints_position(names, command_list)
        t += dt
        if t > 2 * np.pi:
            break
        rate.sleep()

    print("Joint position control example finished.")
