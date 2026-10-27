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
File: 103-move_to_joint_position.py
Brief: Move the robot from its current joints position to the target joints position with blocked motion mode.

Overview
========
1. set the target joints position
2. check if the target joints position exceeds the limit
3. move to target joints position

Key Points
----------
* Units: joints position 'rad'
"""

from astribot_sdk.core.astribot_api.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    
    astribot.move_to_home()

    # The unit of head, torso, left arm and right arm is rad, and the unit of the effector is dimensionless, 100 means fully closed, 0 means fully open
    targets = {
        astribot.head_name           : [0.1, 0.2],
        astribot.torso_name          : [0.29, -0.58, 0.29, 0.0],
        astribot.arm_left_name       : [0.1754, -0.4918, -1.4068, 1.5557, -0.8239, 0.2965, 0.0410],
        astribot.arm_right_name      : [-0.1754, -0.4918,  1.4068, 1.5557,  0.8239, 0.2965,-0.0410],
        astribot.effector_left_name  : [50.0],
        astribot.effector_right_name : [50.0],
    }
    names, command_list = zip(*targets.items())
    
    # Check if the desired joint position exceeds the joint limits
    lower_limit, upper_limit = astribot.get_joints_position_limit(list(names))
    result = all(a > b for row1, row2 in zip(command_list, lower_limit) for a, b in zip(row1, row2)) \
        and all(a > b for row1, row2 in zip(upper_limit, command_list) for a, b in zip(row1, row2))
    if result:
        astribot.move_joints_position(list(names), list(command_list), duration=3.0)
    else:
        print("The desired joint position exceeds the joint limit.")
