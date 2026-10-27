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
File: 206-move_joints_waypoints.py
Brief: Follow sequence of joints' waypoints.

Overview
========
1. set a series of waypoints and arrival times
2. call the function to reach the path points in sequence within the set time

Key Points
----------
* No need to give a point at time 0, the default time is 0 at the current position
"""

from astribot_sdk.core.astribot_api.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    
    astribot.move_to_home()

    names = [astribot.torso_name]
    waypoints = list()
    time_list = list()
    waypoints.append([[ 0.5, -1.0,  0.5, 0.0]])
    time_list.append(2.0)
    waypoints.append([[ 0.2, -0.4,  0.2, 0.0]])
    time_list.append(3.5)
    waypoints.append([[ 0.5, -1.0,  0.5, 0.0]])
    time_list.append(6.0)
    waypoints.append([[ 0.2, -0.4,  0.2, 0.0]])
    time_list.append(7.5)

    astribot.move_joints_waypoints(names, waypoints, time_list)
    astribot.move_to_home()
