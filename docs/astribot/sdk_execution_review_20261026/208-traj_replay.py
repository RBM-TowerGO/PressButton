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
File: 208-traj_replay.py
Brief: Replay a trajectory.

Overview
========
1. read the recorded track file
2. Set the path points and time according to the content in the file
3. call the function to reach the path points in sequence within the set time

Key Points
----------
* If joy_controller=True, user can control forward playback or reverse playback by using the left and right buttons on the top of the handle
"""

import os
import sys
import h5py
from astribot_sdk.core.astribot_api.astribot_client import Astribot
from astribot_ros_middleware import astribot_ros_middleware

if __name__ == '__main__':
    # Connect astribot
    if len(sys.argv) !=2:
        Abs_Path= os.path.abspath(os.path.dirname(__file__))
        path_to_hdf5 = os.path.join(Abs_Path,"example_data","dancing.hdf5")
        print(f"\033[33mNo HDF5 file given — using default: {path_to_hdf5}\033[0m")
    else:
        path_to_hdf5 = sys.argv[1]
        print(f"\033[32mThe path to the hdf5 file is: {path_to_hdf5}\033[0m")

    astribot = Astribot()
    astribot.move_to_home()

    with h5py.File(path_to_hdf5, 'r') as root:
        joints_action_obs = root['joints_dict/joints_position_command'][()].tolist()
        time_obs = root['time'][()].tolist()
        
    names_list = ['astribot_torso', 'astribot_arm_left', 'astribot_gripper_left',
             'astribot_arm_right', 'astribot_gripper_right', 'astribot_head']  
    whole_body_config={
        "astribot_chassis":3,
        "astribot_torso":4,
        "astribot_arm_left":7,
        "astribot_gripper_left":1,
        "astribot_arm_right":7,
        "astribot_gripper_right":1,
        "astribot_head":2
    }
    whole_body_index_dict = {
        "astribot_chassis": (0, 3),
        "astribot_torso": (3, 7),
        "astribot_arm_left": (7, 14),
        "astribot_gripper_left": (14, 15),
        "astribot_arm_right": (15, 22),
        "astribot_gripper_right": (22, 23),
        "astribot_head": (23, 25)
    }

    waypoints, time_list = list(), list()
    last_time = None
    frame_stride = 2  # 新增参数：每隔几帧采样一次，1表示不跳帧

    for index in range(0, len(joints_action_obs), frame_stride):
        action = joints_action_obs[index]
        current_time = time_obs[index]

        # add the time
        if last_time is None:
            time_list.append(2.0)  # 起始时间
        else:
            time_list.append(time_list[-1] + current_time - last_time)
        last_time = current_time

        # add the way points
        waypoint = list()
        for name in names_list:
            waypoint.append(action[whole_body_index_dict[name][0]:whole_body_index_dict[name][1]])
        waypoints.append(waypoint)
        
    while astribot_ros_middleware.ok():
        astribot.move_joints_waypoints(names_list, waypoints, time_list, use_wbc=False, joy_controller=False)

    astribot_ros_middleware.shutdown()
