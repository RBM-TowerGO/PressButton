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
File: 999.stop_robot.py
Brief: Code for Stop Astribot Robotics.
"""

import time
import threading
from astribot_sdk.core.astribot_api.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()

    #Robot should in like pack pose or others, so that robot can move
    move_to_home_thread = threading.Thread(target=astribot.move_to_home(duration=5.0))
    move_to_home_thread.start()

    time.sleep(2.5)

    print(astribot.stop_robot())

    time.sleep(2.5)

    print(astribot.restart_robot())
    astribot.move_to_home(duration=2.0)
