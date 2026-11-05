#!/usr/bin/env python3
"""S1B ROS/SDK environment strings and local port check shared by the agent and keyboard launchers."""
import shlex
import socket

ENV='; '.join('source '+p for p in ['/opt/astribot_ros/middle_ware/local_setup.bash','/opt/astribot_ros/software/local_setup.bash','/opt/astribot_ros/third_pkg/local_setup.bash','/opt/astribot_ros/robot_system_ctrl/ros_env.sh'])
CAMERA_ENV='export ASTRIBOT_ROS_INSTALL_PATH=/opt/astribot_ros/software ASTRIBOT_CAMERA_INSTALL_PATH=/opt/astribot_ros/software ASTRIBOT_CAMERA_INFO_PATH=/etc/config/camera_info.json'


def check_port_available(port):
    # Match the HTTP/SSH listeners: previous connections in TIME_WAIT should
    # not prevent a new session, but another live listener must still fail.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',port));sock.listen(1)


def sdk_environment(config):
    """SDK dependencies only; preserve the established ROS/DDS environment."""
    root=config['sdk_root']
    python_paths=[root,root+'/third_party/astribot_ros_middleware_py']
    library_paths=[root+'/'+suffix for suffix in (
        'astribot_sdk/core/common/robotics_library_py',
        'astribot_sdk/core/common/whole_body_control/third_party',
        'third_party/drake/lib','third_party/third_pkg')]
    return '; '.join([
        'export ASTRIBOT_SDK_ROOT='+shlex.quote(root),
        'export ROBOT_TYPE="${ROBOT_TYPE:-S1}"',
        'export PYTHONPATH='+shlex.quote(':'.join(python_paths))+':"${PYTHONPATH:-}"',
        'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}":'+shlex.quote(':'.join(library_paths)),
    ])
