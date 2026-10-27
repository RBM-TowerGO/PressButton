#!/usr/bin/env python3
"""Launch split S1B observation, workstation inference and guarded control services."""
import json
import os
from pathlib import Path
import shlex
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/astribot/pressb_readonly_p2'
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


def bridge_failure(log_path):
    """Keep the robot's root cause visible in the workstation terminal."""
    lines=Path(log_path).read_text(errors='replace').splitlines()
    diagnosis=[]
    for line in reversed(lines):
        try:summary=json.loads(line)
        except ValueError:continue
        if not isinstance(summary,dict) or summary.get('event')!='finished':continue
        if summary.get('execution_fault') not in (None,'operator_stop','control_loop_shutdown','bridge_shutdown'):
            diagnosis.append('execution_fault='+str(summary['execution_fault']))
        diagnosis.extend(str(reason) for reason in summary.get('fatal',[]))
        feedback=summary.get('stop_feedback')
        if feedback and feedback.get('error'):
            names=feedback['joint_names']
            diagnosis.append('stop_feedback='+json.dumps(dict(
                samples=feedback['sample_count'],max_feedback_gap_ms=feedback.get('max_feedback_gap_ns',0)/1e6,
                position_violations=[names[i] for i in feedback.get('position_violations',[])],
                velocity_violations=[names[i] for i in feedback.get('velocity_violations',[])])))
        break
    # Put the actionable fault first; a large JSON summary or repeated SDK
    # warnings may otherwise hide it when terminal output is truncated.
    tail=[line for line in lines[-24:] if not line.startswith('{')]
    return 'Robot bridge failed; '+str(log_path)+'\n'+'\n'.join(diagnosis+tail[-12:])


def wait_for_start(check_processes, stream=None):
    """Wait for a fresh S key without Enter; restore terminal on every exit."""
    import select
    import termios
    import tty
    stream = sys.stdin if stream is None else stream
    if not stream.isatty():
        raise RuntimeError('需要交互终端：模型加载后按 S 才能开始。')
    fd = stream.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        print('模型已就绪。按 S 开始（无需回车）；Ctrl-C 取消。', flush=True)
        while True:
            check_processes()
            if select.select([fd], [], [], .2)[0]:
                key = os.read(fd, 1)
                if not key:
                    raise RuntimeError('终端输入已关闭，取消启动。')
                if key in (b'S', b's'):
                    check_processes()
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def main():
    from s1b_split_launch import main as split_main
    split_main()


if __name__=='__main__':main()
