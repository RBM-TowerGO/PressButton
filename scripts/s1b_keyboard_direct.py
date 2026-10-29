#!/usr/bin/env python3
"""Direct SDK Cartesian keyboard control, optionally launched on S1B over SSH.

No offline calibration file or planar-service verification flags are consumed.
The startup SDK end-effector pose defines the plane; orientation stays fixed.
"""
import argparse
import builtins
import copy
import math
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
import uuid

# Native SDK destructors can abort during Python interpreter teardown. Keep the
# client alive until the standalone robot process exits after explicit cleanup.
_SDK_KEEPALIVE = []


def connect_robot(factory, frequency):
    original_input = builtins.input

    def interactive_answer(*args, **kwargs):
        try:
            return original_input(*args, **kwargs)
        except (EOFError, KeyboardInterrupt):
            # The SDK's broad exception fallback can acquire rights on EOF.
            # A disconnected terminal must instead decline ownership.
            return ''

    builtins.input = interactive_answer
    try:
        return factory(freq=frequency, high_control_rights=False, node_name='pressb_keyboard_direct')
    finally:
        builtins.input = original_input


def target_for_key(key, pose, initial_x, step_m):
    target = list(pose)
    if key == 'w':target[2] += step_m
    elif key == 's':target[2] -= step_m
    elif key == 'a':target[1] += step_m
    elif key == 'd':target[1] -= step_m
    elif key == 'f':target[0] = initial_x + .1
    elif key == 'r':target[0] = initial_x
    else:return None
    return target


GRIPPER_KEYS = {'c': 100., 'o': 0.}


def send_targets(robot, names, current, torso, gripper_target):
    robot.set_cartesian_pose(names, [list(current), torso], control_way='filter',
        use_wbc=False, add_default_torso=False, remote_wbc_option=False)
    if gripper_target is not None:
        robot.set_joints_position(['astribot_gripper_right'], [[gripper_target]],
            control_way='filter', use_wbc=False, add_default_torso=False)


def launch_remote(args):
    from start_s1b_inference import ENV, sdk_environment
    remote = '/tmp/pressb_keyboard_direct_'+uuid.uuid4().hex+'.py'
    subprocess.run(['scp','-q',str(Path(__file__).resolve()),args.host+':'+remote],check=True)
    sdk_env = sdk_environment(dict(sdk_root=args.sdk_root))
    command = ('trap '+shlex.quote('rm -f '+shlex.quote(remote))+' EXIT; '+ENV+'; '+sdk_env+'; '+
               'python3 '+shlex.quote(remote)+' --step-mm '+shlex.quote(str(args.step_mm))+
               ' --speed-mm-s '+shlex.quote(str(args.speed_mm_s)))
    return subprocess.call(['ssh','-t',args.host,'bash -lc '+shlex.quote(command)])


def run_robot(args):
    from astribot_sdk.core.astribot_api.astribot_client import Astribot
    frequency = 250.
    robot = connect_robot(Astribot, frequency)
    _SDK_KEEPALIVE.append(robot)
    stop = threading.Event()
    lock = threading.RLock()
    errors = []
    current = goal = start = None
    started = duration = 0.
    busy = False
    gripper_target = None  # Preserve the existing opening until C or O is pressed.
    thread = None
    saved = None
    fd = sys.stdin.fileno()

    def stop_robot(*unused):
        stop.set()
        with lock:
            if robot.get_control_rights_status():
                robot.stop_robot()

    try:
        if not robot.get_control_rights_status():
            raise RuntimeError('未取得控制权。重新运行后，可在 SDK 接管提示中输入 yes；或先退出原控制程序。')
        robot.set_head_follow_effector(False)
        names = ['astribot_arm_right', 'astribot_torso']
        poses = robot.get_current_cartesian_pose(names, frame='chassis')
        current, torso = copy.deepcopy(poses)
        initial_x = current[0]
        goal = list(current)
        start = list(current)

        def servo():
            nonlocal current, busy
            deadline = time.monotonic()
            try:
                while not stop.is_set():
                    with lock:
                        if stop.is_set():break
                        if not robot.get_control_rights_status():
                            raise RuntimeError('SDK control rights lost')
                        if busy:
                            s = min(1., max(0., (time.monotonic()-started)/duration))
                            blend = 10*s**3-15*s**4+6*s**5
                            current = [start[i]+blend*(goal[i]-start[i]) for i in range(3)] + start[3:]
                            if s >= 1:busy = False
                        send_targets(robot, names, current, torso, gripper_target)
                    deadline += 1/frequency
                    # Skip missed clock ticks rather than burst-send old targets.
                    if deadline < time.monotonic():deadline = time.monotonic()
                    stop.wait(max(0., deadline-time.monotonic()))
            except BaseException as exc:
                errors.append(str(exc))
                stop_robot()

        signal.signal(signal.SIGINT, stop_robot)
        signal.signal(signal.SIGTERM, stop_robot)
        signal.signal(signal.SIGHUP, stop_robot)
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        print(f'\r\n直连 SDK：W/S 上下，A/D 左右，每次 {args.step_mm:g} mm。',flush=True)
        print('F 前伸至初始平面前 100 mm 并保持；R 返回初始平面。空格/Q/Ctrl-C 停止退出。',flush=True)
        print('C 闭合右夹爪；O 打开右夹爪（移动时也可操作）。',flush=True)
        print('采用机器人底盘坐标：+X 前、+Y 左、+Z 上。末端朝向保持原状。',flush=True)
        thread = threading.Thread(target=servo,daemon=True)
        thread.start()
        while not stop.is_set():
            if not select.select([fd],[],[],.05)[0]:continue
            keys = os.read(fd,64).decode(errors='ignore').lower()
            if not keys or any(k in keys for k in (' ', 'q', '\x03')):
                stop_robot();break
            if '\x1b' in keys:continue
            actions = [key for key in keys if key in 'wasdfrco']
            if not actions:continue
            with lock:
                key = actions[-1]
                if key in GRIPPER_KEYS:
                    gripper_target = GRIPPER_KEYS[key]
                    print('\r\n右夹爪：'+('闭合' if key == 'c' else '打开'),flush=True)
                    continue
                if busy:
                    print('\r\n正在移动；空格停止。',flush=True)
                    continue
                # Lateral jogging belongs to the initial plane, never the extended plane.
                if key in 'wasd' and abs(current[0]-initial_x) > 1e-6:
                    print('\r\n先按 R 返回平面，再上下左右移动。',flush=True)
                    continue
                target = target_for_key(key,current,initial_x,args.step_mm/1000)
                start, goal = list(current), target
                distance = math.dist(start[:3],goal[:3])
                duration = max(.2,1.875*distance/(args.speed_mm_s/1000))
                started = time.monotonic()
                busy = True
                print('\r\n目标 XYZ (m): '+str([round(x,4) for x in goal[:3]]),flush=True)
    finally:
        # Restore the terminal even if the native SDK stop raises.
        if saved is not None:termios.tcsetattr(fd,termios.TCSADRAIN,saved)
        try:
            stop_robot()
        finally:
            if thread:thread.join(timeout=2)
            if thread and thread.is_alive():
                raise RuntimeError('SDK call did not return after stop')
            interface = robot.astribot_interface
            if robot.get_control_rights_status():interface.shutdown_control_rights_srv()
            interface.executor.shutdown(timeout_sec=2.)
            import rclpy
            if rclpy.ok():rclpy.shutdown()
            interface.spin_thread.join(timeout=2.)
            if interface.spin_thread.is_alive():
                raise RuntimeError('SDK ROS spin thread did not stop')
            interface.shutdown()
    if errors:raise RuntimeError(errors[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', help='copy this standalone script and open a remote TTY, e.g. S1B')
    parser.add_argument('--sdk-root', default='/home/astribot/astribot_sdk_aarch64')
    parser.add_argument('--step-mm', type=float, default=2.)
    parser.add_argument('--speed-mm-s', type=float, default=20.)
    args = parser.parse_args()
    if not sys.stdin.isatty():parser.error('需要交互终端')
    if any(not math.isfinite(v) or v<=0 for v in (args.step_mm,args.speed_mm_s)):
        parser.error('step and speed must be positive finite values')
    if args.host:
        raise SystemExit(launch_remote(args))
    code = 0
    try:
        run_robot(args)
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print('键盘控制退出：'+str(exc), file=sys.stderr, flush=True)
        code = 1
    # run_robot has stopped/released control and shut down ROS. Avoid the
    # vendor's second destruction of already-destroyed nodes/native threads.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == '__main__':
    main()
