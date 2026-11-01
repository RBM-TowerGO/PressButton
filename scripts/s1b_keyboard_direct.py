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
import json
from http.client import HTTPConnection, HTTPException

# Native SDK destructors can abort during Python interpreter teardown. Keep the
# client alive until the standalone robot process exits after explicit cleanup.
_SDK_KEEPALIVE = []


class SDKOutput:
    """Fold only the known repeated native warning, including C++ fd writes."""
    marker = b'calling frequency of the set function'

    def __init__(self):
        self.count = 0
        sys.stdout.flush()
        sys.stderr.flush()
        self.out = os.dup(1)
        self.err = os.dup(2)
        reader, writer = os.pipe()
        os.dup2(writer, 1)
        os.dup2(writer, 2)
        os.close(writer)
        self.reader = reader
        self.thread = threading.Thread(target=self.drain, daemon=True)
        self.thread.start()

    def drain(self):
        with os.fdopen(self.reader, 'rb') as source:
            for line in source:
                if self.marker in line:
                    self.count += 1
                    if self.count > 1:
                        continue
                    line += '重复的 SDK 频率警告将折叠，其他日志正常显示。\n'.encode()
                os.write(self.out, line)

    def close(self):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.out, 1)
        os.dup2(self.err, 2)
        self.thread.join(timeout=2)
        if self.count > 1:
            os.write(self.out, f'本次折叠 {self.count-1} 条重复频率警告。\n'.encode())
        if not self.thread.is_alive():
            os.close(self.out)
            os.close(self.err)


def connect_robot(factory, frequency):
    original_input = builtins.input

    def interactive_answer(*args, **kwargs):
        try:
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            while True:
                answer = original_input(*args, **kwargs).strip().lower()
                if answer in ('yes', '', 'no', 'n'):
                    return 'yes' if answer == 'yes' else ''
                print('请输入完整的 yes 后回车以接管，或直接回车取消；尚未进入移动模式。',flush=True)
        except (EOFError, KeyboardInterrupt, OSError, termios.error):
            # The SDK's broad exception fallback can acquire rights on EOF.
            # A disconnected terminal must instead decline ownership.
            return ''

    builtins.input = interactive_answer
    try:
        return factory(freq=frequency, high_control_rights=False, node_name='pressb_keyboard_direct')
    finally:
        builtins.input = original_input


def target_for_key(key, pose, initial_x, step_m, forward_mm=None):
    target = list(pose)
    if key == 'w':target[2] += step_m
    elif key == 's':target[2] -= step_m
    elif key == 'a':target[1] += step_m
    elif key == 'd':target[1] -= step_m
    elif key == 'f':
        distance = float(forward_mm) if forward_mm is not None else float('nan')
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError('前伸距离必须是大于 0 的有限数字（mm）')
        target[0] = initial_x + distance/1000
    elif key == 'r':target[0] = initial_x
    else:return None
    return target


GRIPPER_KEYS = {'c': 100., 'o': 0.}


def interpolate_pose(start, goal, elapsed, duration):
    fraction = min(1., max(0., elapsed/duration))
    blend = 10*fraction**3-15*fraction**4+6*fraction**5
    return [start[i]+blend*(goal[i]-start[i]) for i in range(3)]+start[3:], fraction >= 1.


def send_targets(robot, names, current, torso, gripper_target):
    if gripper_target is None:
        robot.set_cartesian_pose(names, [list(current), torso], control_way='filter',
            use_wbc=False, add_default_torso=False, remote_wbc_option=False)
    else:
        # SDK example 110: mixed Cartesian/joint targets in ONE periodic set.
        robot.set_different_type_command(list(names)+['astribot_gripper_right'],
            ['cartesian', 'cartesian', 'joints'], [list(current), torso, [gripper_target]],
            control_way='filter', use_wbc=False)


def run_cartesian_loop(robot, frequency, stop, lock, targets, on_error, on_sent=None):
    """The keyboard command loop; callers only supply the current targets."""
    deadline = time.monotonic()
    try:
        while not stop.is_set():
            with lock:
                if stop.is_set():break
                if not robot.get_control_rights_status():
                    raise RuntimeError('SDK control rights lost')
                send_targets(robot, *targets())
                if on_sent:on_sent()
            deadline += 1/frequency
            if deadline < time.monotonic():deadline = time.monotonic()+1/frequency
            stop.wait(max(0., deadline-time.monotonic()))
    except BaseException as exc:
        on_error(exc)


def launch_remote(args):
    from start_s1b_inference import ENV, sdk_environment, check_port_available
    remote_dir = '/tmp/pressb_keyboard_'+uuid.uuid4().hex
    remote = remote_dir+'/s1b_keyboard_direct.py'
    sources = [str(Path(__file__).resolve())]
    forwarding = []
    if args.view_cameras:
        check_port_available(args.view_port)
        sources += [str(Path(__file__).with_name(name).resolve()) for name in ('s1b_camera_view.py','s1b_depth_metadata.cpp')]
        forwarding = ['-o','ExitOnForwardFailure=yes','-L',f'127.0.0.1:{args.view_port}:127.0.0.1:{args.view_port}']
    subprocess.run(['ssh',args.host,'mkdir -p '+shlex.quote(remote_dir)],check=True)
    subprocess.run(['scp','-q',*sources,args.host+':'+remote_dir+'/'],check=True)
    sdk_env = sdk_environment(dict(sdk_root=args.sdk_root))
    command = ('trap '+shlex.quote('rm -rf '+shlex.quote(remote_dir))+' EXIT; '+ENV+'; '+sdk_env+'; '+
               'python3 '+shlex.quote(remote)+' --step-mm '+shlex.quote(str(args.step_mm))+
               ' --speed-mm-s '+shlex.quote(str(args.speed_mm_s))+
               ' --control-hz '+shlex.quote(str(args.control_hz)))
    if args.view_cameras:
        command += ' --view-cameras --view-port '+str(args.view_port)
        print('正在启动相机网页，请等待“网页已就绪”提示。',flush=True)
    return run_ssh(['ssh','-t',*forwarding,args.host,'bash -lc '+shlex.quote(command)])


def run_ssh(command):
    """Hide only refused browser channels while the service starts/stops."""
    process=subprocess.Popen(command,stderr=subprocess.PIPE)
    def stderr():
        for line in process.stderr:
            if b'open failed: connect failed: Connection refused' in line:
                continue
            sys.stderr.buffer.write(line);sys.stderr.buffer.flush()
    reader=threading.Thread(target=stderr,daemon=True);reader.start()
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=4)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        reader.join(timeout=2)
        process.stderr.close()


def wait_for_viewer(viewer, port, timeout=20.):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if viewer.poll() is not None:
            raise RuntimeError('相机网页进程启动失败，退出码 '+str(viewer.returncode))
        connection=HTTPConnection('127.0.0.1',port,timeout=.3)
        try:
            connection.request('GET','/health')
            response=connection.getresponse()
            data=json.loads(response.read())
            if response.status==200 and data.get('service')=='s1b-camera-view-v1' and data.get('pid')==viewer.pid:
                return
        except (OSError, ValueError, HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(.1)
    raise RuntimeError('相机网页启动超时，端口尚未就绪')


def read_forward_distance(fd, terminal_settings, stop):
    """Temporarily use normal terminal line editing; never hold the servo lock."""
    termios.tcsetattr(fd,termios.TCSANOW,terminal_settings)
    try:
        while not stop.is_set():
            print('\n前伸距离（mm，相对初始平面；回车执行，直接回车取消）：',end='',flush=True)
            line=b''
            while not stop.is_set():
                if not select.select([fd],[],[],.05)[0]:continue
                chunk=os.read(fd,4096)
                if not chunk:return None
                line+=chunk
                if b'\n' in line or b'\r' in line:break
            if stop.is_set():return None
            value=line.decode(errors='replace').strip()
            if not value:
                print('已取消。',flush=True)
                return None
            try:
                distance=float(value)
                if math.isfinite(distance) and distance>0:return distance
            except ValueError:pass
            print('请输入大于 0 的数字，单位 mm，例如 35。',flush=True)
    finally:
        tty.setcbreak(fd)


def run_robot(args, viewer=None):
    from astribot_sdk.core.astribot_api.astribot_client import Astribot
    frequency = args.control_hz
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

        def targets():
            nonlocal current, busy
            if busy:
                current, complete = interpolate_pose(start, goal, time.monotonic()-started, duration)
                if complete:busy = False
            return names, current, torso, gripper_target

        def failed(exc):
            errors.append(str(exc));stop_robot()

        def servo():
            run_cartesian_loop(robot, frequency, stop, lock, targets, failed)

        signal.signal(signal.SIGINT, stop_robot)
        signal.signal(signal.SIGTERM, stop_robot)
        signal.signal(signal.SIGHUP, stop_robot)
        saved = termios.tcgetattr(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        tty.setcbreak(fd)
        print(f'\r\n直连 SDK：W/S 上下，A/D 左右，每次 {args.step_mm:g} mm。',flush=True)
        print('F 输入前伸距离（mm，相对初始平面），回车执行；R 返回初始平面。空格/Q/Ctrl-C 停止退出。',flush=True)
        print('C 闭合右夹爪；O 打开右夹爪（移动时也可操作）。',flush=True)
        print('采用机器人底盘坐标：+X 前、+Y 左、+Z 上。末端朝向保持原状。',flush=True)
        thread = threading.Thread(target=servo,daemon=True)
        thread.start()
        while not stop.is_set():
            if viewer is not None and viewer.poll() is not None:
                print('\r\n相机网页进程已退出，键盘控制继续；请检查 camera_view.log。',flush=True)
                viewer = None
            if not select.select([fd],[],[],.05)[0]:continue
            keys = os.read(fd,1).decode(errors='ignore').lower()
            if not keys or any(k in keys for k in (' ', 'q', '\x03')):
                stop_robot();break
            for key in keys:
                if key not in 'wasdfrco':continue
                forward_mm = None
                if key == 'f':
                    with lock:
                        if busy:
                            print('\r\n正在移动，请稍后按 F。',flush=True)
                            continue
                    forward_mm=read_forward_distance(fd,saved,stop)
                    if forward_mm is None or stop.is_set():continue
                with lock:
                    if key in GRIPPER_KEYS:
                        gripper_target = GRIPPER_KEYS[key]
                        print('\r\n右夹爪：'+('闭合' if key == 'c' else '打开'),flush=True)
                        continue
                    if busy:
                        print('\r\n正在移动，请稍后按键。',flush=True)
                        continue
                    # Lateral jogging belongs to the initial plane, never the extended plane.
                    if key in 'wasd' and abs(current[0]-initial_x) > 1e-6:
                        print('\r\n先按 R 返回平面，再上下左右移动。',flush=True)
                        continue
                    target = target_for_key(key,current,initial_x,args.step_mm/1000,forward_mm)
                    start, goal = list(current), target
                    distance = math.dist(start[:3],goal[:3])
                    duration = max(.2,1.875*distance/(args.speed_mm_s/1000))
                    started = time.monotonic()
                    busy = True
                    print('\r\n目标 XYZ（mm）：'+str([round(x*1000,1) for x in goal[:3]]),flush=True)
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
    parser.add_argument('--control-hz', type=float, default=250., help='SDK and command loop frequency; independent of movement speed')
    parser.add_argument('--view-cameras', action='store_true', help='show head/right RGB and aligned right centre depth on a local webpage')
    parser.add_argument('--view-port', type=int, default=8765, help='local webpage/SSH tunnel port')
    args = parser.parse_args()
    if not sys.stdin.isatty():parser.error('需要交互终端')
    if any(not math.isfinite(v) or v<=0 for v in (args.step_mm,args.speed_mm_s,args.control_hz)):
        parser.error('step, speed and frequency must be positive finite values')
    if not 1<=args.view_port<=65535:parser.error('invalid view port')
    if args.host:
        raise SystemExit(launch_remote(args))
    code = 0
    output = SDKOutput()
    viewer = viewer_log = None
    try:
        if args.view_cameras:
            viewer_script=Path(__file__).with_name('s1b_camera_view.py')
            viewer_log=Path(__file__).with_name('camera_view.log').open('w')
            viewer=subprocess.Popen([sys.executable,str(viewer_script),'--port',str(args.view_port)],
                                    stdin=subprocess.DEVNULL,stdout=viewer_log,stderr=viewer_log)
            wait_for_viewer(viewer,args.view_port)
            print(f'网页已就绪：http://127.0.0.1:{args.view_port}',flush=True)
        run_robot(args, viewer)
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print('键盘控制退出：'+str(exc), file=sys.stderr, flush=True)
        code = 1
    finally:
        if viewer is not None:
            viewer.terminate()
            try:viewer.wait(timeout=4)
            except subprocess.TimeoutExpired:
                viewer.kill();viewer.wait(timeout=2)
            if viewer.returncode not in (0,-signal.SIGTERM):
                viewer_log.flush()
                print(Path(__file__).with_name('camera_view.log').read_text()[-3000:],file=sys.stderr,flush=True)
        if viewer_log is not None:viewer_log.close()
        output.close()
    # run_robot has stopped/released control and shut down ROS. Avoid the
    # vendor's second destruction of already-destroyed nodes/native threads.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == '__main__':
    main()
