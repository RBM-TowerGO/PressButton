#!/usr/bin/env python3
"""Direct SDK Cartesian keyboard control, optionally launched on S1B over SSH.

No offline calibration file or planar-service verification flags are consumed.
Each round accepts XYZ displacements in mm; orientation stays fixed.
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


from s1b_keyboard_control import (
    GRIPPER_KEYS, interpolate_pose, interpolate_head, target_for_xyz,
    send_targets, run_cartesian_loop, CommandExecutor, keyboard_command,
)


def launch_remote(args):
    from start_s1b_inference import ENV, sdk_environment, check_port_available
    remote_dir = '/tmp/pressb_keyboard_'+uuid.uuid4().hex
    remote = remote_dir+'/s1b_keyboard_direct.py'
    sources = [str(Path(__file__).resolve()), str(Path(__file__).with_name("s1b_keyboard_control.py"))]
    recording=getattr(args,'record',False)
    if recording:
        args.view_cameras=True
        sources.append(str(Path(__file__).with_name('s1b_keyboard_record.py')))
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
               ' --control-hz '+shlex.quote(str(args.control_hz))+
               ' --head-speed-deg-s '+shlex.quote(str(getattr(args,'head_speed_deg_s',20.))))
    if args.view_cameras:
        command += ' --view-cameras --view-port '+str(args.view_port)
        print('正在启动相机网页，请等待“网页已就绪”提示。',flush=True)
    if recording:
        remote_record='~/.local/share/pressb/keyboard_sessions/'+uuid.uuid4().hex
        command += ' --record --record-dir '+shlex.quote(remote_record)
        local_record=Path(args.record_dir).expanduser()
        if local_record.exists():raise FileExistsError('记录目录已存在：'+str(local_record))
        try:local_record.mkdir(parents=True)
        except OSError as exc:print('本地记录目录暂不可写：'+str(exc),file=sys.stderr,flush=True)
        print('机器人持久记录目录：'+args.host+':'+remote_record,flush=True)
    try:
        return run_ssh(['ssh','-t',*forwarding,args.host,'bash -lc '+shlex.quote(command)])
    finally:
        if recording:
            try:
                local_record.mkdir(parents=True,exist_ok=True)
                # scp rejects a remote basename of '.' ("unexpected filename").
                # Recording artifacts have non-hidden names; copy their contents.
                result=subprocess.run(['scp','-rq',args.host+':'+remote_record+'/*',str(local_record)+'/'],check=False)
                if result.returncode:raise RuntimeError('scp exit '+str(result.returncode))
                print('记录已复制到：'+str(local_record),flush=True)
            except Exception as exc:
                print('记录复制失败：'+str(exc)+'；机器人原件保留在 '+args.host+':'+remote_record+
                      '，可用 scp -r 手动复制。',file=sys.stderr,flush=True)



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


def read_line(fd, stop, on_input=None):
    """Canonical terminal input, interruptible by the existing stop handler."""
    line=b''
    while not stop.is_set():
        if not select.select([fd],[],[],.05)[0]:continue
        chunk=os.read(fd,4096)
        if not chunk:
            if on_input:on_input('')
            return 'q'
        line+=chunk
        if b'\n' in line or b'\r' in line:
            raw=line.decode(errors='replace')
            if on_input:on_input(raw)
            return raw.strip().lower()
    return 'q'


def read_xyz(fd, stop, recorder=None):
    values=[]
    for axis,direction in (('X','前'),('Y','左'),('Z','上')):
        while not stop.is_set():
            # SDKOutput forwards complete lines, so prompts must end in newline.
            print(f'{axis} 位移（mm，正数向{direction}）：',flush=True)
            value=(read_line(fd,stop,lambda raw:recorder.event('input' if raw else 'input_eof',axis=axis,raw=raw))
                   if recorder else read_line(fd,stop))
            if value=='q':return 'q'
            if not values and value in GRIPPER_KEYS:return value
            if not values and value=='h':return read_head_angles(fd,stop,recorder)
            if not value:
                if recorder:recorder.event('input_cancelled',partial_xyz_mm=list(values))
                print('已取消本轮。',flush=True)
                return None
            try:
                distance=float(value)
                if math.isfinite(distance):
                    values.append(distance);break
            except ValueError:pass
            if recorder:recorder.event('invalid_input',axis=axis,value=value)
            print('请输入有限数字，单位 mm；支持正数、负数和 0。',flush=True)
        if stop.is_set():return 'q'
    return values


def read_head_angles(fd, stop, recorder=None):
    angles=[]
    for axis in ('头部左右转角 yaw','头部俯仰角 pitch'):
        while not stop.is_set():
            print(f'{axis}（度，绝对角度；空输入取消）：',flush=True)
            value=(read_line(fd,stop,lambda raw:recorder.event('input',axis=axis,raw=raw))
                   if recorder else read_line(fd,stop))
            if value=='q':return 'q'
            if not value:
                if recorder:recorder.event('input_cancelled',partial_head_deg=angles)
                print('已取消头部设置。',flush=True)
                return None
            try:
                angle=float(value)
                if math.isfinite(angle):
                    angles.append(angle);break
            except ValueError:pass
            if recorder:recorder.event('invalid_input',axis=axis,value=value)
            print('请输入有限数字，单位为度。',flush=True)
        if stop.is_set():return 'q'
    return dict(head_deg=angles)



def run_robot(args, viewer=None):
    from astribot_sdk.core.astribot_api.astribot_client import Astribot
    robot = connect_robot(Astribot, args.control_hz)
    _SDK_KEEPALIVE.append(robot)
    stop = threading.Event()
    executor = None
    saved = None
    fd = sys.stdin.fileno()
    recorder = getattr(args, 'recorder', None)
    task = None
    stop_reasons = []

    def stop_robot(*unused):
        if not stop.is_set():
            stop_reasons.append('signal_'+str(unused[0]) if unused else 'operator_stop')
        stop.set()
        if executor:
            executor.stop()
        elif robot.get_control_rights_status():
            robot.stop_robot()

    def drain_input(timeout=0.):
        if select.select([fd], [], [], timeout)[0]:
            value = os.read(fd, 4096)
            quitting = not value or value.strip().lower() == b'q'
            if quitting:
                stop_robot()
            if recorder:
                recorder.event('stop_input' if quitting else 'ignored_input', raw=value.decode(errors='replace'))
            return True
        return False

    try:
        if not robot.get_control_rights_status():
            raise RuntimeError('未取得控制权。重新运行后，可在 SDK 接管提示中输入 yes；或先退出原控制程序。')
        robot.set_head_follow_effector(False)
        signal.signal(signal.SIGINT, stop_robot)
        signal.signal(signal.SIGTERM, stop_robot)
        signal.signal(signal.SIGHUP, stop_robot)
        saved = termios.tcgetattr(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        executor = CommandExecutor(robot, args.control_hz, args.speed_mm_s,
                                   getattr(args, 'head_speed_deg_s', 20.), stop_event=stop)
        # The terminal and recorder use the executor's cancellation event.
        stop = executor.stop_event
        print('每轮依次输入 X、Y、Z 位移（mm），Z 回车后一次执行。', flush=True)
        print('空输入取消本轮；q 回车或 Ctrl-C 停止退出；X 提示处 h 设置头部角度，c/o 回车闭合/打开夹爪。', flush=True)
        print('底盘坐标：+X 前、+Y 左、+Z 上；位移从上一命令目标累加，朝向不变。', flush=True)
        if recorder:
            recorder.session['initial'] = recorder.capture('initial', cancel=stop)
        while not stop.is_set():
            if viewer is not None and viewer.poll() is not None:
                print('相机网页进程已退出，键盘控制继续；请检查 camera_view.log。', flush=True)
                viewer = None
            value = read_xyz(fd, stop, recorder) if recorder else read_xyz(fd, stop)
            if value == 'q' or stop.is_set():
                stop_robot()
                break
            if value is None:
                continue
            command = keyboard_command(value)
            kind = command['type']
            prepared = executor.prepare(command)
            task = None
            if recorder:
                detail = {k: v for k, v in command.items() if k != 'type'}
                if kind == 'head':
                    detail.update(start_head_deg=[math.degrees(v) for v in prepared['head_start']],
                                  head_speed_deg_s=executor.head_speed_deg_s)
                elif kind == 'gripper':
                    detail['target'] = GRIPPER_KEYS[command['key']]
                recorder.begin(kind, detail, prepared['start'], prepared['goal'],
                               None if kind == 'head' else args.speed_mm_s, stop)
                while not stop.is_set() and drain_input():
                    pass
                if stop.is_set():
                    break
            task = executor.start(prepared)
            if recorder:
                recorder.started(task['started'])
            if kind == 'gripper':
                print('右夹爪：'+('闭合' if command['key'] == 'c' else '打开'), flush=True)
            elif kind == 'head':
                print('头部目标 yaw/pitch（度）：'+str(command['head_deg']), flush=True)
            else:
                print('本轮位移 XYZ（mm）：'+str(command['xyz_mm']), flush=True)
            while not stop.is_set() and not task['done'].is_set():
                drain_input(.02)
            if recorder:
                recorder.event('pending_input_discarded')
                recorder.finish('completed' if task['ended'] else 'interrupted', task['ended'],
                                reason=executor.error or ('operator_stop' if stop.is_set() else None), cancel=stop)
                while not stop.is_set() and drain_input():
                    pass
            if kind == 'gripper' and not recorder:
                continue
            termios.tcflush(fd, termios.TCIFLUSH)
            if not stop.is_set():
                print('本轮指令发送完成。' if kind == 'gripper' else '本轮插值指令发送完成。', flush=True)
    finally:
        if saved is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        try:
            stop_robot()
        finally:
            if executor:
                executor.close()
            if recorder:
                recorder.event('control_stopped', reasons=stop_reasons,
                               errors=[executor.error] if executor and executor.error else [])
                if recorder.active:
                    ended = task['ended'] if task else None
                    recorder.finish('completed' if ended else 'interrupted', ended,
                                    reason=executor.error if executor and executor.error else 'operator_stop')
            interface = robot.astribot_interface
            if robot.get_control_rights_status():
                interface.shutdown_control_rights_srv()
            interface.executor.shutdown(timeout_sec=2.)
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
            interface.spin_thread.join(timeout=2.)
            if interface.spin_thread.is_alive():
                raise RuntimeError('SDK ROS spin thread did not stop')
            interface.shutdown()
    if executor and executor.error:
        raise RuntimeError(executor.error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', help='copy this standalone script and open a remote TTY, e.g. S1B')
    parser.add_argument('--sdk-root', default='/home/astribot/astribot_sdk_aarch64')
    parser.add_argument('--step-mm', type=float, default=2., help='compatibility only; ignored by XYZ input')
    parser.add_argument('--speed-mm-s', type=float, default=20.)
    parser.add_argument('--head-speed-deg-s',type=float,default=20.,help='head interpolation peak speed in degrees/s')
    parser.add_argument('--control-hz', type=float, default=250., help='SDK and command loop frequency; independent of movement speed')
    parser.add_argument('--record',action='store_true',help='save operations and before/after camera captures')
    parser.add_argument('--record-dir',help='new recording directory; implies --record')
    parser.add_argument('--view-cameras', action='store_true', help='show head/right RGB and aligned right centre depth on a local webpage')
    parser.add_argument('--view-port', type=int, default=8765, help='local webpage/SSH tunnel port')
    args = parser.parse_args()
    if not sys.stdin.isatty():parser.error('需要交互终端')
    if any(not math.isfinite(v) or v<=0 for v in (args.speed_mm_s,args.control_hz,args.head_speed_deg_s)):
        parser.error('speed and frequency must be positive finite values')
    if not 1<=args.view_port<=65535:parser.error('invalid view port')
    args.record=args.record or args.record_dir is not None
    if args.record:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        args.view_cameras=True
        args.record_dir=args.record_dir or str(Path('outputs/keyboard_sessions')/(
            datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]))
        if Path(args.record_dir).expanduser().exists():parser.error('记录目录已存在：'+args.record_dir)
    if args.host:
        raise SystemExit(launch_remote(args))
    code = 0
    output = SDKOutput()
    viewer = viewer_log = None
    args.recorder=None;exit_reason='operator_exit'
    try:
        if args.record:
            from s1b_keyboard_record import Recorder
            args.recorder=Recorder(args.record_dir,f'http://127.0.0.1:{args.view_port}',
                dict(speed_mm_s=args.speed_mm_s,head_speed_deg_s=args.head_speed_deg_s,control_hz=args.control_hz,frame='chassis',units='mm'))
        if args.view_cameras:
            viewer_script=Path(__file__).with_name('s1b_camera_view.py')
            viewer_log=Path(__file__).with_name('camera_view.log').open('w')
            viewer=subprocess.Popen([sys.executable,str(viewer_script),'--port',str(args.view_port)],
                                    stdin=subprocess.DEVNULL,stdout=viewer_log,stderr=viewer_log)
            wait_for_viewer(viewer,args.view_port)
            print(f'网页已就绪：http://127.0.0.1:{args.view_port}',flush=True)
        run_robot(args, viewer)
    except KeyboardInterrupt:
        code = 130;exit_reason='KeyboardInterrupt'
    except Exception as exc:
        print('键盘控制退出：'+str(exc), file=sys.stderr, flush=True)
        code = 1;exit_reason=str(exc)
    finally:
        if args.recorder:args.recorder.close(exit_reason)
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
