#!/usr/bin/env python3
"""Robot-local v2 motion/observation service; --host deploys with an interactive SDK prompt."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def launch(args):
    from start_s1b_inference import ENV,CAMERA_ENV,sdk_environment,check_port_available
    from s1b_keyboard_direct import run_ssh
    for port in (args.port,args.view_port):check_port_available(port)
    root=Path(__file__).resolve().parents[1]
    remote='/tmp/pressb_agent_'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'agent.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            tar.add(root/'src/pressb',arcname='src/pressb',filter=lambda info:None if '__pycache__' in info.name else info)
            for name in ('s1b_agent_service.py','s1b_keyboard_direct.py','s1b_camera_view.py','s1b_depth_metadata.cpp'):
                tar.add(root/'scripts'/name,arcname='scripts/'+name)
        subprocess.run(['ssh',args.host,'mkdir -p '+shlex.quote(remote)],check=True)
        try:
            subprocess.run(['scp','-q',str(archive),args.host+':'+remote+'/agent.tar.gz'],check=True)
            subprocess.run(['ssh',args.host,'tar -xzf '+shlex.quote(remote+'/agent.tar.gz')+' -C '+shlex.quote(remote)],check=True)
            argv=['python3',remote+'/scripts/s1b_agent_service.py','--port',str(args.port),'--view-port',str(args.view_port),
                '--output',args.output,'--control-hz',str(args.control_hz),'--max-move-mm',str(args.max_move_mm),
                '--max-forward-mm',str(args.max_forward_mm)]
            command=ENV+'; '+CAMERA_ENV+'; '+sdk_environment(dict(sdk_root=args.sdk_root))+'; '+shlex.join(argv)
            result=run_ssh(['ssh','-t','-o','ExitOnForwardFailure=yes',
                '-L',f'127.0.0.1:{args.port}:127.0.0.1:{args.port}',
                '-L',f'127.0.0.1:{args.view_port}:127.0.0.1:{args.view_port}',
                args.host,'bash -lc '+shlex.quote(command)])
            return result
        finally:subprocess.run(['ssh',args.host,'rm -rf '+shlex.quote(remote)],check=False)


def run(args):
    from s1b_keyboard_direct import connect_robot,wait_for_viewer,_SDK_KEEPALIVE
    from pressb.agent_loop.sdk_backend import SDKBackend
    from pressb.agent_loop.service import Camera,Service,make_server
    viewer=robot=monitor=backend=service=server=None
    done=False
    def stop(*unused):
        nonlocal done
        done=True
        if service:service.stop()
        elif backend:backend.stop()
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    log=None
    try:
        out=Path(args.output).expanduser();out.mkdir(parents=True,exist_ok=True)
        log=(out/'camera_view.log').open('ab')
        viewer=subprocess.Popen([sys.executable,str(Path(__file__).with_name('s1b_camera_view.py')),
                                 '--port',str(args.view_port)],stdout=log,stderr=log)
        wait_for_viewer(viewer,args.view_port)
        print(f'相机网页已就绪：http://127.0.0.1:{args.view_port}',flush=True)
        from astribot_sdk.core.astribot_api.astribot_client import Astribot
        from pressb.s1b_deployment.robot_state import RobotState
        robot=connect_robot(Astribot,args.control_hz);_SDK_KEEPALIVE.append(robot)
        if not robot.get_control_rights_status():raise RuntimeError('未取得 SDK 控制权')
        print('运动模式：与键盘相同的连续 Cartesian 发送；等待和到位后不切换关节控制。',flush=True)
        robot.set_head_follow_effector(False)
        monitor=RobotState(lambda row:None,shared_context=True,control_only=True)
        deadline=time.monotonic()+5
        while True:
            try:monitor.snapshot();break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(.05)
        # Finish the gripper-only command before starting the arm stream.
        # No competing joint set calls may run during Cartesian control.
        robot.close_effector(['astribot_gripper_right'],duration=1.0)
        backend=SDKBackend(robot,monitor,args.control_hz)
        service=Service(backend,Camera(f'http://127.0.0.1:{args.view_port}'),out,args.max_move_mm,args.max_forward_mm)
        server=make_server(service,args.port);server.timeout=.1
        print(f'Agent API 已就绪：http://127.0.0.1:{args.port}/v2/session\n默认 20 mm/s；在另一终端启动 s1b_agent.py。',flush=True)
        fault_saved=False
        while not done and not service.shutdown_requested.is_set():
            if viewer.poll() is not None:raise RuntimeError('相机服务退出，详情见 '+str(out/'camera_view.log'))
            if backend.error:service.backend_failed()
            if service.phase=='FAULT' and not fault_saved:
                from pressb.agent_loop.common import atomic_json
                diagnostic=backend.diagnostic()
                diagnostic['error']=backend.error or service.fault
                path=service.root/'fault.json'
                atomic_json(path,diagnostic);fault_saved=True
                print('运动已停止：'+str(diagnostic['error'])+'\n完整故障记录：'+str(path)+
                      '\nAPI 保留故障查询；客户端停止或 Ctrl-C 后退出。',file=sys.stderr,flush=True)
            server.handle_request()
    except Exception as exc:
        from pressb.agent_loop.common import atomic_json
        diagnostic=dict(error=str(exc),traceback=traceback.format_exc(),
            servo_traceback=getattr(backend,'error_traceback',None),
            torso=getattr(backend,'torso_diagnostics',{}))
        path=(service.root if service else Path(args.output).expanduser())/'fault.json'
        try:atomic_json(path,diagnostic)
        except OSError:pass
        print('完整故障记录：'+str(path),file=sys.stderr,flush=True)
        raise
    finally:
        failures=[]
        if backend is not None and backend.error:
            from pressb.agent_loop.common import atomic_json
            path=(service.root if service else Path(args.output).expanduser())/'fault.json'
            try:atomic_json(path,backend.diagnostic())
            except OSError as exc:failures.append('fault log: '+str(exc))
        for close in (server.server_close if server else None,service.close if service else backend.close if backend else None,
                      monitor.close if monitor else None):
            if close:
                try:close()
                except Exception as exc:failures.append(str(exc))
        if robot:
            try:
                if robot.get_control_rights_status():
                    robot.stop_robot();robot.astribot_interface.shutdown_control_rights_srv()
                interface=robot.astribot_interface
                interface.executor.shutdown(timeout_sec=2)
                import rclpy
                if rclpy.ok():rclpy.shutdown()
                interface.spin_thread.join(timeout=2);interface.shutdown()
            except Exception as exc:failures.append(str(exc))
        if viewer:
            viewer.terminate()
            try:viewer.wait(4)
            except subprocess.TimeoutExpired:viewer.kill();viewer.wait()
        if log:log.close()
        if failures:print('退出清理：'+'; '.join(failures),file=sys.stderr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host');p.add_argument('--sdk-root',default='/home/astribot/astribot_sdk_aarch64')
    p.add_argument('--port',type=int,default=6210);p.add_argument('--view-port',type=int,default=8765)
    p.add_argument('--output',default='~/.local/share/pressb/agent_sessions')
    p.add_argument('--control-hz',type=float,default=250)
    p.add_argument('--max-move-mm',type=float,default=20);p.add_argument('--max-forward-mm',type=float,default=100)
    args=p.parse_args()
    import math
    if any(not math.isfinite(v) or v<=0 for v in (args.control_hz,args.max_move_mm,args.max_forward_mm)):
        p.error('频率和行程必须为正有限数')
    if args.port==args.view_port or any(not 1<=v<=65535 for v in (args.port,args.view_port)):p.error('端口无效或重复')
    if args.host:return launch(args)
    from s1b_keyboard_direct import SDKOutput
    output=SDKOutput()
    try:run(args)
    finally:output.close()
    return 0


if __name__=='__main__':
    code=0
    try:code=main()
    except (Exception,KeyboardInterrupt) as exc:print('服务退出：'+str(exc),file=sys.stderr);code=1
    finally:
        sys.stdout.flush();sys.stderr.flush()
    # Native SDK destructor is unsafe after explicit shutdown (same as keyboard entry).
    os._exit(code)
