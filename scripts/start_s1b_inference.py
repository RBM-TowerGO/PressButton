#!/usr/bin/env python3
"""One command: load a verified policy and infer from S1B RGB/state in shadow."""
import argparse
import http.client
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/astribot/pressb_readonly_p2'
ENV='; '.join('source '+p for p in ['/opt/astribot_ros/middle_ware/local_setup.bash','/opt/astribot_ros/software/local_setup.bash','/opt/astribot_ros/third_pkg/local_setup.bash','/opt/astribot_ros/robot_system_ctrl/ros_env.sh'])
CAMERA_ENV='export ASTRIBOT_ROS_INSTALL_PATH=/opt/astribot_ros/software ASTRIBOT_CAMERA_INSTALL_PATH=/opt/astribot_ros/software ASTRIBOT_CAMERA_INFO_PATH=/etc/config/camera_info.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=int,choices=[2000,8000],default=2000)
    p.add_argument('--host',default='S1B')
    p.add_argument('--duration',type=float,default=600,help='seconds of live inference; Ctrl-C stops early')
    a=p.parse_args()
    if not 0<a.duration<=86400:p.error('duration must be in (0,86400]')
    os.chdir(ROOT);token=str(time.time_ns());out=ROOT/'outputs/s1b_deployment'/f'live_{a.checkpoint}_{token}';out.mkdir(parents=True)
    remote_out=REMOTE+'/outputs/live_'+token
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',a.host]
    local=[];remote_pidfiles=[];files=[];bridge=None
    def run(command,**kw):return subprocess.run(command,check=True,**kw)
    def remote(command,**kw):return run(ssh+[command],**kw)
    def start_remote(name,command):
        pidfile='/tmp/pressb_inference_'+token+'_'+name+'.pid';remote_pidfiles.append(pidfile)
        log=(out/(name+'.log')).open('w');files.append(log)
        shell=ENV+'; cd '+shlex.quote(REMOTE)+'; '+CAMERA_ENV+'; export PRESSB_LIVE_TOKEN='+token+'; echo $$ > '+shlex.quote(pidfile)+'; exec '+command
        process=subprocess.Popen(ssh+[shell],stdout=log,stderr=subprocess.STDOUT);local.append(process)
        return process
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        # Refuse to replace an existing service/tunnel or unrelated camera run.
        probe="import socket;s=socket.socket();s.bind(('127.0.0.1',8796));s.close()"
        import socket
        with socket.socket() as s:s.bind(('127.0.0.1',8796))
        remote('python3 -c '+shlex.quote(probe))
        camera_probe="""import json
from pathlib import Path
found={}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=p.joinpath('cmdline').read_bytes().decode().split(chr(0))
  if Path(args[0]).name in ('astribot_camera_main','astribot_camera_sub'):found[Path(args[0]).name]=args[0]
 except (OSError,UnicodeError,IndexError):pass
print(json.dumps(found))
"""
        cameras=json.loads(subprocess.check_output(ssh+['python3 -c '+shlex.quote(camera_probe)],text=True))
        for name,path in cameras.items():
            if path!=REMOTE+'/camera_minimal/lib/astribot_camera/'+name:raise RuntimeError('Another camera installation is active; leave it untouched: '+path)
        remote('mkdir -p '+REMOTE+'/scripts '+REMOTE+'/configs '+REMOTE+'/src/pressb/s1b_deployment')
        package=ROOT/'src/pressb/s1b_deployment'
        run(['scp','-q',*[str(package/(n+'.py')) for n in ['__init__','contract','protocol','observation','scheduler','shadow']],a.host+':'+REMOTE+'/src/pressb/s1b_deployment/'])
        run(['scp','-q',str(ROOT/'scripts/s1b_robot_bridge.py'),a.host+':'+REMOTE+'/scripts/'])
        run(['scp','-q',*[str(ROOT/'configs'/n) for n in ['s1b_deployment_v1.yaml','s1b_deployment_assets_v1.json','s1b_observation_v1.json']],a.host+':'+REMOTE+'/configs/'])
        print('正在加载 checkpoint '+str(a.checkpoint)+'；日志：'+str(out),flush=True)
        log=(out/'policy.log').open('w');files.append(log)
        server=subprocess.Popen([sys.executable,str(ROOT/'scripts/s1b_policy_server.py'),'--checkpoint',str(a.checkpoint)],stdout=log,stderr=subprocess.STDOUT);local.append(server)
        for name,arg in [('astribot_camera_main','camera'),('astribot_camera_sub','compress_camera')]:
            if name not in cameras:
                process=start_remote(name,shlex.quote(REMOTE+'/camera_minimal/lib/astribot_camera/'+name)+' '+arg)
                time.sleep(2)
                if process.poll() is not None:raise RuntimeError(name+' failed; see '+str(out/(name+'.log')))
        tunnel_log=(out/'tunnel.log').open('w');files.append(tunnel_log)
        tunnel=subprocess.Popen(['ssh','-N','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-R','127.0.0.1:8796:127.0.0.1:8796',a.host],stdout=tunnel_log,stderr=subprocess.STDOUT);local.append(tunnel)
        until=time.monotonic()+120;health=None
        while time.monotonic()<until:
            if server.poll() is not None:raise RuntimeError('Model load failed; see '+str(out/'policy.log'))
            if tunnel.poll() is not None:raise RuntimeError('SSH reverse tunnel failed')
            conn=http.client.HTTPConnection('127.0.0.1',8796,timeout=1)
            try:
                conn.request('GET','/health');response=conn.getresponse()
                if response.status==200:health=json.loads(response.read());break
            except OSError:pass
            finally:conn.close()
            time.sleep(.5)
        if not health or health['step']!=a.checkpoint:raise RuntimeError('Model readiness/identity check failed')
        (out/'health.json').write_text(json.dumps(health,indent=2)+'\n')
        print('模型已就绪；开始 S1B 双相机实时推理（shadow，不发送运动）。Ctrl-C 可停止。',flush=True)
        bridge=start_remote('bridge','python3 '+shlex.quote(REMOTE+'/scripts/s1b_robot_bridge.py')+' --config '+shlex.quote(REMOTE+'/configs/s1b_deployment_v1.yaml')+' --mapping '+shlex.quote(REMOTE+'/configs/s1b_observation_v1.json')+' --mode shadow --duration '+str(a.duration)+' --output '+shlex.quote(remote_out))
        # The bridge's asset identity path is relative to its project root.
        # Use a shell prefix for the working directory while preserving exec PID.
        last_progress=None
        while bridge.poll() is None:
            try:
                lines=(out/'bridge.log').read_text().splitlines()
                latest=json.loads(lines[-1]) if lines else {}
                if latest.get('event')=='progress' and latest.get('requests')!=last_progress:
                    last_progress=latest['requests'];print('实时推理：'+str(last_progress)+' 次请求，RTT '+str(round(latest['rtt_ms'],1))+' ms',flush=True)
            except (ValueError,OSError):pass
            if server.poll() is not None or tunnel.poll() is not None:raise RuntimeError('Inference service or tunnel exited')
            time.sleep(1)
        if bridge.returncode:raise RuntimeError('Robot bridge failed; see '+str(out/'bridge.log'))
        remote('cat '+shlex.quote(remote_out+'/summary.json'),stdout=(out/'summary.json').open('w'))
        summary=json.loads((out/'summary.json').read_text())
        if not summary['valid_requests']:raise RuntimeError('No valid predictions; see robot/source logs')
        print('推理结束：'+str(summary['valid_requests'])+' 次有效请求；日志 '+str(out),flush=True)
    except KeyboardInterrupt:
        print('正在停止并清理本次推理进程。',flush=True)
    finally:
        # Only processes bearing this invocation's token may be signalled.
        cleanup="""import os,signal
from pathlib import Path
for name in %r:
 p=Path(name)
 if not p.exists():continue
 try:
  pid=int(p.read_text().strip());env=Path('/proc')/str(pid)/'environ'
  if ('PRESSB_LIVE_TOKEN='+%r).encode() in env.read_bytes().split(bytes([0])):os.kill(pid,signal.SIGINT)
 except (OSError,ValueError):pass
"""%(remote_pidfiles,token)
        subprocess.run(ssh+['python3 -c '+shlex.quote(cleanup)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
        time.sleep(.5)
        for process in reversed(local):
            if process.poll() is None:process.terminate()
        for process in local:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for stream in files:stream.close()
if __name__=='__main__':main()
