"""Orchestrate isolated robot services, independent tunnels and workstation inference."""
import argparse
import http.client
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import start_s1b_inference as shared

ROOT=shared.ROOT
REMOTE=shared.REMOTE


def tunnel_command(host,ports):
    # Explicitly disable SSH multiplexing: image traffic must not share the
    # control channel's TCP connection through an existing ControlMaster.
    command=['ssh','-N','-o','BatchMode=yes','-o','ConnectTimeout=10',
        '-o','ControlMaster=no','-o','ControlPath=none','-o','Compression=no',
        '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=3']
    for port in ports:command+=['-L',f'127.0.0.1:{port}:127.0.0.1:{port}']
    return command+[host]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=int,choices=[2000,4500,8000],default=2000)
    p.add_argument('--config',default=str(ROOT/'configs/s1b_deployment_v1.yaml'))
    p.add_argument('--host',default='S1B')
    p.add_argument('--duration',type=float,default=600)
    p.add_argument('--mode',choices=['shadow','execute'],default='shadow')
    p.add_argument('--execution-config',default=str(ROOT/'configs/s1b_execution_model_v1.json'))
    p.add_argument('--operator-present',action='store_true')
    p.add_argument('--estop-ready',action='store_true')
    p.add_argument('--request-control-handover',action='store_true')
    p.add_argument('--auto-start',action='store_true',help='noninteractive SHADOW validation only')
    for name,port in [('head',55555),('right',55557),('state',55558),('control',6200),('policy',8796)]:
        p.add_argument('--'+name+'-port',type=int,default=port)
    p.add_argument('--execution-strategy',choices=['synchronous','continuous'],default='synchronous')
    a=p.parse_args()
    if a.mode=='execute' and (not a.operator_present or not a.estop_ready):
        p.error('execute requires --operator-present and --estop-ready')
    if a.auto_start and a.mode!='shadow':p.error('--auto-start is restricted to shadow mode')
    if not a.auto_start and not sys.stdin.isatty():p.error('需要交互终端：模型加载后按 S 才能开始。')
    if not 0<a.duration<=86400:p.error('duration must be in (0,86400]')
    ports=[a.head_port,a.right_port,a.state_port,a.control_port,a.policy_port]
    if len(set(ports))!=5 or any(not 1024<=n<=65535 for n in ports):p.error('ports must be distinct and in [1024,65535]')
    sys.path.insert(0,str(ROOT/'src'))
    from pressb.s1b_deployment.cameras import missing_camera_drivers
    execution=None
    if a.mode=='execute':
        from pressb.s1b_deployment.live_execution import validate_execution_config
        execution=json.loads(Path(a.execution_config).read_text());validate_execution_config(execution)
    import zmq
    os.chdir(ROOT)
    from pressb.s1b_deployment.contract import read_config
    deployment=read_config(a.config)
    identity_path=Path(deployment['model']['identity']).resolve()
    identity=json.loads(identity_path.read_text())
    if str(a.checkpoint) not in identity['models']:p.error('checkpoint is not in selected model identity')
    token='split_'+str(time.time_ns());out=ROOT/'outputs/s1b_deployment'/f'live_{a.checkpoint}_{token}'
    out.mkdir(parents=True);remote_out=REMOTE+'/outputs/'+token
    (out/'launch.json').write_text(json.dumps(dict(vars(a),session_id=token,remote_output=remote_out,pyzmq=zmq.__version__),indent=2)+'\n')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',a.host]
    processes={};files=[];pidfiles=[];worker=None;cleanup_errors=[];service_errors=[];summary=None;remote_prepared=False
    warning_position=0
    def show_control_warnings():
        nonlocal warning_position
        path=out/'control.log'
        if not path.exists():return
        with path.open(errors='replace') as stream:
            stream.seek(warning_position)
            while True:
                line=stream.readline()
                if not line or not line.endswith('\n'):break
                warning_position=stream.tell()
                try:row=json.loads(line)
                except ValueError:continue
                if not isinstance(row,dict) or row.get('event')!='warning':continue
                message=row['check']
                if row.get('part'):message+=':'+row['part']
                if row.get('reason'):message+=' '+str(row['reason'])
                if 'drift' in row:message+=f" drift={row['drift']:.6f}, limit={row['limit']:.6f}"
                if 'starved_ms' in row:message+=f" starved={row['starved_ms']:.1f} ms（保持最后目标，不中断）"
                else:message+='（仅记录，不中断测试）'
                print('WARNING: '+message,flush=True)
    def remote(command,**kw):return subprocess.run(ssh+[command],check=True,**kw)
    def launch(name,command):
        log=(out/(name+'.log')).open('w');files.append(log)
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        processes[name]=proc;return proc
    def start_remote(name,command,extra=''):
        pidfile='/tmp/pressb_inference_'+token+'_'+name+'.pid';pidfiles.append(pidfile)
        env='export PYTHONPATH='+shlex.quote(REMOTE+'/deps/split')+':"${PYTHONPATH:-}"'
        shell=(shared.ENV+'; cd '+shlex.quote(REMOTE)+'; '+shared.CAMERA_ENV+'; '+env+'; '+extra+
            '; export PRESSB_LIVE_TOKEN='+shlex.quote(token)+'; echo $$ > '+shlex.quote(pidfile)+'; exec '+command)
        # Avoid an empty shell command between two semicolons.
        shell=shell.replace('; ;',';')
        return launch(name,ssh+[shell])
    def healthy():
        for name,proc in processes.items():
            if proc.poll() is not None:raise RuntimeError(name+' exited; '+shared.bridge_failure(out/(name+'.log')))
    def stop(*_):raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    print('分离式链路；执行策略：'+a.execution_strategy+'；日志：'+str(out),flush=True)
    try:
        for port in ports:
            shared.check_port_available(port)
        probe='import socket,sys;sys.path.insert(0,'+repr(REMOTE+'/deps/split')+');import zmq\n'
        probe+='for port in '+repr(ports[:-1])+":\n with socket.socket() as s:\n  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n  try:s.bind(('127.0.0.1',port));s.listen(1)\n  except OSError as exc:raise RuntimeError('robot_port_unavailable:'+str(port)) from exc\n"
        remote('python3 -c '+shlex.quote(probe),timeout=15)
        camera_probe="""import json
from pathlib import Path
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=p.joinpath('cmdline').read_bytes().decode().split(chr(0))
  if Path(args[0]).name in ('astribot_camera_main','astribot_camera_sub'):found.append(dict(pid=int(p.name),path=args[0]))
 except (OSError,UnicodeError,IndexError):pass
print(json.dumps(found))
"""
        cameras=json.loads(subprocess.check_output(ssh+['python3 -c '+shlex.quote(camera_probe)],text=True))
        missing_camera_drivers(cameras,False)  # Reject duplicate drivers before starting anything.
        (out/'camera_processes.json').write_text(json.dumps(cameras,indent=2)+'\n')
        remote('mkdir -p '+shlex.quote(remote_out)+' '+REMOTE+'/scripts '+REMOTE+'/configs '+REMOTE+'/src/pressb/s1b_deployment')
        remote_prepared=True
        def copy(paths,destination):subprocess.run(['scp','-q',*[str(x) for x in paths],a.host+':'+destination],check=True)
        copy(sorted((ROOT/'src/pressb/s1b_deployment').glob('*.py')),REMOTE+'/src/pressb/s1b_deployment/')
        copy([ROOT/'scripts'/name for name in ('s1b_observation_service.py','s1b_control_service.py','s1b_camera_probe.py')],REMOTE+'/scripts/')
        copy([ROOT/'configs'/name for name in ('s1b_observation_v1.json','s1b_deployment_assets_v1.json')],REMOTE+'/configs/')
        remote_identity=REMOTE+'/configs/identity_'+token+'.json'
        copy([identity_path],remote_identity)
        def camera_ready(stage,timeout):
            command=shared.ENV+'; cd '+shlex.quote(REMOTE)+f'; python3 scripts/s1b_camera_probe.py --timeout {timeout}'
            result=subprocess.run(ssh+[command],capture_output=True,text=True,timeout=timeout+15)
            (out/('camera_'+stage+'.log')).write_text(result.stdout+result.stderr)
            if result.returncode not in (0,2):
                raise RuntimeError('camera_probe_failed; '+str(out/('camera_'+stage+'.log')))
            report=json.loads(result.stdout.splitlines()[-1])
            (out/('camera_'+stage+'.json')).write_text(json.dumps(report,indent=2)+'\n')
            return report
        camera_status=camera_ready('existing',3)
        for camera in cameras:
            print('复用已有相机驱动：'+camera['path']+'（PID '+str(camera['pid'])+'）',flush=True)
        for name,arg in missing_camera_drivers(cameras,camera_status['ready']):
            start_remote(name,shlex.quote(REMOTE+'/camera_minimal/lib/astribot_camera/'+name)+' '+arg)
            time.sleep(2);healthy()
        if not camera_status['ready']:camera_status=camera_ready('started',8)
        if not camera_status['ready']:
            raise RuntimeError('camera_streams_not_ready:'+json.dumps(camera_status['streams'])+'; '+str(out))
        print('双相机图像预检通过（实际 JPEG、尺寸、时间戳与连续帧）。',flush=True)
        if execution:
            copy([Path(a.execution_config).resolve()],REMOTE+'/configs/execution_'+token+'.json')
            probe='from astribot_sdk.core.astribot_api.astribot_client import Astribot; print("SDK_IMPORT_OK_NO_CLIENT_CREATED")'
            with (out/'sdk_preflight.log').open('w') as log:
                remote(shared.ENV+'; '+shared.sdk_environment(execution)+'; python3 -c '+shlex.quote(probe),stdout=log,stderr=subprocess.STDOUT,timeout=30)
        launch('policy',[sys.executable,str(ROOT/'scripts/s1b_policy_server.py'),'--checkpoint',str(a.checkpoint),'--port',str(a.policy_port),'--warmup','--config',str(Path(a.config).resolve())])
        common=' --service-id '+shlex.quote(token)+' --mapping configs/s1b_observation_v1.json'
        start_remote('images','python3 scripts/s1b_observation_service.py --kind images'+common+
            f' --head-port {a.head_port} --right-port {a.right_port} --output '+shlex.quote(remote_out+'/images'))
        start_remote('states','python3 scripts/s1b_observation_service.py --kind states'+common+
            f' --state-port {a.state_port} --output '+shlex.quote(remote_out+'/states'))
        for name,group in [('image_tunnel',[a.head_port,a.right_port]),('state_tunnel',[a.state_port]),('control_tunnel',[a.control_port])]:
            launch(name,tunnel_command(a.host,group))
        until=time.monotonic()+180;model=None
        while time.monotonic()<until:
            healthy();conn=http.client.HTTPConnection('127.0.0.1',a.policy_port,timeout=1)
            try:
                conn.request('GET','/health');response=conn.getresponse()
                if response.status==200:model=json.loads(response.read());break
            except OSError:pass
            finally:conn.close()
            time.sleep(.2)
        if not model or model['step']!=a.checkpoint:raise RuntimeError('model_not_ready')
        (out/'health.json').write_text(json.dumps(model,indent=2)+'\n')
        if not a.auto_start:shared.wait_for_start(healthy)
        else:print('自动开始 shadow 验证（无 SDK 控制）。',flush=True)
        command=(f'python3 scripts/s1b_control_service.py --session-id {token} --mode {a.mode} --checkpoint {a.checkpoint}'
            f' --execution-strategy {a.execution_strategy} --identity {shlex.quote(remote_identity)} --port {a.control_port} --duration {a.duration+120} --output '+shlex.quote(remote_out+'/control'))
        if execution:
            command+=' --execution-config '+shlex.quote(REMOTE+'/configs/execution_'+token+'.json')+' --operator-present --estop-ready'
            if a.request_control_handover:command+=' --request-control-handover'
        start_remote('control',command,shared.sdk_environment(execution) if execution else '')
        args=[sys.executable,str(ROOT/'scripts/s1b_split_inference.py'),'--session-id',token,
            '--execution-strategy',a.execution_strategy,'--identity',str(identity_path),'--checkpoint',str(a.checkpoint),'--mode',a.mode,'--duration',str(a.duration),'--output',str(out/'workstation')]
        for name in ('head','right','state','control','policy'):args+=['--'+name+'-port',str(getattr(a,name+'_port'))]
        worker=launch('inference',args);last=None;startup_reported=False
        while worker.poll() is None:
            show_control_warnings()
            for name,proc in processes.items():
                if name=='inference':continue
                code=proc.poll()
                if code is not None:
                    if name=='control' and code==0:
                        worker.wait(timeout=5);break
                    raise RuntimeError(name+' exited; '+shared.bridge_failure(out/(name+'.log')))
            try:
                rows=(out/'inference.log').read_text().splitlines()
                if not startup_reported and any('"event": "initializing_control"' in line for line in rows):
                    print('观测、校时和预检推理已通过；正在等待机器人控制端初始化。',flush=True)
                    startup_reported=True
                row=json.loads(rows[-1]) if rows else {}
                if row.get('event')=='progress' and row['requests']!=last:
                    last=row['requests']
                    if a.execution_strategy=='synchronous':
                        print(f"同步推理：{last} 次，已完成 {row['completed_chunks']} 段",flush=True)
                    else:print(f"实时推理：{last} 次，预测剩余 {row['horizon_remaining_ms']:.1f} ms",flush=True)
            except (ValueError,OSError):pass
            time.sleep(.5)
        if worker.returncode:
            try:processes['control'].wait(timeout=3)
            except subprocess.TimeoutExpired:pass
            raise RuntimeError(shared.bridge_failure(out/'inference.log')+'\n'+shared.bridge_failure(out/'control.log'))
        processes['control'].wait(timeout=10)
        if processes['control'].returncode:raise RuntimeError(shared.bridge_failure(out/'control.log'))
        summary=json.loads((out/'workstation/summary.json').read_text())
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if not summary['complete']:raise RuntimeError('split_inference_incomplete')
    except KeyboardInterrupt:
        print('正在停止本次分离式推理进程。',flush=True)
    finally:
        original_error=sys.exc_info()[0]
        if worker and worker.poll() is None:
            worker.send_signal(signal.SIGINT)
            try:worker.wait(timeout=4)
            except subprocess.TimeoutExpired:pass
        cleanup="""import os,signal
from pathlib import Path
for name in %r:
 p=Path(name)
 if not p.exists():continue
 try:
  pid=int(p.read_text());env=(Path('/proc')/str(pid)/'environ').read_bytes()
  if ('PRESSB_LIVE_TOKEN='+%r).encode() in env.split(bytes([0])):os.kill(pid,signal.SIGINT)
 except (OSError,ValueError):pass
"""%(list(reversed(pidfiles)),token)
        try:
            if pidfiles:subprocess.run(ssh+['python3 -c '+shlex.quote(cleanup)],timeout=15,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except Exception as exc:cleanup_errors.append('remote_cleanup:'+repr(exc))
        for name in ('control','images','states','astribot_camera_sub','astribot_camera_main'):
            proc=processes.get(name)
            if proc:
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:cleanup_errors.append(name+'_shutdown_timeout')
                if name in ('images','states','control') and proc.returncode not in (None,0):
                    service_errors.append(name+'_exit_'+str(proc.returncode))
        for proc in reversed(list(processes.values())):
            if proc.poll() is None:proc.terminate()
        for proc in processes.values():
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for stream in files:stream.close()
        show_control_warnings()
        try:
            if remote_prepared:
                subprocess.run(['scp','-q','-r',a.host+':'+remote_out,str(out/'robot')],timeout=30,check=True)
        except Exception as exc:cleanup_errors.append('remote_logs:'+repr(exc))
        if summary is None and (out/'workstation/summary.json').exists():
            summary=json.loads((out/'workstation/summary.json').read_text())
        if summary:
            try:
                robot_summary=json.loads((out/'robot/control/summary.json').read_text())
                summary['robot_control']=robot_summary
                summary['shadow_coverage_verified']=(a.mode=='shadow' and a.execution_strategy=='continuous'
                    and robot_summary['accepted_predictions']>0
                    and robot_summary['coverage_gaps']==0 and not robot_summary['fatal'])
                summary['shadow_synchronous_chunks_verified']=(a.mode=='shadow' and a.execution_strategy=='synchronous'
                    and robot_summary['scheduler'].get('completed_chunks',0)>0 and not robot_summary['fatal'])
                if robot_summary['fatal']:service_errors.append('control_fatal:'+str(robot_summary['fatal']))
                for kind in ('images','states'):
                    source_summary=json.loads((out/'robot'/kind/'summary.json').read_text())
                    if source_summary['fatal']:service_errors.append(kind+'_fatal:'+str(source_summary['fatal']))
            except Exception as exc:cleanup_errors.append('summary_collection:'+repr(exc))
            summary['cleanup_errors']=cleanup_errors
            summary['service_errors']=service_errors
            summary['complete']=summary['complete'] and not cleanup_errors and not service_errors
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'cleanup.json').write_text(json.dumps(dict(errors=cleanup_errors,service_errors=service_errors,
            returncodes={name:proc.returncode for name,proc in processes.items()}),indent=2)+'\n')
        if service_errors:
            print('服务运行异常：'+'; '.join(service_errors),flush=True)
        if cleanup_errors:
            print('清理或日志收集异常：'+'; '.join(cleanup_errors),flush=True)
        if (cleanup_errors or service_errors) and original_error is None:
            raise RuntimeError('split_services_or_cleanup_failed; '+str(out/'cleanup.json'))
        if summary and not cleanup_errors and not service_errors and original_error is None:
            print('推理结束：'+str(summary['valid_requests'])+' 次；日志 '+str(out),flush=True)


if __name__=='__main__':main()
