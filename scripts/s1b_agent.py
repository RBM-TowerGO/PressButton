#!/usr/bin/env python3
"""Wrist-only Thor model client with per-action terminal confirmation."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.agent_loop.client import RobotClient,build_request,propose,proposal_still_valid
from pressb.agent_loop.common import atomic_json,request,pose_error,depth_estimate


@contextmanager
def tunnel(host, port):
    if not host:
        yield;return
    from start_s1b_inference import check_port_available
    check_port_available(port)
    process=subprocess.Popen(['ssh','-N','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
        '-L',f'127.0.0.1:{port}:127.0.0.1:8000',host])
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Thor SSH 隧道启动失败')
            try:request(f'http://127.0.0.1:{port}','/v1/models',timeout=1);break
            except (OSError,ValueError):time.sleep(.2)
        else:raise RuntimeError('Thor 模型连接超时')
        yield
    finally:
        process.terminate()
        try:process.wait(3)
        except subprocess.TimeoutExpired:process.kill();process.wait()


def run(args):
    if not args.replay:
        try:request(args.robot_base,'/v2/session',timeout=3)
        except (OSError,ValueError) as exc:
            raise RuntimeError('无法连接机器人 API '+args.robot_base+
                '。请先在另一终端运行 python scripts/s1b_agent_service.py --host S1B，'
                '确认显示“Agent API 已就绪”且进程没有退出。若出现 torso_drift，请先处理服务端错误。'
                ' 原因：'+str(exc)) from exc
    output=Path(args.output).expanduser();output.mkdir(parents=True,exist_ok=False)
    prompt=Path(args.prompt).read_text()
    model_base=args.model_base or f'http://127.0.0.1:{args.thor_port}/v1'
    if args.replay:
        folder=Path(args.replay);obs=json.loads((folder/'manifest.json').read_text())
        session=json.loads(Path(args.session).read_text()) if args.session else dict(max_move_mm=20,max_forward_mm=100,speed_mm_s=20)
        with tunnel(None if args.model_base else args.thor_host,args.thor_port):
            proposal=propose(model_base,build_request(args.goal,obs,(folder/'right.jpg').read_bytes(),session,[],prompt),obs,session,output)
            print(json.dumps(proposal,ensure_ascii=False,indent=2))
        return
    client=None;history=[]
    with tunnel(None if args.model_base else args.thor_host,args.thor_port):
        try:
            client=RobotClient(args.robot_base,output)
            atomic_json(output/'session.json',client.session)
            atomic_json(output/'task.json',dict(goal=args.goal,model_base=model_base,prompt=prompt))
            obs,folder=client.observe()
            for index in range(args.max_rounds):
                rd=output/'rounds'/f'{index:04d}'
                body=build_request(args.goal,obs,(folder/'right.jpg').read_bytes(),client.session,history,prompt)
                print(f'第 {index+1} 轮：请求 Thor；右腕图像 {folder / "right.jpg"}',flush=True)
                try:proposal=propose(model_base,body,obs,client.session,rd)
                except (OSError,ValueError) as exc:
                    atomic_json(rd/'error.json',dict(error=str(exc)))
                    print('模型请求未产生有效动作：'+str(exc))
                    if input('[r] 重新观察并重试；其他键退出：').strip().lower()!='r':break
                    obs,folder=client.observe();continue
                print(json.dumps(proposal,ensure_ascii=False,indent=2),flush=True)
                name=proposal['name']
                if name=='report_done':
                    answer=input('模型报告目标触发。确认完成？[y] 完成；[r] 重新观察；其他键退出：').strip().lower()
                    atomic_json(rd/'confirmation.json',dict(answer=answer))
                    if answer=='y':
                        atomic_json(output/'result.json',dict(status='human_confirmed_done',evidence=proposal['arguments']['evidence']));break
                    if answer=='r':obs,folder=client.observe();continue
                    break
                if name in ('request_observation','report_unable'):
                    answer=input('[r] 重新观察；其他键退出：').strip().lower()
                    atomic_json(rd/'confirmation.json',dict(answer=answer))
                    if answer!='r':break
                    obs,folder=client.observe();continue
                answer=input('速度 20 mm/s。[y] 执行；[r] 重新观察；其他键退出：').strip().lower()
                atomic_json(rd/'confirmation.json',dict(answer=answer,decision_observation_id=obs['observation_id']))
                if answer=='r':obs,folder=client.observe();continue
                if answer!='y':break
                # Do not reuse a stale model-era observation as the execution token.
                fresh,fresh_folder=client.observe()
                if not proposal_still_valid(obs,fresh,proposal):
                    distance,angle=pose_error(obs['state']['pose_m_xyzw'],fresh['state']['pose_m_xyzw'])
                    print(f'本轮动作未执行：位姿变化 {distance:.2f} mm / {angle:.2f}°；'
                          f'测距点版本 {obs["selection"]["revision"]} → {fresh["selection"]["revision"]}。重新请求建议并确认。')
                    atomic_json(rd/'invalidated.json',dict(execution_observation_id=fresh['observation_id'],
                        decision_observation_id=obs['observation_id'],position_change_mm=distance,
                        rotation_change_deg=angle,old_selection=obs['selection'],new_selection=fresh['selection'],
                        old_forward_estimate=depth_estimate(obs),new_forward_estimate=depth_estimate(fresh)))
                    obs,folder=fresh,fresh_folder;continue
                atomic_json(rd/'execution.json',dict(decision_observation_id=obs['observation_id'],execution_observation_id=fresh['observation_id']))
                result,(obs,folder)=client.execute(proposal,fresh)
                history.append(dict(proposal=proposal,result={k:v for k,v in result['motion'].items() if k!='state_trace'}))
                print('运动指令发送完成，观测已保存：'+str(folder))
            else:print('已达到轮数上限，停止本次任务。')
        finally:
            if client:
                try:client.stop()
                except Exception as exc:print('停止请求未确认：'+str(exc),file=sys.stderr)


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--goal',required=True);p.add_argument('--output',required=True)
    p.add_argument('--thor-host',default='Thor');p.add_argument('--thor-port',type=int,default=18000)
    p.add_argument('--model-base',help='existing /v1 URL; disables automatic Thor tunnel')
    p.add_argument('--robot-base',default='http://127.0.0.1:6210')
    p.add_argument('--prompt',default=str(root/'docs/astribot/planar_control/agent_system_prompt.txt'))
    p.add_argument('--max-rounds',type=int,default=20)
    p.add_argument('--replay',help='saved observation directory; never connects to robot')
    p.add_argument('--session',help='session.json for replay limits')
    args=p.parse_args()
    if args.max_rounds<=0:p.error('--max-rounds must be positive')
    try:run(args)
    except (Exception,KeyboardInterrupt) as exc:
        print('Agent 退出：'+str(exc),file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
