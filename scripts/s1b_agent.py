#!/usr/bin/env python3
"""Demonstration-conditioned dual-camera agent using the keyboard command executor."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import os
import queue
import select
import signal
import threading
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pressb.agent_loop.client import RobotClient,build_request,propose
from pressb.agent_loop.common import atomic_json,request,SESSION_SCHEMA
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.failure_case import FailureCase


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


class StopControl:
    """Terminal/signal cancellation stays responsive during all network requests."""
    def __init__(self):
        self.cancel = threading.Event()
        self.closed = threading.Event()
        self.client = None
        self.ack = None
        self.error = None
        self.handlers = {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                self.handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda *args: self.cancel.set())
        self.watcher = threading.Thread(target=self._watch, daemon=True)
        self.watcher.start()
        return self

    def _watch(self):
        try:
            fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        except (OSError, ValueError):
            fd = None
        pending = b''
        while not self.closed.is_set():
            if self.cancel.is_set():
                if self.client:
                    try:
                        self.ack = self.client.stop()
                    except Exception as exc:
                        self.error = str(exc)
                return
            if fd is None:
                self.closed.wait(.05)
            elif select.select([fd], [], [], .05)[0]:
                data = os.read(fd, 4096)
                if not data:
                    self.cancel.set()
                pending += data
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if line.strip().lower() == b'q':
                        self.cancel.set()

    def call(self, function, *args, **kwargs):
        if self.cancel.is_set():
            raise InterruptedError('operator_stopped')
        result = queue.Queue()
        def worker():
            try:
                result.put((True, function(*args, **kwargs)))
            except BaseException as exc:
                result.put((False, exc))
        threading.Thread(target=worker, daemon=True).start()
        while not self.cancel.is_set():
            try:
                ok, value = result.get(timeout=.05)
            except queue.Empty:
                continue
            if self.cancel.is_set():
                break
            if not ok:
                raise value
            return value
        raise InterruptedError('operator_stopped')

    def __exit__(self, *args):
        self.closed.set()
        self.watcher.join(.2)
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)


def run(args):
    if not args.replay:
        try:
            session = request(args.robot_base, '/v2/session', timeout=3)
        except (OSError, ValueError) as exc:
            raise RuntimeError('无法连接机器人 API '+args.robot_base+
                '。请先运行 python scripts/s1b_agent_service.py --host S1B。 原因：'+str(exc)) from exc
        if session.get('schema') != SESSION_SCHEMA:
            raise ValueError('keyboard_session_schema_required: 请部署新版共用键盘执行器服务')
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=False)
    client = None
    outcome = dict(status='error', model_reported_done=False, human_verified_success=None,
                   human_verified_retracted=None, completed_motion_commands=0)
    try:
        prompt = Path(args.prompt).read_text()
        demo = Demonstration(args.demo) if getattr(args, 'demo', None) else None
        if demo:
            demo.save(output)
        failure = FailureCase(args.failure_case) if getattr(args, 'failure_case', None) else None
        if failure:
            failure.save(output)
        image_count = (len(demo.images) if demo else 0) + (len(failure.images) if failure else 0) + 2
        print(f'上下文：成功/历史示范 {len(demo.images) if demo else 0} 张，'
              f'失败案例 {len(failure.images) if failure else 0} 张，当前双视角 2 张；共 {image_count} 张。', flush=True)
        model_base = args.model_base or f'http://127.0.0.1:{args.thor_port}/v1'
        atomic_json(output/'task.json', dict(goal=args.goal, model_base=model_base, prompt=prompt,
            demo=str(demo.root) if demo else None, expected_images_per_request=image_count,
            failure_case=str(failure.root) if failure else None,
            failure_analysis_version=failure.review['analysis_version'] if failure else None,
            execution='automatic_one_command_at_a_time'))
        with tunnel(None if args.model_base else args.thor_host, args.thor_port), StopControl() as control:
            try:
                if args.replay:
                    folder = Path(args.replay)
                    obs = json.loads((folder/'manifest.json').read_text())
                    session = json.loads(Path(args.session).read_text()) if args.session else dict(schema=SESSION_SCHEMA,
                        speed_mm_s=20., head_speed_deg_s=20., control_hz=250.)
                    body = build_request(args.goal, obs, (folder/'right.jpg').read_bytes(), session, [], prompt,
                                         (folder/'head.jpg').read_bytes(), demo, failure)
                    proposal = control.call(propose, model_base, body, obs, session, output/'replay', cancel=control.cancel)
                    print(json.dumps(proposal, ensure_ascii=False, indent=2))
                    outcome.update(status='replay_only', proposal=proposal)
                    return
                client = RobotClient(args.robot_base, output, control.cancel)
                control.client = client
                atomic_json(output/'session.json', client.session)
                obs, folder = control.call(client.observe)
                history = []
                print('自动闭环已启动；q 回车或 Ctrl-C 随时停止。', flush=True)
                for index in range(args.max_rounds):
                    rd = output/'rounds'/f'{index:04d}'
                    body = build_request(args.goal, obs, (folder/'right.jpg').read_bytes(), client.session,
                                         history, prompt, (folder/'head.jpg').read_bytes(), demo, failure)
                    print(f'第 {index+1} 轮：请求 Qwen；当前双视角 {folder}', flush=True)
                    proposal = control.call(propose, model_base, body, obs, client.session, rd, cancel=control.cancel)
                    print(json.dumps(proposal, ensure_ascii=False, indent=2), flush=True)
                    name = proposal['name']
                    if name == 'report_done':
                        outcome.update(status='model_reported_done', model_reported_done=True,
                                       evidence=proposal['arguments']['evidence'])
                        break
                    if name == 'report_unable':
                        outcome.update(status='model_unable', reason=proposal['arguments']['reason'])
                        break
                    if name == 'request_observation':
                        history.append(dict(proposal=proposal))
                        obs, folder = control.call(client.observe)
                        continue
                    atomic_json(rd/'execution.json', dict(decision_observation_id=obs['observation_id'],
                                execution_observation_id=obs['observation_id'], mode='automatic'))
                    result, (obs, folder) = control.call(client.execute, proposal, obs)
                    atomic_json(rd/'result.json', result)
                    outcome['completed_motion_commands'] += 1
                    history.append(dict(proposal=proposal, result={k:v for k,v in result['motion'].items() if k!='state_trace'}))
                    print('命令发送完成，新观测已保存：'+str(folder), flush=True)
                else:
                    outcome.update(status='max_rounds', reason='round_budget_exhausted')
            finally:
                if client:
                    try:
                        outcome['stop'] = client.stop()
                    except Exception as exc:
                        outcome['stop_error'] = str(exc)
                        print('停止请求未确认：'+str(exc), file=sys.stderr)
    except (Exception, KeyboardInterrupt) as exc:
        outcome.update(status='operator_stopped' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'error',
                       error=str(exc))
        raise
    finally:
        atomic_json(output/'result.json', outcome)


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--goal',required=True);p.add_argument('--output',required=True)
    p.add_argument('--thor-host',default='Thor');p.add_argument('--thor-port',type=int,default=18000)
    p.add_argument('--model-base',help='existing /v1 URL; disables automatic Thor tunnel')
    p.add_argument('--robot-base',default='http://127.0.0.1:6210')
    p.add_argument('--prompt',default=str(root/'docs/astribot/planar_control/agent_system_prompt.txt'))
    p.add_argument('--max-rounds',type=int,default=20)
    p.add_argument('--demo',help='original keyboard session directory with operations and before/after images')
    failure_group = p.add_mutually_exclusive_group()
    failure_group.add_argument('--failure-case', help='reviewed failure run directory (case_review.json and agent/)')
    failure_group.add_argument('--no-failure-case', action='store_const', const=None, dest='failure_case',
                               help='explicitly omit failure context (standalone Agent default)')
    p.add_argument('--replay',help='saved observation directory; never connects to robot')
    p.add_argument('--session',help='keyboard-agent session.json for replay')
    args=p.parse_args()
    if args.max_rounds<=0:p.error('--max-rounds must be positive')
    try:run(args)
    except (Exception,KeyboardInterrupt) as exc:
        print('Agent 退出：'+str(exc),file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
