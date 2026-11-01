#!/usr/bin/env python3
"""One terminal: robot service, Qwen agent, and local live dashboard."""
import argparse
import codecs
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop.common import atomic_json, request, SESSION_SCHEMA
from pressb.agent_loop.failure_case import FailureCase
from s1b_agent_dashboard import Dashboard, serve
from start_s1b_inference import check_port_available


def wait_for_service(process, base, stop, timeout=180):
    deadline = time.monotonic()+timeout
    while not stop.is_set():
        if process.poll() is not None:
            raise RuntimeError('机器人服务启动失败或已退出；请查看终端 / 网页日志。')
        try:
            session = request(base, '/v2/session', timeout=.5)
        except (OSError, ValueError):
            session = None
        if session is not None:
            if session.get('schema') != SESSION_SCHEMA:
                raise RuntimeError('机器人服务版本不匹配')
            return session
        if time.monotonic() >= deadline:
            raise TimeoutError('等待机器人服务超过180秒；SDK接管提示需要在终端输入 yes。')
        stop.wait(.2)
    raise InterruptedError('operator_stopped')


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def relay_output(stream, emit):
    # SDK input prompts need not end in a newline. Forward available bytes
    # immediately while preserving UTF-8 characters split across reads.
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    while chunk := stream.read1(4096):
        emit(decoder.decode(chunk))
    emit(decoder.decode(b'', final=True))


def launch(args):
    if not sys.stdin.isatty():
        raise RuntimeError('请在交互终端运行，以便输入 SDK 控制权接管确认。')
    ports = (args.robot_port, args.camera_port, args.web_port, args.thor_port)
    if len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError('端口必须互不相同且在1–65535之间')
    for port in ports:
        check_port_available(port)
    if not (Path(args.demo).expanduser()/'session.json').is_file():
        raise ValueError('找不到示范 session.json：'+args.demo)
    failure_path = getattr(args, 'failure_case', None)
    if failure_path:
        # Fail before SDK takeover if any required case evidence is missing.
        FailureCase(failure_path)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    directory = Path(args.output).expanduser() if args.output else ROOT/'outputs'/'agent_web'/(stamp+'_'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    directory = directory.resolve()
    stop = threading.Event()
    close_dashboard = threading.Event()
    lock = threading.Lock()
    robot_base = f'http://127.0.0.1:{args.robot_port}'
    camera_base = f'http://127.0.0.1:{args.camera_port}'
    state = dict(phase='starting', goal=args.goal, output=str(directory))
    log = (directory/'terminal.log').open('a', buffering=1)
    server = service = agent = None
    readers = []
    handlers = {}
    code = 0

    def update(**values):
        state.update(values)
        atomic_json(directory/'launcher.json', state)

    def emit(text):
        with lock:
            print(text, end='', flush=True)
            log.write(text)

    def spawn(argv, stdin):
        process = subprocess.Popen(argv, cwd=ROOT, stdin=stdin, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        reader = threading.Thread(target=relay_output, args=(process.stdout, emit), daemon=True)
        reader.start()
        readers.append(reader)
        return process

    def interrupt(*unused):
        stop.set()
        close_dashboard.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, interrupt)
        update()
        server, web_thread = serve(Dashboard(directory, camera_base, stop), args.web_port)
        emit(f'网页：http://127.0.0.1:{server.server_port}\n记录目录：{directory}\n'
             '请在本终端处理 SDK 接管提示；服务就绪后 Agent 将自动运行。\n')
        try:
            update(phase='waiting_for_service')
            service = spawn([sys.executable, str(ROOT/'scripts/s1b_agent_service.py'), '--host', args.host,
                '--port', str(args.robot_port), '--view-port', str(args.camera_port)], sys.stdin)
            session = wait_for_service(service, robot_base, stop)
            update(phase='running', session_id=session['session_id'])
            # Only SSH/service reads this terminal. q reaches the robot service;
            # its exit immediately cancels the Agent, including a pending model call.
            agent_argv = [sys.executable, str(ROOT/'scripts/s1b_agent.py'), '--goal', args.goal,
                '--demo', str(Path(args.demo).expanduser().resolve()), '--output', str(directory/'agent'),
                '--robot-base', robot_base, '--thor-host', args.thor_host,
                '--thor-port', str(args.thor_port), '--max-rounds', str(args.max_rounds)]
            if failure_path:
                agent_argv.extend(['--failure-case', str(Path(failure_path).expanduser().resolve())])
            agent = spawn(agent_argv, subprocess.DEVNULL)
            while agent.poll() is None and not stop.is_set():
                if service.poll() is not None:
                    stop.set()
                    update(phase='stopping', reason='robot_service_exited')
                    break
                stop.wait(.1)
            if stop.is_set():
                update(phase='stopping', reason=state.get('reason', 'operator_stopped'))
                stop_process(agent)
            else:
                code = agent.returncode
            update(phase='finished' if code == 0 else 'error', agent_exit_code=agent.poll())
        except (Exception, KeyboardInterrupt) as exc:
            code = 1
            stop.set()
            update(phase='error', error=str(exc))
            emit('启动 / 执行已停止：'+str(exc)+'\n')
        finally:
            if service is not None and service.poll() is None:
                try:
                    request(robot_base, '/v2/stop', {}, timeout=2)
                except (OSError, ValueError):
                    pass
            stop_process(agent)
            if service is not None:
                try:
                    service.wait(5)
                except subprocess.TimeoutExpired:
                    stop_process(service)
            for reader in readers:
                reader.join(2)
            # At this point no owned control/model child remains active.
            emit('机器人 / Agent 进程已结束。网页保留最后画面和日志；Ctrl-C 关闭网页。\n')
        while not close_dashboard.wait(.2):
            pass
    finally:
        stop_process(agent)
        stop_process(service)
        if server:
            server.shutdown()
            server.server_close()
            web_thread.join(2)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        log.close()
    return code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='S1B')
    parser.add_argument('--goal', default='按下电梯外呼向下按钮并撤回')
    parser.add_argument('--demo', default=str(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047'))
    failure_group = parser.add_mutually_exclusive_group()
    failure_group.add_argument('--failure-case',
        default=str(ROOT/'outputs/agent_web/20261101_174626_c319289f'),
        help='reviewed failure run directory; defaults to the 20261101 failed trial')
    failure_group.add_argument('--no-failure-case', action='store_const', const=None, dest='failure_case',
                               help='explicitly omit failure context')
    parser.add_argument('--output', help='new session directory; default outputs/agent_web/<timestamp_id>')
    parser.add_argument('--robot-port', type=int, default=6210)
    parser.add_argument('--camera-port', type=int, default=8765)
    parser.add_argument('--web-port', type=int, default=8766)
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--thor-port', type=int, default=18000)
    parser.add_argument('--max-rounds', type=int, default=20)
    args = parser.parse_args(argv)
    if args.max_rounds <= 0:
        parser.error('--max-rounds must be positive')
    return args


def main():
    args = parse_args()
    try:
        return launch(args)
    except (Exception, KeyboardInterrupt) as exc:
        print('启动退出：'+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
