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
DEFAULT_PROMPT = ROOT/'docs/astribot/planar_control/agent_system_prompt.txt'
LOCALIZED_PROMPT = ROOT/'docs/astribot/planar_control/agent_system_prompt_localized.txt'
sys.path.insert(0, str(ROOT/'src'))
from pressb.agent_loop.common import atomic_json, request, SESSION_SCHEMA
from pressb.agent_loop.demonstration import Demonstration
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
            raise TimeoutError('等待机器人服务超时；请查看终端 / 网页中的 SDK 和服务启动日志。')
        stop.wait(.2)
    raise InterruptedError('operator_stopped')


def aim_depth_window(camera_base, point, stop, timeout=10):
    """Point the wrist depth window at the merged-end pixel for this run only.

    The press phase trusts depth only when it is read at the calibrated aim pixel.
    The selection is not saved, so the robot's stored measuring point is unchanged
    after the run. Returns the confirmed center pixel; raises before any agent runs.
    """
    request(camera_base, '/api/selection', dict(point_normalized=list(point)), timeout=3)
    deadline = time.monotonic()+timeout
    while not stop.is_set():
        snapshot = request(camera_base, '/api/snapshot', timeout=2)
        selected = (snapshot.get('selection') or {}).get('point_normalized')
        center = (snapshot.get('depth') or {}).get('center_px')
        if selected == list(point) and center is not None:
            return center
        if time.monotonic() >= deadline:
            raise TimeoutError('测距点未切换到合拢末端像素；请检查相机网页。')
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
        raise RuntimeError('请在交互终端运行，以便使用 q 回车停止机器人服务。')
    auto_confirm = getattr(args, 'auto_confirm', True)
    ports = (args.robot_port, args.camera_port, args.web_port, args.thor_port)
    if len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError('端口必须互不相同且在1–65535之间')
    for port in ports:
        check_port_available(port)
    if args.demo is not None and not (Path(args.demo).expanduser()/'session.json').is_file():
        raise ValueError('找不到示范 session.json：'+args.demo)
    aim_point = None
    if getattr(args, 'fingertip_calibration', None):
        from pressb.agent_loop import calibrated_tip
        # Validate before SDK takeover; the merged-end point also sets the depth window.
        aim_point = calibrated_tip.load(Path(args.fingertip_calibration).expanduser(), args.fingertip_state)['aim']
    decision_mode = getattr(args, 'decision_mode', 'localized')
    prompt_path = Path(getattr(args, 'prompt', None) or
                       (LOCALIZED_PROMPT if decision_mode == 'localized' else DEFAULT_PROMPT)).expanduser().resolve()
    try:
        prompt_path.read_text()
    except OSError as exc:
        raise ValueError('无法读取 Agent 提示词：'+str(prompt_path)) from exc
    demo_last_steps = getattr(args, 'demo_last_steps', None)
    failure_path = getattr(args, 'failure_case', None)
    if failure_path:
        # Fail before SDK takeover if any required case evidence is missing.
        FailureCase(failure_path)
    # Validate the selected activation evidence before starting any robot process.
    demo = Demonstration(args.demo) if args.demo is not None else None
    if demo_last_steps is not None and demo is not None:
        demo.select_activation_tail(demo_last_steps)
    demo_images = len(demo.images) if demo else 0
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    directory = Path(args.output).expanduser() if args.output else ROOT/'outputs'/'agent_web'/(stamp+'_'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    directory = directory.resolve()
    stop = threading.Event()
    close_dashboard = threading.Event()
    lock = threading.Lock()
    robot_base = f'http://127.0.0.1:{args.robot_port}'
    camera_base = f'http://127.0.0.1:{args.camera_port}'
    state = dict(phase='starting', goal=args.goal, output=str(directory),
                 decision_mode=decision_mode, demo=str(demo.root) if demo else None, demo_last_steps=demo_last_steps,
                 demo_selection=demo.selection if demo else None, failure_case=failure_path, prompt=str(prompt_path),
                 depth_aim_point=aim_point,
                 execution_policy='yz_alignment_before_forward' if decision_mode == 'localized' else 'legacy_validated',
                 forward_guard_policy='wrist_tip_overlap' if decision_mode == 'localized' else 'wrist_horizontal',
                 progress_monitor_format=('pressb-localized-progress-v3' if decision_mode == 'localized' else
                                          'pressb-progress-v1' if decision_mode == 'direct' else None),
                 view_acquisition_format='pressb-view-acquisition-v1' if decision_mode == 'direct' else None,
                 localization_format='pressb-localization-v3' if decision_mode == 'localized' else None,
                 hypothesis_review_format='pressb-hypothesis-review-v1' if decision_mode == 'localized' else None,
                 forward_guard_format='pressb-wrist-tip-overlap-v1' if decision_mode == 'localized' else None,
                 input_format='pressb-conversation-v5' if decision_mode == 'localized' else None,
                 auto_confirm=auto_confirm)
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
        emit(f'网页：http://127.0.0.1:{server.server_port}\n记录目录：{directory}\n')
        emit(('SDK 控制权接管将自动回复 yes；' if auto_confirm else '请在本终端处理 SDK 接管提示；')
             + '服务就绪后 Agent 将自动运行。\n')
        if decision_mode == 'localized':
            emit(f'定位闭环：成功示范 {demo_images} 张、失败参考 {"启用" if failure_path else "关闭"}；'
                 '每轮独立双图定位→原始偏差和实际反馈→模型自主决策。thinking 关闭。\n'
                 '先做Y/Z对齐；每个正X候选检查当前腕图物理指端是否落入按钮框（横纵两轴）。\n'
                 '未知则拒绝整条候选并重新决策；通过后原XYZ不变，无正X步长上限；负X/Y/Z不受此条件限制。\n'
                 '每个新观测首次前推及触发无进展复核时，会增加串行模型请求。\n')
        elif decision_mode == 'grounded':
            emit(f'短闭环：成功示范 {demo_images} 张、失败参考 {"启用" if failure_path else "关闭"}；'
                 '当前双图独立判读后决策，必要时单独验证撤回；thinking 关闭。\n')
        elif decision_mode == 'direct':
            emit('直接模式：先检查头部取景，取景不足时只允许头部调整/重新观察/结束；'
                 '每轮记录进展，异常时复核，判断冲突时恢复取景。thinking 关闭。\n')
        try:
            update(phase='waiting_for_service')
            service_argv = [sys.executable, str(ROOT/'scripts/s1b_agent_service.py'), '--host', args.host,
                '--port', str(args.robot_port), '--view-port', str(args.camera_port)]
            if auto_confirm:
                service_argv.append('--auto-confirm')
            service = spawn(service_argv, sys.stdin)
            session = wait_for_service(service, robot_base, stop)
            if aim_point is not None:
                center = aim_depth_window(camera_base, aim_point, stop)
                update(depth_aim_center_px=center)
                emit(f'测距点已设为合拢末端像素 {tuple(center)}（仅本次运行，不保存）。\n')
            update(phase='running', session_id=session['session_id'])
            # Only SSH/service reads this terminal. q reaches the robot service;
            # its exit immediately cancels the Agent, including a pending model call.
            agent_argv = [sys.executable, str(ROOT/'scripts/s1b_agent.py'), '--goal', args.goal,
                '--output', str(directory/'agent'),
                '--prompt', str(prompt_path),
                '--robot-base', robot_base, '--thor-host', args.thor_host,
                '--thor-port', str(args.thor_port), '--max-rounds', str(args.max_rounds),
                '--context-budget-tokens', str(getattr(args, 'context_budget_tokens', 65536)),
                '--decision-mode', decision_mode]
            if getattr(args, 'fingertip_calibration', None):
                agent_argv.extend(['--fingertip-calibration', str(Path(args.fingertip_calibration).expanduser().resolve()),
                                   '--fingertip-state', args.fingertip_state,
                                   '--press-overpress-mm', str(args.press_overpress_mm)])
            if demo is not None:
                agent_argv.extend(['--demo', str(Path(args.demo).expanduser().resolve())])
            if demo_last_steps is not None and demo is not None:
                agent_argv.extend(['--demo-last-steps', str(demo_last_steps)])
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
    parser.add_argument('--auto-confirm', action=argparse.BooleanOptionalAction, default=True,
                        help='automatically answer yes to SDK takeover prompts (default: enabled)')
    parser.add_argument('--goal', default='按下电梯外呼向下按钮并撤回')
    parser.add_argument('--prompt', help='Agent system prompt file; default follows decision mode')
    demo_group = parser.add_mutually_exclusive_group()
    demo_group.add_argument('--demo', default=None,
                            help='keyboard session to show the model as a demonstration; default: none, e.g. '
                                 'outputs/keyboard_sessions/20261101_182424_352b5047')
    demo_group.add_argument('--no-demo', action='store_const', const=None, dest='demo',
                            help='run without any demonstration images in the model context (default)')
    parser.add_argument('--demo-last-steps', type=int,
                        help='optionally select completed steps ending at reviewed activation; default: full demonstration')
    failure_group = parser.add_mutually_exclusive_group()
    failure_group.add_argument('--failure-case',
        default=argparse.SUPPRESS,
        help='reviewed failure run; localized/grounded omit by default, direct/separated keep the prior default')
    failure_group.add_argument('--no-failure-case', action='store_const', const=None, dest='failure_case',
                               default=argparse.SUPPRESS, help='explicitly omit failure context')
    parser.add_argument('--output', help='new session directory; default outputs/agent_web/<timestamp_id>')
    parser.add_argument('--robot-port', type=int, default=6210)
    parser.add_argument('--camera-port', type=int, default=8765)
    parser.add_argument('--web-port', type=int, default=8766)
    parser.add_argument('--thor-host', default='Thor')
    parser.add_argument('--thor-port', type=int, default=18000)
    parser.add_argument('--context-budget-tokens', type=int, default=65536)
    parser.add_argument('--decision-mode', choices=('localized', 'grounded', 'separated', 'direct'), default='localized')
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--fingertip-calibration', help='human-marked fingertip JSON; the merged-end closed pixel becomes the fixed alignment and contact point')
    parser.add_argument('--fingertip-state', choices=('closed', 'open'), default='open',
                        help='gripper state at start; recorded only, never changes the aim point')
    parser.add_argument('--press-overpress-mm', type=float, default=5.0,
                        help='closed-gripper press target is live wrist depth 150 mm minus this (default 5 mm)')
    args = parser.parse_args(argv)
    if args.prompt is None:
        args.prompt = str(LOCALIZED_PROMPT if args.decision_mode == 'localized' else DEFAULT_PROMPT)
    if not hasattr(args, 'failure_case'):
        args.failure_case = (None if args.decision_mode in ('localized', 'grounded') else
                            str(ROOT/'outputs/agent_web/20261101_184824_dcb23038'))
    if args.demo_last_steps is not None and args.demo_last_steps <= 0:
        parser.error('--demo-last-steps must be positive')
    if args.context_budget_tokens <= 0:
        parser.error('--context-budget-tokens must be positive')
    if args.max_rounds <= 0:
        parser.error('--max-rounds must be positive')
    if not 0 <= args.press_overpress_mm <= 15:
        parser.error('--press-overpress-mm must be between 0 and 15')
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
