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
from pressb.agent_loop.client import RobotClient,propose
from pressb.agent_loop.conversation import Conversation, image_count as count_images
from pressb.agent_loop.short_loop import ShortConversation
from pressb.agent_loop.model_context import FORMAT
from pressb.agent_loop import calibrated_tip
from pressb.agent_loop import press_phase
from pressb.agent_loop import model_provider
from pressb.agent_loop.common import atomic_json,request,SESSION_SCHEMA
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.failure_case import FailureCase

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT/'docs/astribot/planar_control/agent_system_prompt.txt'
LOCALIZED_PROMPT = ROOT/'docs/astribot/planar_control/agent_system_prompt_localized.txt'
END_TO_END_PROMPT = ROOT/'docs/astribot/planar_control/agent_system_prompt_end_to_end.txt'
DECISION_MODES = ('localized', 'grounded', 'separated', 'direct', 'end_to_end')


def conversation_class(mode):
    if mode == 'end_to_end':
        from pressb.agent_loop.end_to_end import EndToEndConversation
        return EndToEndConversation
    if mode == 'localized':
        from pressb.agent_loop.localized_conversation import LocalizedConversation
        return LocalizedConversation
    return ShortConversation if mode == 'grounded' else Conversation


def default_prompt(mode):
    if mode == 'end_to_end':
        return END_TO_END_PROMPT
    return LOCALIZED_PROMPT if mode == 'localized' else DEFAULT_PROMPT


def forward_guard_policy(mode):
    if mode == 'end_to_end':
        return 'none'
    return 'wrist_tip_overlap' if mode == 'localized' else 'wrist_horizontal'


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


def _report_range(conversation, oid):
    """Print this observation's freeze decision, made inside prepare_decision."""
    result = getattr(conversation, 'range_evidence', {}).get(oid)
    if not result:
        return
    if result.get('froze'):
        print(f'测距已冻结：z_b = {result["median_m"]:.4f} m，'
              f'行程上限 ≤ {press_phase.describe()["budget_m"]*1000:.1f} mm。', flush=True)
    elif result.get('role') == 'live':
        state = press_phase.describe(oid)
        live = f'{result["median_m"]*1000:.1f} mm' if result['median_m'] is not None else '无'
        print(f'实时测距：{live}（{"可信" if result["trusted"] else "不可信："+", ".join(result["reasons"])}），'
              f'目标 {state["target_m"]*1000:.0f} mm，本轮可前推 ≤ {state["remaining_m"]*1000:.1f} mm'
              f'（{state["bound"]}）。', flush=True)
    elif not result['trusted'] and press_phase.phase() == press_phase.ALIGN:
        print('测距未采用：'+', '.join(result['reasons']), flush=True)


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
    active_rd = None
    decision_mode = getattr(args, 'decision_mode', 'localized')
    outcome = dict(status='error', model_reported_done=False, human_verified_success=None,
                   human_verified_retracted=None, completed_motion_commands=0)
    try:
        prompt_path = Path(getattr(args, 'prompt', None) or default_prompt(decision_mode)).expanduser().resolve()
        prompt = prompt_path.read_text()
        demo = Demonstration(args.demo) if getattr(args, 'demo', None) else None
        tail = getattr(args, 'demo_last_steps', None)
        if tail is not None and demo:
            demo.select_activation_tail(tail)
        if demo:
            demo.save(output)
        failure = FailureCase(args.failure_case) if getattr(args, 'failure_case', None) else None
        if failure:
            failure.save(output)
        image_count = (len(demo.images) if demo else 0) + (len(failure.images) if failure else 0) + 2
        print(f'上下文：成功/历史示范 {len(demo.images) if demo else 0} 张，'
              f'失败案例 {len(failure.images) if failure else 0} 张，当前双视角 2 张；首轮共 {image_count} 张，'
              + ('后续保留当前双图和短运行记忆。' if decision_mode == 'grounded' else '后续追加。'), flush=True)
        calibrated_tip.deactivate()
        press_phase.reset()
        tip_calibration, press_limits = None, None
        if getattr(args, 'fingertip_calibration', None):
            tip_calibration = calibrated_tip.activate(args.fingertip_calibration, args.fingertip_state)
            if decision_mode == 'end_to_end':
                # Only the pixel is given to the model as a reference; no press-phase gate.
                print('指端标定：合拢末端像素 '+str([round(v, 4) for v in tip_calibration['point_normalized']])
                      +' 作为参考信息提供给模型，不做按压阶段限制。', flush=True)
            else:
                press_limits = press_phase.configure(
                    overpress_m=args.press_overpress_mm/1000.0,
                    max_travel_m=args.press_max_travel_mm/1000.0)
                press_phase.set_gripper(args.fingertip_state)
                print('指端标定：对准与接触点固定为合拢末端像素 '
                      +str([round(v, 4) for v in tip_calibration['point_normalized']])
                      +'（夹爪实测状态 '+tip_calibration['gripper_state']+'），模型只定位按钮。', flush=True)
                print('按压分阶段：张开时在瞄准像素读按钮深度 z_b 并冻结为行程上限；闭合后每轮用实时深度闭环，'
                      f'前推到读数 {150-args.press_overpress_mm:.0f} mm（150 mm 接触 + {args.press_overpress_mm:.0f} mm 过压），'
                      '且不超过冻结上限。', flush=True)
        provider_name = getattr(args, 'model_provider', 'thor')
        provider = model_provider.configure(provider_name, model=getattr(args, 'model_name', None),
                                            base=args.model_base if provider_name != 'thor' else None,
                                            key_env=getattr(args, 'model_api_key_env', None))
        external = provider_name != 'thor'
        model_base = provider['base'] if external else (args.model_base or f'http://127.0.0.1:{args.thor_port}/v1')
        print(f'模型：{provider["provider"]} / {provider["model"]} @ {model_base}'
              + ('（外部 API：本次图像与提示词会发送到该服务）' if external else ''), flush=True)
        atomic_json(output/'task.json', dict(goal=args.goal, model_base=model_base, model_provider=provider, prompt=prompt,
            prompt_path=str(prompt_path),
            fingertip_calibration=tip_calibration,
            press_phase=press_limits,
            demo=str(demo.root) if demo else None, initial_image_count=image_count,
            failure_case=str(failure.root) if failure else None,
            input_format=('pressb-conversation-e2e-v1' if decision_mode == 'end_to_end' else
                          'pressb-conversation-v5' if decision_mode == 'localized' else
                          'pressb-conversation-v3' if decision_mode == 'grounded' else
                          'pressb-conversation-v2' if decision_mode == 'separated' else FORMAT),
            demo_selection=getattr(demo,'selection',None),
            decision_mode=decision_mode,
            execution_policy=('model_action_executed_as_given' if decision_mode == 'end_to_end' else
                              'yz_alignment_before_forward' if decision_mode == 'localized' else 'legacy_validated'),
            forward_guard_policy=forward_guard_policy(decision_mode),
            progress_monitor_format=('pressb-localized-progress-v3' if decision_mode == 'localized' else
                                     'pressb-progress-v1' if decision_mode == 'direct' else None),
            view_acquisition_format='pressb-view-acquisition-v1' if decision_mode == 'direct' else None,
            perception_format='pressb-grounding-v2' if decision_mode == 'grounded' else None,
            localization_format='pressb-localization-v3' if decision_mode == 'localized' else None,
            hypothesis_review_format='pressb-hypothesis-review-v1' if decision_mode == 'localized' else None,
            forward_guard_format='pressb-wrist-tip-overlap-v1' if decision_mode == 'localized' else None,
            control_policy_version='pressb-control-plan-v1' if decision_mode == 'grounded' else None,
            context_budget_tokens=getattr(args, 'context_budget_tokens', None),
            failure_analysis_version=failure.review['analysis_version'] if failure else None,
            execution='automatic_one_command_at_a_time'))
        print('决策模式：'+decision_mode+('；每轮一次模型请求：直接看当前双图和记录输出动作，无独立定位/前推检查/复核。thinking 首次闭合夹爪前关闭，之后低档开启至运行结束。'
              if decision_mode == 'end_to_end' else '；每轮独立双图定位→原始坐标偏差和实际反馈→模型自主决策；'
              '坐标显式换算为0～1；连续同向且无明确进展时额外只读复核，再重新决策；thinking 首次闭合夹爪前关闭，之后低档开启至运行结束。'
              if decision_mode == 'localized' else '；当前双图定位按钮/指尖→计算图像偏差与阶段建议→动作决策；撤回后按需核验。thinking 首次闭合夹爪前关闭，之后低档开启至运行结束。'
              if decision_mode == 'grounded' else '；每轮先独立判读再选择动作，增加一次串行模型请求。'
              if decision_mode == 'separated' else '；先检查头部取景，再直接决策；异常时复核，判断冲突时恢复取景。thinking 首次闭合夹爪前关闭，之后低档开启至运行结束。'), flush=True)
        if decision_mode == 'end_to_end':
            print('执行策略：模型输出的命令按原样执行，不检查、不裁剪、不替换；程序只校验命令格式。', flush=True)
        elif decision_mode == 'localized':
            print('执行策略：先用Y/Z对齐；每个正X候选都检查当前腕图，箭头所指物理指端须在按钮框内（横纵均满足）。'
                  '未知不放行，拒绝整条候选后重新决策；无正X步长上限，不裁剪XYZ；负X后撤和Y/Z不受此条件限制。'
                  '每个新观测首次前推额外请求腕图判读；无进展时仍额外复核原假设。', flush=True)
        else:
            print('旧模式前推限制：正X发送前仅检查当前腕图横向，按钮须位于可辨认的两指之间；'
                  '未知不放行。不检查高度，候选前推增加一次独立腕图请求。', flush=True)
        with tunnel(None if (args.model_base or external) else args.thor_host, args.thor_port), StopControl() as control:
            try:
                if args.replay:
                    folder = Path(args.replay)
                    obs = json.loads((folder/'manifest.json').read_text())
                    session = json.loads(Path(args.session).read_text()) if args.session else dict(schema=SESSION_SCHEMA,
                        speed_mm_s=20., head_speed_deg_s=20., control_hz=250.)
                    conversation = conversation_class(decision_mode)(args.goal, obs, (folder/'head.jpg').read_bytes(),
                        (folder/'right.jpg').read_bytes(), session, prompt, demo, failure,
                        getattr(args, 'context_budget_tokens', None), decision_mode=decision_mode,
                        **(dict(monitor_progress=True, acquire_view=True) if decision_mode == 'direct' else {}))
                    active_rd = output/'replay'
                    body = control.call(conversation.prepare_decision, model_base, active_rd, control.cancel)
                    proposal = control.call(propose, model_base, body, obs, session, output/'replay',
                                            cancel=control.cancel, conversation=conversation)
                    print(json.dumps(proposal, ensure_ascii=False, indent=2))
                    outcome.update(status='replay_only', proposal=proposal)
                    return
                client = RobotClient(args.robot_base, output, control.cancel,
                    **({'forward_guard_policy': forward_guard_policy(decision_mode)}
                       if decision_mode in ('localized', 'end_to_end') else {}))
                control.client = client
                atomic_json(output/'session.json', client.session)
                obs, folder = control.call(client.observe)
                conversation = conversation_class(decision_mode)(args.goal, obs, (folder/'head.jpg').read_bytes(),
                    (folder/'right.jpg').read_bytes(), client.session, prompt, demo, failure,
                    getattr(args, 'context_budget_tokens', None), decision_mode=decision_mode,
                    **(dict(monitor_progress=True, acquire_view=True) if decision_mode == 'direct' else {}))
                if hasattr(conversation, 'max_press_attempts'):
                    conversation.max_press_attempts = args.max_press_attempts
                    print(f'按压未亮灯时不允许提前放弃：最多尝试 {args.max_press_attempts} 次按压后才接受 report_unable。', flush=True)
                print('自动闭环已启动；q 回车或 Ctrl-C 随时停止。', flush=True)
                for index in range(args.max_rounds):
                    rd = output/'rounds'/f'{index:04d}'
                    active_rd = rd
                    if press_phase.active():
                        atomic_json(rd/'press_phase.json', press_phase.describe())
                    print(f'第 {index+1} 轮：'+('直接决策' if decision_mode == 'end_to_end' else
                          '独立定位画面中' if decision_mode == 'localized' else
                          '独立判读画面中' if decision_mode in ('separated','grounded') else '整理动作上下文'), flush=True)
                    body = control.call(conversation.prepare_decision, model_base, rd, control.cancel)
                    if press_phase.active():
                        _report_range(conversation, obs['observation_id'])
                    print(f'第 {index+1} 轮：请求 {provider["model"]}；{count_images(body["messages"])} 张图，'
                          + ('短运行记忆' if decision_mode == 'grounded' else f'压缩代次 {conversation.generation}')
                          + f'；当前双视角 {folder}', flush=True)
                    proposal = control.call(propose, model_base, body, obs, client.session, rd,
                                            cancel=control.cancel, conversation=conversation)
                    print(json.dumps(proposal, ensure_ascii=False, indent=2), flush=True)
                    client.check_running()
                    conversation.check_observation(obs['observation_id'])
                    if hasattr(conversation, 'check_execution'):
                        conversation.check_execution(proposal, obs['observation_id'])
                    name = proposal['name']
                    if name == 'report_done':
                        outcome.update(status='model_reported_done', model_reported_done=True,
                                       evidence=proposal['arguments']['evidence'])
                        break
                    if name == 'report_unable':
                        outcome.update(status='model_unable', reason=proposal['arguments']['reason'])
                        break
                    if name == 'request_observation':
                        obs, folder = control.call(client.observe)
                        conversation.append_result(proposal, None, obs, (folder/'head.jpg').read_bytes(),
                                                   (folder/'right.jpg').read_bytes())
                        atomic_json(rd/'observation_result.json', dict(observation_id=obs['observation_id']))
                        continue
                    atomic_json(rd/'execution.json', dict(decision_observation_id=obs['observation_id'],
                                execution_observation_id=obs['observation_id'], mode='automatic'))
                    result, (obs, folder) = control.call(client.execute, proposal, obs,
                        forward_check=getattr(conversation, 'forward_guard_decision', None))
                    atomic_json(rd/'result.json', result)
                    outcome['completed_motion_commands'] += 1
                    if name == 'propose_gripper':
                        state_now = 'closed' if proposal['arguments']['key'] == 'c' else 'open'
                        calibrated_tip.set_state(state_now)
                        # The press starts at the first executed close: from here every model
                        # request thinks (low effort) until the run ends, even after reopening.
                        if state_now == 'closed' and model_provider.enable_thinking():
                            outcome['thinking_enabled_at_round'] = index+1
                            atomic_json(rd/'thinking.json', dict(enabled=True, round=index+1,
                                        fields=model_provider.describe()['thinking']['fields']))
                            print('按压开始：thinking 已开启（低档），保持到本次运行结束。', flush=True)
                        if press_phase.active():
                            press_phase.set_gripper(state_now)
                            print('夹爪状态：'+state_now+('；测距预算已冻结' if state_now == 'closed'
                                  else '；重新开始测距'), flush=True)
                    if press_phase.active() and name == 'propose_move':
                        press_phase.record_move(proposal['arguments']['xyz_mm'])
                    conversation.append_result(proposal, result, obs, (folder/'head.jpg').read_bytes(),
                                               (folder/'right.jpg').read_bytes())
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
        if active_rd is not None:
            if not (active_rd/'error.json').exists():
                atomic_json(active_rd/'error.json', dict(stage='round', error_type=type(exc).__name__, error=str(exc)))
            context_path = active_rd/'decision_context.json'
            if context_path.exists():
                context = json.loads(context_path.read_text())
                context['stage'] = 'interrupted' if outcome['status'] == 'operator_stopped' else 'failed'
                atomic_json(context_path, context)
        raise
    finally:
        atomic_json(output/'result.json', outcome)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--goal',required=True);p.add_argument('--output',required=True)
    p.add_argument('--thor-host',default='Thor');p.add_argument('--thor-port',type=int,default=18000)
    p.add_argument('--model-base',help='existing /v1 URL; disables automatic Thor tunnel')
    p.add_argument('--model-provider',choices=sorted(model_provider.PROVIDERS),default='thor',
                   help='thor (local Qwen, default), deepseek (deepseek-flash, needs DEEPSEEK_API_KEY) or openai_compatible')
    p.add_argument('--model-name',help='override the provider default model name')
    p.add_argument('--model-api-key-env',help='environment variable holding the API key; the key itself is never saved')
    p.add_argument('--robot-base',default='http://127.0.0.1:6210')
    p.add_argument('--prompt',help='system prompt file; default follows decision mode')
    p.add_argument('--context-budget-tokens', type=int, default=None, help='optional context token limit; default: no limit, no compaction')
    p.add_argument('--decision-mode', choices=DECISION_MODES, default='localized',
                   help='end_to_end: one model call per round that looks at the images and outputs the action')
    p.add_argument('--max-rounds',type=int,default=40)
    p.add_argument('--max-press-attempts',type=int,default=3,
                   help='report_unable is rejected after a missed press until this many press attempts were made')
    p.add_argument('--fingertip-calibration',help='human-marked fingertip JSON (s1b-fingertip-point-v1); the merged-end closed pixel becomes the fixed alignment and contact point and the model locates only the button')
    p.add_argument('--fingertip-state',choices=('closed','open'),default='open',help='gripper state at start, recorded only so the program knows whether depth at the aim pixel still sees the button; it never changes the aim point')
    p.add_argument('--press-overpress-mm',type=float,default=5.0,help='push past the 150 mm contact reading by this much; live depth target is 150 minus this (default 5 mm)')
    p.add_argument('--press-max-travel-mm',type=float,default=85.0,help='hard ceiling on the closed-gripper forward stroke (default 85 mm)')
    p.add_argument('--demo',help='original keyboard session directory with operations and before/after images')
    p.add_argument('--demo-last-steps',type=int,help='optionally select last N completed actions through reviewed activation; default: full demonstration')
    failure_group = p.add_mutually_exclusive_group()
    failure_group.add_argument('--failure-case', help='reviewed failure run directory (case_review.json and agent/)')
    failure_group.add_argument('--no-failure-case', action='store_const', const=None, dest='failure_case',
                               help='explicitly omit failure context (standalone Agent default)')
    p.add_argument('--replay',help='saved observation directory; never connects to robot')
    p.add_argument('--session',help='keyboard-agent session.json for replay')
    args=p.parse_args(argv)
    if args.prompt is None:
        args.prompt = str(default_prompt(args.decision_mode))
    if args.context_budget_tokens is not None and args.context_budget_tokens<=0:p.error('--context-budget-tokens must be positive')
    if args.max_rounds<=0:p.error('--max-rounds must be positive')
    if args.max_press_attempts<0:p.error('--max-press-attempts must not be negative')
    if args.demo_last_steps is not None and args.demo_last_steps<=0:p.error('--demo-last-steps must be positive')
    return args


def main():
    args = parse_args()
    try:run(args)
    except (Exception,KeyboardInterrupt) as exc:
        print('Agent 退出：'+str(exc),file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
