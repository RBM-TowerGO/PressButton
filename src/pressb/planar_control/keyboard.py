"""Local terminal jogging. Forward is a fixed 100 mm offset, never cumulative."""
import select
import os
import sys
import termios
import threading
import tty
import uuid


def submit_key(controller, key, step_mm=2.):
    moves = {'w': (0, step_mm), 's': (0, -step_mm),
             'a': (-step_mm, 0), 'd': (step_mm, 0)}
    if key not in (*moves, 'f', 'r'):
        raise ValueError('unknown_key')
    observation = controller.observe()
    args = dict(observation_id=observation['observation_id'], command_id=str(uuid.uuid4()))
    if key in moves:
        du, dv = moves[key]
        return controller.submit('move', **args, du_mm=du, dv_mm=dv)
    if key == 'f':
        return controller.submit('advance', **args, normal_mm=100.)
    return controller.submit('retract', **args)


def run_keyboard(controller, step_mm=2., check=lambda: None):
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    worker = None
    last_status = None
    print(f'W/S 上下，A/D 左右（每次 {step_mm:g} mm）；F 向前 100 mm 并保持；R 退回平面。')
    print('空格停止并锁定；Q/Ctrl-C 退出。前伸后须先 R 撤回才能平移；动作中不排队。', flush=True)

    def action(key):
        try:
            receipt = submit_key(controller, key, step_mm)
            print('\r\n接受：'+receipt['task_id'], flush=True)
        except Exception as exc:
            print('\r\n拒绝：'+str(exc), flush=True)

    try:
        tty.setcbreak(fd)
        while not controller.cancel.is_set():
            check()
            state = controller.get_state()
            status = (state['phase'], str(state['result']))
            if status != last_status:
                print('\r\n'+state['phase']+' '+str(state['result'] or ''), flush=True)
                last_status = status
            if not select.select([sys.stdin], [], [], .05)[0]:
                continue
            keys = os.read(fd, 64).decode(errors='ignore').lower()
            if not keys or any(key in keys for key in (' ', 'q', '\x03')):
                controller.stop('keyboard_stop')
                break
            if '\x1b' in keys:
                continue
            actionable = [key for key in keys if key in 'wasdfr']
            if not actionable:
                continue
            key = actionable[-1]
            if state['phase'] != 'READY' or (worker and worker.is_alive()):
                print('\r\n正在执行，忽略移动键；空格可停止。', flush=True)
                continue
            # Observation/FK can take time; keep the terminal stop key responsive.
            worker = threading.Thread(target=action, args=(key,), daemon=True)
            worker.start()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        controller.stop('keyboard_exit')
        if worker:
            worker.join(timeout=2)
