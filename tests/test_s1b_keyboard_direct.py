import pytest
from s1b_keyboard_direct import target_for_key


def test_cartesian_keyboard_axes_and_fixed_orientation():
    p=[.4,-.2,1.,0.,0.,0.,1.]
    for key,index,delta in [('w',2,.002),('s',2,-.002),('a',1,.002),('d',1,-.002)]:
        result=target_for_key(key,p,.4,.002)
        assert result[index]==pytest.approx(p[index]+delta)
        assert result[3:]==p[3:]
        assert p==[.4,-.2,1.,0.,0.,0.,1.]


@pytest.mark.parametrize('distance',[2,35.5,100])
def test_forward_uses_entered_mm_and_retract_preserves_uv(distance):
    p=[.4,-.2,1.,0.,0.,0.,1.]
    forward=target_for_key('f',p,.4,.002,distance)
    assert forward[0]==pytest.approx(.4+distance/1000)
    assert target_for_key('f',forward,.4,.002,distance)==forward
    assert target_for_key('r',forward,.4,.002)==p
    assert target_for_key('?',p,.4,.002) is None


@pytest.mark.parametrize('answer', ['yes', ''])
def test_sdk_ownership_prompt_is_interactive(monkeypatch, answer):
    import builtins
    from s1b_keyboard_direct import connect_robot
    reader=lambda: answer
    monkeypatch.setattr(builtins,'input',reader)
    def factory(**kwargs):
        assert kwargs['high_control_rights'] is False
        return builtins.input()
    assert connect_robot(factory,250)==answer
    assert builtins.input is reader


def test_sdk_eof_declines_instead_of_triggering_takeover_fallback(monkeypatch):
    import builtins
    from s1b_keyboard_direct import connect_robot
    def eof():raise EOFError()
    monkeypatch.setattr(builtins,'input',eof)
    assert connect_robot(lambda **kw: builtins.input(),250)==''
    assert builtins.input is eof


def test_sdk_constructor_failure_restores_input(monkeypatch):
    import builtins
    from s1b_keyboard_direct import connect_robot
    reader=lambda:'yes'
    monkeypatch.setattr(builtins,'input',reader)
    def fail(**kw):raise RuntimeError('init failure')
    with pytest.raises(RuntimeError,match='init failure'):
        connect_robot(fail,250)
    assert builtins.input is reader


def test_gripper_commands_preserve_arm_and_only_target_right_gripper():
    from s1b_keyboard_direct import GRIPPER_KEYS, send_targets
    class SDK:
        def __init__(self):self.cartesian=[];self.mixed=[]
        def set_cartesian_pose(self,*args,**kwargs):self.cartesian.append((args,kwargs))
        def set_different_type_command(self,*args,**kwargs):self.mixed.append((args,kwargs))
    sdk=SDK()
    names=['astribot_arm_right','astribot_torso']
    arm=[.4,-.2,1.,0,0,0,1];torso=[0,0,.5,0,0,0,1]
    send_targets(sdk,names,arm,torso,None)
    assert not sdk.mixed
    for key,expected in [('c',100.),('o',0.)]:
        send_targets(sdk,names,arm,torso,GRIPPER_KEYS[key])
        args,options=sdk.mixed[-1]
        assert args==(names+['astribot_gripper_right'],['cartesian','cartesian','joints'],[arm,torso,[expected]])
        assert options==dict(control_way='filter',use_wbc=False)
    assert len(sdk.cartesian)==1
    assert len(sdk.mixed)==2  # exactly one native set per tick, including gripper changes


def test_native_frequency_warning_is_folded_but_other_errors_survive():
    import os
    from pathlib import Path
    import subprocess
    import sys
    code = """
import os
from s1b_keyboard_direct import SDKOutput
output=SDKOutput()
for _ in range(100):
    os.write(2,b'WARNING: The calling frequency of the set function does not match.\\n')
os.write(1,b'RIGHT GRIPPER OPEN\\n')
os.write(2,b'ERROR: lost control rights\\n')
output.close()
"""
    result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(Path('scripts').resolve())),capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert result.stdout.count('calling frequency of the set function')==1
    assert 'RIGHT GRIPPER OPEN' in result.stdout
    assert 'ERROR: lost control rights' in result.stdout
    assert '99' in result.stdout


def test_web_option_uploads_sidecar_and_forwards_loopback(monkeypatch):
    from types import SimpleNamespace
    import s1b_keyboard_direct as module
    import start_s1b_inference
    calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda command,**kwargs:calls.append(command))
    monkeypatch.setattr(module,'run_ssh',lambda command:calls.append(command) or 0)
    monkeypatch.setattr(start_s1b_inference,'check_port_available',lambda port:None)
    args=SimpleNamespace(host='S1B',sdk_root='/sdk',step_mm=5,speed_mm_s=50,control_hz=250,
                         view_cameras=True,view_port=8765)
    assert module.launch_remote(args)==0
    assert any(str(x).endswith('s1b_camera_view.py') for x in calls[1])
    assert any(str(x).endswith('s1b_depth_metadata.cpp') for x in calls[1])
    assert '127.0.0.1:8765:127.0.0.1:8765' in calls[-1]
    assert '--view-cameras --view-port 8765' in calls[-1][-1]
    calls.clear();args.view_cameras=False
    module.launch_remote(args)
    assert '-L' not in calls[-1]
    assert not any('s1b_camera_view.py' in str(x) for x in calls[1])


def test_mistyped_ownership_answer_reprompts(monkeypatch):
    import builtins
    from s1b_keyboard_direct import connect_robot
    answers=iter(['ddddddddyes','yes'])
    monkeypatch.setattr(builtins,'input',lambda:next(answers))
    assert connect_robot(lambda **kw:builtins.input(),250)=='yes'


def test_only_refused_web_channels_are_silenced(capfd):
    import sys
    from s1b_keyboard_direct import run_ssh
    code="import sys;print('channel 2: open failed: connect failed: Connection refused',file=sys.stderr);print('Permission denied',file=sys.stderr)"
    assert run_ssh([sys.executable,'-c',code])==0
    captured=capfd.readouterr()
    assert 'Connection refused' not in captured.err
    assert 'Permission denied' in captured.err


def test_viewer_exit_is_reported_before_sdk_start():
    from types import SimpleNamespace
    from s1b_keyboard_direct import wait_for_viewer
    dead=SimpleNamespace(poll=lambda:1,returncode=1,pid=123)
    with pytest.raises(RuntimeError,match='启动失败'):
        wait_for_viewer(dead,8765,timeout=.1)


@pytest.mark.parametrize('distance',[None,0,-1,float('nan'),float('inf')])
def test_forward_requires_valid_distance(distance):
    with pytest.raises(ValueError):
        target_for_key('f',[.4,-.2,1.,0,0,0,1],.4,.002,distance)


def test_keyboard_input_loop_wasd_and_forward_prompt(monkeypatch,capsys):
    """Exercise run_robot's actual input/servo loops, with hardware calls stubbed."""
    import sys
    import time
    from types import SimpleNamespace
    import s1b_keyboard_direct as module
    pose=[.4,-.2,1.,0.,0.,0.,1.]
    robot=SimpleNamespace(
        get_control_rights_status=lambda:True,
        set_head_follow_effector=lambda value:None,
        get_current_cartesian_pose=lambda *a,**kw:[list(pose),list(pose)],
        stop_robot=lambda:None)
    sent=[]
    robot.set_cartesian_pose=lambda names,poses,**kw:sent.append(list(poses[0]))
    robot.astribot_interface=SimpleNamespace(
        shutdown_control_rights_srv=lambda:None,
        executor=SimpleNamespace(shutdown=lambda **kw:None),
        spin_thread=SimpleNamespace(join=lambda **kw:None,is_alive=lambda:False),
        shutdown=lambda:None)
    monkeypatch.setitem(sys.modules,'astribot_sdk.core.astribot_api.astribot_client',SimpleNamespace(Astribot=lambda **kw:robot))
    monkeypatch.setitem(sys.modules,'rclpy',SimpleNamespace(ok=lambda:False))
    monkeypatch.setattr(module,'connect_robot',lambda *a:robot)
    monkeypatch.setattr(module.sys,'stdin',SimpleNamespace(fileno=lambda:12345))
    monkeypatch.setattr(module.signal,'signal',lambda *a:None)
    monkeypatch.setattr(module.termios,'tcgetattr',lambda *a:[])
    monkeypatch.setattr(module.termios,'tcsetattr',lambda *a:None)
    monkeypatch.setattr(module.termios,'tcflush',lambda *a:None)
    monkeypatch.setattr(module.tty,'setcbreak',lambda *a:None)
    monkeypatch.setattr(module.select,'select',lambda *a:([12345],[],[]))
    keys=iter([b'w',b's',b'a',b'd',b'f',b'\n',b'w',b'f',b'35\n',b'r',b'd',b'q'])
    def read(fd,n):
        time.sleep(.28)
        return next(keys)
    monkeypatch.setattr(module.os,'read',read)
    module.run_robot(SimpleNamespace(control_hz=100,step_mm=2,speed_mm_s=1000))
    output=capsys.readouterr().out
    targets=[line.split('目标 XYZ（mm）：')[1] for line in output.splitlines() if '目标 XYZ（mm）：' in line]
    assert targets==[
        '[400.0, -200.0, 1002.0]', '[400.0, -200.0, 1000.0]',
        '[400.0, -198.0, 1000.0]', '[400.0, -200.0, 1000.0]',
        '[400.0, -200.0, 1002.0]',  # W works after cancelling F with an empty line.
        '[435.0, -200.0, 1002.0]', '[400.0, -200.0, 1002.0]',
        '[400.0, -202.0, 1002.0]']

    import ast
    for target in targets:
        xyz=[v/1000 for v in ast.literal_eval(target)]
        assert any(command[:3]==pytest.approx(xyz) for command in sent)


@pytest.mark.parametrize('typed,expected',[(b'35\n',35.),(b'2.5\n',2.5),(b'\n',None)])
def test_forward_normal_terminal_input(typed,expected):
    import os
    import termios
    import threading
    import tty
    from s1b_keyboard_direct import read_forward_distance
    master,slave=os.openpty()
    original=termios.tcgetattr(slave)
    tty.setcbreak(slave)
    writer=threading.Timer(.05,lambda:os.write(master,typed))
    try:
        writer.start()
        assert read_forward_distance(slave,original,threading.Event())==expected
        assert not termios.tcgetattr(slave)[3]&termios.ICANON
    finally:
        writer.join();os.close(master);os.close(slave)
