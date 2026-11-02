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


@pytest.mark.parametrize('auto_confirm', [False, True])
def test_sdk_constructor_failure_restores_input(monkeypatch, auto_confirm):
    import builtins
    from s1b_keyboard_direct import connect_robot
    reader=lambda:'yes'
    monkeypatch.setattr(builtins,'input',reader)
    def fail(**kw):raise RuntimeError('init failure')
    with pytest.raises(RuntimeError,match='init failure'):
        connect_robot(fail,250,auto_confirm=auto_confirm)
    assert builtins.input is reader


def test_sdk_auto_confirmation_does_not_consume_terminal_input(monkeypatch, capsys):
    import builtins
    import s1b_keyboard_direct as keyboard
    reader=lambda *a, **k:pytest.fail('auto confirmation must not read stdin')
    monkeypatch.setattr(builtins, 'input', reader)
    monkeypatch.setattr(keyboard.termios, 'tcflush',
                        lambda *a:pytest.fail('auto confirmation must preserve queued stop input'))
    def factory(**kwargs):
        assert kwargs['high_control_rights'] is False
        assert kwargs['freq'] == 250
        return builtins.input('SDK takeover? ')
    assert keyboard.connect_robot(factory, 250, auto_confirm=True) == 'yes'
    assert builtins.input is reader
    assert 'SDK takeover? yes' in capsys.readouterr().out


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
    assert any(str(x).endswith('s1b_keyboard_control.py') for x in calls[1])
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


@pytest.mark.parametrize('scenario',['normal','stop_input','stop_motion','discard_input','q_motion','sdk_error','head'])
@pytest.mark.parametrize('recording',[False,True])
def test_xyz_input_and_servo_loop(monkeypatch,capsys,scenario,recording,tmp_path):
    import sys
    import time
    from types import SimpleNamespace
    import s1b_keyboard_direct as module
    pose=[.4,-.2,1.,0.,0.,0.,1.]
    stopped=[];sent=[];grippers=[];heads=[]
    robot=SimpleNamespace(
        get_control_rights_status=lambda:True,
        set_head_follow_effector=lambda value:None,
        get_current_cartesian_pose=lambda *a,**kw:[list(pose),list(pose)],
        stop_robot=lambda:stopped.append(True))
    def cartesian(names,poses,**kw):
        if scenario=='sdk_error' and poses[0][2]>1.:raise RuntimeError('native_send_failed')
        sent.append(list(poses[0]))
    robot.set_cartesian_pose=cartesian
    robot.get_current_joints_position=lambda names:[[0.,0.]]
    def head_set(names,values,**kw):
        assert names==['astribot_head']
        assert kw==dict(control_way='filter',use_wbc=False,add_default_torso=False)
        heads.append(list(values[0]))
    robot.set_joints_position=head_set
    def mixed(names,types,poses,**kw):
        sent.append(list(poses[0]));grippers.append(poses[2][0])
    robot.set_different_type_command=mixed
    robot.astribot_interface=SimpleNamespace(
        shutdown_control_rights_srv=lambda:None,
        executor=SimpleNamespace(shutdown=lambda **kw:None),
        spin_thread=SimpleNamespace(join=lambda **kw:None,is_alive=lambda:False),
        shutdown=lambda:None)
    monkeypatch.setitem(sys.modules,'astribot_sdk.core.astribot_api.astribot_client',SimpleNamespace(Astribot=lambda **kw:robot))
    monkeypatch.setitem(sys.modules,'rclpy',SimpleNamespace(ok=lambda:False))
    monkeypatch.setattr(module,'connect_robot',lambda *a:robot)
    monkeypatch.setattr(module.sys,'stdin',SimpleNamespace(fileno=lambda:12345))
    handlers={}
    monkeypatch.setattr(module.signal,'signal',lambda sig,fn:handlers.update({sig:fn}))
    monkeypatch.setattr(module.termios,'tcgetattr',lambda *a:[])
    monkeypatch.setattr(module.termios,'tcsetattr',lambda *a:None)
    flushes=[]
    monkeypatch.setattr(module.termios,'tcflush',lambda *a:flushes.append(True))
    discarded=[]
    def select(*args):
        time.sleep(.01)
        if scenario=='stop_motion':handlers[module.signal.SIGINT]()
        if scenario in ('discard_input','q_motion') and not discarded:return [12345],[],[]
        return [],[],[]
    def during_motion(*args):
        discarded.append(True)
        return b'q\n' if scenario=='q_motion' else b'999\n'
    monkeypatch.setattr(module.os,'read',during_motion)
    monkeypatch.setattr(module.select,'select',select)
    entries=iter((['h','15','-10'] if scenario=='head' else [])+['20','-5','10','c','-20','5','-10','o','0','0','0','q'])
    count=[0]
    def read(fd,stop,on_input=None):
        time.sleep(.03)
        if scenario=='stop_input':
            handlers[module.signal.SIGINT]()
            return 'q'
        index=max(0,count[0]-3) if scenario=='head' else count[0]
        if index<3:
            assert all(p==pose for p in sent)  # Even X+Y do not start motion.
        if index==3:assert sent[-1][:3]==pytest.approx([.42,-.205,1.01])
        if index==7:assert sent[-1]==pytest.approx(pose)
        count[0]+=1
        value=next(entries)
        if on_input:on_input(value+'\n')
        return value
    monkeypatch.setattr(module,'read_line',read)
    rec=None
    if recording:
        from s1b_keyboard_record import Recorder
        from test_s1b_keyboard_record import make_bundle
        rec=Recorder(tmp_path/'record','http://unused',{},capture_timeout=.03)
        def fetch(timeout):
            if rec.active:
                op=rec.active
                expected=op['start_command_pose_m_xyzw'] if op['status']=='prepared' else op['target_command_pose_m_xyzw']
                if op['status'] in ('prepared','completed'):assert sent[-1]==pytest.approx(expected)
            return make_bundle()
        monkeypatch.setattr(rec,'_fetch',fetch)
    args=SimpleNamespace(control_hz=100,step_mm=999,speed_mm_s=1000,recorder=rec)
    if scenario=='sdk_error':
        with pytest.raises(RuntimeError,match='native_send_failed'):module.run_robot(args)
    else:module.run_robot(args)
    if rec:rec.close()
    output=capsys.readouterr().out
    if scenario in ('stop_input','stop_motion','q_motion','sdk_error'):
        assert stopped
        assert '本轮插值指令发送完成' not in output
        assert count[0]==(0 if scenario=='stop_input' else 3)
        if rec and rec.operations:assert rec.operations[0]['status']=='interrupted'
        return
    if scenario=='discard_input':assert discarded
    assert output.count('本轮位移 XYZ（mm）')==3
    assert output.count('本轮插值指令发送完成')==(4 if scenario=='head' else 3)
    if scenario=='head':
        import math
        assert heads[-1]==pytest.approx([math.radians(15),math.radians(-10)])
        assert all(-math.radians(10)-1e-8<=p[1]<=0 for p in heads)
    assert set(grippers)=={0.,100.}
    assert stopped and len(flushes)==(6 if recording else 4)+(1 if scenario=='head' else 0)
    if rec:
        assert len(rec.operations)==(6 if scenario=='head' else 5)
        if scenario=='head':
            assert rec.operations[0]['kind']=='head'
            assert rec.operations[0]['command']['head_deg']==[15.,-10.]
            assert rec.operations[0]['completion_basis']=='command_interpolation'
        assert all(op['status']=='completed' for op in rec.operations)
        assert all(op['before']['status']==op['after']['status']=='complete' for op in rec.operations)
    # Every command stays on one combined XYZ line, preserving orientation.
    for p in sent:
        fraction=(p[0]-.4)/.02
        assert p[1]==pytest.approx(-.2-.005*fraction)
        assert p[2]==pytest.approx(1.+.01*fraction)
        assert p[3:]==pose[3:]


@pytest.mark.parametrize('lines,expected',[
    (['20','-5','10'],[20.,-5.,10.]),
    (['1000000','-.5','0'],[1000000.,-.5,0.]),
    (['bad','nan','inf','1','oops','2','-inf','3'],[1.,2.,3.]),
    ([''],None),(['1',''],None),(['1','2',''],None),
    (['q'],'q'),(['1','q'],'q'),(['1','2','q'],'q'),
    (['c'],'c'),(['o'],'o'),
])
def test_xyz_prompt_validation_and_cancellation(monkeypatch,lines,expected):
    import threading
    import s1b_keyboard_direct as module
    values=iter(lines)
    monkeypatch.setattr(module,'read_line',lambda *a:next(values))
    assert module.read_xyz(0,threading.Event())==expected


def test_xyz_unbounded_mm_conversion():
    from s1b_keyboard_direct import target_for_xyz
    pose=[.4,-.2,1.,0,0,0,1]
    assert target_for_xyz(pose,[1000000,-500000,0])==pytest.approx([1000.4,-500.2,1.,0,0,0,1])
    assert pose==[.4,-.2,1.,0,0,0,1]


@pytest.mark.parametrize('typed,expected',[(b'35\n-5\n2.5\n',[35.,-5.,2.5]),
                                         (b'3x\x7f5\n0\n0\n',[35.,0.,0.]),
                                         (b'20\n\n',None)])
def test_xyz_normal_terminal_input(typed,expected):
    import os
    import threading
    from s1b_keyboard_direct import read_xyz
    master,slave=os.openpty()
    writer=threading.Timer(.05,lambda:os.write(master,typed))
    try:
        writer.start()
        assert read_xyz(slave,threading.Event())==expected
    finally:
        writer.join();os.close(master);os.close(slave)


def test_line_wait_responds_to_stop():
    import os
    import threading
    from s1b_keyboard_direct import read_line
    master,slave=os.openpty();stop=threading.Event()
    timer=threading.Timer(.05,stop.set)
    try:
        timer.start()
        assert read_line(slave,stop)=='q'
    finally:
        timer.join();os.close(master);os.close(slave)


@pytest.mark.parametrize('lines,expected',[
    (['h','0','-15'],{'head_deg':[0.,-15.]}),
    (['h','nan','10','inf','-5'],{'head_deg':[10.,-5.]}),
    (['h',''],None),(['h','10',''],None),(['h','10','q'],'q'),
])
def test_head_prompt(monkeypatch,lines,expected):
    import threading
    import s1b_keyboard_direct as module
    values=iter(lines)
    monkeypatch.setattr(module,'read_line',lambda *a:next(values))
    assert module.read_xyz(0,threading.Event())==expected
