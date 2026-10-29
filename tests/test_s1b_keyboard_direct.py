import pytest
from s1b_keyboard_direct import target_for_key


def test_cartesian_keyboard_axes_and_fixed_orientation():
    p=[.4,-.2,1.,0.,0.,0.,1.]
    for key,index,delta in [('w',2,.002),('s',2,-.002),('a',1,.002),('d',1,-.002)]:
        result=target_for_key(key,p,.4,.002)
        assert result[index]==pytest.approx(p[index]+delta)
        assert result[3:]==p[3:]
        assert p==[.4,-.2,1.,0.,0.,0.,1.]


def test_forward_is_absolute_100mm_and_retract_preserves_uv():
    p=[.4,-.2,1.,0.,0.,0.,1.]
    forward=target_for_key('f',p,.4,.002)
    assert forward[0]==pytest.approx(.5)
    assert target_for_key('f',forward,.4,.002)==forward
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
        def __init__(self):self.cartesian=[];self.joints=[]
        def set_cartesian_pose(self,*args,**kwargs):self.cartesian.append((args,kwargs))
        def set_joints_position(self,*args,**kwargs):self.joints.append((args,kwargs))
    sdk=SDK()
    names=['astribot_arm_right','astribot_torso']
    arm=[.4,-.2,1.,0,0,0,1];torso=[0,0,.5,0,0,0,1]
    send_targets(sdk,names,arm,torso,None)
    assert not sdk.joints
    for key,expected in [('c',100.),('o',0.)]:
        send_targets(sdk,names,arm,torso,GRIPPER_KEYS[key])
        args,options=sdk.joints[-1]
        assert args==(['astribot_gripper_right'],[[expected]])
        assert options==dict(control_way='filter',use_wbc=False,add_default_torso=False)
    assert all(call==sdk.cartesian[0] for call in sdk.cartesian)
