from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import s1b_agent_run as launcher


def test_grounded_defaults_use_full_success_recording():
    args = launcher.parse_args(['--decision-mode', 'grounded'])
    assert args.decision_mode == 'grounded'
    assert args.demo_last_steps is None
    assert args.failure_case is None
    assert Path(args.demo).name == '20261101_182424_352b5047'
    assert Path(args.prompt) == ROOT/'docs/astribot/planar_control/agent_system_prompt.txt'
    assert args.auto_confirm is True


def test_direct_command_auto_confirms_unless_explicitly_disabled():
    argv = ['--decision-mode', 'direct', '--demo-last-steps', '3', '--no-failure-case',
            '--prompt', 'docs/astribot/planar_control/agent_system_prompt_red_marker.txt']
    assert launcher.parse_args(argv).auto_confirm is True
    assert launcher.parse_args(argv + ['--no-auto-confirm']).auto_confirm is False


def test_one_key_main_forwards_grounded_defaults_without_launching(monkeypatch):
    captured=[]
    monkeypatch.setattr(sys,'argv',['s1b_agent_run.py','--decision-mode','grounded'])
    monkeypatch.setattr(launcher,'launch',lambda args:captured.append(args) or 0)
    assert launcher.main()==0
    assert len(captured)==1
    assert captured[0].decision_mode=='grounded' and captured[0].demo_last_steps is None
    assert captured[0].failure_case is None


@pytest.mark.parametrize('mode', ['direct', 'separated'])
def test_legacy_modes_keep_case_defaults_and_explicit_overrides(mode):
    args = launcher.parse_args(['--decision-mode', mode])
    assert args.demo_last_steps is None
    assert Path(args.failure_case).name == '20261101_184824_dcb23038'
    args = launcher.parse_args(['--decision-mode', mode, '--no-failure-case', '--demo-last-steps', '2'])
    assert args.failure_case is None and args.demo_last_steps == 2


def test_grounded_accepts_explicit_failure_case_and_selection():
    args = launcher.parse_args(['--decision-mode', 'grounded', '--failure-case', 'chosen', '--demo-last-steps', '4'])
    assert args.failure_case == 'chosen' and args.demo_last_steps == 4
    assert launcher.parse_args(['--no-failure-case']).failure_case is None
    with pytest.raises(SystemExit):
        launcher.parse_args(['--no-failure-case', '--failure-case', 'chosen'])
    for count in ['0', '-1']:
        with pytest.raises(SystemExit):
            launcher.parse_args(['--demo-last-steps', count])


def test_invalid_demo_fails_before_starting_robot(tmp_path, monkeypatch):
    from pressb.agent_loop.common import atomic_json
    root=tmp_path/'demo';atomic_json(root/'session.json',dict(schema='wrong'))
    monkeypatch.setattr(launcher.sys,'stdin',SimpleNamespace(isatty=lambda:True))
    monkeypatch.setattr(launcher,'check_port_available',lambda port:None)
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('robot must not start'))
    args=launcher.parse_args(['--demo',str(root)])
    with pytest.raises(ValueError,match='keyboard_demo_schema_required'):
        launcher.launch(args)


def test_missing_prompt_fails_before_starting_robot(tmp_path, monkeypatch):
    from pressb.agent_loop.common import atomic_json
    root=tmp_path/'demo';atomic_json(root/'session.json',{})
    monkeypatch.setattr(launcher.sys,'stdin',SimpleNamespace(isatty=lambda:True))
    monkeypatch.setattr(launcher,'check_port_available',lambda port:None)
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('robot must not start'))
    args=launcher.parse_args(['--demo',str(root),'--decision-mode','direct','--no-failure-case',
                             '--demo-last-steps','3','--prompt',str(tmp_path/'missing.txt')])
    with pytest.raises(ValueError,match='无法读取 Agent 提示词'):
        launcher.launch(args)


@pytest.mark.parametrize('entry', ['s1b_agent.py', 's1b_agent_run.py'])
def test_agent_entry_help_lists_grounded_and_tail(entry):
    result=subprocess.run([sys.executable,str(ROOT/'scripts'/entry),'--help'],
                          capture_output=True,text=True,timeout=10)
    assert result.returncode==0
    assert '--decision-mode {localized,grounded,separated,direct}' in result.stdout
    assert '--demo-last-steps' in result.stdout
    assert '--prompt' in result.stdout
    assert 'default: full demonstration' in ' '.join(result.stdout.split())


def test_standalone_main_defaults_archive_full_demo_before_network(tmp_path,monkeypatch):
    import s1b_agent as agent
    def forbidden(*a,**k):
        pytest.fail('preflight must not contact robot or model')
    def halt(*a,**k):
        raise RuntimeError('offline-preflight-boundary')
    monkeypatch.setattr(agent,'request',forbidden)
    monkeypatch.setattr(agent,'RobotClient',forbidden)
    monkeypatch.setattr(agent,'tunnel',halt)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['s1b_agent.py','--goal','test','--output',str(output),
        '--decision-mode','grounded',
        '--demo',str(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047'),
        '--replay',str(tmp_path/'unused_observation')])
    assert agent.main()==1
    task=json.loads((output/'task.json').read_text())
    assert task['decision_mode']=='grounded' and task['input_format']=='pressb-conversation-v3'
    assert task['failure_case'] is None and task['initial_image_count']==13
    assert task['demo_selection']['mode']=='all'
    assert task['demo_selection']['selected_steps']==[f'{step:06d}' for step in range(1,12)]
    assert task['demo_selection']['model_image_count']==11
    assert json.loads((output/'result.json').read_text())['error']=='offline-preflight-boundary'
