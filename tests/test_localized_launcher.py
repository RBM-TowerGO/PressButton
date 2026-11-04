"""Localized entry defaults and model-only replay never take robot control."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import s1b_agent as agent
import s1b_agent_run as launcher


def test_localized_is_default_with_full_demo_and_separate_prompt():
    for args in (launcher.parse_args([]),
                 agent.parse_args(['--goal', 'test', '--output', '/tmp/not-created'])):
        assert args.decision_mode == 'localized'
        assert args.demo_last_steps is None
        assert args.failure_case is None
        assert args.context_budget_tokens is None
        assert args.max_rounds == 40
        assert Path(args.prompt) == agent.LOCALIZED_PROMPT == launcher.LOCALIZED_PROMPT
    # The one-key launcher runs without a demonstration unless one is given.
    assert launcher.parse_args([]).demo is None
    assert launcher.parse_args(['--no-demo']).demo is None
    assert Path(launcher.parse_args(['--demo', 'x/20261101_182424_352b5047']).demo).name == '20261101_182424_352b5047'
    with pytest.raises(SystemExit):
        launcher.parse_args(['--demo', 'x', '--no-demo'])


@pytest.mark.parametrize('mode', ['localized', 'grounded', 'separated', 'direct'])
def test_explicit_case_and_prompt_choices_are_preserved(mode):
    options = ['--decision-mode', mode, '--prompt', 'custom prompt.txt',
               '--failure-case', 'reviewed-case', '--demo-last-steps', '5',
               '--context-budget-tokens', '50000', '--max-rounds', '7']
    for args in (launcher.parse_args(options),
                 agent.parse_args(['--goal', 'test', '--output', '/tmp/not-created'] + options)):
        assert args.decision_mode == mode
        assert args.prompt == 'custom prompt.txt'
        assert args.failure_case == 'reviewed-case'
        assert args.demo_last_steps == 5
        assert args.context_budget_tokens == 50000 and args.max_rounds == 7
    assert launcher.parse_args(['--decision-mode', mode, '--no-failure-case']).failure_case is None


@pytest.mark.parametrize('mode', ['grounded', 'separated', 'direct'])
def test_legacy_modes_keep_their_original_default_prompt(mode):
    for args in (launcher.parse_args(['--decision-mode', mode]),
                 agent.parse_args(['--goal', 'test', '--output', '/tmp/not-created', '--decision-mode', mode])):
        assert Path(args.prompt) == launcher.DEFAULT_PROMPT == agent.DEFAULT_PROMPT


def test_one_key_defaults_are_forwarded_without_starting_services(monkeypatch):
    captured = []
    monkeypatch.setattr(sys, 'argv', ['s1b_agent_run.py'])
    monkeypatch.setattr(launcher, 'launch', lambda args: captured.append(args) or 0)
    assert launcher.main() == 0
    assert len(captured) == 1
    assert captured[0].decision_mode == 'localized'
    assert captured[0].demo_last_steps is None
    assert captured[0].auto_confirm is True
    assert captured[0].prompt == str(launcher.LOCALIZED_PROMPT)


def test_localized_preflight_archives_default_case_before_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('model-only preflight must not contact robot or model')
    monkeypatch.setattr(agent, 'request', forbidden)
    monkeypatch.setattr(agent, 'RobotClient', forbidden)
    def boundary(*args, **kwargs):
        raise RuntimeError('offline-localized-boundary')
    monkeypatch.setattr(agent, 'tunnel', boundary)
    output = tmp_path/'run'
    args = agent.parse_args(['--goal', 'press', '--output', str(output),
        '--demo', str(ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047'),
        '--replay', str(tmp_path/'unused-observation')])
    with pytest.raises(RuntimeError, match='offline-localized-boundary'):
        agent.run(args)
    task = json.loads((output/'task.json').read_text())
    assert task['decision_mode'] == 'localized'
    assert task['input_format'] == 'pressb-conversation-v5'
    assert task['localization_format'] == 'pressb-localization-v3'
    assert task['progress_monitor_format'] == 'pressb-localized-progress-v3'
    assert task['execution_policy'] == 'yz_alignment_before_forward'
    assert task['forward_guard_policy'] == 'wrist_tip_overlap'
    assert task['forward_guard_format'] == 'pressb-wrist-tip-overlap-v1'
    assert task['failure_case'] is None and task['initial_image_count'] == 13
    assert task['demo_selection']['mode'] == 'all'
    assert task['demo_selection']['selected_steps'] == [f'{step:06d}' for step in range(1,12)]
    assert task['demo_selection']['model_image_count'] == 11
    model_demo=json.loads((output/'demonstration/model_reference.json').read_text())
    assert len(model_demo['operations']) == 11
    assert model_demo['operations'][-1]['status'] == 'interrupted'
    assert model_demo['operations'][-1]['after']['missing_images'] == ['head','right']
    assert model_demo['session']['final']['status'] == 'missing'
    assert task['prompt_path'] == str(agent.LOCALIZED_PROMPT)
    assert task['prompt'] == agent.LOCALIZED_PROMPT.read_text()
    assert json.loads((output/'result.json').read_text())['completed_motion_commands'] == 0


def test_saved_observation_uses_localized_prepare_before_action_and_no_robot(tmp_path, monkeypatch):
    folder = tmp_path/'saved'; folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps({'observation_id': 'saved-observation'}))
    (folder/'head.jpg').write_bytes(b'saved-head')
    (folder/'right.jpg').write_bytes(b'saved-wrist')
    order = []
    class LocalizedSpy:
        def __init__(self, goal, obs, head, wrist, session, prompt, *args, **kwargs):
            assert kwargs == {'decision_mode': 'localized'}
            assert obs['observation_id'] == 'saved-observation'
            assert (head, wrist) == (b'saved-head', b'saved-wrist')
            assert session['schema'] == agent.SESSION_SCHEMA
            assert prompt == agent.LOCALIZED_PROMPT.read_text()
        def prepare_decision(self, base, directory, cancel):
            order.append('localized_prepare')
            return {'messages': [{'role': 'user', 'content': 'localized-evidence'}]}
    def choose(mode):
        assert mode == 'localized'
        return LocalizedSpy
    def propose(base, body, obs, session, directory, **kwargs):
        assert body['messages'][0]['content'] == 'localized-evidence'
        order.append('action')
        return {'name': 'report_unable', 'arguments': {'reason': 'offline replay'}}
    def forbidden(*args, **kwargs):
        pytest.fail('saved-observation replay must not instantiate or contact robot')
    monkeypatch.setattr(agent, 'conversation_class', choose)
    monkeypatch.setattr(agent, 'propose', propose)
    monkeypatch.setattr(agent, 'tunnel', lambda *args: nullcontext())
    monkeypatch.setattr(agent, 'request', forbidden)
    monkeypatch.setattr(agent, 'RobotClient', forbidden)
    output = tmp_path/'run'
    agent.run(agent.parse_args(['--goal', 'press', '--output', str(output), '--replay', str(folder)]))
    assert order == ['localized_prepare', 'action']
    result = json.loads((output/'result.json').read_text())
    assert result['status'] == 'replay_only' and result['completed_motion_commands'] == 0


@pytest.mark.parametrize('mode', ['localized', 'grounded', 'separated', 'direct', 'end_to_end'])
def test_live_entry_selects_client_policy_without_real_network(mode, tmp_path, monkeypatch):
    """The current wrist overlap policy reaches the execution client, not just the log."""
    folder = tmp_path/'observation'; folder.mkdir()
    (folder/'head.jpg').write_bytes(b'head')
    (folder/'right.jpg').write_bytes(b'wrist')
    obs = {'observation_id': 'current'}
    session = {'schema': agent.SESSION_SCHEMA}
    constructors = []
    class ClientSpy:
        def __init__(self, base, output, cancel, **kwargs):
            constructors.append(kwargs)
            self.session = session
        def observe(self):
            return obs, folder
        def check_running(self):
            pass
        def stop(self):
            return {'stopped': True}
    class ConversationSpy:
        generation = 0
        def __init__(self, *args, **kwargs):
            pass
        def prepare_decision(self, *args):
            return {'messages': []}
        def check_observation(self, oid):
            assert oid == 'current'
    monkeypatch.setattr(agent, 'request', lambda *args, **kwargs: session)
    monkeypatch.setattr(agent, 'RobotClient', ClientSpy)
    monkeypatch.setattr(agent, 'tunnel', lambda *args: nullcontext())
    monkeypatch.setattr(agent, 'conversation_class', lambda mode: ConversationSpy)
    monkeypatch.setattr(agent, 'propose', lambda *args, **kwargs:
        {'name': 'report_unable', 'arguments': {'reason': 'offline wiring test'}})
    output = tmp_path/'run'
    agent.run(agent.parse_args(['--goal', 'press', '--output', str(output),
                               '--decision-mode', mode, '--max-rounds', '1']))
    policy = {'localized': 'wrist_tip_overlap', 'end_to_end': 'none'}.get(mode)
    assert constructors == [{'forward_guard_policy': policy} if policy else {}]
    task = json.loads((output/'task.json').read_text())
    assert task['forward_guard_policy'] == (policy or 'wrist_horizontal')
    assert task['execution_policy'] == {'localized': 'yz_alignment_before_forward',
                                        'end_to_end': 'model_action_executed_as_given'}.get(mode, 'legacy_validated')
    assert json.loads((output/'result.json').read_text())['completed_motion_commands'] == 0


def test_recursive_deployment_includes_localized_modules_without_network(monkeypatch):
    import tarfile
    import s1b_agent_service as service
    import s1b_keyboard_direct as keyboard
    import start_s1b_inference as inference
    modules = ['localized_conversation', 'model_localization', 'spatial_localization',
               'spatial_progress', 'press_phase', 'wrist_overlap_guard']
    required = [f'src/pressb/agent_loop/{module}.py' for module in modules]
    required.append('scripts/s1b_keyboard_control.py')
    packaged = {}
    def fake_run(argv, **kwargs):
        if argv[0] == 'scp':
            with tarfile.open(argv[2]) as archive:
                for name in required:
                    packaged[name] = archive.extractfile(name).read()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(service.subprocess, 'run', fake_run)
    monkeypatch.setattr(keyboard, 'run_ssh', lambda *args: 0)
    monkeypatch.setattr(inference, 'check_port_available', lambda *args: None)
    monkeypatch.setattr(inference, 'sdk_environment', lambda *args: 'true')
    args = SimpleNamespace(host='no-robot', port=6210, view_port=8765, output='/tmp/test',
        control_hz=250, speed_mm_s=20, head_speed_deg_s=20, sdk_root='/unused', auto_confirm=False)
    assert service.launch(args) == 0
    for name in required:
        assert packaged[name] == (ROOT/name).read_bytes()
        compile(packaged[name], name, 'exec')


CALIBRATION = ROOT/'docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json'


def test_depth_window_is_pointed_at_merged_end_without_saving(monkeypatch):
    import threading
    aim = json.loads(CALIBRATION.read_text())['points']['closed']['point_normalized']
    calls, state = [], dict(point=[0.70, 0.80])
    def fake(base, route, payload=None, timeout=10):
        calls.append(route)
        if route == '/api/selection':
            state['point'] = payload['point_normalized']
            return dict(point_normalized=state['point'])
        assert route == '/api/snapshot'
        return dict(selection=dict(point_normalized=state['point']), depth=dict(center_px=[343, 248]))
    monkeypatch.setattr(launcher, 'request', fake)
    assert launcher.aim_depth_window('http://cam', aim, threading.Event()) == [343, 248]
    assert calls[0] == '/api/selection' and '/api/selection/save' not in calls


def test_depth_window_switch_times_out_instead_of_running_on_wrong_point(monkeypatch):
    import threading
    monkeypatch.setattr(launcher, 'request', lambda base, route, payload=None, timeout=10:
        dict(selection=dict(point_normalized=[0.70, 0.80]), depth=dict(center_px=[450, 286])))
    with pytest.raises(TimeoutError):
        launcher.aim_depth_window('http://cam', [0.5368, 0.6908], threading.Event(), timeout=0.3)


def test_depth_window_survives_a_camera_page_that_is_still_starting(monkeypatch):
    import threading
    aim = [0.5368, 0.6908]
    state = dict(point=[0.70, 0.80], failures=3, snapshots=0)
    def fake(base, route, payload=None, timeout=10):
        if route == '/api/selection':
            if state['failures']:
                state['failures'] -= 1
                raise ValueError('HTTP 503: starting')
            state['point'] = payload['point_normalized']
            return dict(point_normalized=state['point'])
        state['snapshots'] += 1
        return dict(selection=dict(point_normalized=state['point']), depth=dict(center_px=[343, 248]))
    monkeypatch.setattr(launcher, 'request', fake)
    assert launcher.aim_depth_window('http://cam', aim, threading.Event(), timeout=5, resend_s=0.01) == [343, 248]


def test_depth_window_continues_when_point_is_set_but_depth_not_ready(monkeypatch):
    import threading
    aim = [0.5368, 0.6908]
    monkeypatch.setattr(launcher, 'request', lambda base, route, payload=None, timeout=10:
        dict(point_normalized=aim) if route == '/api/selection' else
        dict(selection=dict(point_normalized=aim), depth=dict(median_m=None, status='深度已过期')))
    assert launcher.aim_depth_window('http://cam', aim, threading.Event(), timeout=0.3) is None


def test_no_demo_reaches_agent_with_zero_demonstration_images(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, 'request', lambda *a, **k: pytest.fail('no network'))
    monkeypatch.setattr(agent, 'RobotClient', lambda *a, **k: pytest.fail('no robot'))
    monkeypatch.setattr(agent, 'tunnel', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline-boundary')))
    output = tmp_path/'run'
    args = agent.parse_args(['--goal', 'press', '--output', str(output),
        '--fingertip-calibration', str(CALIBRATION), '--replay', str(tmp_path/'unused')])
    from pressb.agent_loop import calibrated_tip, press_phase
    try:
        with pytest.raises(RuntimeError, match='offline-boundary'):
            agent.run(args)
    finally:
        # agent.run activates module-level calibration state; do not leak it to other tests.
        calibrated_tip.deactivate()
        press_phase.reset()
    task = json.loads((output/'task.json').read_text())
    assert task['demo'] is None and task['initial_image_count'] == 2
