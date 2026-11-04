"""End-to-end: align open, freeze range, close, press on live depth within the frozen range.

Only the model localizer is stubbed; the conversation, phased guard and permission
recheck before HTTP are the live code paths.
"""
import json

import pytest

from pressb.agent_loop import calibrated_tip, client, press_phase, wrist_overlap_guard as overlap
from test_conversation import jpeg, response
from test_localized_conversation import make, action, observation
from test_model_localization import view

CALIBRATION = 'docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json'


@pytest.fixture(autouse=True)
def _reset():
    calibrated_tip.deactivate()
    press_phase.reset()
    yield
    calibrated_tip.deactivate()
    press_phase.reset()


def _aim():
    return calibrated_tip.aim_point()


def _box_on_aim(half=.05):
    x, y = _aim()
    return [x-half, y-half, x+half, y+half]


def _locations(oid, aligned=True):
    box = _box_on_aim() if aligned else [.05, .05, .15, .15]
    wrist = view(button_box=box, contact_visibility='unknown', contact_tip=None, arrow_box=None)
    return dict(coordinate_units='normalized_01', head=view(), wrist=wrist)


def _obs(n, z):
    o = observation(n)
    o['depth'] = dict(median_m=z, status='ok', valid_pixels=25)
    o['selection'] = dict(point_normalized=_aim())
    return o


def _setup(jpeg, monkeypatch, aligned=lambda oid: True):
    calibrated_tip.activate(CALIBRATION, 'open')
    press_phase.configure()
    c, _ = make(jpeg, monkeypatch, data=lambda oid: _locations(oid, aligned(oid)))
    c.manifests[c.current_id] = _obs(0, 0.200)
    return c


def _step(c, jpeg, n, proposal, z):
    c.check_execution(proposal, c.current_id)
    result = dict(status='completed', sends_completed=True,
                  request=dict(type=proposal['name'].removeprefix('propose_'), command_id='cmd-'+str(n),
                               **{k: v for k, v in proposal['arguments'].items() if k != 'reason'}),
                  motion=dict(action_completed=True), capture=dict(observation_id='live-'+str(n)))
    c.append_result(proposal, result, _obs(n, z), jpeg, jpeg)


def test_align_freeze_close_press_withdraw(jpeg, monkeypatch, tmp_path):
    c = _setup(jpeg, monkeypatch)
    overlap_checks = []
    monkeypatch.setattr(overlap, 'inspect', lambda *a, **k: overlap_checks.append(1) or pytest.fail(
        'press phase must not ask the model to box the occluded button'))

    # Round 0: aligned and in band -> freeze; model sees stage "close".
    body = c.prepare_decision('fake', tmp_path/'0')
    assert c.range_evidence['live-0']['froze'] and press_phase.phase() == press_phase.PRESS
    assert c.press_status['live-0']['stage'] == 'close'
    assert 'CURRENT_PRESS_PHASE' in json.dumps(body['messages'])

    # Forward while still open is rejected without any model visual request.
    replies = iter([action(c, xyz=[10, 0, 0]), action(c, 'propose_gripper', key='c')])
    monkeypatch.setattr(client, 'request', lambda *a, **k: next(replies))
    proposal = client.propose('fake', body, _obs(0, .2), c.session, tmp_path/'0', conversation=c)
    assert proposal['name'] == 'propose_gripper' and proposal['arguments']['key'] == 'c'
    gate = json.loads((tmp_path/'0/forward_guard/candidate-0.json').read_text())['decision']
    assert gate['policy'] == overlap.PRESS_POLICY and 'gripper_not_closed' in gate['reasons']
    _step(c, jpeg, 1, proposal, 0.172)         # closed: aim pixel still reads the button
    press_phase.set_gripper('closed')

    # Round 1: closed; live 172 mm -> 27 mm to the 145 mm target, under the 55 mm ceiling.
    body = c.prepare_decision('fake', tmp_path/'1')
    assert c.range_evidence['live-1']['role'] == 'live' and c.range_evidence['live-1']['trusted']
    status = c.press_status['live-1']
    assert status['stage'] == 'press' and status['bound'] == 'live'
    assert status['remaining_mm'] == pytest.approx(27.0) and status['target_depth_mm'] == 145.0
    assert press_phase.describe()['z_b_m'] == pytest.approx(0.200)

    # An overshoot is rejected (not clipped); a stroke within the live bound passes.
    replies = iter([action(c, xyz=[30, 0, 0]), action(c, xyz=[20, 0, 0])])
    monkeypatch.setattr(client, 'request', lambda *a, **k: next(replies))
    proposal = client.propose('fake', body, _obs(1, None), c.session, tmp_path/'1', conversation=c)
    assert proposal['arguments']['xyz_mm'] == [20, 0, 0]
    assert c.forward_guard_decision['policy'] == overlap.PRESS_POLICY
    assert c.forward_guard_decision['basis']['live_from'] == 'live-1'
    # The same recheck RobotClient.execute runs immediately before HTTP.
    client.require_forward_permission(proposal, _obs(1, None), c.forward_guard_decision,
                                      policy='wrist_tip_overlap')
    # Arm fell 2 mm short of the command: live depth reads 154 mm, not 152.
    _step(c, jpeg, 2, proposal, 0.154)
    press_phase.record_move(proposal['arguments']['xyz_mm'])   # agent records after sending

    # Round 2: live 154 -> 9 mm left; the frozen ceiling (35 mm) does not bind.
    body = c.prepare_decision('fake', tmp_path/'2')
    assert c.press_status['live-2']['remaining_mm'] == pytest.approx(9.0)
    assert press_phase.evaluate(10, 'live-2')['allowed'] is False
    replies = iter([action(c, xyz=[10, 0, 0]), action(c, xyz=[9, 0, 0])])
    monkeypatch.setattr(client, 'request', lambda *a, **k: next(replies))
    proposal = client.propose('fake', body, _obs(2, None), c.session, tmp_path/'2', conversation=c)
    assert proposal['arguments']['xyz_mm'] == [9, 0, 0]
    _step(c, jpeg, 3, proposal, 0.145)
    press_phase.record_move(proposal['arguments']['xyz_mm'])

    # Round 3: target reached -> model is told to stop and verify.
    c.prepare_decision('fake', tmp_path/'3')
    assert c.press_status['live-3']['stage'] == 'press_reached'
    assert press_phase.evaluate(1, 'live-3')['allowed'] is False
    assert overlap_checks == []

    # Withdraw needs no permission at all.
    client.require_forward_permission(dict(name='propose_move', arguments=dict(xyz_mm=[-30, 0, 0])),
                                      _obs(2, None), None, policy='wrist_tip_overlap')


def test_misaligned_in_band_depth_does_not_freeze_and_keeps_alignment_gate(jpeg, monkeypatch, tmp_path):
    c = _setup(jpeg, monkeypatch, aligned=lambda oid: False)
    c.prepare_decision('fake', tmp_path/'0')
    ev = c.range_evidence['live-0']
    assert not ev['trusted'] and 'aim_point_not_inside_current_button_box' in ev['reasons']
    assert press_phase.phase() == press_phase.ALIGN
    assert c.press_status['live-0']['stage'] == 'align_open'
    # Alignment phase still uses the model wrist-overlap check for positive X.
    assert not client._press_phase_press()


def test_stale_permission_from_a_previous_observation_is_rejected(jpeg, monkeypatch, tmp_path):
    c = _setup(jpeg, monkeypatch)
    c.prepare_decision('fake', tmp_path/'0')
    press_phase.set_gripper('closed')
    xyz = [10, 0, 0]
    old = dict(overlap.press_gate('live-0', xyz[0]), authorized_command=dict(type='move', xyz_mm=xyz))
    with pytest.raises(ValueError, match='live_and_frozen_range'):
        overlap.validate_permission(dict(name='propose_move', arguments=dict(xyz_mm=xyz)),
                                    dict(observation_id='live-1'), old)


def _closed_press_setup(jpeg, monkeypatch, tmp_path):
    """Freeze at 200 mm with the gripper open, then close; returns (conversation, localizer calls)."""
    calibrated_tip.activate(CALIBRATION, 'open')
    press_phase.configure()
    c, calls = make(jpeg, monkeypatch, data=lambda oid: _locations(oid))
    c.manifests[c.current_id] = _obs(0, 0.200)
    c.prepare_decision('fake', tmp_path/'0')
    assert press_phase.phase() == press_phase.PRESS
    monkeypatch.setattr(client, 'request', lambda *a, **k: action(c, 'propose_gripper', key='c'))
    proposal = client.propose('fake', c.body(), _obs(0, .2), c.session, tmp_path/'0', conversation=c)
    _step(c, jpeg, 1, proposal, 0.198)
    press_phase.set_gripper('closed')
    return c, calls


def test_closed_gripper_press_decides_directly_from_images(jpeg, monkeypatch, tmp_path):
    c, calls = _closed_press_setup(jpeg, monkeypatch, tmp_path)
    before = len(calls)
    body = c.prepare_decision('fake', tmp_path/'1')
    # No separate button localization, direction rule or progress comparison in this stage.
    assert len(calls) == before and 'live-1' not in c.localizations
    assert 'live-1' not in c.direction_hints and 'live-1' in c.direct_press
    text = json.dumps(body['messages'])
    assert 'DIRECT_PRESS_STAGE' in text and 'CURRENT_PRESS_PHASE' in text
    move = next(t for t in body['tools'] if t['function']['name'] == 'propose_move')
    assert 'DIRECT PRESS STAGE' in move['function']['description']
    assert json.loads((tmp_path/'1/decision_context.json').read_text())['press_stage'] == 'direct_press'
    # Live depth still bounds the stroke: 198 - 145 = 53 mm.
    assert c.press_status['live-1']['remaining_mm'] == pytest.approx(53.0)

    # Press, miss, then retry WITHOUT reopening: withdraw, shift Y/Z, press again.
    for n, xyz, z in ((2, [50, 0, 0], 0.148), (3, [-15, 0, 0], 0.163), (4, [0, -4, 3], 0.163),
                      (5, [16, 0, 0], 0.147)):
        replies = iter([action(c, xyz=xyz)])
        monkeypatch.setattr(client, 'request', lambda *a, **k: next(replies))
        proposal = client.propose('fake', c.prepare_decision('fake', tmp_path/str(n)), _obs(n-1, z),
                                  c.session, tmp_path/str(n), conversation=c)
        assert proposal['arguments']['xyz_mm'] == xyz
        if c.forward_guard_decision is not None:
            client.require_forward_permission(proposal, _obs(n-1, None), c.forward_guard_decision,
                                              policy='wrist_tip_overlap')
        _step(c, jpeg, n, proposal, z)
        press_phase.record_move(xyz)
    assert len(calls) == before and press_phase.phase() == press_phase.PRESS
    from pressb.agent_loop import press_retry
    assert press_retry.attempts(c.turns) == ['live-2', 'live-5']


def test_reopening_returns_to_localized_alignment(jpeg, monkeypatch, tmp_path):
    c, calls = _closed_press_setup(jpeg, monkeypatch, tmp_path)
    c.prepare_decision('fake', tmp_path/'1')
    monkeypatch.setattr(client, 'request', lambda *a, **k: action(c, 'propose_gripper', key='o'))
    proposal = client.propose('fake', c.body(), _obs(1, .198), c.session, tmp_path/'1', conversation=c)
    _step(c, jpeg, 2, proposal, 0.198)
    press_phase.set_gripper('open')
    before = len(calls)
    c.prepare_decision('fake', tmp_path/'2')
    assert len(calls) == before+1 and 'live-2' in c.localizations and 'live-2' not in c.direct_press
