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


HEAD_OK = dict(verdict='aligned', head_view='Closed end is centered in front of the target button.',
               wrist_view='Button ring visible evenly around the closed tips.')
HEAD_OFF = dict(verdict='offset', head_view='Closed end sits right of and below the target button.',
                wrist_view='Button ring visible only on the upper-left of the closed tips.')


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

    # Round 1: closed; first approach: live 172 mm -> 12 mm to the 160 mm standoff stop.
    body = c.prepare_decision('fake', tmp_path/'1')
    assert c.range_evidence['live-1']['role'] == 'live' and c.range_evidence['live-1']['trusted']
    status = c.press_status['live-1']
    assert status['stage'] == 'press' and status['bound'] == 'live'
    assert status['head_alignment_required'] is False   # first approach is free
    assert status['remaining_mm'] == pytest.approx(12.0) and status['target_depth_mm'] == 145.0
    assert status['approach'] is True and status['approach_stop_depth_mm'] == 160.0
    assert press_phase.describe()['z_b_m'] == pytest.approx(0.200)

    # An overshoot is rejected (not clipped); a stroke within the live bound passes.
    replies = iter([action(c, xyz=[27, 0, 0]), action(c, xyz=[10, 0, 0])])
    monkeypatch.setattr(client, 'request', lambda *a, **k: next(replies))
    proposal = client.propose('fake', body, _obs(1, None), c.session, tmp_path/'1', conversation=c)
    assert proposal['arguments']['xyz_mm'] == [10, 0, 0]
    rejected = json.loads((tmp_path/'1'/'validation-0.json').read_text())['error']
    assert 'approach_stops_at_160mm' in rejected   # the old target-based 27 mm reaches the panel
    assert c.forward_guard_decision['policy'] == overlap.PRESS_POLICY
    assert c.forward_guard_decision['basis']['live_from'] == 'live-1'
    # The same recheck RobotClient.execute runs immediately before HTTP.
    client.require_forward_permission(proposal, _obs(1, None), c.forward_guard_decision,
                                      policy='wrist_tip_overlap')
    # Arm fell 2 mm short of the command: live depth reads 164 mm, not 162.
    _step(c, jpeg, 2, proposal, 0.164)
    press_phase.record_move(proposal['arguments']['xyz_mm'])   # agent records after sending

    # Round 2: approach done; live 164 -> 19 mm to the 145 mm target; frozen ceiling (45 mm) does not bind.
    body = c.prepare_decision('fake', tmp_path/'2')
    assert c.press_status['live-2']['remaining_mm'] == pytest.approx(19.0)
    assert c.press_status['live-2']['stage'] == 'head_align'   # check before the final press
    assert press_phase.evaluate(20, 'live-2')['allowed'] is False
    replies = iter([action(c, xyz=[20, 0, 0], head_alignment=HEAD_OK),
                    action(c, xyz=[19, 0, 0], head_alignment=HEAD_OK)])
    monkeypatch.setattr(client, 'request', lambda *a, **k: next(replies))
    proposal = client.propose('fake', body, _obs(2, None), c.session, tmp_path/'2', conversation=c)
    assert proposal['arguments']['xyz_mm'] == [19, 0, 0]
    _step(c, jpeg, 3, proposal, 0.145)
    press_phase.record_move(proposal['arguments']['xyz_mm'])

    # Round 3: target reached -> the fixed withdraw step starts.
    c.prepare_decision('fake', tmp_path/'3')
    assert c.press_status['live-3']['stage'] == 'press_reached_withdraw'
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
    # First approach: live 198 mm down to the 160 mm standoff stop = 38 mm.
    assert c.press_status['live-1']['remaining_mm'] == pytest.approx(38.0)

    # A press that stops short of the target is not complete: withdraw, shift and press stay free.
    for n, xyz, z in ((2, [38, 0, 0], 0.160), (3, [-15, 0, 0], 0.175), (4, [0, -4, 3], 0.175),
                      (5, [25, 0, 0], 0.150)):
        replies = iter([action(c, xyz=xyz, head_alignment=HEAD_OK if xyz[0] > 0 else HEAD_OFF)])
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


def _propose(c, monkeypatch, tmp_path, n, z_before, *replies):
    queue = iter(replies)
    monkeypatch.setattr(client, 'request', lambda *a, **k: next(queue))
    return client.propose('fake', c.prepare_decision('fake', tmp_path/str(n)), _obs(n-1, z_before),
                          c.session, tmp_path/str(n), conversation=c)


def test_missed_press_withdraws_10mm_then_reopens_and_realigns(jpeg, monkeypatch, tmp_path):
    c, calls = _closed_press_setup(jpeg, monkeypatch, tmp_path)
    # Round 1: approach the full 38 mm to the 160 mm standoff stop; not a completed press.
    proposal = _propose(c, monkeypatch, tmp_path, 2, .198, action(c, xyz=[38, 0, 0]))
    _step(c, jpeg, 2, proposal, 0.160)
    press_phase.record_move(proposal['arguments']['xyz_mm'])
    press_phase.record_retry(proposal)
    assert press_phase.retry_step() is None

    # Round 2: head check confirmed, final press 15 mm to the 145 mm target.
    proposal = _propose(c, monkeypatch, tmp_path, 3, .160, action(c, xyz=[15, 0, 0], head_alignment=HEAD_OK))
    assert c.press_status['live-2']['stage'] == 'head_align'
    _step(c, jpeg, 3, proposal, 0.145)
    press_phase.record_move(proposal['arguments']['xyz_mm'])
    press_phase.record_retry(proposal)

    # Round 3: target reached. Anything except exactly [-10, 0, 0] is rejected and repaired.
    proposal = _propose(c, monkeypatch, tmp_path, 4, .145,
                        action(c, xyz=[-20, 0, 0]), action(c, xyz=[-10, 3, 0]),
                        action(c, 'propose_gripper', key='o'), action(c, xyz=[-10, 0, 0]))
    assert c.press_status['live-3']['stage'] == 'press_reached_withdraw'
    assert proposal['arguments']['xyz_mm'] == [-10, 0, 0]
    errors = [json.loads((tmp_path/'4'/f'validation-{i}.json').read_text())['error'] for i in range(3)]
    assert all('press_retry_withdraw_required' in e for e in errors)
    _step(c, jpeg, 4, proposal, 0.155)
    press_phase.record_move(proposal['arguments']['xyz_mm'])
    assert press_phase.record_retry(proposal) == press_phase.REOPEN

    # Round 4: withdrawn. Closed re-press, Y/Z shift or re-close is rejected; opening is accepted.
    proposal = _propose(c, monkeypatch, tmp_path, 5, .155,
                        action(c, xyz=[10, 0, 0]), action(c, xyz=[0, -4, 0]),
                        action(c, 'propose_gripper', key='c'), action(c, 'propose_gripper', key='o'))
    assert c.press_status['live-4']['stage'] == 'withdrawn_check_light'
    assert proposal == dict(name='propose_gripper', arguments=dict(key='o', reason=proposal['arguments']['reason']))
    errors = [json.loads((tmp_path/'5'/f'validation-{i}.json').read_text())['error'] for i in range(3)]
    assert all('press_retry_reopen_required' in e for e in errors)
    _step(c, jpeg, 5, proposal, 0.155)
    press_phase.set_gripper('open')

    # Round 5: open again -> fresh localization, alignment stage, sequence cleared; closing starts a new attempt.
    before = len(calls)
    c.prepare_decision('fake', tmp_path/'6')
    assert len(calls) == before+1 and 'live-5' in c.localizations and 'live-5' not in c.direct_press
    assert press_phase.retry_step() is None and press_phase.describe()['completed_presses'] == 1
    from pressb.agent_loop import press_retry
    assert press_retry.attempts(c.turns) == ['live-3']


def test_light_on_after_withdrawal_may_report_done():
    press_phase.configure()
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([20, 0, 0])     # approach done
    assert press_phase.note_press_status(0.0) == press_phase.WITHDRAW
    with pytest.raises(ValueError, match='withdraw_required'):
        press_phase.check_retry(dict(name='report_done', arguments={}))
    press_phase.record_retry(dict(name='propose_move', arguments=dict(xyz_mm=[-10, 0, 0])))
    press_phase.check_retry(dict(name='report_done', arguments={}))
    press_phase.check_retry(dict(name='propose_move', arguments=dict(xyz_mm=[-15, 0, 0])))


def test_head_alignment_after_first_approach_with_multiple_yz_rounds(jpeg, monkeypatch, tmp_path):
    c, _ = _closed_press_setup(jpeg, monkeypatch, tmp_path)
    # Round 1: first approach needs no head verdict but stops at the 160 mm standoff:
    # 45 mm (it would reach the panel) is rejected, 38 mm passes.
    proposal = _propose(c, monkeypatch, tmp_path, 2, .198, action(c, xyz=[45, 0, 0]), action(c, xyz=[38, 0, 0]))
    assert c.press_status['live-1']['stage'] == 'press'
    assert 'approach_stops_at_160mm' in json.loads((tmp_path/'2'/'validation-0.json').read_text())['error']
    assert proposal['arguments']['xyz_mm'] == [38, 0, 0] and 'live-1' not in c.head_checks
    _step(c, jpeg, 2, proposal, .160)
    press_phase.record_move(proposal['arguments']['xyz_mm'])
    assert press_phase.head_alignment_required()

    # Round 2: standoff reached, 15 mm left -> head_align. head_alignment is required; pushing needs aligned.
    proposal = _propose(c, monkeypatch, tmp_path, 3, .160,
                        action(c, xyz=[15, 0, 0]),                              # no verdict
                        action(c, xyz=[15, 0, 0], head_alignment=HEAD_OFF),     # final press while offset
                        action(c, xyz=[15, 0, 2], head_alignment=HEAD_OK),      # final press with Y/Z
                        action(c, xyz=[0, -3, 2], head_alignment=HEAD_OFF))     # fine-tune: accepted
    status = c.press_status['live-2']
    assert status['stage'] == 'head_align' and status['remaining_mm'] == pytest.approx(15.0)
    move = next(t for t in c.body()['tools'] if t['function']['name'] == 'propose_move')['function']
    assert 'head_alignment' in move['parameters']['required']
    errors = [json.loads((tmp_path/'3'/f'validation-{i}.json').read_text())['error'] for i in range(3)]
    assert 'head_alignment_required' in errors[0]
    assert 'head_alignment_not_confirmed' in errors[1]
    assert 'head_aligned_press_must_be_pure_x' in errors[2]
    assert proposal['arguments']['xyz_mm'] == [0, -3, 2] and 'head_alignment' not in proposal['arguments']
    _step(c, jpeg, 3, proposal, .160)
    press_phase.record_move(proposal['arguments']['xyz_mm'])
    assert c.turns[-1]['ledger']['head_alignment']['verdict'] == 'offset'

    # Round 3: still unclear -> another Y/Z round.
    proposal = _propose(c, monkeypatch, tmp_path, 4, .160,
                        action(c, xyz=[0, -2, 0], head_alignment=dict(verdict='unclear', head_view='Partly hidden.', wrist_view='Button hidden.')))
    _step(c, jpeg, 4, proposal, .160)
    press_phase.record_move(proposal['arguments']['xyz_mm'])
    assert press_phase.describe()['head_shifts'] == 2

    # Round 4: aligned -> pure positive X final press within the live limit is accepted.
    proposal = _propose(c, monkeypatch, tmp_path, 5, .160, action(c, xyz=[15, 0, 0], head_alignment=HEAD_OK))
    assert c.press_status['live-4']['head_shifts'] == 2 and c.forward_guard_decision['allowed']
    assert c.head_checks['live-4']['verdict'] == 'aligned'
    _step(c, jpeg, 5, proposal, .145)
    press_phase.record_move(proposal['arguments']['xyz_mm'])
    c.prepare_decision('fake', tmp_path/'6')
    assert c.press_status['live-5']['stage'] == 'press_reached_withdraw'


def test_head_align_advises_reopen_after_many_rounds():
    press_phase.configure()
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([0, 2, 0])      # before the first approach: not a head round
    press_phase.record_move([40, 0, 0])
    for _ in range(press_phase.HEAD_ALIGN_ROUNDS):
        press_phase.record_move([0, 2, 0])
    assert press_phase.describe()['head_shifts'] == press_phase.HEAD_ALIGN_ROUNDS
    from pressb.agent_loop.localized_conversation import press_status
    status = press_status(press_phase.describe())
    assert status['stage'] == 'head_align' and 'open the gripper (o)' in status['instruction']


def test_retry_after_reopen_repeats_approach_then_head_alignment():
    press_phase.configure()
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    assert press_phase.check_head_alignment('propose_move', [40, 0, 0], None) is None
    press_phase.record_move([40, 0, 0])
    press_phase.record_move([0, 2, 0])
    assert press_phase.note_press_status(0.0) == press_phase.WITHDRAW
    # The fixed withdraw/reopen steps need no head verdict.
    assert not press_phase.head_alignment_required()
    assert press_phase.check_head_alignment('propose_move', [-10, 0, 0], None) is None
    press_phase.record_retry(dict(name='propose_move', arguments=dict(xyz_mm=[-10, 0, 0])))
    press_phase.set_gripper('open')
    # Realign open, freeze again, close: first approach is free again, then the head check.
    press_phase._state['z_b_m'] = .19
    press_phase.set_gripper('closed')
    state = press_phase.describe()
    assert not state['head_alignment_required'] and state['head_shifts'] == 0
    from pressb.agent_loop.localized_conversation import press_status
    assert press_status(state)['stage'] == 'press'
    assert press_phase.check_head_alignment('propose_move', [35, 0, 0], None) is None
    press_phase.record_move([35, 0, 0])
    assert press_status(press_phase.describe())['stage'] == 'head_align'
    with pytest.raises(ValueError, match='head_alignment_not_confirmed'):
        press_phase.check_head_alignment('propose_move', [5, 0, 0], HEAD_OFF)


def test_alignment_verdict_needs_both_head_and_wrist_descriptions():
    press_phase.configure()
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([30, 0, 0])
    with pytest.raises(ValueError, match='head_alignment_required'):
        press_phase.check_head_alignment('propose_move', [5, 0, 0],
                                         dict(verdict='aligned', head_view='Centered.'))
    with pytest.raises(ValueError, match='head_alignment_required'):
        press_phase.check_head_alignment('propose_move', [5, 0, 0],
                                         dict(verdict='aligned', head_view='Centered.', wrist_view=' '))
    assert press_phase.check_head_alignment('propose_move', [5, 0, 0], HEAD_OK) == HEAD_OK
    from pressb.agent_loop.localized_conversation import HEAD_ALIGNMENT_SCHEMA
    assert HEAD_ALIGNMENT_SCHEMA['required'] == ['verdict', 'head_view', 'wrist_view']


def test_aim_centered_uses_central_part_of_box():
    box = [.4, .4, .6, .8]                          # center (.5, .6), size .2 x .4
    assert press_phase.aim_centered(box, [.5, .6])
    assert press_phase.aim_centered(box, [.54, .69])       # just within CENTER_FRACTION of the size
    assert not press_phase.aim_centered(box, [.5, .42])    # inside the box, on the upper rim
    assert not press_phase.aim_centered(box, [.58, .6])
    assert not press_phase.aim_centered(None, [.5, .6])


def _reopen_after_completed_press():
    press_phase.configure()
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([40, 0, 0])
    press_phase.record_move([10, 0, 0])
    assert press_phase.note_press_status(0.0) == press_phase.WITHDRAW
    withdraw = dict(name='propose_move', arguments=dict(xyz_mm=list(press_phase.RETRY_WITHDRAW_MM)))
    press_phase.record_move(withdraw['arguments']['xyz_mm'])
    press_phase.record_retry(withdraw)
    press_phase.set_gripper('open')


def test_retry_does_not_freeze_on_rim_until_realigned():
    calibrated_tip.activate(CALIBRATION, 'open')
    _reopen_after_completed_press()
    assert press_phase.realign_required() and press_phase.describe()['realign_required']
    # Hand still inside the close band and the aim point inside the box, but only on its rim.
    result = press_phase.observe(_obs(6, .158), calibrated_tip.aim_pixel(), calibrated_tip.image_size(),
                                 aligned=True, centered=False)
    assert 'retry_realign_required' in result['reasons'] and press_phase.phase() == press_phase.ALIGN
    # Pure X approach does not count as realignment; an executed Y/Z move does.
    press_phase.record_move([5, 0, 0])
    assert press_phase.realign_required()
    press_phase.record_move([0, 0, 6])
    assert not press_phase.realign_required()
    result = press_phase.observe(_obs(7, .158), calibrated_tip.aim_pixel(), calibrated_tip.image_size(),
                                 aligned=True, centered=False)
    assert result.get('froze') and press_phase.phase() == press_phase.PRESS


def test_retry_freezes_without_a_move_when_aim_is_centered():
    calibrated_tip.activate(CALIBRATION, 'open')
    _reopen_after_completed_press()
    result = press_phase.observe(_obs(6, .158), calibrated_tip.aim_pixel(), calibrated_tip.image_size(),
                                 aligned=True, centered=True)
    assert result.get('froze') and not press_phase.realign_required()
    # The next reopen after another completed press asks again; a first attempt never does.
    press_phase.configure()
    press_phase.set_gripper('open')
    assert not press_phase.realign_required()


def test_conversation_passes_centered_and_tells_model_to_realign(jpeg, monkeypatch, tmp_path):
    calibrated_tip.activate(CALIBRATION, 'open')
    x, y = _aim()
    rim = [x-.05, y-.01, x+.05, y+.09]              # aim point inside, on the upper rim
    c, _ = make(jpeg, monkeypatch, data=lambda oid: dict(coordinate_units='normalized_01', head=view(),
                wrist=view(button_box=rim, contact_visibility='unknown', contact_tip=None, arrow_box=None)))
    _reopen_after_completed_press()
    c.manifests[c.current_id] = _obs(0, .158)
    c.prepare_decision('fake', tmp_path/'0')
    status = c.press_status[c.current_id]
    assert status['stage'] == 'align_open' and status['realign_required']
    assert 'retry_realign_required' in status['this_observation_range']['reasons']
    assert 'RETRY REALIGNMENT' in status['instruction'] and press_phase.phase() == press_phase.ALIGN


def test_final_press_drop_is_off_by_default():
    press_phase.configure()
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([40, 0, 0])
    assert press_phase.drop_mm() == 0 and press_phase.final_press_z_mm() == 0
    with pytest.raises(ValueError, match='head_aligned_press_must_be_pure_x'):
        press_phase.check_head_alignment('propose_move', [5, 0, -10], HEAD_OK)
    press_phase.record_move([5, 0, 0])
    assert press_phase.note_press_status(0.0) == press_phase.WITHDRAW
    assert press_phase.retry_withdraw_mm() == [-10, 0, 0]


def test_configured_drop_is_restored_by_the_fixed_withdraw():
    press_phase.configure(final_drop_m=.010)
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([40, 0, 0])             # first closed approach
    assert press_phase.final_press_z_mm() == -10
    with pytest.raises(ValueError, match='head_aligned_press_yz_fixed'):
        press_phase.check_head_alignment('propose_move', [5, 0, 0], HEAD_OK)
    press_phase.record_move([5, 0, -10])
    # Applied once: further strokes, Y/Z fine-tuning or a free withdrawal keep Z=0.
    assert press_phase.final_press_z_mm() == 0
    with pytest.raises(ValueError, match='Z=0'):
        press_phase.check_head_alignment('propose_move', [3, 0, -10], HEAD_OK)
    press_phase.record_move([0, 2, 0])
    press_phase.record_move([-5, 0, 0])
    press_phase.record_move([5, 0, 0])
    assert press_phase.final_press_z_mm() == 0
    # Target reached: the fixed withdraw rises back by the drop.
    assert press_phase.note_press_status(0.0) == press_phase.WITHDRAW
    assert press_phase.retry_withdraw_mm() == [-10, 0, 10]
    with pytest.raises(ValueError, match=r'\[-10, 0, 10\]'):
        press_phase.check_retry(dict(name='propose_move', arguments=dict(xyz_mm=[-10, 0, 0])))
    withdraw = dict(name='propose_move', arguments=dict(xyz_mm=[-10, 0, 10]))
    press_phase.check_retry(withdraw)
    press_phase.record_move([-10, 0, 10])           # agent records the move before the retry step
    assert press_phase.record_retry(withdraw) == press_phase.REOPEN
    assert press_phase.final_press_z_mm() == -10
    press_phase.set_gripper('open')
    press_phase.set_gripper('closed')
    assert press_phase.final_press_z_mm() == -10 and press_phase.retry_withdraw_mm() == [-10, 0, 0]


def test_configured_drop_unrestored_is_not_applied_twice_after_reopening():
    press_phase.configure(final_drop_m=.010)
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([40, 0, 0])
    press_phase.record_move([5, 0, -10])            # dropped, target not reached
    press_phase.record_move([-20, 0, 0])
    press_phase.set_gripper('open')                 # reopened without the restoring withdraw
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([40, 0, 0])
    assert press_phase.final_press_z_mm() == 0
    press_phase.check_head_alignment('propose_move', [5, 0, 0], HEAD_OK)


def test_configured_drop_reaches_model_text():
    from pressb.agent_loop.localized_conversation import press_status, _drop_note
    press_phase.configure()
    assert _drop_note() == ''
    press_phase.configure(final_drop_m=.0075)
    press_phase.set_gripper('closed')
    press_phase._state['z_b_m'] = .2
    press_phase.record_move([40, 0, 0])
    status = press_status(press_phase.describe())
    assert status['final_press_z_mm'] == -7.5 and 'Z=-7.5' in status['instruction']
    assert '7.5 mm DOWN' in _drop_note()
