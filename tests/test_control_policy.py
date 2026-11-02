"""Stage advice uses measured camera stability and actual command evidence only."""
import copy

import pytest

from pressb.agent_loop.control_policy import build_plan


def assessment(oid='now', error=(1.0, -3.0), **changes):
    x, y = error
    result = dict(format='pressb-grounding-v2', observation_id=oid,
        head_visible='yes', wrist_visible='yes', contact_finger='head_image_right',
        head_horizontal='center' if abs(x) <= .25 else 'right' if x > 0 else 'left',
        head_vertical='level' if abs(y) <= .25 else 'below' if y > 0 else 'above',
        wrist_relation='center', lamp='off', lamp_view='head',
        geometry=dict(head_button_box=[.5, .4, .6, .5], head_finger_tip=[.45, .75],
            error_button_units=list(error), delta_image_norm=[x*.1, y*.1],
            magnitude=dict(horizontal='medium', vertical='large')))
    result.update(changes)
    return result


def plan(error=(1, -3), previous_error=None, **kwargs):
    return build_plan(assessment(error=error), {'head_deg': [0, 25]},
        previous_assessment=assessment('before', previous_error) if previous_error else None,
        previous_state={'head_deg': [0, 25]}, latest_command={'type': 'move', 'xyz_mm': [0, 0, 10]},
        **kwargs)


def test_large_error_selects_axis_and_advisory_step():
    value = plan()
    assert value['format'] == 'pressb-control-plan-v1'
    assert value['observation_id'] == 'now'
    assert value['stage'] == 'align' and value['focus'] == 'vertical'
    assert value['alignment_size'] == {'horizontal': 'medium', 'vertical': 'large'}
    assert value['step_hint_mm'] == 20
    assert 'not a calibrated' in ' '.join(value['instructions'])
    assert 'command' not in value and 'xyz_mm' not in value


@pytest.mark.parametrize('error,focus,step', [((.4, .1), 'horizontal', 5),
    ((1.5, -.3), 'horizontal', 10), ((.1, -3), 'vertical', 20), ((1, -1), 'vertical', 10)])
def test_magnitude_hints_are_not_repeated_fixed_micro_steps(error, focus, step):
    value = plan(error)
    assert (value['focus'], value['step_hint_mm']) == (focus, step)


def test_legacy_categories_have_unknown_size_and_vertical_priority():
    current = assessment(geometry=None)
    value = build_plan(current, {})
    assert value['stage'] == 'align' and value['focus'] == 'vertical'
    assert value['step_hint_mm'] is None
    assert value['alignment_size'] == dict(horizontal='unknown', vertical='unknown')


def test_visible_known_offset_can_be_addressed_while_other_axis_unknown():
    value = build_plan(assessment(geometry=None, head_vertical='unknown'), {})
    assert value['stage'] == 'align' and value['focus'] == 'horizontal'


@pytest.mark.parametrize('changes', [dict(head_visible='no'), dict(contact_finger='unknown'),
    dict(head_horizontal='unknown', head_vertical='unknown', geometry=None)])
def test_unknown_target_or_finger_gets_view(changes):
    value = build_plan(assessment(**changes), {})
    assert value['stage'] == 'get_view'
    assert value['preferred_contact'] == 'head_image_right'


def test_centered_projection_does_not_claim_contact_or_bypass_guard():
    value = plan((0, 0))
    assert value['stage'] == 'approach'
    words = ' '.join(value['instructions'])
    assert 'does not prove 3D contact alignment' in words
    assert 'independent current-WRIST horizontal' in words


def test_advice_does_not_introduce_height_or_depth_gate():
    current = assessment(error=(0, -3), wrist_relation='center')
    invalid = build_plan(current, {'head_deg': [0, 25], 'depth': {'valid': False}})
    valid = build_plan(current, {'head_deg': [0, 25], 'depth': {'valid': True, 'depth_mm': 131}})
    assert invalid == valid
    words = ' '.join(invalid['instructions'])
    assert 'no height or depth validity prerequisite' in words
    assert 'separately justified approach is still allowed' in words
    assert 'do not hide positive X inside a correction label' in words


def test_evidence_has_priority_and_keeps_model_provenance():
    evidence = dict(light_on=dict(observation_id='lit', source='model_not_human'),
                    withdrawal=dict(observation_id='retreated', source='model_not_human'))
    value = build_plan({}, {}, evidence=evidence)
    assert value['stage'] == 'complete'
    assert value['evidence'] == evidence and value['evidence'] is not evidence
    assert 'not human confirmation' in ' '.join(value['instructions'])
    evidence['withdrawal'] = None
    assert build_plan({}, {}, evidence=evidence)['stage'] == 'withdraw'
    assert build_plan({}, {}, evidence={'withdrawal': {'observation_id': 'r'}})['stage'] == 'get_view'


def test_improvement_requires_same_finger_stable_head_and_actual_move():
    value = plan((.8, -2), (1, -3))
    progress = value['geometry_progress']
    assert progress['comparable']
    assert progress['axes']['horizontal']['trend'] == 'unchanged'
    assert progress['axes']['vertical']['trend'] == 'improved'
    assert progress['actual_command'] == {'type': 'move', 'xyz_mm': [0, 0, 10]}
    assert value['stage'] == 'align'


@pytest.mark.parametrize('new,old,flag', [((1, -3), (1, -1), 'substantial_worsening'),
    ((-1, -1), (1, -1), 'large_sign_reversal')])
def test_large_worsening_or_sign_reversal_requires_recheck(new, old, flag):
    value = plan(new, old)
    assert value['stage'] == 'recheck'
    assert value['geometry_progress']['needs_recheck']
    assert any(v[flag] for v in value['geometry_progress']['axes'].values())


def test_small_jitter_and_small_crossing_do_not_claim_overshoot():
    value = plan((-.3, -.4), (.3, -.3))
    assert not value['geometry_progress']['needs_recheck']
    assert value['stage'] == 'align'


@pytest.mark.parametrize('command', [None, {}, {'type': 'head', 'head_deg': [0, 30]},
    {'type': 'gripper', 'key': 'o'}, {'type': 'move', 'xyz_mm': [0, 0, 0]},
    {'type': 'move', 'xyz_mm': [True, 0, 1]}])
def test_only_actual_nonzero_hand_moves_can_support_trend(command):
    value = build_plan(assessment(error=(1, -3)), {'head_deg': [0, 25]},
        assessment('before', (1, -1)), {'head_deg': [0, 25]}, command)
    assert not value['geometry_progress']['comparable']
    assert value['geometry_progress']['reason'] == 'no_recorded_nonzero_hand_move'
    assert value['stage'] == 'align'


@pytest.mark.parametrize('state,previous', [({}, {'head_deg': [0, 25]}),
    ({'head_deg': [0, 25]}, {}), ({'head_deg': [0, 26.1]}, {'head_deg': [0, 25]}),
    ({'head_deg': [2, 25]}, {'head_deg': [0, 25]})])
def test_camera_changes_or_missing_pose_cannot_masquerade_as_hand_progress(state, previous):
    value = build_plan(assessment(), state, assessment('before', (1, -1)), previous,
        {'type': 'move', 'xyz_mm': [0, 0, 10]})
    assert not value['geometry_progress']['comparable']
    assert value['geometry_progress']['reason'] == 'measured_head_pose_unknown_or_changed'


@pytest.mark.parametrize('finger', ['head_image_left', 'unknown'])
def test_contact_identity_change_prevents_comparison(finger):
    value = build_plan(assessment(), {'head_deg': [0, 25]},
        assessment('before', contact_finger=finger), {'head_deg': [0, 25]},
        {'type': 'move', 'xyz_mm': [0, 0, 10]})
    assert not value['geometry_progress']['comparable']


@pytest.mark.parametrize('box', [[.5, .4, .62, .52], [.5, .4, .65, .5],
    [.5, .4, .6, .4], [0, 0, 1.1, .2]])
def test_unreliable_scale_shape_or_invalid_box_prevents_precision_claims(box):
    current, before = assessment(), assessment('before', (1, -1))
    current['geometry']['head_button_box'] = box
    value = build_plan(current, {'head_deg': [0, 25]}, before, {'head_deg': [0, 25]},
        {'type': 'move', 'xyz_mm': [0, 0, 10]})
    assert not value['geometry_progress']['comparable']
    assert value['stage'] == 'align'


def test_previous_head_target_and_actual_measurement_stay_separate():
    target = dict(head_deg=[0, 30], measured_head_deg=[0, 25], observation_id='head-after', command_id='head-cmd')
    value = build_plan(assessment(head_visible='no'), {'head_deg': [0, 25]}, last_head_target=target)
    info = value['head_target_info']
    assert info['last_absolute_target_deg'] == [0, 30]
    assert info['current_measured_head_deg'] == [0, 25]
    assert info['measured_pose_stable_since_last_target']
    assert info['avoid_repeating_same_target']
    assert 'different sensible target' in info['guidance']
    assert 'command' not in value
    assert not build_plan(assessment(), {}, last_head_target=target)['head_target_info']['avoid_repeating_same_target']


def test_inputs_not_mutated_and_plan_copies_archive_values():
    current, before = assessment(), assessment('before')
    state, old_state = {'head_deg': [0, 25]}, {'head_deg': [0, 25]}
    command = {'type': 'move', 'xyz_mm': [0, 0, 10]}
    evidence, head = {'light_on': {'observation_id': 'lit'}}, {'head_deg': [0, 30]}
    args = [current, state, before, old_state, command, evidence, head]
    original = copy.deepcopy(args)
    value = build_plan(*args)
    assert args == original
    value['evidence']['light_on']['observation_id'] = 'changed'
    value['head_target_info']['last_absolute_target_deg'][1] = 0
    assert args == original
