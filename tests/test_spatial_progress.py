import copy

import pytest

from pressb.agent_loop import spatial_localization as loc, spatial_progress as progress
from test_spatial_localization import SIZES, located, raw, unknown


def assessment(oid, tip=(.65, .55), **changes):
    point = located(contact_tip=list(tip), **changes)
    return loc.validate_raw(raw(oid, head=point, wrist=point), oid, SIZES)


def compare(before=None, after=None, commands=None, old_head=(0, 30), new_head=(0, 30)):
    return progress.compare(before or assessment('a'), after or assessment('b', (.55, .45)),
                            dict(head_deg=list(old_head)), dict(head_deg=list(new_head)),
                            [dict(type='move', xyz_mm=[0, 0, 10])] if commands is None else commands)


def test_progress_uses_coordinates_and_does_not_generate_robot_actions():
    result = compare()
    assert all(result['views'][view]['comparable'] for view in ('head', 'wrist'))
    assert result['views']['wrist']['axes']['vertical']['trend'] == 'improved'
    assert result['views']['head']['axes']['horizontal']['trend'] == 'improved'
    assert not result['needs_recheck']
    assert not any(key in result for key in ('xyz_mm', 'command', 'step_hint_mm'))


def test_head_motion_resets_only_head_when_interval_contains_hand_move():
    result = compare(new_head=(0, 35), commands=[dict(type='move', xyz_mm=[0, 0, 10]), dict(type='head', head_deg=[0,35])])
    assert not result['views']['head']['comparable']
    assert result['views']['wrist']['comparable']
    assert compare(new_head=(0,31))['views']['head']['comparable']


def test_gripper_and_ambiguous_identity_invalidate_comparison():
    result = compare(commands=[dict(type='move', xyz_mm=[0, 0, 10]), dict(type='gripper', key='o')])
    assert all(v['reason'] == 'gripper_geometry_changed' for v in result['views'].values())
    after = assessment('b'); after['contact_reference'] = 'red_marker'
    assert not compare(after=after)['views']['wrist']['comparable']
    after = loc.validate_raw(raw('b', head=unknown(), wrist=located()), 'b', SIZES)
    assert not compare(after=after)['views']['head']['comparable']
    assert compare(after=after)['views']['wrist']['comparable']


def test_scale_change_is_unknown_not_direction_reversal():
    after = assessment('b', button_box=[.3, .1, .5, .3])
    result = compare(after=after)
    assert all(v['reason'] == 'button_scale_or_shape_changed_over_20_percent' for v in result['views'].values())
    assert not result['needs_recheck']


def test_large_sign_reversal_and_worsening_trigger_alignment_review():
    reversed_point = assessment('b', (.15, .0))
    result = compare(after=reversed_point)
    assert result['needs_recheck']
    assert result['views']['wrist']['axes']['vertical']['large_sign_reversal']
    result = compare(after=assessment('b', (.8, .9)))
    assert result['needs_recheck']
    assert result['views']['head']['axes']['vertical']['substantial_worsening']


def test_x_approach_projection_is_not_a_gate_or_review_metric():
    result = compare(after=assessment('b', (.8, .9)), commands=[dict(type='move', xyz_mm=[10,0,0])])
    assert result['views']['wrist']['axes']['vertical']['trend'] == 'worsened'
    assert not result['has_alignment_motion'] and not result['needs_recheck']
    assert 'allowed' not in result


def test_no_motion_missing_state_and_input_immutability():
    before, after = assessment('a'), assessment('b')
    original = copy.deepcopy([before, after])
    result = progress.compare(before, after, {}, {}, [])
    assert result['views']['head']['reason'] == 'measured_head_pose_unknown_or_changed'
    assert result['views']['wrist']['reason'] == 'no_recorded_nonzero_hand_move'
    assert [before, after] == original
    assert progress.compare(None, after, {}, {}, [])['views']['head']['reason'] == 'no_previous_localization'


def test_unknown_partial_geometry_does_not_compare_old_known_points():
    after = assessment('b', button_visibility='partial')
    result = compare(after=after)
    assert not any(v['comparable'] for v in result['views'].values())


@pytest.mark.parametrize('side', ['before', 'after'])
@pytest.mark.parametrize('field,value', [('usable_views', ['wrist']), ('recovery_views', ['head'])])
def test_review_disabled_coordinates_do_not_become_progress_evidence(side, field, value):
    before, after = assessment('a'), assessment('b', (.55, .45))
    target = before if side == 'before' else after
    target[field] = value
    result = compare(before=before, after=after)
    assert not result['views']['head']['comparable']
    assert result['views']['head']['reason'] == 'object_identity_or_geometry_unknown'
    assert result['views']['wrist']['comparable']
    assert result['views']['wrist']['axes']['vertical']['trend'] == 'improved'
    # Preserve the coordinates in the caller's archive without trusting them.
    assert target['head']['contact_tip'] is not None


def test_legacy_geometry_without_availability_lists_remains_compatible():
    before, after = assessment('a'), assessment('b', (.55, .45))
    del before['usable_views']
    del after['usable_views']
    result = compare(before=before, after=after)
    assert all(view['comparable'] for view in result['views'].values())
