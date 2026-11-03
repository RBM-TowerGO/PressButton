import copy
import json

import pytest

from pressb.agent_loop.wrist_direction_rules import build_hint


def location(**wrist_changes):
    wrist = dict(coordinate_units='normalized_01', button_visibility='full',
                 contact_visibility='full', button_box=[.4, .3, .6, .5],
                 contact_tip=[.7, .6])
    wrist.update(wrist_changes)
    return dict(observation_id='current-7', coordinate_units='normalized_01', wrist=wrist)


@pytest.mark.parametrize('tip,signs', [
    ([.7, .6], dict(Y=1, Z=1)),   # Button left and above the physical tip.
    ([.3, .6], dict(Y=-1, Z=1)),  # Button right and above.
    ([.7, .2], dict(Y=1, Z=-1)),  # Button left and below.
    ([.3, .2], dict(Y=-1, Z=-1)), # Button right and below.
])
def test_four_quadrants_use_robot_signs_not_image_motion_signs(tip, signs):
    hint = build_hint(location(contact_tip=tip))
    assert hint['signs'] == signs
    assert hint['can_assess_interval'] is True
    assert hint['observation_id'] == 'current-7'
    assert hint['camera'] == 'wrist'
    assert hint['source'] == 'program_rule_applied_to_model_estimates_not_visual_truth'
    assert 'nonzero millimeter magnitudes' in hint['instruction']
    assert 'do not substitute negative Z for positive Z' in hint['instruction']


@pytest.mark.parametrize('tip,signs', [
    ([.4, .3], dict(Y=0, Z=0)), ([.6, .5], dict(Y=0, Z=0)),
    ([.5, .4], dict(Y=0, Z=0)), ([.4, .2], dict(Y=0, Z=-1)),
    ([.7, .5], dict(Y=1, Z=0)),
])
def test_interval_boundaries_are_inclusive_without_center_matching(tip, signs):
    assert build_hint(location(contact_tip=tip))['signs'] == signs


@pytest.mark.parametrize('field,bad', [
    ('button_box', None), ('button_box', [.4, .3, .4, .5]),
    ('button_box', [.6, .3, .4, .5]), ('button_box', [.4, .5, .6, .3]),
    ('button_box', [.4, .3, .6, .3]), ('button_box', [.4, .3, 1.01, .5]),
    ('button_box', [.4, .3, float('nan'), .5]),
    ('button_box', [.4, .3, float('inf'), .5]), ('button_box', [.4, .3]),
    ('contact_tip', None), ('contact_tip', [-.1, .6]),
    ('contact_tip', [.7, float('-inf')]), ('contact_tip', [True, .6]),
    ('contact_tip', ['.7', .6]), ('contact_tip', '[0.7, 0.6]'),
])
def test_bad_geometry_stays_unknown_and_json_safe(field, bad):
    hint = build_hint(location(**{field: bad}))
    assert hint['signs'] == dict(Y=None, Z=None)
    assert hint['can_assess_interval'] is False
    assert hint[field] is None
    assert 'unknown is not zero' in hint['caveat']
    json.dumps(hint, allow_nan=False)


@pytest.mark.parametrize('field', ['button_visibility', 'contact_visibility'])
@pytest.mark.parametrize('visibility', ['unknown', 'not_visible', 'yes', None, ['full']])
def test_unreadable_visibility_does_not_infer_signs_from_numbers(field, visibility):
    hint = build_hint(location(**{field: visibility}))
    assert hint['signs'] == dict(Y=None, Z=None)
    assert not hint['can_assess_interval']


@pytest.mark.parametrize('units', [None, 'pixels', 'normalized_1000', ['normalized_01']])
def test_missing_or_noncanonical_shared_units_are_not_guessed(units):
    value = location()
    value['coordinate_units'] = units
    hint = build_hint(value)
    assert hint['signs'] == dict(Y=None, Z=None)
    assert hint['coordinate_units'] is None
    assert hint['button_box'] is hint['contact_tip'] is None


def test_wrist_units_must_agree_but_can_inherit_explicit_shared_units():
    value = location(coordinate_units='normalized_1000')
    assert not build_hint(value)['can_assess_interval']
    del value['wrist']['coordinate_units']
    assert build_hint(value)['signs'] == dict(Y=1, Z=1)
    del value['coordinate_units']
    assert not build_hint(value)['can_assess_interval']


@pytest.mark.parametrize('oid', [None, '', '  ', 7])
def test_missing_observation_binding_produces_unknown_signs(oid):
    value = location()
    value['observation_id'] = oid
    hint = build_hint(value)
    assert hint['observation_id'] is None
    assert hint['signs'] == dict(Y=None, Z=None)


def test_head_cannot_replace_missing_wrist_and_unreadable_inputs_are_tolerated():
    value = location()
    value['head'] = value.pop('wrist')
    for candidate in (value, None, [], 'not a location'):
        hint = build_hint(candidate)
        assert hint['signs'] == dict(Y=None, Z=None)
        assert not hint['can_assess_interval']


@pytest.mark.parametrize('tip,expected', [([.7, .6], dict(Y=1, Z=1)),
                                          ([.5, .4], dict(Y=0, Z=0))])
def test_partial_face_keeps_conditional_visible_interval_without_forward_permission(tip, expected):
    hint = build_hint(location(button_visibility='partial', contact_visibility='partial', contact_tip=tip))
    assert hint['signs'] == expected
    assert hint['can_assess_interval'] is True
    assert hint['button_visibility'] == hint['contact_visibility'] == 'partial'
    assert 'visible portion, not the full face' in hint['caveat']
    assert 'tip is only partly visible' in hint['caveat']
    assert 'not positive-X permission' in hint['caveat']
    assert not {'allowed', 'forward_permission', 'xyz_mm'} & hint.keys()


def test_pure_hint_does_not_mutate_or_alias_the_localization():
    value = location()
    original = copy.deepcopy(value)
    hint = build_hint(value)
    assert value == original
    assert hint == build_hint(value)
    hint['button_box'][0] = 0
    hint['contact_tip'][0] = 0
    assert value == original
