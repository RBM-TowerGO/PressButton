import json

import pytest

from pressb.agent_loop import calibrated_tip, model_localization as locator, wrist_overlap_guard as guard

CALIBRATION = 'docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json'
SIZES = dict(head=[1280, 720], wrist=[640, 360])


@pytest.fixture(autouse=True)
def _reset():
    calibrated_tip.deactivate()
    yield
    calibrated_tip.deactivate()


def _raw(tip=None):
    wrist = dict(button_visibility='full', contact_visibility='unknown' if tip is None else 'full',
                 button_box=[100, 50, 200, 150], arrow_box=None, contact_tip=tip, note='')
    head = dict(button_visibility='unknown', contact_visibility='unknown', button_box=None,
                arrow_box=None, contact_tip=None, note='')
    return dict(coordinate_units='normalized_1000', wrist=wrist, head=head)


def test_inactive_keeps_model_tip():
    view = locator.normalize(_raw([120, 100]), 'o1', SIZES)['wrist']
    assert view['contact_tip'] == [0.12, 0.1] and 'tip_source' not in view


def test_active_overrides_wrist_tip_only_and_records_source():
    info = calibrated_tip.activate(CALIBRATION, 'open')
    out = locator.normalize(_raw([120, 100]), 'o1', SIZES)
    wrist = out['wrist']
    assert wrist['contact_tip'] == info['point_normalized'] and wrist['contact_visibility'] == 'full'
    assert wrist['tip_source']['sha256'] == info['sha256']
    assert out['head']['contact_tip'] is None
    dx = wrist['derived_relation']['dx']
    assert dx == pytest.approx(0.15-info['point_normalized'][0])


def test_aim_point_is_merged_end_and_ignores_activated_state():
    """Closing the gripper moves the finger, not the button, so the aim point is fixed."""
    closed = calibrated_tip.load(CALIBRATION, 'closed')['points']['closed']
    for state in ('closed', 'open'):
        info = calibrated_tip.activate(CALIBRATION, state)
        assert info['aim_state'] == 'closed'
        assert info['point_normalized'] == closed
        assert info['gripper_state'] == state  # recorded, but not used as the aim point


def test_state_switch_does_not_move_the_aim_point():
    calibrated_tip.activate(CALIBRATION, 'open')
    before = calibrated_tip.aim_point()
    calibrated_tip.set_state('closed')
    assert calibrated_tip.aim_point() == before
    calibrated_tip.set_state('open')
    assert calibrated_tip.aim_point() == before


def test_missing_closed_point_is_rejected(tmp_path):
    """The merged-end point is required even when the gripper starts open."""
    with pytest.raises(ValueError):calibrated_tip.load(CALIBRATION, 'half')
    data = json.load(open(CALIBRATION));del data['points']['closed']
    f = tmp_path/'c.json';f.write_text(json.dumps(data))
    for state in ('closed', 'open'):
        with pytest.raises(ValueError):calibrated_tip.load(f, state)


def test_guard_uses_calibrated_tip_and_recomputes_consistently():
    calibrated_tip.activate(CALIBRATION, 'open')
    point = calibrated_tip.current_point()
    box = [int(point[0]*1000)-30, int(point[1]*1000)-30, int(point[0]*1000)+30, int(point[1]*1000)+30]
    data = dict(coordinate_units='normalized_1000', button_visibility='full', button_box=box,
                contact_visibility='unknown', contact_tip=None, note='x')
    result = guard.validate_raw(data, 'o1', [640, 360])
    assert result['tip_inside_button'] == 'yes' and result['contact_tip_visible'] == 'yes'
    assert guard.evaluate(result, 'o1')['allowed'] is True
    far = dict(data, button_box=[0, 0, 100, 100])
    assert guard.validate_raw(far, 'o1', [640, 360])['tip_inside_button'] == 'no'


def test_prompt_note_only_when_active():
    assert 'CALIBRATED TIP' not in json.dumps(guard.build_request('g', _obs()))
    calibrated_tip.activate(CALIBRATION, 'open')
    assert 'CALIBRATED TIP' in json.dumps(guard.build_request('g', _obs()))


def _obs():
    image = dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,eA=='))
    return dict(role='user', content=[dict(type='text', text=json.dumps(dict(observation_id='o1'))), image, image])
