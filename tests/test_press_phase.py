"""The merged-gripper press: align open, close, then press within a frozen range."""
import math

import pytest

from pressb.agent_loop import calibrated_tip, press_phase as phase, wrist_overlap_guard as guard

CALIBRATION = 'docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json'


@pytest.fixture(autouse=True)
def _reset():
    calibrated_tip.deactivate()
    phase.reset()
    yield
    calibrated_tip.deactivate()
    phase.reset()


def _obs(z, oid='o1', point=None, valid=25, status='ok', size=(640, 360)):
    return dict(observation_id=oid, depth=dict(median_m=z, status=status, valid_pixels=valid),
                selection=dict(point_normalized=point if point is not None else calibrated_tip.aim_point()),
                image_size=list(size))


def _open_at(z, aligned=True, **kw):
    calibrated_tip.activate(CALIBRATION, 'open')
    phase.configure()
    return phase.observe(_obs(z, **kw), calibrated_tip.aim_pixel(), calibrated_tip.image_size(), aligned=aligned)


def _approach_done():
    """Skip the standoff-limited first approach; these tests cover the final-press bounds."""
    phase._state['pressed_since_close'] = True


def _again(z, aligned=True, **kw):
    return phase.observe(_obs(z, **kw), calibrated_tip.aim_pixel(), calibrated_tip.image_size(), aligned=aligned)


@pytest.mark.parametrize('aligned', [False, None])
def test_range_is_not_frozen_unless_the_button_covers_the_aim_point(aligned):
    """Depth at the aim pixel may be the panel until the button is aligned there."""
    result = _open_at(0.200, aligned=aligned)
    assert not result['trusted'] and 'aim_point_not_inside_current_button_box' in result['reasons']
    assert phase.phase() == phase.ALIGN
    assert _again(0.200, oid='o2')['froze']


def test_trusted_open_range_freezes_and_sets_press_phase():
    result = _open_at(0.200)
    assert result['trusted'] and result['froze']
    assert phase.phase() == phase.PRESS
    # Frozen ceiling: 200 - (150 - 5 overpress) = 55 mm
    assert phase.describe()['budget_m'] == pytest.approx(0.055)
    assert phase.target_m() == pytest.approx(0.145)


def test_later_observation_never_overwrites_the_frozen_range():
    _open_at(0.200, oid='o1')
    later = _again(0.210, oid='o2')
    assert later['trusted'] and not later.get('froze')
    assert phase.describe()['z_b_m'] == pytest.approx(0.200)
    assert phase.describe()['frozen_from'] == 'o1'


@pytest.mark.parametrize('kwargs,reason', [
    (dict(z=0.320), 'range_outside_close_band'),      # still too far to close
    (dict(z=0.140), 'range_outside_close_band'),      # closer than the tip plane
    (dict(z=None), 'no_depth_sample'),
    (dict(z=0.200, valid=1), 'too_few_valid_depth_pixels'),
    (dict(z=0.200, status='深度已过期'), 'depth_status'),
])
def test_untrusted_samples_do_not_freeze(kwargs, reason):
    result = _open_at(**kwargs)
    assert not result['trusted'] and not result.get('froze')
    assert any(r.startswith(reason) for r in result['reasons']), result['reasons']
    assert phase.phase() == phase.ALIGN


def test_measurement_point_must_sit_on_the_merged_end_pixel():
    result = _open_at(0.200, point=[0.7277, 0.7967])   # the OPEN fingertip, not the aimed end
    assert not result['trusted'] and 'measurement_point_not_on_merged_end' in ' '.join(result['reasons'])


def test_missing_calibration_never_freezes():
    phase.configure()
    result = phase.observe(_obs(0.200), None, None, aligned=True)
    assert not result['trusted'] and 'no_merged_end_calibration' in result['reasons']


def test_forward_requires_closed_gripper():
    _open_at(0.200)
    value = phase.evaluate(10)
    assert not value['allowed'] and 'gripper_not_closed' in value['reasons']
    phase.set_gripper('closed')
    assert phase.evaluate(10)['allowed']


def test_stroke_is_bounded_by_frozen_range_and_hard_ceiling():
    _open_at(0.200)
    phase.set_gripper('closed')
    _approach_done()
    assert phase.evaluate(55)['allowed']
    too_far = phase.evaluate(56)
    assert not too_far['allowed'] and 'stroke_exceeds_press_limit' in too_far['reasons'][0]
    assert too_far['bound'] == 'frozen_only_no_live_depth'


def test_hard_ceiling_binds_when_the_band_would_allow_more():
    """max_travel_m is a backstop: with the default band it is unreachable, not wrong."""
    calibrated_tip.activate(CALIBRATION, 'open')
    phase.configure(range_max_m=0.500)
    _again(0.300)
    phase.set_gripper('closed')
    assert phase.describe()['budget_m'] == pytest.approx(0.085)
    assert not phase.evaluate(90)['allowed']


def test_consumed_stroke_reduces_remaining_and_blocks_overshoot():
    _open_at(0.200)
    phase.set_gripper('closed')
    phase.record_move([55, 0, 0])
    assert phase.evaluate(1)['remaining_m'] == pytest.approx(0.0)
    assert not phase.evaluate(1)['allowed']


def test_withdrawal_returns_budget_including_beyond_the_freeze_point():
    _open_at(0.200)
    phase.set_gripper('closed')
    phase.record_move([55, 0, 0])
    phase.record_move([-20, 0, 0])
    assert phase.evaluate(20)['allowed'] and not phase.evaluate(21)['allowed']
    # Withdrawing 100 after a net 35 leaves the surface 65 mm further than at freeze,
    # so a repeated press may cover the full 55 mm budget plus those 65 mm.
    phase.record_move([-100, 0, 0])
    assert phase.describe()['consumed_m'] == pytest.approx(-0.065)
    assert phase.evaluate(120)['allowed'] and not phase.evaluate(121)['allowed']


def _posed(z, oid, x):
    value = _obs(z, oid=oid)
    value['state'] = dict(pose_m_xyzw=[x, -0.227, 1.2, 0, 0, 0.707, 0.707])
    return value


def test_stroke_is_consumed_by_measured_travel_not_by_command():
    """2026-11-04 run: 46 mm commanded, 41.8 mm measured under contact, depth 151 mm."""
    calibrated_tip.activate(CALIBRATION, 'open')
    phase.configure()
    aim, size = calibrated_tip.aim_pixel(), calibrated_tip.image_size()
    assert phase.observe(_posed(0.1914, 'freeze', 0.4069), aim, size, aligned=True)['froze']
    phase.set_gripper('closed')
    phase.record_move([46, 0, 0])
    phase.observe(_posed(0.1510, 'pressed', 0.4487), aim, size)
    value = phase.evaluate(4, 'pressed')
    assert value['consumed_source'] == 'measured_pose'
    assert value['consumed_m'] == pytest.approx(0.0418)
    assert value['commanded_consumed_m'] == pytest.approx(0.046)
    # The frozen ceiling no longer cuts the press at 0.4 mm: 46.4 - 41.8 = 4.6 mm remain,
    # still capped by live depth (151 - 145 = 6 mm).
    assert value['remaining_m'] == pytest.approx(0.0046) and value['bound'] == 'frozen_ceiling'
    assert value['allowed'] and not phase.evaluate(5, 'pressed')['allowed']
    assert phase.describe('pressed')['consumed_source'] == 'measured_pose'


def test_without_measured_pose_the_commanded_stroke_is_used():
    _open_at(0.200)
    phase.set_gripper('closed')
    phase.record_move([40, 0, 0])
    value = phase.evaluate(10, 'no-pose')
    assert value['consumed_source'] == 'commanded'
    assert value['remaining_m'] == pytest.approx(0.015)


def test_open_gripper_motion_is_not_counted_as_press_stroke():
    _open_at(0.200)
    phase.record_move([30, 0, 0])
    assert phase.describe()['consumed_m'] == 0.0


def test_rejected_stroke_is_not_consumed():
    _open_at(0.200)
    phase.set_gripper('closed')
    assert not phase.evaluate(200)['allowed']
    assert phase.describe()['consumed_m'] == 0.0


def test_press_gate_skips_the_model_and_binds_observation():
    _open_at(0.200)
    phase.set_gripper('closed')
    decision = guard.press_gate('o1', 10)
    assert decision['policy'] == guard.PRESS_POLICY and decision['allowed'] is True
    assert decision['observation_id'] == 'o1' and decision['geometry'] is None


def test_press_permission_accepts_allowed_and_rejects_overshoot():
    _open_at(0.200)
    phase.set_gripper('closed')
    xyz = [30.0, 0.0, 0.0]
    proposal = dict(name='propose_move', arguments=dict(xyz_mm=xyz))
    decision = dict(guard.press_gate('o1', xyz[0]), authorized_command=dict(type='move', xyz_mm=xyz))
    assert guard.validate_permission(proposal, dict(observation_id='o1'), decision) is None
    far = [200.0, 0.0, 0.0]
    bad = dict(guard.press_gate('o1', far[0]), authorized_command=dict(type='move', xyz_mm=far))
    with pytest.raises(ValueError, match='live_and_frozen_range'):
        guard.validate_permission(dict(name='propose_move', arguments=dict(xyz_mm=far)),
                                  dict(observation_id='o1'), bad)


def test_negative_x_and_planar_moves_need_no_press_permission():
    _open_at(0.200)
    phase.set_gripper('closed')
    for xyz in ([-5, 0, 0], [0, 10, 0], [0, 0, -10]):
        guard.validate_permission(dict(name='propose_move', arguments=dict(xyz_mm=xyz)),
                                  dict(observation_id='o1'), None)
    guard.validate_permission(dict(name='propose_gripper', arguments=dict(key='o')),
                              dict(observation_id='o1'), None)


def test_opening_resumes_ranging_and_can_freeze_a_new_range():
    _open_at(0.200)
    phase.set_gripper('closed')
    phase.record_move([55, 0, 0])
    phase.set_gripper('open')
    # Reopening starts a new attempt: the old range and stroke no longer apply.
    assert phase.phase() == phase.ALIGN
    assert phase.describe()['z_b_m'] is None and phase.describe()['consumed_m'] == 0.0
    result = _again(0.190)
    assert result['froze'] and phase.describe()['z_b_m'] == pytest.approx(0.190)
    assert phase.describe()['budget_m'] == pytest.approx(0.045)


def test_inactive_phase_passes_nothing_and_describe_is_none():
    assert phase.describe() is None and phase.phase() == phase.BLOCKED
    assert not phase.evaluate(1)['allowed']
    assert phase.observe(_obs(0.2)) is None


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        phase.configure(nonsense=1)
    with pytest.raises(ValueError):
        phase.configure(overpress_m=-1)
    with pytest.raises(ValueError):
        phase.configure(range_min_m=0.3, range_max_m=0.2)


def _closed_at(z_b=0.200):
    _open_at(z_b, oid='o0')
    phase.set_gripper('closed')


def test_live_depth_bounds_the_stroke_to_the_overpress_target():
    """2026-11-04 run: closed, 60 mm pushed, live 169.4 mm; target 145 mm."""
    _closed_at(0.2266)
    phase.record_move([60, 0, 0])
    live = _again(0.1694, oid='o5', aligned=False)    # box not required once closed
    assert live['role'] == 'live' and live['trusted'], live['reasons']
    value = phase.evaluate(0, 'o5')
    # frozen: 226.6 - 145 - 60 = 21.6 mm; live: 169.4 - 145 = 24.4 mm -> frozen binds
    assert value['bound'] == 'frozen_ceiling'
    assert value['remaining_m'] == pytest.approx(0.0216)
    assert value['live_remaining_m'] == pytest.approx(0.0244)


def test_live_depth_shortens_the_stroke_when_closer_than_commanded():
    _closed_at(0.200)
    phase.record_move([20, 0, 0])
    _again(0.150, oid='o2', aligned=False)
    value = phase.evaluate(6, 'o2')
    # live 150 - 145 = 5 mm < frozen 55 - 20 = 35 mm
    assert value['bound'] == 'live' and value['remaining_m'] == pytest.approx(0.005)
    assert not value['allowed'] and phase.evaluate(5, 'o2')['allowed']


def test_live_depth_never_extends_beyond_the_frozen_ceiling():
    """A stalled reading (arm blocked, depth not falling) must not grant more travel."""
    _closed_at(0.200)
    phase.record_move([55, 0, 0])
    _again(0.190, oid='o3', aligned=False)
    value = phase.evaluate(1, 'o3')
    assert value['remaining_m'] == pytest.approx(0.0) and not value['allowed']


def test_live_depth_from_another_observation_is_never_used():
    _closed_at(0.200)
    _again(0.150, oid='o2', aligned=False)
    assert phase.evaluate(10, 'o3')['bound'] == 'frozen_only_no_live_depth'
    assert phase.evaluate(10, 'o3')['allowed']        # frozen 55 mm still applies
    assert phase.evaluate(10, None)['live_m'] is None


@pytest.mark.parametrize('kwargs,reason', [
    (dict(z=0.100), 'live_range_implausible'),        # reads the fingertips themselves
    (dict(z=None), 'no_depth_sample'),
    (dict(z=0.170, valid=1), 'too_few_valid_depth_pixels'),
    (dict(z=0.170, point=[0.7277, 0.7967]), 'measurement_point_not_on_merged_end'),
])
def test_untrusted_live_depth_falls_back_to_the_frozen_bound(kwargs, reason):
    _closed_at(0.200)
    phase.record_move([40, 0, 0])
    result = _again(oid='o4', aligned=False, **kwargs)
    assert not result['trusted'] and any(r.startswith(reason) for r in result['reasons'])
    value = phase.evaluate(10, 'o4')
    assert value['bound'] == 'frozen_only_live_depth_untrusted'
    assert value['remaining_m'] == pytest.approx(0.015)


def test_live_reading_never_changes_the_frozen_range():
    _closed_at(0.200)
    _approach_done()
    _again(0.230, oid='o2', aligned=False)
    assert phase.describe()['z_b_m'] == pytest.approx(0.200)
    assert phase.describe('o2')['remaining_m'] == pytest.approx(0.055)


def test_overpress_is_configurable_and_bounded():
    calibrated_tip.activate(CALIBRATION, 'open')
    phase.configure(overpress_m=0.0)
    assert phase.target_m() == pytest.approx(0.150)
    with pytest.raises(ValueError):
        phase.configure(overpress_m=0.2)


def test_press_gate_binds_live_depth_to_its_observation():
    _closed_at(0.200)
    _approach_done()
    _again(0.150, oid='o2', aligned=False)
    decision = guard.press_gate('o2', 5)
    assert decision['allowed'] and decision['basis']['bound'] == 'live'
    assert decision['basis']['live_from'] == 'o2'
    xyz = [5.0, 0.0, 0.0]
    decision['authorized_command'] = dict(type='move', xyz_mm=xyz)
    # The same permission is not valid for a later observation with different evidence.
    with pytest.raises(ValueError):
        guard.validate_permission(dict(name='propose_move', arguments=dict(xyz_mm=xyz)),
                                  dict(observation_id='o3'), decision)


def test_first_closed_approach_stops_at_the_standoff():
    """2026-11-05: the first stroke reached the panel; it must stop at contact + standoff."""
    _open_at(0.211)
    phase.set_gripper('closed')
    _again(0.211, oid='o2', aligned=False)
    # Live 211 mm, approach stop 150 + 10 = 160 mm -> 51 mm, not the 66 mm to the 145 mm target.
    value = phase.evaluate(52, 'o2')
    assert value['approach'] and value['stop_m'] == pytest.approx(0.160)
    assert value['remaining_m'] == pytest.approx(0.051) and not value['allowed']
    assert 'approach_stops_at_160mm' in value['reasons'][0]
    assert phase.evaluate(51, 'o2')['allowed']
    # Frozen ceiling also ends at the stop: 211 - 160 = 51 mm without live depth.
    assert phase.evaluate(0, 'none')['remaining_m'] == pytest.approx(0.051)
    # Reaching the stop is not a completed press.
    assert phase.note_press_status(0.0) is None
    phase.record_move([51, 0, 0])
    assert not phase.in_approach() and phase.head_alignment_required()
    _again(0.160, oid='o3', aligned=False)
    # Final press: 160 - 145 = 15 mm.
    assert phase.evaluate(0, 'o3')['remaining_m'] == pytest.approx(0.015)


def test_closing_inside_the_standoff_starts_the_head_check_at_once():
    _open_at(0.158)
    phase.set_gripper('closed')
    _again(0.157, oid='o2', aligned=False)
    assert phase.in_approach() and phase.evaluate(0, 'o2')['remaining_m'] == 0
    assert phase.note_approach('o2') and phase.head_alignment_required()
    assert phase.evaluate(0, 'o2')['remaining_m'] == pytest.approx(0.012)


def test_standoff_is_configurable_and_bounded():
    phase.configure(standoff_m=0.005)
    assert phase.approach_stop_m() == pytest.approx(0.155)
    with pytest.raises(ValueError, match='standoff'):
        phase.configure(standoff_m=0.080)
