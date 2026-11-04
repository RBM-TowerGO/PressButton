"""Phased permission for the merged-gripper press: align open, close, then press.

The merged gripper end is a fixed wrist-camera pixel. While the gripper is OPEN
that pixel shows the button face, so the depth read there is the button surface
distance z_b; it is frozen once trusted and the button covers the pixel.

After CLOSING, the 2026-11-04 run showed the aim pixel still returns valid depth
that falls about 1:1 with forward motion, so the press is closed loop on live
depth. The target is the measured contact reading (150 mm) minus an overpress,
because the button must be pushed in, not just touched:

  ALIGN  positive X needs the merged-end pixel inside the current button box
  PRESS  positive X needs both
         live:   x <= z_live - (tip_plane - overpress)     (this observation only)
         frozen: x <= z_b - (tip_plane - overpress) - stroke used since the freeze,
                 clamped to a hard maximum

The stroke used is the MEASURED X of this observation minus the measured X at the
freeze. Under contact the arm stops a few mm short of its command (2026-11-04:
46 mm commanded, 41.7 mm measured, depth 151 mm), so counting commands would spend
stroke the hand never travelled. Only without measured poses is the commanded net
stroke since closing used.

The frozen bound is the ceiling: a live reading can only make the stroke
shorter, so a depth that stalls or reads the wrong surface never pushes beyond
what the open-gripper range allowed. Without a trusted live reading for the
current observation only the frozen bound applies, and that is recorded.

This module holds only program rules over recorded numbers. It never inspects
images, generates XYZ, calls a robot or changes a command.
"""
import copy
import math

FORMAT = 'pressb-press-phase-v2'
SOURCE = 'program_rule_on_live_depth_within_frozen_range'
ALIGN, PRESS, BLOCKED = 'align', 'press', 'blocked'

# Measured 2026-11-04: with the gripper closed and the button pressed, the right
# wrist camera reads 150 mm at the aim pixel. This is the calibrated contact plane.
TIP_PLANE_M = 0.150
DEFAULT_LIMITS = dict(overpress_m=0.005, max_travel_m=0.085,
                      range_min_m=TIP_PLANE_M + 0.003, range_max_m=TIP_PLANE_M + 0.080,
                      live_min_m=TIP_PLANE_M - 0.020,
                      min_valid_pixels=5, aim_tolerance_px=8)

_state = None


def configure(**limits):
    """Override defaults and reset the run. Unknown keys are rejected."""
    global _state
    unknown = set(limits) - set(DEFAULT_LIMITS)
    if unknown:
        raise ValueError('unknown_press_limit:'+','.join(sorted(unknown)))
    merged = dict(DEFAULT_LIMITS, **limits)
    for key, value in merged.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid_press_limit:'+key)
    if merged['max_travel_m'] <= 0 or merged['overpress_m'] >= TIP_PLANE_M:
        raise ValueError('invalid_press_limit')
    if merged['range_min_m'] >= merged['range_max_m']:
        raise ValueError('invalid_press_range_bounds')
    if merged['live_min_m'] >= merged['range_max_m']:
        raise ValueError('invalid_press_live_bounds')
    _state = dict(limits=merged, gripper_state='open', z_b_m=None, frozen_from=None,
                  consumed_m=0.0, live=None, freeze_x_m=None, measured={},
                  retry_step=None, completed_presses=0)
    return describe()


def _measured_x(observation):
    """Measured chassis X of the end effector in this observation, or None."""
    pose = ((observation or {}).get('state') or {}).get('pose_m_xyzw')
    if not isinstance(pose, (list, tuple)) or len(pose) != 7:
        return None
    x = pose[0]
    return float(x) if type(x) in (int, float) and math.isfinite(x) else None


def _consumed(observation_id):
    """Forward stroke used since the freeze, and how it was obtained.

    Measured: this observation's measured X minus the measured X at the freeze.
    Under contact the arm falls a few mm short of its command, so the commanded sum
    would spend stroke the hand never travelled. Without both measured poses the
    commanded net stroke since closing is used instead.
    """
    start = _state['freeze_x_m']
    now = _state['measured'].get(observation_id) if observation_id is not None else None
    if start is not None and now is not None:
        return now - start, 'measured_pose'
    return _state['consumed_m'], 'commanded'


def reset():
    global _state
    _state = None


def active():
    return _state is not None


def target_m():
    """Live depth reading at which the press is complete: contact plane minus overpress."""
    limits = _state['limits'] if _state else DEFAULT_LIMITS
    return TIP_PLANE_M - limits['overpress_m']


def set_gripper(state):
    """Record an executed c/o. Closing freezes nothing by itself; see observe."""
    if _state is None:
        return
    if state not in ('closed', 'open'):
        raise ValueError('press_gripper_state_must_be_closed_or_open')
    if state == 'open':
        # Reopening starts a new attempt: realign and measure a new range.
        _state.update(z_b_m=None, frozen_from=None, consumed_m=0.0, live=None, freeze_x_m=None,
                      retry_step=None)
    _state['gripper_state'] = state


# After a completed closed press the next steps are fixed, whatever the light shows:
#   withdraw  only propose_move xyz_mm = RETRY_WITHDRAW_MM (the end leaves the panel)
#   reopen    report_done if the light is still on, otherwise open the gripper (o) and
#             realign from fresh localization, then close and press again. No closed
#             re-press, Y/Z shift or re-close in between; further -X withdrawal is allowed.
# Opening clears the step. Program rule on recorded depth and commands only.
RETRY_WITHDRAW_MM = (-10, 0, 0)
WITHDRAW, REOPEN = 'withdraw', 'reopen'


def retry_step():
    return _state['retry_step'] if _state else None


def note_press_status(remaining_m):
    """Start the retry sequence once the closed press has reached its target."""
    if (_state is None or _state['gripper_state'] != 'closed' or _state['z_b_m'] is None
            or _state['retry_step'] is not None or remaining_m is None or remaining_m > 0.0005):
        return retry_step()
    _state['retry_step'] = WITHDRAW
    _state['completed_presses'] += 1
    return WITHDRAW


def check_retry(proposal):
    """Raise ValueError (repairable rejection) for an action outside the retry sequence."""
    step = retry_step()
    if step is None:
        return
    name, args = proposal.get('name'), proposal.get('arguments') or {}
    if step == WITHDRAW:
        if name == 'propose_move' and list(args.get('xyz_mm') or []) == list(RETRY_WITHDRAW_MM):
            return
        raise ValueError(
            'press_retry_withdraw_required: the closed press reached its target. The next action must be '
            f'exactly propose_move xyz_mm={list(RETRY_WITHDRAW_MM)} (withdraw 10 mm, no Y/Z), whether or '
            'not the light looks on. Return that command now.')
    if name in ('report_done', 'report_unable', 'request_observation', 'propose_head'):
        return
    if name == 'propose_gripper' and args.get('key') == 'o':
        return
    xyz = list(args.get('xyz_mm') or [])
    if name == 'propose_move' and len(xyz) == 3 and xyz[0] < 0 and xyz[1] == 0 and xyz[2] == 0:
        return
    raise ValueError(
        'press_retry_reopen_required: the hand has withdrawn 10 mm after a completed press. If the target '
        'light is clearly on in the current HEAD image, report_done. Otherwise do NOT press, shift or '
        'close again with the gripper closed: open the gripper now (propose_gripper key o); the program '
        'then localizes the button again so you can realign, close and press.')


def record_retry(proposal):
    """Advance the sequence after a command was executed."""
    if _state is None or _state['retry_step'] != WITHDRAW:
        return retry_step()
    if (proposal.get('name') == 'propose_move'
            and list((proposal.get('arguments') or {}).get('xyz_mm') or []) == list(RETRY_WITHDRAW_MM)):
        _state['retry_step'] = REOPEN
    return retry_step()


def _selection_px(observation, image_size):
    selection = (observation or {}).get('selection') or {}
    point = selection.get('point_normalized')
    size = list(image_size or [])
    if (not isinstance(point, list) or len(point) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
        return None
    if len(size) != 2 or any(type(v) not in (int, float) or v <= 0 for v in size):
        return None
    return [point[0]*(size[0]-1), point[1]*(size[1]-1)]


def _sample_reasons(observation, aim_px, image_size, limits):
    """Checks shared by the open freeze and the closed live reading."""
    depth = (observation or {}).get('depth') or {}
    reasons = []
    median = depth.get('median_m')
    if (type(median) not in (int, float)) or not math.isfinite(median):
        median = None
    if median is None:
        reasons.append('no_depth_sample')
    if depth.get('status') not in (None, 'ok'):
        reasons.append('depth_status:'+str(depth.get('status')))
    valid = depth.get('valid_pixels')
    if type(valid) is not int or valid < limits['min_valid_pixels']:
        reasons.append('too_few_valid_depth_pixels')
    if aim_px is not None:
        point = _selection_px(observation, image_size)
        if point is None:
            reasons.append('measurement_point_unknown')
        else:
            gap = math.hypot(point[0]-aim_px[0], point[1]-aim_px[1])
            if gap > limits['aim_tolerance_px']:
                reasons.append('measurement_point_not_on_merged_end:%.1fpx' % gap)
    else:
        reasons.append('no_merged_end_calibration')
    return median, reasons


def observe(observation, aim_px=None, image_size=None, aligned=None):
    """Record this observation's depth at the merged-end pixel.

    Open gripper, no range yet: freeze z_b once trusted and the button box covers
    the aim pixel; never overwritten afterwards.
    Closed gripper with a frozen range: keep it as the live reading for THIS
    observation only. The button box is not required (closed tips may hide part of
    the button), but the same pixel, validity and plausible-range checks apply.
    """
    if _state is None:
        return None
    limits = _state['limits']
    median, reasons = _sample_reasons(observation, aim_px, image_size, limits)
    oid = (observation or {}).get('observation_id')
    measured_x = _measured_x(observation)
    if oid is not None and measured_x is not None:
        _state['measured'][oid] = measured_x
    closed_press = _state['gripper_state'] == 'closed' and _state['z_b_m'] is not None
    if closed_press:
        if median is not None and not limits['live_min_m'] <= median <= limits['range_max_m']:
            reasons.append('live_range_implausible:%.4fm' % median)
    else:
        if median is not None and not limits['range_min_m'] <= median <= limits['range_max_m']:
            reasons.append('range_outside_close_band:%.4fm' % median)
        if aligned is not True:
            # Depth at the aim pixel describes the BUTTON only when the button covers it;
            # otherwise it may be the panel or background and must not set the stroke.
            reasons.append('aim_point_not_inside_current_button_box')
    trusted = not reasons
    result = dict(observation_id=oid, gripper_state=_state['gripper_state'], median_m=median,
                  trusted=trusted, reasons=reasons, role='live' if closed_press else 'freeze')
    if closed_press:
        _state['live'] = dict(observation_id=oid, median_m=median, trusted=trusted, reasons=list(reasons))
    elif trusted and _state['gripper_state'] == 'open' and _state['z_b_m'] is None:
        _state['z_b_m'] = float(median)
        _state['frozen_from'] = oid
        _state['freeze_x_m'] = measured_x
        _state['consumed_m'] = 0.0
        result['froze'] = True
    return result


def phase():
    if _state is None:
        return BLOCKED
    if _state['z_b_m'] is None:
        return ALIGN
    return PRESS


def _budget():
    limits = _state['limits']
    raw = _state['z_b_m'] - target_m()
    return max(0.0, min(raw, limits['max_travel_m']))


def _live_for(observation_id):
    live = _state['live']
    if live is None or observation_id is None or live['observation_id'] != observation_id:
        return None
    return live


def evaluate(x_mm=0.0, observation_id=None):
    """Program decision for one positive-X request; the caller keeps its own XYZ.

    ``observation_id`` binds the live depth: a reading from any other observation
    (before the last move, or before closing) is never used.
    """
    value = dict(format=FORMAT, source=SOURCE, allowed=False, phase=phase(),
                 tip_plane_m=TIP_PLANE_M, target_m=target_m(), z_b_m=None, frozen_from=None,
                 budget_m=0.0, consumed_m=0.0, frozen_remaining_m=0.0, live_m=None,
                 live_from=None, live_remaining_m=None, bound='none', remaining_m=0.0,
                 requested_m=0.0, reasons=[])
    if _state is None:
        value['reasons'].append('press_phase_not_configured')
        return value
    state = _state
    value.update(gripper_state=state['gripper_state'], z_b_m=state['z_b_m'],
                 frozen_from=state['frozen_from'], consumed_m=state['consumed_m'])
    if state['z_b_m'] is None:
        value['reasons'].append('no_trusted_open_gripper_range')
        return value
    budget = _budget()
    consumed, consumed_source = _consumed(observation_id)
    value.update(consumed_m=consumed, consumed_source=consumed_source,
                 commanded_consumed_m=state['consumed_m'])
    frozen_remaining = max(0.0, budget - consumed)
    remaining, bound = frozen_remaining, 'frozen_only_no_live_depth'
    live = _live_for(observation_id)
    if live is not None:
        value.update(live_from=live['observation_id'], live_m=live['median_m'],
                     live_reasons=list(live['reasons']))
        if live['trusted']:
            live_remaining = max(0.0, live['median_m'] - target_m())
            value['live_remaining_m'] = live_remaining
            remaining = min(frozen_remaining, live_remaining)
            bound = 'live' if live_remaining <= frozen_remaining else 'frozen_ceiling'
        else:
            bound = 'frozen_only_live_depth_untrusted'
    requested = max(0.0, float(x_mm))/1000.0
    value.update(budget_m=budget, frozen_remaining_m=frozen_remaining, remaining_m=remaining,
                 bound=bound, requested_m=requested)
    if state['gripper_state'] != 'closed':
        value['reasons'].append('gripper_not_closed')
    if requested <= 0:
        value['reasons'].append('not_a_forward_request')
    if requested > remaining + 1e-9:
        value['reasons'].append('stroke_exceeds_press_limit:%.4fm>%.4fm(%s)' % (requested, remaining, bound))
    value['allowed'] = not value['reasons']
    return value


def record_move(xyz_mm):
    """Track net forward X since closing, after a command was actually sent.

    Withdrawal returns budget, so pressing again after backing off is measured from
    the frozen surface, not from the furthest point ever reached. Withdrawing past
    the freeze point makes this negative: the surface is then that much further away,
    so a repeated press can still reach it. Live depth still bounds every stroke.
    """
    if _state is None or _state['gripper_state'] != 'closed':
        return 0.0 if _state is None else _state['consumed_m']
    try:
        x = float(list(xyz_mm)[0])
    except (TypeError, ValueError, IndexError):
        return _state['consumed_m']
    if math.isfinite(x):
        _state['consumed_m'] = _state['consumed_m'] + x/1000.0
    return _state['consumed_m']


def describe(observation_id=None):
    if _state is None:
        return None
    value = dict(format=FORMAT, phase=phase(), gripper_state=_state['gripper_state'],
                 z_b_m=_state['z_b_m'], frozen_from=_state['frozen_from'],
                 budget_m=_budget() if _state['z_b_m'] is not None else 0.0,
                 consumed_m=_state['consumed_m'], limits=copy.deepcopy(_state['limits']),
                 tip_plane_m=TIP_PLANE_M, target_m=target_m(), retry_step=_state['retry_step'],
                 completed_presses=_state['completed_presses'])
    if observation_id is not None and _state['z_b_m'] is not None:
        current = evaluate(0.0, observation_id)
        value.update(remaining_m=current['remaining_m'], bound=current['bound'],
                     live_m=current['live_m'], live_from=current['live_from'],
                     live_remaining_m=current['live_remaining_m'],
                     consumed_m=current['consumed_m'], consumed_source=current['consumed_source'])
    return value
