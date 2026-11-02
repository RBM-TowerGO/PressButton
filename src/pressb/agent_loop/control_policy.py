"""Render current visual judgments as control advice, never robot commands.

Image errors are measured in the model's estimated button dimensions.  Neither
these estimates nor the suggested step sizes are calibrated robot geometry.
"""
import copy
import math


FORMAT = 'pressb-control-plan-v1'
DEFAULT_CONTACT = 'head_image_right'
_FINGERS = ('head_image_right', 'head_image_left')
_STEPS = {'small': 5, 'medium': 10, 'large': 20, 'unknown': None}


def _numbers(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) for x in value):
        return None
    return list(value)


def _head(state):
    return _numbers((state or {}).get('head_deg'), 2)


def _stable(a, b):
    return a is not None and b is not None and max(abs(x-y) for x, y in zip(a, b)) <= 1.0


def _geometry(assessment):
    geometry = (assessment or {}).get('geometry') or {}
    error = _numbers(geometry.get('error_button_units'), 2)
    box = _numbers(geometry.get('head_button_box'), 4)
    if box is None or not all(0 <= x <= 1 for x in box):
        return error, None
    width, height = box[2]-box[0], box[3]-box[1]
    return error, (width, height) if min(width, height) > 0 else None


def _magnitude(error):
    if error is None:
        return 'unknown'
    size = abs(error)
    if size <= 0.5 or math.isclose(size, 0.5, rel_tol=0, abs_tol=1e-12):
        return 'small'
    if size <= 2 or math.isclose(size, 2, rel_tol=0, abs_tol=1e-12):
        return 'medium'
    return 'large'


def _progress(current, state, previous, previous_state, command):
    result = dict(comparable=False, reason='no_previous_assessment', needs_recheck=False,
                  previous_observation_id=(previous or {}).get('observation_id'),
                  current_observation_id=current.get('observation_id'), axes={},
                  source='comparison_of_model_estimates_not_measured_hand_error')
    if not previous:
        return result
    if (current.get('contact_finger') not in _FINGERS
            or previous.get('contact_finger') != current.get('contact_finger')):
        result['reason'] = 'selected_finger_unknown_or_changed'
        return result
    if current.get('head_visible') != 'yes' or previous.get('head_visible') != 'yes':
        result['reason'] = 'head_target_not_visible_in_both'
        return result
    if not _stable(_head(state), _head(previous_state)):
        result['reason'] = 'measured_head_pose_unknown_or_changed'
        return result
    xyz = _numbers((command or {}).get('xyz_mm'), 3)
    if (command or {}).get('type') != 'move' or xyz is None or not any(xyz):
        result['reason'] = 'no_recorded_nonzero_hand_move'
        return result
    now, current_size = _geometry(current)
    before, previous_size = _geometry(previous)
    if now is None or before is None or current_size is None or previous_size is None:
        result['reason'] = 'geometry_missing_or_invalid'
        return result
    # A scale/aspect jump can be a locator error or perspective change.  Neither
    # permits a reliable conclusion about hand correction from these two frames.
    current_scale = (*current_size, current_size[0]*current_size[1])
    previous_scale = (*previous_size, previous_size[0]*previous_size[1])
    ratios = [max(a, b)/min(a, b) for a, b in zip(current_scale, previous_scale)]
    if max(ratios) > 1.2:
        result['reason'] = 'button_scale_or_shape_changed_over_20_percent'
        return result
    result.update(comparable=True, reason='same_finger_stable_head_and_comparable_button_scale',
                  actual_command=dict(type='move', xyz_mm=copy.deepcopy(xyz)))
    for axis, old, new in zip(('horizontal', 'vertical'), before, now):
        change = abs(new)-abs(old)
        flip = old*new < 0 and min(abs(old), abs(new)) > 0.5
        worse = change > max(0.5, abs(old)*0.25)
        status = 'unchanged' if abs(change) <= 0.25 else 'worsened' if change > 0 else 'improved'
        result['axes'][axis] = dict(previous_error_button_units=old,
            current_error_button_units=new, absolute_error_change_button_units=round(change, 4),
            trend=status, large_sign_reversal=flip, substantial_worsening=worse)
        result['needs_recheck'] |= flip or worse
    return result


def _head_info(last_target, state, view_inadequate):
    last_target = last_target if isinstance(last_target, dict) else {}
    target = _numbers(last_target.get('head_deg'), 2)
    measured = _head(state)
    prior_measured = _numbers(last_target.get('measured_head_deg'), 2)
    return dict(last_absolute_target_deg=target, current_measured_head_deg=measured,
        measured_after_last_target_deg=prior_measured,
        last_target_observation_id=last_target.get('observation_id'),
        command_id=last_target.get('command_id'),
        measured_pose_stable_since_last_target=(_stable(measured, prior_measured)
            if measured is not None and prior_measured is not None else None),
        avoid_repeating_same_target=bool(view_inadequate and target is not None),
        guidance=('Choose head yaw/pitch from current framing. +30 degrees pitch is an '
                  'absolute look-down example, not a fixed target or an increment. If '
                  'this view remains inadequate after the last target, choose a different '
                  'sensible target; repeating it does not provide a new viewpoint. '
                  'Head rotation does not move the hand or change the wrist camera.'))


def build_plan(current_assessment, current_state, previous_assessment=None,
               previous_state=None, latest_command=None, evidence=None,
               last_head_target=None):
    """Return advisory stages from recorded observations, with explicit uncertainty.

    ``latest_command`` must be the command actually executed between observations,
    not a replay suggestion or prose reason.  No result here approves a movement;
    the independent current-wrist horizontal guard still checks positive X.
    """
    current, state = current_assessment or {}, current_state or {}
    evidence = evidence or {}
    error, _ = _geometry(current)
    sizes = dict(zip(('horizontal', 'vertical'),
        [_magnitude(x) for x in error] if error is not None else ['unknown', 'unknown']))
    horizontal, vertical = current.get('head_horizontal', 'unknown'), current.get('head_vertical', 'unknown')
    offsets = dict(horizontal=horizontal in ('left', 'right'), vertical=vertical in ('above', 'below'))
    visible = current.get('head_visible') == 'yes' and current.get('contact_finger') in _FINGERS
    view_inadequate = not visible or (not any(offsets.values())
        and (horizontal != 'center' or vertical != 'level'))
    progress = _progress(current, state, previous_assessment, previous_state, latest_command)
    plan = dict(format=FORMAT, observation_id=current.get('observation_id'),
        source='program_advice_from_model_judgments_not_human_confirmation',
        preferred_contact=DEFAULT_CONTACT, selected_contact=current.get('contact_finger', 'unknown'),
        stage=None, focus=None, alignment_size=sizes, step_hint_mm=None,
        geometry_progress=progress, head_target_info=_head_info(last_head_target, state, view_inadequate),
        evidence=copy.deepcopy({k: evidence.get(k) for k in ('light_on', 'withdrawal')}),
        expected_change=None, instructions=[])
    if evidence.get('light_on') and evidence.get('withdrawal'):
        plan.update(stage='complete', expected_change='report_this_runs_light_and_later_withdrawal_evidence')
        plan['instructions'].append('Report completion with this run\'s light-on and subsequent withdrawal observation IDs; these remain model judgments, not human confirmation.')
    elif evidence.get('light_on'):
        plan.update(stage='withdraw', expected_change='subsequent_visual_withdrawal_evidence')
        plan['instructions'].append('The recorded light event supports moving to withdrawal; obtain a later visual observation showing withdrawal before reporting complete.')
    elif not visible:
        plan.update(stage='get_view', expected_change='head_target_and_selected_finger_identifiable')
        plan['instructions'].append('Improve HEAD framing to locate the button and preferred HEAD-image RIGHT finger; a fresh observation alone does not alter the viewpoint.')
    elif progress['needs_recheck']:
        plan.update(stage='recheck', expected_change='resolve_large_reversal_or_worsening_before_repeating_correction')
        plan['instructions'].append('Recheck the button and selected fingertip locations and the actual last command: estimated image error reversed substantially or worsened. Do not mechanically repeat or reverse a correction; explain the current visual evidence.')
    elif any(offsets.values()):
        axes = [axis for axis in ('vertical', 'horizontal') if offsets[axis]]
        # Vertical wins ties and is the fallback when legacy categories lack size.
        focus = max(axes, key=lambda axis: abs(error[0 if axis == 'horizontal' else 1])) if error is not None else axes[0]
        plan.update(stage='align', focus=focus, step_hint_mm=_STEPS[sizes[focus]],
                    expected_change=f'reduce_{focus}_image_error')
        plan['instructions'].append(f'Address the {focus} image offset and inspect its change after the actual movement. Check the other axis too; a coarse relation that remains unchanged does not establish lack of progress.')
        if plan['step_hint_mm'] is not None:
            plan['instructions'].append(f'Consider {plan["step_hint_mm"]} mm as an adjustable trial step for this {sizes[focus]} visual error; this is not a calibrated pixel-to-mm conversion, minimum, cap, or forced action.')
    elif not view_inadequate:
        plan.update(stage='approach', expected_change='new_contact_or_light_evidence_from_current_views')
        plan['instructions'].append('Current HEAD projection is centered/level; it does not prove 3D contact alignment. Assess a separate approach using live views and the independent current-WRIST horizontal check.')
    else:
        plan.update(stage='get_view', expected_change='resolve_unknown_head_alignment')
        plan['instructions'].append('The remaining HEAD alignment is unknown; obtain a view that identifies the target relative to the selected finger.')
    plan['instructions'] += [
        'For a correction purpose focus on hand Y/Z; do not hide positive X inside a correction label. Robot RIGHT is -Y and UP is +Z, but image right is not automatically robot right.',
        'Stages and step sizes are advice, not new execution gates. A separately justified approach is still allowed when the independent current-WRIST horizontal-between-fingers guard passes; no height or depth validity prerequisite is added.',
        'Open single-finger contact is allowed. Unknown depth or an unconfirmed light event does not itself forbid forward motion. Sending a command is not proof of arrival, contact, or success.',
    ]
    return plan
