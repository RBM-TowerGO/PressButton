"""Compare image errors from recorded coordinates, never synthesize a motion."""
import copy
import math


FORMAT = 'pressb-spatial-progress-v1'
SOURCE = 'program_comparison_of_model_coordinates'


def _numbers(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != size or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        return None
    return list(value)


def _geometry(assessment, view):
    value = (assessment or {}).get(view) or {}
    if value.get('button_visibility') != 'full' or value.get('contact_visibility') != 'full':
        return None, None
    error = _numbers((value.get('derived_relation') or {}).get('error_button_units'), 2)
    box = _numbers(value.get('button_box'), 4)
    tip = _numbers(value.get('contact_tip'), 2)
    if box is None or tip is None or min(box) < 0 or max(box) > 1 or min(tip) < 0 or max(tip) > 1:
        return None, None
    width, height = box[2]-box[0], box[3]-box[1]
    return (error, (width, height)) if min(width, height) > 0 else (None, None)


def compare(previous, current, previous_state, current_state, commands):
    """Compare an actual interval separately for HEAD and WRIST.

    commands contains only the actual commands sent in this interval. Empty means
    observation-only. X approach is not required to reduce image alignment error.
    """
    previous, current = previous or {}, current or {}
    result = dict(format=FORMAT, source=SOURCE,
                  previous_observation_id=previous.get('observation_id'),
                  observation_id=current.get('observation_id'), views={}, needs_recheck=False,
                  limitation='Image estimates only; not physical arrival, contact, activation or a robot direction.')
    commands = commands if isinstance(commands, list) else []
    actual = [{k: copy.deepcopy(v) for k, v in c.items() if k in ('type', 'xyz_mm', 'head_deg', 'key')}
              for c in commands if isinstance(c, dict)]
    result['actual_commands'] = actual
    moves = [_numbers(c.get('xyz_mm'), 3) for c in actual if c.get('type') == 'move']
    valid_moves = [values for values in moves if values is not None and any(values)]
    alignment_motion = any(values[1] or values[2] for values in valid_moves)
    result['has_alignment_motion'] = bool(alignment_motion)
    for view in ('head', 'wrist'):
        entry = dict(comparable=False, reason='no_previous_localization', axes={}, needs_recheck=False)
        result['views'][view] = entry
        if not previous:
            continue
        if previous.get('observation_id') == current.get('observation_id'):
            entry['reason'] = 'same_observation'
            continue
        reference = current.get('contact_reference')
        if reference not in ('white_arrow', 'red_marker', 'selected_fingertip') or previous.get('contact_reference') != reference:
            entry['reason'] = 'contact_identity_unknown_or_changed'
            continue
        if len(actual) != len(commands) or any(c.get('type') not in ('move', 'head', 'gripper') for c in actual) or any(v is None for v in moves):
            entry['reason'] = 'invalid_recorded_command'
            continue
        if any(c.get('type') == 'gripper' for c in actual):
            entry['reason'] = 'gripper_geometry_changed'
            continue
        if view == 'head':
            old_head = _numbers((previous_state or {}).get('head_deg'), 2)
            new_head = _numbers((current_state or {}).get('head_deg'), 2)
            if old_head is None or new_head is None or max(abs(a-b) for a, b in zip(old_head, new_head)) > 1.0:
                entry['reason'] = 'measured_head_pose_unknown_or_changed'
                continue
            if any(c.get('type') == 'head' for c in actual):
                entry['reason'] = 'head_command_changes_head_baseline'
                continue
        old, old_size = _geometry(previous, view)
        new, new_size = _geometry(current, view)
        if old is None or new is None or old_size is None or new_size is None:
            entry['reason'] = 'object_identity_or_geometry_unknown'
            continue
        ratios = [max(a, b)/min(a, b) for a, b in zip(
            (*old_size, old_size[0]*old_size[1]), (*new_size, new_size[0]*new_size[1]))]
        if max(ratios) > 1.2 and not math.isclose(max(ratios), 1.2, abs_tol=1e-12):
            entry['reason'] = 'button_scale_or_shape_changed_over_20_percent'
            continue
        if not valid_moves:
            entry['reason'] = 'no_recorded_nonzero_hand_move'
            continue
        entry.update(comparable=True, reason='same_identifiable_contact_and_comparable_projection')
        for axis, before, after in zip(('horizontal', 'vertical'), old, new):
            change = abs(after)-abs(before)
            flip = before*after < 0 and min(abs(before), abs(after)) > 0.5
            worse = change > max(0.5, abs(before)*0.25)
            trend = 'unchanged' if abs(change) <= 0.25 or math.isclose(abs(change), .25, abs_tol=1e-12) else 'worsened' if change > 0 else 'improved'
            entry['axes'][axis] = dict(previous_error_button_units=before, current_error_button_units=after,
                                      absolute_error_change_button_units=change, trend=trend,
                                      large_sign_reversal=flip, substantial_worsening=worse)
            # Projection is not an X-approach metric. It cannot authorize or veto X.
            entry['needs_recheck'] |= bool(alignment_motion and (flip or worse))
        result['needs_recheck'] |= entry['needs_recheck']
    return result
