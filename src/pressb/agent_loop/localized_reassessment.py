"""Detect repeated executed motion and request a read-only hypothesis review.

Image errors are model estimates, never calibrated XYZ or contact measurements.
The only program decision here is whether a fresh review is required. No robot
client is imported, and no motion is generated, clipped, or substituted.
"""
import copy
import math

from .grounding import _enum, _image_content, _inspect, _observation_parts, _request_body, _tool
from .model_context import text
from . import model_localization as locator

FORMAT = 'pressb-hypothesis-review-v1'
REPEAT_COUNT = 3
MIN_RELATIVE_IMPROVEMENT = .10
MIN_BUTTON_UNITS_IMPROVEMENT = .10
AXES = ('X', 'Y', 'Z')


def actual(result):
    result = result or {}
    replay = result.get('replay') or {}
    command = replay.get('recorded_actual_command') if replay else result.get('command')
    return {k: copy.deepcopy(v) for k, v in (command or {}).items()
            if k in ('type', 'xyz_mm', 'head_deg', 'key')}


def _error(location, view):
    data = (location or {}).get(view) or {}
    if (data.get('button_visibility') != 'full' or data.get('contact_visibility') != 'full'):
        return None
    relation = locator.derive(data, data.get('image_size'))
    values = relation.get('error_button_units')
    if not values or any(v is None or not math.isfinite(v) for v in values):
        return None
    return values


def monitor(turns, localizations, current_id):
    """Inspect actual completed commands, including replay's recorded commands.

An observation-only round neither increments nor clears a motion streak.
Duplicate command IDs do not count twice. Head/gripper changes and failed or
interrupted motions end the comparable interval. Partial boxes are not used as
full-button centers. Thresholds only schedule review; they never approve motion.
"""
    moves, events, seen = [], [], set()
    for turn in reversed(turns):
        ledger = turn.get('ledger', turn)
        result = ledger.get('result') or {}
        command = actual(result)
        if not command:
            continue
        if result.get('status') != 'completed' or result.get('sends_completed') is not True:
            break
        cid = result.get('command_id')
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        if command.get('type') != 'move':
            break
        xyz = command.get('xyz_mm')
        if (not isinstance(xyz, list) or len(xyz) != 3 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in xyz)):
            break
        event = dict(command_id=cid, before_observation_id=ledger['before_observation_id'],
                     after_observation_id=ledger['after_observation_id'], command=command)
        events.append(event)
        if any(xyz):
            moves.append(event)
    candidates = []
    for axis, name in enumerate(AXES):
        streak = []
        sign = None
        for item in moves:
            value = item['command']['xyz_mm'][axis]
            if not value or (sign is not None and value * sign <= 0):
                break
            sign = 1 if value > 0 else -1
            streak.append(item)
        if len(streak) >= REPEAT_COUNT:
            candidates.append((axis, name, sign, streak))
    signal = dict(format=FORMAT, source='program_review_trigger_not_contact_judgment',
                  observation_id=current_id, review_required=False, reasons=[],
                  repeat_count=REPEAT_COUNT, windows=[])
    for axis, name, sign, streak in candidates:
        window = list(reversed(streak[:REPEAT_COUNT]))
        baseline = window[0]['before_observation_id']
        before, after = localizations.get(baseline), localizations.get(current_id)
        views = {}
        for view in locator.VIEWS:
            old, new = _error(before, view), _error(after, view)
            comparable = bool(old is not None and new is not None and before.get('contact_reference') == after.get('contact_reference'))
            effect = 'unknown'
            if comparable:
                a, b = math.hypot(*old), math.hypot(*new)
                margin = max(MIN_BUTTON_UNITS_IMPROVEMENT, a * MIN_RELATIVE_IMPROVEMENT)
                effect = 'improved' if a-b >= margin else 'worsened' if b-a >= margin else 'unchanged'
            views[view] = dict(before_error_button_units=old, current_error_button_units=new,
                               projection_effect=effect)
        effects = [item['projection_effect'] for item in views.values()]
        reasons = []
        if 'improved' not in effects or 'worsened' in effects:
            reasons.append('repeated_'+name+'_without_consistent_projection_improvement')
        # Even decreasing 2D error cannot establish a forward contact/activation.
        if name == 'X' and sign > 0:
            reasons.append('repeated_forward_requires_contact_hypothesis_review')
        if reasons:
            signal['windows'].append(dict(axis=name, sign=sign, count=len(streak),
                total_mm=sum(item['command']['xyz_mm'][axis] for item in streak),
                baseline_observation_id=baseline, commands=window, views=views))
            signal['reasons'].extend(reasons)
    if signal['windows']:
        signal['review_required'] = True
        signal['baseline_observation_id'] = signal['windows'][0]['baseline_observation_id']
        # Test a reviewed decision's prediction against its own command boundary.
        for turn in reversed(turns):
            ledger = turn.get('ledger', turn)
            command = actual(ledger.get('result'))
            if command.get('type') != 'move' or not any(command.get('xyz_mm', [])):
                continue
            if ledger.get('reassessment'):
                signal['previous_prediction'] = copy.deepcopy(ledger['reassessment'])
                signal['baseline_observation_id'] = ledger['before_observation_id']
            break
        events.reverse()
        start = next(i for i, event in enumerate(events) if event['before_observation_id'] == signal['baseline_observation_id'])
        signal['commands_between_images'] = events[start:]
    return signal


PROPERTIES = dict(
    observed_effect=_enum('improved', 'unchanged', 'worsened', 'unknown'),
    distance_only_hypothesis=_enum('supported', 'unsupported', 'unknown'),
    light_change=_enum('yes', 'no', 'unknown'),
    prediction_check=_enum('not_supplied', 'met', 'not_met', 'unknown'),
    evidence=dict(type='string'),
    alternative_explanation=dict(type='string'),
    next_check=dict(type='string'),
)
TOOL = _tool('review_motion_hypothesis', 'Read-only test of repeated motion against BEFORE/CURRENT images.', PROPERTIES)
PROMPT = '''Independently compare BEFORE and CURRENT HEAD/WRIST images of the
down-call button and the white-arrow physical finger end. Recorded commands were
actually sent; that alone proves neither alignment, contact nor success.
Recorded xyz_mm is chassis-frame millimeters: +X forward, -X withdraw,
+Y robot-left, -Y robot-right, +Z raise, -Z lower. Z is NOT forward/retraction.
Test the assumption that repeating the same motion will reach the button. For
forward motion, is insufficient forward distance really the only remaining issue?
Use the per-camera canonical coordinates as fallible estimates. Image x is right,
y down. Partial box centers are not full button centers. Unknown remains unknown.
Discuss changed visibility, lateral/vertical projection error and the marked
physical end. Do not silently accept earlier claims of alignment. A visible button,
larger button, or smaller fixed-point depth is not evidence of contact alignment.
Neither is a smaller error_button_units by itself: its denominator is a model
estimated button-box size. Compare the SAME physical button and finger end in
the images and pixel offsets; changing boxes, tip identity or clipping may make
the comparison unknown. A restored view alone does not justify reversing the
motion that restored it while the target remains on the same side.
light_change=yes requires this pair visibly showing this target turn on. No light
change does not by itself prove misalignment. Explain evidence in one short
sentence, give an alternative explanation and a falsifiable visual next_check.
If previous_prediction is supplied, explicitly test its expected_visual_change
against this image pair and use prediction_check=met/not_met/unknown. State in
evidence whether the prior assumption survived this check; do not merely repeat
the intended effect. Without a previous prediction use not_supplied.
Do not select XYZ, invent a pixel-to-millimeter transform, or issue robot commands.
Return review_motion_hypothesis only. All conclusions are model judgments.'''


def build_request(goal, before, current, signal, localizations):
    old_id = _observation_parts(before)[0]
    oid = _observation_parts(current)[0]
    if old_id != signal.get('baseline_observation_id') or oid != signal.get('observation_id') or old_id == oid:
        raise ValueError('hypothesis_review_observation_mismatch')
    commands = signal['commands_between_images']
    if not commands or commands[0]['before_observation_id'] != old_id:
        raise ValueError('hypothesis_review_command_boundary_mismatch')
    content = [text(dict(goal=goal, review_trigger=signal,
                        actual_commands=commands, movement='Recorded commands, not newly proposed actions.',
                        before_locations=locator.model_view(localizations[old_id]),
                        current_locations=locator.model_view(localizations[oid])))]
    content += _image_content(before, 'BEFORE') + _image_content(current, 'CURRENT')
    return _request_body(PROMPT, content, TOOL)


def validate(response, before_id, observation_id, has_prediction=False):
    choices = response.get('choices') or []
    if len(choices) != 1 or choices[0].get('finish_reason') != 'tool_calls':
        raise ValueError('one_complete_hypothesis_review_required')
    calls = choices[0].get('message', {}).get('tool_calls') or []
    if len(calls) != 1 or calls[0].get('function', {}).get('name') != 'review_motion_hypothesis':
        raise ValueError('one_read_only_hypothesis_tool_required')
    data = locator._object(calls[0]['function'].get('arguments'))
    if not isinstance(data, dict) or set(data) != set(PROPERTIES):
        raise ValueError('hypothesis_review_fields_required')
    for key, spec in PROPERTIES.items():
        if not isinstance(data[key], str) or not data[key].strip() or ('enum' in spec and data[key] not in spec['enum']):
            raise ValueError('invalid_hypothesis_review:'+key)
    if (data['prediction_check'] == 'not_supplied') == has_prediction:
        raise ValueError('hypothesis_prediction_check_required' if has_prediction else 'no_previous_prediction_supplied')
    return dict(format=FORMAT, source='model_not_human', review_id='review-'+observation_id,
                observation_id=observation_id, previous_observation_id=before_id,
                evidence_observation_ids=[before_id, observation_id], **data)


def inspect(base, body, observation_id, before_id, directory, cancel=None):
    import json
    payload = json.loads(body['messages'][1]['content'][0]['text'])
    has_prediction = bool(payload['review_trigger'].get('previous_prediction'))
    return _inspect(base, body, observation_id, directory,
                    lambda r: validate(r, before_id, observation_id, has_prediction),
                    'review_motion_hypothesis', cancel, before_id)


REASSESSMENT_PROPERTIES = dict(
    observation_id=dict(type='string'), review_id=dict(type='string'),
    response_to_review=_enum('accept', 'disagree_with_image_evidence'),
    evidence=dict(type='string', description='Current camera/object evidence; explain any disagreement explicitly.'),
    expected_visual_change=dict(type='string', description='What should visibly change after this action, beyond send completion?'),
    if_no_change=dict(type='string', description='Which assumption will be reconsidered if that expected change is absent?'),
)


def extend_tools(tools, review):
    if not review:
        return tools
    for item in tools:
        spec = item['function']
        if spec['name'] in ('report_done', 'report_unable'):
            continue
        spec['parameters']['properties']['reassessment'] = dict(type='object',
            properties=copy.deepcopy(REASSESSMENT_PROPERTIES), required=list(REASSESSMENT_PROPERTIES), additionalProperties=False)
        spec['parameters']['required'].append('reassessment')
    return tools


def validate_reassessment(data, review):
    if not isinstance(data, dict) or set(data) != set(REASSESSMENT_PROPERTIES):
        raise ValueError('fresh_reassessment_required: cite current review and state evidence, expected_visual_change, if_no_change')
    for key, spec in REASSESSMENT_PROPERTIES.items():
        if not isinstance(data[key], str) or not data[key].strip() or ('enum' in spec and data[key] not in spec['enum']):
            raise ValueError('invalid_reassessment:'+key)
    if data['observation_id'] != review['observation_id'] or data['review_id'] != review['review_id']:
        raise ValueError('reassessment_observation_mismatch')
    return copy.deepcopy(data)
