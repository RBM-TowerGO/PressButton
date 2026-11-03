"""Conditional wrist direction hints from canonical model estimates.

This pure helper does not inspect images, generate XYZ, call a robot, cap motion,
or gate execution. The current image must still establish the objects' identities.
"""
import math


FORMAT = 'pressb-wrist-direction-hint-v1'
SOURCE = 'program_rule_applied_to_model_estimates_not_visual_truth'


def _coordinates(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    if any(type(item) not in (int, float) or not 0 <= item <= 1
           or not math.isfinite(item) for item in value):
        return None
    if length == 4 and (value[0] >= value[2] or value[1] >= value[3]):
        return None
    return list(value)


def _visibility(value):
    return value if isinstance(value, str) and value in (
        'full', 'partial', 'not_visible', 'unknown') else 'unknown'


def _interval_sign(low, high, tip):
    return 1 if high < tip else -1 if low > tip else 0


def build_hint(location):
    """Bind interval-based Y/Z signs to one canonical wrist observation.

    A partial box can support a conditional rule about its visible portion, not
    the complete button. Unknown geometry or visibility never becomes sign zero.
    Head locations cannot supply missing wrist evidence.
    """
    location = location if isinstance(location, dict) else {}
    wrist = location.get('wrist')
    wrist = wrist if isinstance(wrist, dict) else {}
    oid = location.get('observation_id')
    oid = oid if isinstance(oid, str) and oid.strip() else None
    units_valid = (location.get('coordinate_units') == 'normalized_01'
                   and wrist.get('coordinate_units', 'normalized_01') == 'normalized_01')
    box = _coordinates(wrist.get('button_box'), 4) if units_valid else None
    tip = _coordinates(wrist.get('contact_tip'), 2) if units_valid else None
    button_visibility = _visibility(wrist.get('button_visibility'))
    contact_visibility = _visibility(wrist.get('contact_visibility'))
    visibility_known = all(value in ('full', 'partial')
                           for value in (button_visibility, contact_visibility))
    can_assess = bool(oid is not None and units_valid and visibility_known
                      and box is not None and tip is not None)
    signs = dict(Y=None, Z=None)
    if can_assess:
        signs = dict(Y=_interval_sign(box[0], box[2], tip[0]),
                     Z=_interval_sign(box[1], box[3], tip[1]))

    caveats = ['Applies only to the present unmirrored wrist mounting and preserved hand orientation.']
    if button_visibility == 'partial':
        caveats.append('The button box covers only the visible portion, not the full face; '
                       'its interval signs require confirmation in the current image.')
    if contact_visibility == 'partial':
        caveats.append('The physical tip is only partly visible; confirm its identity and endpoint.')
    if not can_assess:
        caveats.append('Missing or invalid observation binding, normalized_01 units, geometry or '
                       'known visibility makes the signs unknown; unknown is not zero.')
    caveats.append('This is not positive-X permission or proof of alignment/contact. '
                   'The executor is unchanged: no XYZ is generated, no displacement cap or hard gate is added.')
    if can_assess:
        instruction = (f'For this observation, use Y={signs["Y"]:+d} and Z={signs["Z"]:+d} '
                       'if the CURRENT WRIST image confirms the same button face and marked physical tip. '
                       'Choose nonzero millimeter magnitudes for nonzero signs; use zero only for an axis '
                       'whose tip is already within the reported interval. '
                       'If Z=+1, raise with positive Z; do not substitute negative Z for positive Z. '
                       'Reacquire or reassess if the image contradicts these estimates.')
    else:
        instruction = ('For this observation, Y/Z signs are unknown, not zero. Reacquire or reassess the '
                       'CURRENT WRIST button face and marked physical tip; do not fill gaps from HEAD. '
                       'Choose magnitudes using the signs only after the current image confirms identities '
                       'and usable wrist geometry.')
    return dict(format=FORMAT, source=SOURCE, observation_id=oid, camera='wrist',
                coordinate_units='normalized_01' if units_valid else None,
                button_visibility=button_visibility, contact_visibility=contact_visibility,
                button_box=box, contact_tip=tip, can_assess_interval=can_assess,
                signs=signs, caveat=' '.join(caveats), instruction=instruction)
