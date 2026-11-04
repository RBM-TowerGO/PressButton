"""Advisory: use the head camera when the wrist cannot see the button.

The wrist camera sits behind the fingers, so the button face is hidden exactly when
the hand closes in. The head camera is not occluded by the hand and is the only view
that can show the target light. This pure helper reads canonical model estimates and
the measured head angle; it never inspects images, generates XYZ, calls the robot or
gates execution. It only tells the model which view is currently usable.
"""
import math

FORMAT = 'pressb-head-view-advice-v1'
SOURCE = 'program_rule_applied_to_model_estimates_not_visual_truth'
LOOK_DOWN_REFERENCE_PITCH_DEG = 30.0
USABLE = ('full', 'partial')


def _visibility(location, camera, field):
    view = location.get(camera) if isinstance(location, dict) else None
    value = view.get(field) if isinstance(view, dict) else None
    return value if value in ('full', 'partial', 'not_visible', 'unknown') else 'unknown'


def _pitch(state):
    head = state.get('head_deg') if isinstance(state, dict) else None
    if (isinstance(head, (list, tuple)) and len(head) == 2 and type(head[1]) in (int, float)
            and math.isfinite(head[1])):
        return float(head[1])
    return None


def build_advice(location, state):
    location = location if isinstance(location, dict) else {}
    wrist = _visibility(location, 'wrist', 'button_visibility')
    head = _visibility(location, 'head', 'button_visibility')
    pitch = _pitch(state)
    wrist_ok, head_ok = wrist == 'full', head in USABLE
    if not head_ok:
        situation = 'head_cannot_see_button'
        instruction = ('The HEAD view does not currently show the button. The wrist view loses the button '
                       'when the finger closes in, and only the head view can show the target light. '
                       'Before any forward press, and whenever the wrist button is partial or lost, '
                       'lower the head (absolute pitch; about +30 degrees is a reference, go further if the '
                       'measured pitch is already near it and the button is still absent) so the panel and '
                       'hand are visible together. Keep Y/Z alignment on the wrist while it is usable.')
    elif not wrist_ok:
        situation = 'wrist_lost_button_head_sees_it'
        instruction = ('The WRIST button is partial, hidden or unknown, but the HEAD view shows it. '
                       'The hand itself usually hides the wrist view, so do not withdraw or reacquire only '
                       'because the wrist lost it: judge contact, whether the target light is on, and the '
                       'remaining gap from the HEAD view, and keep that view useful. Head-image offsets '
                       'are not wrist Y/Z signs.')
    else:
        situation = 'both_views_usable'
        instruction = ('Both views show the button. Keep the head view useful for the press stage '
                       'so the light can be seen when the wrist view is hidden by the hand.')
    return dict(format=FORMAT, source=SOURCE, observation_id=location.get('observation_id'),
                situation=situation, wrist_button_visibility=wrist, head_button_visibility=head,
                measured_head_pitch_deg=pitch, look_down_reference_pitch_deg=LOOK_DOWN_REFERENCE_PITCH_DEG,
                instruction=instruction + ' Advisory only: no XYZ is generated and nothing is gated; '
                                          'the head locations are model estimates.')
