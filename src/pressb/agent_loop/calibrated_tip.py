"""Optional fixed wrist-image fingertip pixel, marked once by a human.

The right wrist camera is rigidly mounted, so for a given gripper state the
physical contact end is a constant image point. When activated, the program
supplies the WRIST contact tip and the model only locates the button. Off by
default: archives, replays and old modes behave as before.
"""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 's1b-fingertip-point-v1'
STATES = ('closed', 'open')
SOURCE = 'human_calibrated_fixed_pixel'
PROMPT_NOTE = ('\nCALIBRATED TIP: the program supplies the wrist contact tip from a human-marked fixed '
               'pixel. Set wrist contact_visibility to "unknown" and contact_tip to null; do not locate '
               'any finger or fingertip. Locate only the button.')
_active = None


def load(path, state):
    path = Path(path).expanduser()
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('schema') != SCHEMA or data.get('camera') != 'right_wrist_rgbd':
        raise ValueError('fingertip_calibration_schema_mismatch')
    points = {}
    for name, entry in data.get('points', {}).items():
        point = entry.get('point_normalized') if isinstance(entry, dict) else None
        if (name in STATES and isinstance(point, list) and len(point) == 2
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in point)):
            points[name] = [float(v) for v in point]
    if state not in STATES:
        raise ValueError('fingertip_state_must_be_closed_or_open')
    if state not in points:
        raise ValueError('fingertip_calibration_missing_state:'+state)
    return dict(points=points, state=state, path=str(path.resolve()),
                sha256=hashlib.sha256(raw).hexdigest(), source=SOURCE)


def activate(path, state):
    global _active
    _active = load(path, state)
    return describe()


def deactivate():
    global _active
    _active = None


def set_state(state):
    """Switch gripper state (after an executed c/o). Fails loudly if not calibrated."""
    if _active is None:
        return
    if state not in _active['points']:
        raise ValueError('fingertip_calibration_missing_state:'+str(state))
    _active['state'] = state


def active():
    return _active is not None


def current_point():
    return list(_active['points'][_active['state']]) if _active else None


def describe():
    if _active is None:
        return None
    return dict(source=SOURCE, state=_active['state'], point_normalized=current_point(),
                path=_active['path'], sha256=_active['sha256'])


def apply(view):
    """Override a normalized WRIST view's contact tip in place; returns view."""
    if _active is None:
        return view
    view['contact_tip'] = current_point()
    view['contact_visibility'] = 'full'
    view['tip_source'] = describe()
    return view
