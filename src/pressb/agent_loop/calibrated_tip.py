"""Single fixed wrist-image point: the merged (closed) gripper end.

The right wrist camera is rigidly mounted, so the physical contact end of the
merged gripper is a constant image point. Closing the gripper does not move the
button or the camera; it only moves the finger. The reference for Y/Z alignment,
the wrist overlap check and the press is therefore THIS marked point, never the
tip of whichever finger happens to be visible in the current gripper state.

While the gripper is open that pixel shows the button face, so depth read there
describes the button. Once the gripper closes the merged tips fill the pixel and
depth there no longer describes the button. Gripper state is recorded for that
reason only; it never changes which point is used.

Off by default: archives, replays and old modes behave as before.
"""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 's1b-fingertip-point-v1'
STATES = ('closed', 'open')
AIM_STATE = 'closed'
SOURCE = 'human_calibrated_fixed_pixel'
PROMPT_NOTE = ('\nCALIBRATED TIP: the program supplies the wrist contact tip from a human-marked fixed '
               'pixel at the merged gripper end. Set wrist contact_visibility to "unknown" and '
               'contact_tip to null; do not locate any finger or fingertip. Locate only the button.')
_active = None


def load(path, state):
    path = Path(path).expanduser()
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('schema') != SCHEMA or data.get('camera') != 'right_wrist_rgbd':
        raise ValueError('fingertip_calibration_schema_mismatch')
    points, sizes = {}, {}
    for name, entry in data.get('points', {}).items():
        point = entry.get('point_normalized') if isinstance(entry, dict) else None
        if (name in STATES and isinstance(point, list) and len(point) == 2
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in point)):
            points[name] = [float(v) for v in point]
            size = entry.get('image_size')
            if (isinstance(size, list) and len(size) == 2
                    and all(type(v) is int and v > 0 for v in size)):
                sizes[name] = size
    if state not in STATES:
        raise ValueError('fingertip_state_must_be_closed_or_open')
    if AIM_STATE not in points:
        raise ValueError('fingertip_calibration_missing_state:'+AIM_STATE)
    return dict(points=points, aim=points[AIM_STATE], sizes=sizes,
                image_size=sizes.get(AIM_STATE), gripper_state=state, path=str(path.resolve()),
                sha256=hashlib.sha256(raw).hexdigest(), source=SOURCE)


def activate(path, state):
    global _active
    _active = load(path, state)
    return describe()


def deactivate():
    global _active
    _active = None


def set_state(state):
    """Record the executed gripper command. The aim point never changes with it."""
    if _active is None:
        return
    if state not in STATES:
        raise ValueError('fingertip_state_must_be_closed_or_open')
    _active['gripper_state'] = state


def active():
    return _active is not None


def aim_point():
    """The fixed merged-end pixel used for alignment, overlap and pressing."""
    return list(_active['aim']) if _active else None


def aim_pixel():
    """Aim point in calibrating-image pixels, or None when the size was not recorded."""
    if _active is None or _active['image_size'] is None:
        return None
    width, height = _active['image_size']
    return [_active['aim'][0]*(width-1), _active['aim'][1]*(height-1)]


def image_size():
    """Pixel size of the image the aim point was marked in."""
    return list(_active['image_size']) if _active and _active['image_size'] else None


def current_point():
    """Backward-compatible alias; the aim point is state-independent."""
    return aim_point()


def describe():
    if _active is None:
        return None
    return dict(source=SOURCE, aim_state=AIM_STATE, point_normalized=aim_point(),
                gripper_state=_active['gripper_state'], marked_states=sorted(_active['points']),
                path=_active['path'], sha256=_active['sha256'])


def apply(view):
    """Override a normalized WRIST view's contact tip in place; returns view."""
    if _active is None:
        return view
    view['contact_tip'] = aim_point()
    view['contact_visibility'] = 'full'
    view['tip_source'] = describe()
    return view
