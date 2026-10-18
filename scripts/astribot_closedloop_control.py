"""Pure control-contract checks shared by simulation and CPU tests."""
import numpy as np

ARM = np.array([5,8,11,13,15,17,19])
GRIP = np.array([24,25,26,27,30,31])
SIGNS = np.array([-1,-1,1,1,1,-1], dtype=np.float32)
FIXED = np.array([i for i in range(32) if i not in set(ARM) | set(GRIP)])


def validate_prediction(action8, action32, fixed, limits):
    a, full = np.asarray(action8), np.asarray(action32)
    if a.shape != (7,8) or full.shape != (7,32) or not np.isfinite(full).all() or not np.isfinite(a).all():
        raise ValueError('Invalid policy action shape or nonfinite output')
    np.testing.assert_array_equal(full[:,ARM], a[:,:7])
    np.testing.assert_array_equal(full[:,GRIP], a[:,7:8]*SIGNS)
    np.testing.assert_array_equal(full[:,FIXED], np.broadcast_to(np.asarray(fixed,dtype=np.float32)[FIXED], (7,len(FIXED))))
    if np.any(full < limits[:,0]-.02) or np.any(full > limits[:,1]+.02):
        raise ValueError('Predicted joint target exceeds articulation limits with 0.02 rad tolerance')
    return full.astype(np.float32, copy=False)


def interpolate_target(previous, following):
    """Exactly four future ticks; start from previous COMMAND, not measured q."""
    start, end = np.asarray(previous,dtype=np.float32), np.asarray(following,dtype=np.float32)
    return start[None] + (end-start)[None] * (np.arange(1,5,dtype=np.float32)[:,None]/4)


def physical_hit(travel, force, config):
    return bool(travel >= config['press_threshold'] and force > config['contact_threshold_n'])
