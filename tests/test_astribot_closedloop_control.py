import numpy as np
import pytest
from astribot_closedloop_control import validate_prediction, interpolate_target, physical_hit, ARM, GRIP, SIGNS, FIXED


def prediction():
    a=np.arange(56,dtype=np.float32).reshape(7,8)/100
    fixed=np.arange(32,dtype=np.float32)/100
    full=np.tile(fixed,(7,1));full[:,ARM]=a[:,:7];full[:,GRIP]=a[:,7:8]*SIGNS
    return a,full,fixed,np.tile([-3.,3.],(32,1))


def test_mapping_and_limits():
    a,full,fixed,limits=prediction()
    validate_prediction(a,full,fixed,limits)
    for index in (ARM[0],GRIP[0],FIXED[0]):
        changed=full.copy();changed[0,index]+=.1
        with pytest.raises(AssertionError):validate_prediction(a,changed,fixed,limits)
    limits[ARM[0],1]=.1
    with pytest.raises(ValueError):validate_prediction(a,full,fixed,limits)


def test_four_tick_absolute_command_interpolation():
    start=np.full(32,1.,dtype=np.float32);end=np.full(32,2.,dtype=np.float32)
    actual=interpolate_target(start,end)
    np.testing.assert_array_equal(actual[:,0],[1.25,1.5,1.75,2.])
    np.testing.assert_array_equal(actual[-1],end)


def test_success_requires_simultaneous_force_and_displacement():
    cfg=dict(press_threshold=.001,contact_threshold_n=.02)
    assert physical_hit(.001,.021,cfg)
    assert not physical_hit(.002,0.,cfg)
    assert not physical_hit(0.,1.,cfg)
    assert not physical_hit(.001,.02,cfg)
