import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('real_export',Path(__file__).resolve().parents[1]/'scripts/export_s1b_real_lerobot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def source(t):
    n=len(t);s=np.zeros((n,25));c=np.zeros((n,25))
    s[:,15]=np.arange(n)+100;c[:,15]=np.arange(n)
    c[:,22]=np.where(np.arange(n)%2,100,0)
    f={'time':np.asarray(t),'joints_dict/joints_position_state':s,'joints_dict/joints_position_command':c}
    for k in ['joints_dict/state_timestamp','joints_dict/command_timestamp','images_dict/head/rgb_timestamp','images_dict/right/rgb_timestamp']:f[k]=np.asarray(t)
    return f

def test_binary_threshold_and_invalid():
    np.testing.assert_array_equal(m.binary_gripper([0,49.999,50,100]),[0,0,1,1])
    for v in [np.nan,np.inf,-.1,100.1]:
        with pytest.raises(ValueError):m.binary_gripper([v])

def test_future_targets_tail_and_measured_state():
    a=m.build_arrays(source(np.arange(12)/30))
    np.testing.assert_array_equal(a['action'][:,0],np.arange(1,12))
    np.testing.assert_array_equal(a['observation.state'][:,0],np.arange(11)+100)
    assert a['valid_anchor'].sum()==5
    np.testing.assert_array_equal(a['action'][:7,0],np.arange(1,8))
    assert not a['valid_anchor'][-6:].any()

def test_gap_and_qc_invalidate_windows():
    t=np.r_[np.arange(12)/30,1+np.arange(12)/30]
    a=m.build_arrays(source(t))
    for i in np.flatnonzero(a['valid_anchor'][:,0]):
        rows=a['source_observation_row'][i:i+7,0]
        assert not (rows.min()<12<=a['source_action_row'][i+6,0])
    f=source(np.arange(12)/30);f['joints_dict/command_timestamp']=f['time'].copy();f['joints_dict/command_timestamp'][4]+=.1
    assert m.build_arrays(f)['valid_anchor'].sum()==0

def test_split_and_constant_normalization():
    split=m.split_episodes(96)
    assert len(split['train'])==80 and len(split['validation'])==16
    assert not set(split['train']) & set(split['validation'])
    a=m.build_arrays(source(np.arange(12)/30));a['action'][:,8:]=[.1,.2]
    stats=m.normalization([a])['action']
    assert stats['std'][7]==1 and stats['mean'][7]==0
    assert stats['constant'][8:]==[True,True]
    x=(a['action']-stats['mean'])/stats['std']
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x[:,8:],np.zeros((11,2)))
