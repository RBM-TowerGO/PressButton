import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pressb.online_rl.offset_probe import (PANEL_AXES, offsets_mm, group_split,
                                         near_mask, paired_improvement, metrics)


def test_directions_and_rigid_transform_invariance():
    center=np.array([.46,0,1.1])
    expected=np.array([8.,-4.,15.])
    tip=center+PANEL_AXES@expected/1000
    np.testing.assert_allclose(offsets_mm(tip,center),expected,atol=1e-10)
    r=Rotation.from_euler('xyz',[.3,-.6,1.2]).as_matrix()
    shift=np.array([2.,3.,-1.])
    np.testing.assert_allclose(offsets_mm(r@tip+shift,r@center+shift,r@PANEL_AXES),expected,atol=1e-10)
    # Facing the panel from -X: -Y is right, +Z is up, -X is outward.
    np.testing.assert_allclose(offsets_mm(center+[-.005,-.002,.003],center),[2,3,5])


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError): offsets_mm([np.nan,0,0],[0,0,0])
    with pytest.raises(ValueError): offsets_mm([0,0,0],[0,0,0],np.ones((3,3)))


def test_whole_scene_split_reproducible():
    groups=np.repeat(np.arange(100),75)
    a=group_split(groups)
    np.testing.assert_array_equal(a,group_split(groups))
    assert [len(np.unique(groups[a==i])) for i in range(3)]==[60,20,20]
    assert all(len(np.unique(a[groups==g]))==1 for g in np.unique(groups))


def test_group_bootstrap_does_not_overweight_long_episodes():
    groups=np.array([0]*100+[1])
    y=np.zeros((101,2));a=np.ones_like(y);b=a.copy()
    b[:100]=0;b[100]=2
    result=paired_improvement(y,a,b,groups)
    np.testing.assert_allclose(result['improvement_mm'],[0,0])
    assert result['groups']==2


def test_near_and_empty_domain():
    np.testing.assert_array_equal(near_mask(np.array([[0,0,30],[0,31,0],[0,0,-31]])),[True,False,False])
    assert metrics(np.empty((0,2)),np.empty((0,2)),np.array([]))['n']==0
    assert paired_improvement(np.empty((0,2)),np.empty((0,2)),np.empty((0,2)),np.array([]))['groups']==0


def test_capture_detects_physics_advance_and_uses_normalized_latent(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import collect_astribot_offset_probe as collector
    from pressb.online_rl.astribot_features import normalize_latent
    runtime=SimpleNamespace(q=np.zeros((1,32),np.float32),qd=np.zeros((1,32),np.float32),
                            world=SimpleNamespace(current_time=1.))
    images={'head':np.ones((2,2,3),np.uint8),'wrist_right':np.ones((2,2,3),np.uint8)}
    latent=np.arange(2048,dtype=np.float32)
    geometry=dict(label_mm=[1,2,3],contact=False,ever_contact=False,light=False)
    recorder=SimpleNamespace(geometry=lambda:geometry,physics_step=120)
    args=SimpleNamespace(seed=42,endpoint='unused')
    sample=collector.Samples(tmp_path,3,0,args)
    def frozen_predict(*args):return {'action8':np.zeros((7,8),np.float32),'vlm_latent':latent}
    monkeypatch.setattr(collector,'predict',frozen_predict)
    sample.capture(runtime,recorder,0,images=images)
    np.testing.assert_array_equal(sample.arrays[0][3],normalize_latent(latent))
    assert sample.rows[0]['sim_time']==1.
    def advancing_predict(*args):
        runtime.world.current_time+=1/120
        return frozen_predict()
    monkeypatch.setattr(collector,'predict',advancing_predict)
    with pytest.raises(RuntimeError,match='Physics advanced'):
        sample.capture(runtime,recorder,1,images=images)
    assert len(sample.rows)==1


def test_ridge_scaler_does_not_use_validation_or_test():
    from analyze_astribot_offset_probe import fit_ridge
    x=np.arange(120,dtype=float).reshape(40,3)
    y=x[:,:2]*.1
    train=np.arange(40)<20
    val=(np.arange(40)>=20)&(np.arange(40)<30)
    x[30:]+=1e6
    scaler,model,_=fit_ridge(x,y,train,val)
    np.testing.assert_allclose(scaler.mean_,x[train].mean(0))
    assert np.max(np.abs(model.predict(scaler.transform(x[val]))-y[val]))<.1
